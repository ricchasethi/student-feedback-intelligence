"""Tests for the LLM layer's response handling. A fake client stands in for the API, so no
network calls or API key are needed."""

import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import yaml

from src.llm.classify_theme import classify_comment, make_output_model
from src.llm.name_theme import name_theme
from src.llm.summarise_theme import summarise_theme

TAXONOMY = yaml.safe_load(Path("results/theme_taxonomy_v1.yaml").read_text())


class FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text = text
        self.stop_reason = stop_reason
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            model=kwargs["model"],
            _request_id="req_test",
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
            content=[SimpleNamespace(type="text", text=self.text)],
        )


def fake_client(output, stop_reason="end_turn"):
    text = output if isinstance(output, str) else json.dumps(output)
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(text, stop_reason)))


THEME_NAME = {"theme": "Library resources", "description": "Books.", "generic_feedback": False, "confidence": 0.9}


def test_valid_output_is_parsed_and_logged():
    client = fake_client(THEME_NAME)
    parsed, record = name_theme(client, "library, books", ["need more books"])
    assert parsed.theme == "Library resources"
    assert record["status"] == "ok"
    assert record["prompt_version"] == "name_theme_v1"
    assert record["cost_usd"] == (1000 * 4.00 + 200 * 20.00) / 1e6


def test_request_forces_json_schema_and_enables_fallback():
    client = fake_client(THEME_NAME)
    name_theme(client, "library", ["books"])
    request = client.beta.messages.request
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["fallbacks"] == "default"


def test_out_of_range_confidence_is_invalid_output():
    parsed, record = name_theme(fake_client({**THEME_NAME, "confidence": 1.5}), "library", ["books"])
    assert parsed is None
    assert record["status"] == "invalid_output"


def test_truncated_response_is_not_parsed():
    parsed, record = name_theme(fake_client('{"theme": "Libr', stop_reason="max_tokens"), "library", ["books"])
    assert parsed is None
    assert record["status"] == "max_tokens"


def test_refusal_is_recorded():
    parsed, record = name_theme(fake_client("", stop_reason="refusal"), "library", ["books"])
    assert parsed is None
    assert record["status"] == "refusal"


def test_classification_schema_only_allows_taxonomy_themes():
    theme_names = [theme["name"] for theme in TAXONOMY["themes"]]
    schema = anthropic.transform_schema(make_output_model(theme_names))
    assert schema["$defs"]["ThemeLabel"]["properties"]["theme"]["enum"] == theme_names


def test_evidence_not_in_comment_is_flagged():
    comment = "Teaching is good but paper checking is unfair"
    output = {
        "labels": [
            {"theme": "Teaching Quality", "sentiment": "positive", "evidence": "teaching is good", "confidence": 0.9},
            {"theme": "Assessment and Marking", "sentiment": "negative", "evidence": "exams are too hard", "confidence": 0.8},
        ],
        "unlisted_topic": None,
    }
    parsed, record = classify_comment(fake_client(output), comment, TAXONOMY)
    assert len(parsed.labels) == 2
    assert record["ungrounded_evidence"] == "Assessment and Marking"


def test_theme_outside_taxonomy_is_invalid_output():
    output = {
        "labels": [{"theme": "Parking", "sentiment": "negative", "evidence": "parking", "confidence": 0.9}],
        "unlisted_topic": None,
    }
    parsed, record = classify_comment(fake_client(output), "parking is bad", TAXONOMY)
    assert parsed is None
    assert record["status"] == "invalid_output"


def test_summary_with_invented_ids_is_flagged():
    output = {
        "summary": "Students want more books.",
        "key_points": [
            {"point": "Too few books.", "sentiment": "negative", "supporting_comment_ids": [3, 99]},
            {"point": "Good facilities.", "sentiment": "positive", "supporting_comment_ids": []},
        ],
    }
    comments = {3: "books are less", 7: "library facilities are good"}
    _, record = summarise_theme(fake_client(output), "Library", "Books", comments)
    assert record["unknown_ids"] == [99]
    assert record["uncited_points"] == 1


def test_codex_backend_reads_result_file_and_usage(monkeypatch):
    def fake_run(command, cwd, **kwargs):
        Path(cwd, "result.json").write_text(json.dumps(THEME_NAME))
        usage = {"input_tokens": 12000, "output_tokens": 80, "reasoning_output_tokens": 20}
        stdout = json.dumps({"type": "turn.completed", "usage": usage})
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("src.llm.client.subprocess.run", fake_run)
    parsed, record = name_theme(None, "library", ["books"], model="gpt-5.6-luna")
    assert parsed.theme == "Library resources"
    assert record["status"] == "ok"
    assert (record["input_tokens"], record["output_tokens"]) == (12000, 100)


def test_codex_error_is_recorded(monkeypatch):
    def fake_run(command, cwd, **kwargs):
        stdout = json.dumps({"type": "error", "message": "model not supported"})
        return SimpleNamespace(returncode=1, stdout=stdout, stderr="")

    monkeypatch.setattr("src.llm.client.subprocess.run", fake_run)
    parsed, record = name_theme(None, "library", ["books"], model="gpt-5.6-luna")
    assert parsed is None
    assert record["status"] == "codex_error"
    assert record["raw_output"] == "model not supported"
