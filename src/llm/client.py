"""call_llm: the one function every LLM task goes through.

Two backends, chosen by the model name:
- "claude-..." models: the Anthropic API (needs ANTHROPIC_API_KEY, paid per token)
- "gpt-..." models: the local Codex CLI (uses your ChatGPT login, no API key)

Either way the output is forced to match a pydantic model, validated, and logged
(prompt version, model, latency, tokens, cost, status). An invalid response is recorded
with a failure status instead of raising, so one bad answer doesn't stop a batch.
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path

import anthropic
import pydantic

DEFAULT_MODEL = "claude-opus-5-5"

# US$ per million tokens (input, output). Output tokens include thinking.
# Codex models aren't listed: they run on the ChatGPT plan, so cost_usd is NaN.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

# If a safety classifier declines a request, these models can re-run it on another model.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


def run_claude(client, system, user, schema, model, effort, max_tokens):
    """Return (text, status, input tokens, output tokens)."""
    fallback = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if model in FALLBACK_MODELS else {}
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        **fallback,
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    # "refusal": declined by a safety classifier. "max_tokens": cut off, so the JSON is incomplete.
    status = "ok" if response.stop_reason == "end_turn" else response.stop_reason
    return text, status, response.usage.input_tokens, response.usage.output_tokens


def run_codex(prompt, schema, model, effort, timeout_s=600):
    """Return (text, status, input tokens, output tokens)."""
    # Codex is a snap, and snaps can't see /tmp, so the folder goes in the home directory.
    # It's empty: Codex is a coding agent, and here it has no project files (or .env) to read.
    with tempfile.TemporaryDirectory(dir=Path.home(), prefix="codex-run-") as folder:
        Path(folder, "schema.json").write_text(json.dumps(schema))
        command = [
            "codex", "exec",
            "--ephemeral",  # don't save the conversation
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "--json",  # print events, including token usage, as JSON lines
            "--output-schema", "schema.json",
            "--output-last-message", "result.json",
            "--model", model,
            "--config", f"model_reasoning_effort={json.dumps(effort)}",
            "-",  # read the prompt from stdin
        ]
        try:
            done = subprocess.run(
                command, cwd=folder, input=prompt, text=True, capture_output=True, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired:
            return "", "timeout", 0, 0
        result = Path(folder, "result.json")
        text = result.read_text() if result.exists() else ""

    events = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]
    usage = next((event["usage"] for event in events if event["type"] == "turn.completed"), {})
    if done.returncode != 0 or not text:
        error = next((event["message"] for event in events if event["type"] == "error"), done.stderr[-1000:])
        return error, "codex_error", 0, 0

    output_tokens = usage.get("output_tokens", 0) + usage.get("reasoning_output_tokens", 0)
    return text, "ok", usage.get("input_tokens", 0), output_tokens


def call_llm(client, *, task, prompt_version, system, user, output_model, model, effort, max_tokens):
    """Return (parsed output or None, log record)."""
    schema = anthropic.transform_schema(output_model)  # also valid for Codex's --output-schema

    start = time.perf_counter()
    if model.startswith("gpt-"):
        # Codex has no separate system prompt, so the instructions go first in the prompt.
        text, status, input_tokens, output_tokens = run_codex(f"{system}\n\n{user}", schema, model, effort)
    else:
        text, status, input_tokens, output_tokens = run_claude(client, system, user, schema, model, effort, max_tokens)
    latency = time.perf_counter() - start

    input_price, output_price = PRICES.get(model, (float("nan"), float("nan")))
    record = {
        "task": task,
        "prompt_version": prompt_version,
        "model": model,
        "effort": effort,
        "latency_s": round(latency, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": (input_tokens * input_price + output_tokens * output_price) / 1e6,
        "status": status,
        "raw_output": text,
    }
    if status != "ok":
        return None, record

    # The JSON shape is guaranteed, but limits such as 0 <= confidence <= 1 are only checked here.
    try:
        parsed = output_model.model_validate_json(text)
    except pydantic.ValidationError:
        record["status"] = "invalid_output"
        return None, record
    return parsed, record
