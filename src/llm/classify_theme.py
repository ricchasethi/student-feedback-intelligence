"""Task B: multi-label theme x sentiment classification of one comment."""

from typing import Literal

from pydantic import BaseModel, Field

from src.llm.client import DEFAULT_MODEL, call_llm
from src.llm.prompts import (
    CLASSIFY_THEME_SYSTEM,
    CLASSIFY_THEME_VERSION,
    classify_theme_prompt,
    render_taxonomy,
)

GENERAL = "General feedback (no specific theme)"


def make_output_model(theme_names):
    # Theme names come from the taxonomy, so the JSON schema only accepts real themes.
    class ThemeLabel(BaseModel):
        theme: Literal[*theme_names]
        sentiment: Literal["positive", "negative", "neutral"]
        evidence: str
        confidence: float = Field(ge=0, le=1)

    class CommentClassification(BaseModel):
        labels: list[ThemeLabel]
        unlisted_topic: str | None

    return CommentClassification


def normalise(text):
    return " ".join(text.lower().split())


def classify_comment(client, comment, taxonomy, model=DEFAULT_MODEL, effort="low"):
    theme_names = [theme["name"] for theme in taxonomy["themes"]]
    system = CLASSIFY_THEME_SYSTEM.format(taxonomy=render_taxonomy(taxonomy), general=GENERAL)

    parsed, record = call_llm(
        client,
        task="classify_theme",
        prompt_version=CLASSIFY_THEME_VERSION,
        system=system,
        user=classify_theme_prompt(comment),
        output_model=make_output_model(theme_names),
        model=model,
        effort=effort,
        max_tokens=4000,
    )

    if parsed:
        record["unlisted_topic"] = parsed.unlisted_topic
        # The schema can't check that the evidence was really copied from the comment.
        ungrounded = [label.theme for label in parsed.labels if normalise(label.evidence) not in normalise(comment)]
        record["ungrounded_evidence"] = ", ".join(ungrounded)
    return parsed, record
