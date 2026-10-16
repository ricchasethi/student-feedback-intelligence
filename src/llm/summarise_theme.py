"""Task C: grounded summary of one theme, citing the ids of the comments behind every point."""

from typing import Literal

from pydantic import BaseModel

from src.llm.client import DEFAULT_MODEL, call_llm
from src.llm.prompts import (
    SUMMARISE_THEME_SYSTEM,
    SUMMARISE_THEME_VERSION,
    summarise_theme_prompt,
)


class KeyPoint(BaseModel):
    point: str
    sentiment: Literal["positive", "negative", "neutral"]
    supporting_comment_ids: list[int]


class ThemeSummary(BaseModel):
    summary: str
    key_points: list[KeyPoint]


def summarise_theme(client, theme, description, comments, model=DEFAULT_MODEL, effort="medium"):
    """`comments` maps comment_id -> text."""
    parsed, record = call_llm(
        client,
        task="summarise_theme",
        prompt_version=SUMMARISE_THEME_VERSION,
        system=SUMMARISE_THEME_SYSTEM,
        user=summarise_theme_prompt(theme, description, comments),
        output_model=ThemeSummary,
        model=model,
        effort=effort,
        max_tokens=8000,
    )
    record["theme"] = theme

    if parsed:
        # An invented id, or a point with no evidence, makes the summary unauditable.
        cited = {i for point in parsed.key_points for i in point.supporting_comment_ids}
        record["unknown_ids"] = sorted(cited - set(comments))
        record["uncited_points"] = sum(not point.supporting_comment_ids for point in parsed.key_points)
    return parsed, record
