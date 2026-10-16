"""Task A: name a topic from its keywords and representative comments."""

from pydantic import BaseModel, Field

from src.llm.client import DEFAULT_MODEL, call_llm
from src.llm.prompts import NAME_THEME_SYSTEM, NAME_THEME_VERSION, name_theme_prompt


class ThemeName(BaseModel):
    theme: str
    description: str
    generic_feedback: bool
    confidence: float = Field(ge=0, le=1)


def name_theme(client, keywords, comments, model=DEFAULT_MODEL, effort="low"):
    return call_llm(
        client,
        task="name_theme",
        prompt_version=NAME_THEME_VERSION,
        system=NAME_THEME_SYSTEM,
        user=name_theme_prompt(keywords, comments),
        output_model=ThemeName,
        model=model,
        effort=effort,
        max_tokens=4000,
    )
