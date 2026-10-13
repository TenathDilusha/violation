"""Few-shot prompt: zero-shot instructions + 3 fixed examples in the article's language.

Examples live in few_shot_examples.json (2 violations + 1 non-violation per
language). Every call for a given language gets the identical system prompt.
The example articles are excluded from evaluation (EXCLUDE_IDS).
"""

from __future__ import annotations

import json
from pathlib import Path

from binary_detection.prompts import zero_shot

NAME = "few_shot"
JSON_OUTPUT = zero_shot.JSON_OUTPUT
MAX_OUTPUT_TOKENS = zero_shot.MAX_OUTPUT_TOKENS

EXAMPLES = json.loads(
    (Path(__file__).resolve().parent / "few_shot_examples.json").read_text(encoding="utf-8")
)
EXCLUDE_IDS = {ex["article_id"] for examples in EXAMPLES.values() for ex in examples}

USER = zero_shot.USER
parse = zero_shot.parse


def _examples_block(lang: str) -> str:
    parts = []
    for i, ex in enumerate(EXAMPLES[lang], start=1):
        article = USER.format(heading=ex["heading"], content=ex["content"])
        parts.append(f"--- EXAMPLE {i} ---\n{article}\nAnswer: {ex['answer']}")
    return "\n\n".join(parts)


SYSTEM = {
    lang: f"{zero_shot.SYSTEM}\n\nEXAMPLES of articles with the correct answer:\n\n{_examples_block(lang)}"
    for lang in EXAMPLES
}


def build(article: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt); examples match the article's language (default en)."""
    lang = article.get("lang") if article.get("lang") in SYSTEM else "en"
    user = USER.format(
        heading=(article.get("heading") or "").strip(),
        content=(article.get("content") or "").strip(),
    )
    return SYSTEM[lang], user
