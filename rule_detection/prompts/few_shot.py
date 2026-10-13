"""Few-shot rule detection: zero-shot instructions + 3 fixed examples in the article's language.

Examples live in few_shot_examples.json (2 violations + 1 non-violation per
language), each with the rules it breaks and the evidence sentence. Every call
for a given language gets the identical prompt. The example articles are
excluded from evaluation (EXCLUDE_IDS).

Base models see the same examples in the "Broken rule: <rule_id>" format
(the first listed rule, or "none").
"""

from __future__ import annotations

import json
from pathlib import Path

from rule_detection.prompts import zero_shot

NAME = "few_shot"
JSON_OUTPUT = zero_shot.JSON_OUTPUT
MAX_OUTPUT_TOKENS = zero_shot.MAX_OUTPUT_TOKENS
RULE_IDS = zero_shot.RULE_IDS
CANDIDATES = zero_shot.CANDIDATES
EVIDENCE_SUFFIX = zero_shot.EVIDENCE_SUFFIX
parse = zero_shot.parse

EXAMPLES = json.loads(
    (Path(__file__).resolve().parent / "few_shot_examples.json").read_text(encoding="utf-8")
)
EXCLUDE_IDS = {ex["article_id"] for examples in EXAMPLES.values() for ex in examples}


def _answer(ex: dict) -> str:
    return json.dumps({"violated_rules": ex["violated_rules"]}, ensure_ascii=False)


def _chat_examples(lang: str) -> str:
    return "\n\n".join(
        f"--- EXAMPLE {i} ---\n{zero_shot.USER.format(heading=ex['heading'], content=ex['content'])}\n{_answer(ex)}"
        for i, ex in enumerate(EXAMPLES[lang], start=1)
    )


def _choice_examples(lang: str) -> str:
    parts = []
    for i, ex in enumerate(EXAMPLES[lang], start=1):
        rule = ex["violated_rules"][0]["rule_id"] if ex["violated_rules"] else "none"
        parts.append(
            f"--- EXAMPLE {i} ---\n{zero_shot.CHOICE_USER.format(heading=ex['heading'], content=ex['content'])} {rule}"
        )
    return "\n\n".join(parts)


SYSTEM = {
    lang: f"{zero_shot.SYSTEM}\n\nEXAMPLES of articles with the correct reply:\n\n{_chat_examples(lang)}"
    for lang in EXAMPLES
}
CHOICE_SYSTEM = {
    lang: f"{zero_shot.CHOICE_SYSTEM}\n\nEXAMPLES of articles with the correct answer:\n\n{_choice_examples(lang)}"
    for lang in EXAMPLES
}


def _lang(article: dict) -> str:
    return article.get("lang") if article.get("lang") in EXAMPLES else "en"


def build(article: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt); examples match the article's language (default en)."""
    return SYSTEM[_lang(article)], zero_shot.USER.format(**zero_shot._fields(article))


def build_choice(article: dict) -> str:
    return f"{CHOICE_SYSTEM[_lang(article)]}\n\n{zero_shot.CHOICE_USER.format(**zero_shot._fields(article))}"
