"""Zero-shot rule detection: code of conduct + article → the rules it breaks, with evidence.

Chat models return JSON:
    {"violated_rules": [{"rule_id": "...", "evidence": "<sentence from the article>",
                         "explanation": "<one sentence>"}, ...]}
An empty list means the article breaks no rule.

Base models (SinLlama) use build_choice(): the prompt ends in "Broken rule:" and
each candidate in CANDIDATES (every rule_id, plus "none") is scored; the most
likely one is the answer, so SinLlama names at most one rule.
"""

from __future__ import annotations

import json
import re

from binary_detection.prompts.per_rule import _load
from binary_detection.prompts.zero_shot import CODE, DEFINITIONS

NAME = "zero_shot"
JSON_OUTPUT = True
MAX_OUTPUT_TOKENS = 1500

RULE_IDS = list(CODE)
CANDIDATES = [f" {r}" for r in RULE_IDS] + [" none"]
EVIDENCE_SUFFIX = "\nThe sentence from the article that breaks the rule:"

_CONTEXT = f"""You are a media ethics reviewer for Sri Lankan newspapers.
Identify which rules of the Code of Professional Practice below a news article breaks.
Articles may be in English or Sinhala.

CODE OF CONDUCT (JSON, keyed by rule_id):
{json.dumps(CODE, ensure_ascii=False, indent=1)}

DEFINITIONS of the exception names used in the rules:
{json.dumps(DEFINITIONS, ensure_ascii=False, indent=1)}

A rule is broken when all of its violation_conditions hold and none of its exceptions apply."""

SYSTEM = f"""{_CONTEXT}

Reply with one JSON object and nothing else:
{{"violated_rules": [{{"rule_id": "<rule_id>",
                      "evidence": "the sentence from the article that breaks the rule, copied exactly in the article's language",
                      "explanation": "one short sentence in English"}}]}}
List every broken rule once. If the article breaks no rule, reply {{"violated_rules": []}}."""

USER = """ARTICLE
Heading: {heading}

{content}

Which rules does this article break? Reply with the JSON object."""

CHOICE_SYSTEM = f"""{_CONTEXT}

Answer with the rule_id of the rule the article breaks most clearly, or "none" if it breaks no rule."""

CHOICE_USER = """ARTICLE
Heading: {heading}

{content}

Which rule does this article break?
Broken rule:"""


def _fields(article: dict) -> dict:
    return {
        "heading": (article.get("heading") or "").strip(),
        "content": (article.get("content") or "").strip(),
    }


def build(article: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for chat models."""
    return SYSTEM, USER.format(**_fields(article))


def build_choice(article: dict) -> str:
    """Prompt ending in "Broken rule:" for candidate scoring (base models)."""
    return f"{CHOICE_SYSTEM}\n\n{CHOICE_USER.format(**_fields(article))}"


def _rule_id(value) -> str | None:
    text = str(value or "").strip()
    if text in CODE:
        return text
    m = re.search(r"\d+(?:\.\d+)?(?:\.[a-z]+)?", text)
    return m.group(0) if m and m.group(0) in CODE else None


def parse(text: str) -> list[dict] | None:
    """Broken rules as [{rule_id, verdict, evidence, explanation}], or None if not valid JSON."""
    data = _load(text)
    if data is None:
        return None
    items = data.get("violated_rules")
    if not isinstance(items, list):
        return None
    rules, seen = [], set()
    for item in items:
        if isinstance(item, str):
            item = {"rule_id": item}
        if not isinstance(item, dict):
            continue
        rule_id = _rule_id(item.get("rule_id"))
        if rule_id is None or rule_id in seen:
            continue
        seen.add(rule_id)
        rules.append({
            "rule_id": rule_id,
            "verdict": str(item.get("verdict") or "VIOLATION").upper(),
            "evidence": (item.get("evidence") or "").strip(),
            "explanation": (item.get("explanation") or "").strip(),
        })
    return rules
