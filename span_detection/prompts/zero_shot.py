"""Zero-shot span detection: code of conduct + article → where the article breaks it.

Chat models return JSON:
    {"violations": [{"rule_id": "...", "excerpt": "<exact words from the article>",
                     "explanation": "<one sentence>"}, ...]}
One item per place; an empty list means the article breaks no rule. The excerpt is
located in the article text afterwards (span_detection/locate.py) to get offsets.

Base models (SinLlama) score each article sentence instead: sentence_prefix()
holds the code and the article (cached once), SENTENCE_QUESTION asks YES/NO for
one sentence, and RULE_QUESTION + CANDIDATES picks the rule for a flagged sentence.
"""

from __future__ import annotations

import json
import re

from llm.prompts.per_rule import _load
from llm.prompts.zero_shot import CODE, DEFINITIONS

NAME = "zero_shot"
JSON_OUTPUT = True
MAX_OUTPUT_TOKENS = 2000

RULE_IDS = list(CODE)
CANDIDATES = [f" {r}" for r in RULE_IDS]

CONTEXT = f"""You are a media ethics reviewer for Sri Lankan newspapers.
Find the exact places where a news article breaks the Code of Professional Practice below.
Articles may be in English or Sinhala.

CODE OF CONDUCT (JSON, keyed by rule_id):
{json.dumps(CODE, ensure_ascii=False, indent=1)}

DEFINITIONS of the exception names used in the rules:
{json.dumps(DEFINITIONS, ensure_ascii=False, indent=1)}

A rule is broken when all of its violation_conditions hold and none of its exceptions apply."""

EXCERPT_RULES = """Excerpt rules:
- Copy the excerpt character for character from the article, in the article's language. Do not translate,
  paraphrase, correct spelling or add "...".
- Give the shortest part of the article that itself breaks the rule: the name of the victim or relative,
  the words describing the method, the private detail, the loaded wording. A few words up to one sentence.
- If the article breaks a rule in several places, give one item per place."""

SYSTEM = f"""{CONTEXT}

{EXCERPT_RULES}

Reply with one JSON object and nothing else:
{{"violations": [{{"rule_id": "<rule_id>",
                  "excerpt": "exact words from the article",
                  "explanation": "one short sentence in English"}}]}}
If the article breaks no rule, reply {{"violations": []}}."""

USER = """ARTICLE
Heading: {heading}

{content}

Mark every place where this article breaks the code. Reply with the JSON object."""

SENTENCE_QUESTION = """SENTENCE FROM THE ARTICLE: {sentence}
Does this sentence break a rule of the code? Answer YES or NO.
Answer:"""

RULE_QUESTION = """SENTENCE FROM THE ARTICLE: {sentence}
Which rule does this sentence break?
Broken rule:"""


def _fields(article: dict) -> dict:
    return {
        "heading": (article.get("heading") or "").strip(),
        "content": (article.get("content") or "").strip(),
    }


def build(article: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for chat models."""
    return SYSTEM, USER.format(**_fields(article))


def article_block(article: dict) -> str:
    return "ARTICLE\nHeading: {heading}\n\n{content}".format(**_fields(article))


def sentence_prefix(article: dict) -> str:
    """Code + article; SENTENCE_QUESTION / RULE_QUESTION are appended per sentence (base models)."""
    return f"{CONTEXT}\n\n{article_block(article)}\n\n"


def _rule_id(value) -> str | None:
    text = str(value or "").strip()
    if text in CODE:
        return text
    m = re.search(r"\d+(?:\.\d+)?(?:\.[a-z]+)?", text)
    return m.group(0) if m and m.group(0) in CODE else None


def parse(text: str) -> list[dict] | None:
    """[{rule_id, verdict, excerpt, explanation}], or None if the output is not the expected JSON."""
    data = _load(text)
    if data is None or not isinstance(data.get("violations"), list):
        return None
    spans, seen = [], set()
    for item in data["violations"]:
        if not isinstance(item, dict):
            continue
        rule_id = _rule_id(item.get("rule_id"))
        excerpt = str(item.get("excerpt") or "").strip()
        if rule_id is None or (rule_id, excerpt) in seen:
            continue
        seen.add((rule_id, excerpt))
        span = {
            "rule_id": rule_id,
            "verdict": "VIOLATION",
            "excerpt": excerpt,
            "explanation": str(item.get("explanation") or "").strip(),
        }
        if isinstance(item.get("p"), (int, float)):
            span["p"] = item["p"]
        spans.append(span)
    return spans
