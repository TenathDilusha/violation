"""Zero-shot prompt: code of conduct + article → YES / NO."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

NAME = "zero_shot"
JSON_OUTPUT = False
MAX_OUTPUT_TOKENS = 5

# CODE_LANG=si (set by the runners' --code-lang si) gives every prompt the Sinhala code text;
# rule ids, keys, groups and exception names are the same in both files.
CODE_LANG = os.getenv("CODE_LANG", "en")
CODE_DIR = Path(__file__).resolve().parents[2] / "code_of_conduct"
if CODE_LANG == "en":
    CODE = json.loads((CODE_DIR / "code_of_conduct.json").read_text(encoding="utf-8"))
    DEFINITIONS = json.loads((CODE_DIR / "metadata.json").read_text(encoding="utf-8"))["definitions"]
else:
    CODE = json.loads((CODE_DIR / f"code_of_conduct_{CODE_LANG}.json").read_text(encoding="utf-8"))
    DEFINITIONS = json.loads((CODE_DIR / f"definitions_{CODE_LANG}.json").read_text(encoding="utf-8"))

SYSTEM = f"""You are a media ethics reviewer for Sri Lankan newspapers.
Decide whether a news article violates the Code of Professional Practice below.
Articles may be in English or Sinhala.

CODE OF CONDUCT (JSON, keyed by rule_id):
{json.dumps(CODE, ensure_ascii=False, indent=1)}

DEFINITIONS of the exception names used in the rules:
{json.dumps(DEFINITIONS, ensure_ascii=False, indent=1)}

A rule is violated when all of its violation_conditions hold and none of its exceptions apply.
Answer with exactly one word: YES if the article violates at least one rule, otherwise NO."""

USER = """ARTICLE
Heading: {heading}

{content}

Does this article violate any rule in the code of conduct? Answer YES or NO."""


def build(article: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one article."""
    user = USER.format(
        heading=(article.get("heading") or "").strip(),
        content=(article.get("content") or "").strip(),
    )
    return SYSTEM, user


def parse(text: str) -> str:
    m = re.search(r"\b(YES|NO)\b", (text or "").upper())
    return m.group(1) if m else "UNPARSED"
