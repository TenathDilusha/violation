"""Rule-based span detection: a verdict for every rule, with the exact excerpts that break it.

Same verdicts and evidence rules as llm/prompts/per_rule.py; instead of one
evidence sentence each flagged rule lists short exact excerpts:
    {"rules": {"<rule_id>": {"verdict": "...", "excerpts": ["..."], "explanation": "..."}},
     "violation": "YES|NO"}
VIOLATION rules become spans. "verdict" comes
first in each rule so OpenAI token probabilities give P(violation) per rule
(token_confidence).

SinLlama: the rules are flagged with the calibrated single-rule YES/NO prompts
(llm/sinllama.py per_rule, build_rule); then for each flagged rule every
sentence is scored with rule_sentence_prefix() + RULE_SENTENCE_QUESTION.
"""

from __future__ import annotations

import json

from llm.prompts.per_rule import (  # noqa: F401
    CONTEXT, EVIDENCE_SUFFIX, POSITIVE_VERDICTS, RULE_IDS, VERDICT_GUIDE, VERDICTS,
    _load, build_rule, token_confidence,
)
from llm.prompts.zero_shot import CODE, DEFINITIONS
from span_detection.prompts.zero_shot import EXCERPT_RULES, USER, _fields, article_block

NAME = "per_rule"
JSON_OUTPUT = True
MAX_OUTPUT_TOKENS = 5000

SYSTEM = f"""{CONTEXT}

{VERDICT_GUIDE}

{EXCERPT_RULES}

Reply with one JSON object and nothing else, in this shape:
{{
  "rules": {{
    "<rule_id>": {{"verdict": "VIOLATION | NO_VIOLATION",
                  "excerpts": ["exact words from the article that break the rule; [] for NO_VIOLATION"],
                  "explanation": "one short sentence in English; may be empty for NO_VIOLATION"}}
  }},
  "violation": "YES or NO"
}}
Include every rule_id exactly once, in this order: {", ".join(RULE_IDS)}.
"violation" is YES if any rule is VIOLATION, otherwise NO."""

RULE_SENTENCE_QUESTION = """SENTENCE FROM THE ARTICLE: {sentence}
Does this sentence break rule {rule_id}? Answer YES or NO.
Answer:"""


def build(article: dict) -> tuple[str, str]:
    return SYSTEM, USER.format(**_fields(article))


def rule_sentence_prefix(article: dict, rule_id: str) -> str:
    """One rule + article; RULE_SENTENCE_QUESTION is appended per sentence (base models)."""
    rule = CODE[rule_id]
    definitions = {name: DEFINITIONS[name] for name in rule.get("exceptions", []) if name in DEFINITIONS}
    defs = f"DEFINITIONS:\n{json.dumps(definitions, ensure_ascii=False, indent=1)}\n\n" if definitions else ""
    return (
        "You are a media ethics reviewer for Sri Lankan newspapers.\n"
        f"Find the sentences of the news article below that break rule {rule_id} of the Code of Professional Practice.\n\n"
        f"RULE {rule_id} (JSON):\n{json.dumps(rule, ensure_ascii=False, indent=1)}\n{defs}"
        f"{article_block(article)}\n\n"
    )


def parse(text: str) -> list[dict] | None:
    """Spans of the VIOLATION rules, or None if the output is not the expected JSON."""
    data = _load(text)
    if data is None or not isinstance(data.get("rules"), (dict, list)):
        return None
    rules = data["rules"]
    if isinstance(rules, list):
        rules = {r.get("rule_id"): r for r in rules if isinstance(r, dict)}
    spans = []
    for rule_id, r in rules.items():
        if not isinstance(r, dict):
            continue
        verdict = str(r.get("verdict") or "").strip().upper()
        if verdict not in POSITIVE_VERDICTS:
            continue
        excerpts = r.get("excerpts") or r.get("evidence") or [""]
        if isinstance(excerpts, str):
            excerpts = [excerpts]
        for excerpt in dict.fromkeys(str(e).strip() for e in excerpts):
            spans.append({
                "rule_id": str(rule_id),
                "verdict": verdict,
                "excerpt": excerpt,
                "explanation": str(r.get("explanation") or "").strip(),
            })
    return spans
