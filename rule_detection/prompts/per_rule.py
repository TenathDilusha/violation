"""Rule-based detection: the per-rule checklist from binary_detection/prompts/per_rule.py.

Every rule gets a verdict (VIOLATION / NO_VIOLATION) with an evidence quote;
the broken rules are those with VIOLATION.
OpenAI adds P(VIOLATION) per rule from token logprobs;
SinLlama scores each rule in its own prompt (build_rule), calibrated against
an empty article.
"""

from __future__ import annotations

from binary_detection.prompts.per_rule import (  # noqa: F401
    EVIDENCE_SUFFIX,
    JSON_OUTPUT,
    MAX_OUTPUT_TOKENS,
    POSITIVE_VERDICTS,
    RULE_IDS,
    build,
    build_rule,
    parse_details,
    token_confidence,
)

NAME = "per_rule"


def parse(text: str) -> list[dict] | None:
    details = parse_details(text)
    if details is None:
        return None
    return [
        {k: r[k] for k in ("rule_id", "verdict", "evidence", "explanation")}
        for r in details["rules"]
        if r["verdict"] in POSITIVE_VERDICTS
    ]
