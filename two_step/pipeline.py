"""The two two-step pipelines for one article.

codes_then_highlights: CODES names the broken rules; if there are any, LOCATE finds
the words for each of them (a rule with no words found is dropped).
highlights_then_codes: PASSAGES marks the problem words; if there are any, ASSIGN
gives each passage its rule(s) (a passage with no rule is dropped).

Every excerpt is then located in the article text or heading (span_detection
locate_spans). The article is a violation when at least one located highlight remains.
"""

from __future__ import annotations

from typing import Callable

from llm.prompts.per_rule import _load
from llm.prompts.zero_shot import CODE
from span_detection.common import locate_spans
from span_detection.prompts.zero_shot import _rule_id
from two_step.prompts import assign_messages, codes_messages, locate_messages, passages_messages

AskFn = Callable[[str, str], str]
APPROACHES = ("codes_then_highlights", "highlights_then_codes")


def _step(ask: AskFn, messages: tuple[str, str], key: str) -> tuple[str | None, list | None, str | None]:
    """(raw, items, error) of one call whose reply is {key: [...]}."""
    try:
        raw = ask(*messages)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"{type(exc).__name__}: {exc}"
    data = _load(raw)
    if data is None or not isinstance(data.get(key), list):
        return raw, None, "UNPARSED: the output is not the expected JSON"
    return raw, [x for x in data[key] if isinstance(x, dict)], None


def codes_then_highlights(article: dict, ask: AskFn, few: bool) -> dict:
    raw1, items, error = _step(ask, codes_messages(article, few), "broken_rules")
    out = {"stage1": {"raw": raw1, "error": error}, "stage2": None, "spans": None, "error": error}
    if items is None:
        return out
    rules: dict[str, str] = {}
    for item in items:
        rid = _rule_id(item.get("rule_id"))
        if rid and rid not in rules:
            rules[rid] = str(item.get("reason") or "").strip()
    out["stage1"]["rules"] = [{"rule_id": rid, "reason": reason} for rid, reason in rules.items()]
    if not rules:
        out["spans"] = []
        return out

    raw2, items, error = _step(ask, locate_messages(article, list(rules), few), "highlights")
    out["stage2"] = {"raw": raw2, "error": error}
    if items is None:
        out["error"] = error
        return out
    spans, seen = [], set()
    for item in items:
        rid = _rule_id(item.get("rule_id"))
        excerpt = str(item.get("excerpt") or "").strip()
        if rid not in rules or not excerpt or (rid, excerpt) in seen:
            continue
        seen.add((rid, excerpt))
        spans.append({"rule_id": rid, "verdict": "VIOLATION", "excerpt": excerpt,
                      "explanation": str(item.get("explanation") or "").strip() or rules[rid]})
    locate_spans(spans, article)
    found = {s["rule_id"] for s in spans if s["offsets"] or s["heading_offsets"]}
    out["stage2"]["dropped_rules"] = [rid for rid in rules if rid not in found]
    out["spans"] = spans
    return out


def highlights_then_codes(article: dict, ask: AskFn, few: bool) -> dict:
    raw1, items, error = _step(ask, passages_messages(article, few), "passages")
    out = {"stage1": {"raw": raw1, "error": error}, "stage2": None, "spans": None, "error": error}
    if items is None:
        return out
    passages, seen = [], set()
    for item in items:
        excerpt = str(item.get("excerpt") or "").strip()
        if excerpt and excerpt not in seen:
            seen.add(excerpt)
            passages.append({"excerpt": excerpt, "problem": str(item.get("problem") or "").strip()})
    locate_spans(passages, article)
    out["stage1"]["passages"] = passages
    if not passages:
        out["spans"] = []
        return out

    raw2, items, error = _step(ask, assign_messages(article, passages, few), "passages")
    out["stage2"] = {"raw": raw2, "error": error}
    if items is None:
        out["error"] = error
        return out
    assigned: dict[int, dict] = {}
    for item in items:
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if 1 <= index <= len(passages) and index not in assigned:
            rule_ids = item.get("rule_ids") if isinstance(item.get("rule_ids"), list) else []
            assigned[index] = {
                "rule_ids": [r for r in dict.fromkeys(_rule_id(x) for x in rule_ids) if r in CODE],
                "explanation": str(item.get("explanation") or "").strip(),
            }
    spans = []
    for index, p in enumerate(passages, start=1):
        a = assigned.get(index, {"rule_ids": [], "explanation": ""})
        p["rule_ids"], p["assign_explanation"] = a["rule_ids"], a["explanation"]
        for rid in a["rule_ids"]:
            spans.append({"rule_id": rid, "verdict": "VIOLATION", "excerpt": p["excerpt"],
                          "explanation": a["explanation"] or p["problem"],
                          "offsets": p["offsets"], "heading_offsets": p["heading_offsets"]})
    out["stage2"]["dropped_passages"] = [p["excerpt"] for p in passages if not p["rule_ids"]]
    out["spans"] = spans
    return out


RUNNERS = {"codes_then_highlights": codes_then_highlights, "highlights_then_codes": highlights_then_codes}


def run(approach: str, article: dict, ask: AskFn, few: bool) -> dict:
    result = RUNNERS[approach](article, ask, few)
    result["calls"] = 1 + (result["stage2"] is not None)
    return result
