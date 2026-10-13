"""Hand-checked correct answers (rule codes and highlights) from reviewed_suggestions.json.

Used for violation articles that have no row in violations.manual_annotations.
"""

from __future__ import annotations

import json
from pathlib import Path

from span_detection.locate import locate

REVIEW_FILE = Path(__file__).with_name("reviewed_suggestions.json")


def load_review() -> dict:
    """reviewed_suggestions.json by article_id ({} if the file is missing)."""
    if not REVIEW_FILE.exists():
        return {}
    return {k: v for k, v in json.loads(REVIEW_FILE.read_text(encoding="utf-8")).items() if not k.startswith("_")}


def reviewed_spans(article: dict, review: dict) -> list[dict]:
    """The reviewed highlights of one article as [{rule_id, part, start, end}]; excerpts not in the text are skipped."""
    spans = []
    for rid, info in review.get("rules", {}).items():
        for s in info.get("spans", []):
            part = "heading" if s.get("part") == "heading" else "content"
            found = locate(article.get(part) or "", s["excerpt"], min_len=1)
            if found:
                spans.append({"rule_id": rid, "part": part, "start": found[0][0], "end": found[-1][1]})
    return sorted(spans, key=lambda s: (s["part"] != "heading", s["start"], s["rule_id"]))
