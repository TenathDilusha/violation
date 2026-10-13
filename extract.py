"""Export review statuses and violation articles to data/ as CSV + JSON.

Usage:
    python extract.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from db import (
    SCHEMA,
    fetch_assumed_non_violation_sample,
    fetch_non_violation_sample,
    fetch_review_status,
    fetch_violation_articles,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
NEGATIVE_SAMPLE_SIZE = 30
# No English article has been reviewed as a non-violation, so English negatives are a
# random sample of unreviewed articles (status ASSUMED_NON_VIOLATION).
ASSUMED_NEGATIVE_LANGS = {"en": 30}
FEW_SHOT_EXAMPLE_FILES = [
    ROOT / "llm" / "prompts" / "few_shot_examples.json",
    ROOT / "rule_detection" / "prompts" / "few_shot_examples.json",
    ROOT / "span_detection" / "prompts" / "few_shot_examples.json",
]


def few_shot_example_ids() -> tuple[str, ...]:
    ids = set()
    for path in FEW_SHOT_EXAMPLE_FILES:
        for examples in json.loads(path.read_text(encoding="utf-8")).values():
            ids.update(ex["article_id"] for ex in examples)
    return tuple(sorted(ids))


def _serialize(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _serialize(v) for k, v in row.items()})


def write_json(rows: list[dict], path: Path) -> None:
    data = [{k: _serialize(v) for k, v in row.items()} for row in rows]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    statuses = fetch_review_status()
    write_csv(statuses, DATA_DIR / "article_review_status.csv")
    print(f"{SCHEMA}.article_review_status: {len(statuses)} rows")

    articles = fetch_violation_articles()
    write_csv(articles, DATA_DIR / "violation_articles.csv")
    write_json(articles, DATA_DIR / "violation_articles.json")
    missing = sum(1 for a in articles if a["heading"] is None and a["content"] is None)
    print(f"violation articles: {len(articles)} rows ({missing} without a raw_articles match)")

    negatives = fetch_non_violation_sample(NEGATIVE_SAMPLE_SIZE)
    print(f"non-violation sample: {len(negatives)} reviewed rows")
    for lang, n in ASSUMED_NEGATIVE_LANGS.items():
        assumed = fetch_assumed_non_violation_sample(n, lang, exclude=few_shot_example_ids())
        negatives += assumed
        print(f"non-violation sample: {len(assumed)} unreviewed {lang} rows (ASSUMED_NON_VIOLATION)")
    write_csv(negatives, DATA_DIR / "non_violation_articles.csv")
    print(f"written to {DATA_DIR}")


if __name__ == "__main__":
    main()
