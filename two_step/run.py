"""Run the two-step pipelines on the test articles with gpt-4.1.

Four experiments in violations.span_predictions, each approach with the zero-shot
and the few-shot prompts:
    codes_then_highlights_zero_shot, codes_then_highlights_few_shot,
    highlights_then_codes_zero_shot, highlights_then_codes_few_shot
Same articles as span_detection (violation snapshot + --negatives N per
language); the few-shot example articles are left out of all four so they are
compared on the same articles. Each step's output is kept in details; full
traces are appended to two_step/outputs/<model>_<timestamp>.jsonl.

Usage (from the project root):
    python two_step/run.py --negatives 30
    python two_step/run.py --negatives 30 --limit 3 --force
"""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from db import expected_answer, fetch_span_predictions, upsert_span_predictions  # noqa: E402
from binary_detection.common import load_articles, parse_args  # noqa: E402
from two_step import pipeline  # noqa: E402
from two_step.prompts import EXCLUDE_IDS  # noqa: E402

EXPERIMENTS = {
    f"{approach}_{prompt}": (approach, prompt == "few_shot")
    for approach in pipeline.APPROACHES for prompt in ("zero_shot", "few_shot")
}
MAX_OUTPUT_TOKENS = 2000
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _details(result: dict, approach: str, few: bool) -> dict:
    strip = lambda stage: None if stage is None else {k: v for k, v in stage.items() if k != "raw"}  # noqa: E731
    return {
        "approach": approach,
        "prompt": "few_shot" if few else "zero_shot",
        "calls": result["calls"],
        "stage1": strip(result["stage1"]),
        "stage2": strip(result["stage2"]),
    }


def main() -> None:
    from openai import OpenAI

    args = parse_args("openai", os.getenv("OPENAI_MODEL", "gpt-4.1"))
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def ask(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=args.model,
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    articles = [a for a in load_articles(args.negatives, None, args.lang) if a["article_id"] not in EXCLUDE_IDS]
    if args.limit:
        articles = articles[: args.limit]
    done = {
        (p["experiment"], p["article_id"])
        for e in EXPERIMENTS for p in fetch_span_predictions(e)
        if p["model"] == args.model and p["spans"] is not None
    }
    jobs = [(a, e) for a in articles for e in EXPERIMENTS if args.force or (e, a["article_id"]) not in done]
    print(f"openai:{args.model} · two_step · {len(articles)} articles ({len(EXCLUDE_IDS)} example articles excluded) · "
          f"{len(jobs)} article × experiment runs to do", flush=True)
    if not jobs:
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.model.replace('/', '_')}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    lock = threading.Lock()

    def work(article: dict, experiment: str) -> dict:
        approach, few = EXPERIMENTS[experiment]
        result = pipeline.run(approach, article, ask, few)
        spans = result["spans"]
        row = {
            "article_id": article["article_id"],
            "experiment": experiment,
            "provider": "openai",
            "model": args.model,
            "expected": expected_answer(article["status"]),
            "article_status": article["status"],
            "spans": spans,
            "n_spans": None if spans is None else len(spans),
            "n_located": None if spans is None else sum(1 for s in spans if s["offsets"] or s["heading_offsets"]),
            "raw_output": json.dumps({"stage1": result["stage1"]["raw"], "stage2": (result["stage2"] or {}).get("raw")},
                                     ensure_ascii=False),
            "error": result["error"],
            "details": _details(result, approach, few),
            "created_at": datetime.now(timezone.utc),
        }
        with lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**row, "created_at": row["created_at"].isoformat()}, ensure_ascii=False) + "\n")
            upsert_span_predictions([row])
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, a, e) for a, e in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            codes = sorted({s["rule_id"] for s in r["spans"] or [] if s["offsets"] or s["heading_offsets"]})
            print(f"[{i}/{len(jobs)}] {r['article_id']}  {r['experiment']:34} expected={r['expected']}  "
                  f"codes={codes}{'  ERROR ' + r['error'] if r['error'] else ''}", flush=True)
    print(f"saved to {out_path.relative_to(ROOT)} and violations.span_predictions")


if __name__ == "__main__":
    main()
