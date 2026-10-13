"""Shared runner for span detection (run_openai.py, run_gemini.py, run_sinllama.py).

Articles and CLI options are the same as llm/ (violation snapshot + --negatives N
non-violation articles). Every excerpt the model returns is located in the
article text or heading (locate.py) and stored with its character offsets. Each result is
appended to span_detection/outputs/<experiment>_<provider>_<model>_<timestamp>.jsonl
and upserted into violations.span_predictions.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import expected_answer, fetch_span_predictions, upsert_span_predictions  # noqa: E402
from llm.common import experiment_name, load_articles, parse_args  # noqa: E402,F401
from span_detection.locate import locate  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ask(system, user) or ask(article) returns the raw model text, or (raw_text, details)
# where details may hold {"rule_p": {rule_id: probability}}.
AskFn = Callable[..., "str | tuple[str, dict | None]"]


def load_prompt(name: str):
    return importlib.import_module(f"span_detection.prompts.{name}")


def locate_spans(spans: list[dict], article: dict) -> dict:
    """Set "offsets" (in the content) or, failing that, "heading_offsets" on every span.

    Models see the heading too and often quote loaded headline wording. Returns the counts.
    """
    for s in spans:
        s["offsets"] = [list(o) for o in locate(article.get("content") or "", s["excerpt"])]
        s["heading_offsets"] = [] if s["offsets"] else [list(o) for o in locate(article.get("heading") or "", s["excerpt"])]
    return {
        "n_spans": len(spans),
        "n_located": sum(1 for s in spans if s["offsets"] or s["heading_offsets"]),
    }


def run(provider: str, args: argparse.Namespace, ask: AskFn, *, ask_article: bool = False) -> None:
    """ask_article=True calls ask(article) instead of ask(system, user)."""
    prompt = load_prompt(args.prompt)
    name = experiment_name(prompt.NAME, getattr(args, "repeat", 1), getattr(args, "code_lang", "en"))
    excluded = getattr(prompt, "EXCLUDE_IDS", set())
    articles = [a for a in load_articles(args.negatives, None, args.lang) if a["article_id"] not in excluded]
    if args.limit:
        articles = articles[: args.limit]

    done = {
        p["article_id"]
        for p in fetch_span_predictions(name)
        if p["model"] == args.model and p["spans"] is not None
    }
    todo = [a for a in articles if args.force or a["article_id"] not in done]
    print(
        f"{provider}:{args.model} · spans/{name} · {len(articles)} articles"
        f"{f' ({len(excluded)} example articles excluded)' if excluded else ''} · {len(todo)} to run",
        flush=True,
    )
    if not todo:
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_path = OUTPUT_DIR / f"{name}_{provider}_{safe_model}_{stamp}.jsonl"
    lock = threading.Lock()

    def work(article: dict) -> dict:
        row = {
            "article_id": article["article_id"],
            "experiment": name,
            "provider": provider,
            "model": args.model,
            "expected": expected_answer(article["status"]),
            "article_status": article["status"],
            "spans": None,
            "n_spans": None,
            "n_located": None,
            "raw_output": None,
            "error": None,
            "details": None,
        }
        try:
            result = ask(article) if ask_article else ask(*prompt.build(article))
            raw, details = result if isinstance(result, tuple) else (result, None)
            row.update(raw_output=raw.strip(), details=details)
            spans = prompt.parse(raw)
            if spans is None:
                row["error"] = "UNPARSED: the output is not the expected JSON"
            else:
                rule_p = (details or {}).get("rule_p", {})
                for s in spans:
                    if "p" not in s and s["rule_id"] in rule_p:
                        s["p"] = rule_p[s["rule_id"]]
                row.update(spans=spans, **locate_spans(spans, article))
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["created_at"] = datetime.now(timezone.utc)
        with lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**row, "created_at": row["created_at"].isoformat()}, ensure_ascii=False) + "\n")
            upsert_span_predictions([row])
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, a) for a in todo]
        for i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            if r["error"]:
                got = r["error"]
            else:
                got = f"{r['n_spans']} spans ({r['n_located']} found in the text)"
                if r["spans"]:
                    got += ": " + ", ".join(dict.fromkeys(s["rule_id"] for s in r["spans"]))
            print(f"[{i}/{len(todo)}] {r['article_id']}  expected={r['expected']}  {got[:100]}", flush=True)

    print(f"saved to {out_path.relative_to(ROOT)} and violations.span_predictions")
