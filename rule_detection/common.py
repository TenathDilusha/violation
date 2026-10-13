"""Shared runner for rule detection (run_openai.py, run_gemini.py, run_sinllama.py).

Articles and CLI options are the same as binary_detection/ (violation snapshot + --negatives N
non-violation articles). Each result is appended to
rule_detection/outputs/<experiment>_<provider>_<model>_<timestamp>.jsonl and
upserted into violations.rule_predictions.
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

from db import expected_answer, fetch_rule_predictions, upsert_rule_predictions  # noqa: E402
from binary_detection.common import experiment_name, load_articles, parse_args  # noqa: E402,F401

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ask(system, user) or ask(article) returns the raw model text, or (raw_text, details)
# where details may hold {"rule_p": {rule_id: probability}}.
AskFn = Callable[..., "str | tuple[str, dict | None]"]


def load_prompt(name: str):
    return importlib.import_module(f"rule_detection.prompts.{name}")


def run(provider: str, args: argparse.Namespace, ask: AskFn, *, ask_article: bool = False) -> None:
    """ask_article=True calls ask(article) instead of ask(system, user)."""
    prompt = load_prompt(args.prompt)
    name = experiment_name(prompt.NAME, args.repeat, args.code_lang)
    excluded = getattr(prompt, "EXCLUDE_IDS", set())
    articles = [
        a for a in load_articles(args.negatives, None, args.lang) if a["article_id"] not in excluded
    ]
    if args.limit:
        articles = articles[: args.limit]

    done = {
        p["article_id"]
        for p in fetch_rule_predictions(name)
        if p["model"] == args.model and p["rules"] is not None
    }
    todo = [a for a in articles if args.force or a["article_id"] not in done]
    print(
        f"{provider}:{args.model} · rules/{name} · {len(articles)} articles"
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
            "rules": None,
            "rule_ids": None,
            "raw_output": None,
            "error": None,
            "details": None,
        }
        try:
            result = ask(article) if ask_article else ask(*prompt.build(article))
            raw, details = result if isinstance(result, tuple) else (result, None)
            row.update(raw_output=raw.strip(), details=details)
            rules = prompt.parse(raw)
            if rules is None:
                row["error"] = "UNPARSED: the output is not the expected JSON"
            else:
                rule_p = (details or {}).get("rule_p", {})
                for r in rules:
                    if r["rule_id"] in rule_p:
                        r["p"] = rule_p[r["rule_id"]]
                row.update(rules=rules, rule_ids=[r["rule_id"] for r in rules])
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["created_at"] = datetime.now(timezone.utc)
        with lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**row, "created_at": row["created_at"].isoformat()}, ensure_ascii=False) + "\n")
            upsert_rule_predictions([row])
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, a) for a in todo]
        for i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            got = r["error"] or (", ".join(r["rule_ids"]) or "no rule")
            print(f"[{i}/{len(todo)}] {r['article_id']}  expected={r['expected']}  rules: {got[:80]}", flush=True)

    print(f"saved to {out_path.relative_to(ROOT)} and violations.rule_predictions")
