"""Shared runner for the provider scripts (run_openai.py, run_gemini.py).

Each result is appended to llm/outputs/<experiment>_<provider>_<model>_<timestamp>.jsonl
and upserted into violations.llm_predictions.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from db import (  # noqa: E402
    expected_answer,
    fetch_predictions,
    fetch_snapshot_articles,
    fetch_snapshot_negatives,
    upsert_predictions,
)

load_dotenv(ROOT / ".env", override=True)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ask(system, user) returns the raw model text, (raw_text, confidence) where
# confidence is P(YES) in [0, 1], or (raw_text, confidence, extra_details).
AskFn = Callable[[str, str], "str | tuple"]


def parse_args(
    provider: str,
    default_model: str,
    *,
    default_lang: str | None = None,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run a prompt on {provider}.")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--prompt", default="zero_shot", help="module name in llm/prompts/")
    parser.add_argument("--lang", default=default_lang, help="only articles in this language (en / si)")
    parser.add_argument(
        "--negatives", type=int, default=0,
        help="add the first N articles per language of the violations.non_violation_articles snapshot",
    )
    parser.add_argument("--limit", type=int, help="only the first N articles")
    parser.add_argument("--force", action="store_true", help="re-run articles that already have an answer")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="repeat number: 1 is stored as the prompt name, N > 1 as <prompt>.rN",
    )
    parser.add_argument(
        "--code-lang", choices=("en", "si"), default="en",
        help="language of the code of conduct in the prompt; si is stored as <prompt>_si_code and needs --lang si",
    )
    if add_arguments:
        add_arguments(parser)
    args = parser.parse_args()
    if args.code_lang != "en":
        if args.lang != args.code_lang:
            parser.error(f"--code-lang {args.code_lang} is only for {args.code_lang} articles: add --lang {args.code_lang}")
        if "llm.prompts.zero_shot" in sys.modules:
            parser.error("the prompts were imported before --code-lang was read")
        os.environ["CODE_LANG"] = args.code_lang
    return args


def load_articles(negatives: int, limit: int | None, lang: str | None = None) -> list[dict]:
    articles = fetch_snapshot_articles()
    if negatives:
        articles += fetch_snapshot_negatives(negatives)
    if lang:
        articles = [a for a in articles if a["lang"] == lang]
    return articles[:limit] if limit else articles


def load_prompt(name: str):
    return importlib.import_module(f"llm.prompts.{name}")


def experiment_name(prompt_name: str, repeat: int = 1, code_lang: str = "en") -> str:
    """The stored experiment of one run; evaluation/compare.py split_repeat() undoes the repeat suffix."""
    name = prompt_name if code_lang == "en" else f"{prompt_name}_{code_lang}_code"
    return name if repeat == 1 else f"{name}.r{repeat}"


def run(provider: str, args: argparse.Namespace, ask: AskFn, *, ask_article: bool = False) -> None:
    """ask_article=True calls ask(article) instead of ask(system, user)."""
    prompt = load_prompt(args.prompt)
    name = experiment_name(prompt.NAME, args.repeat, args.code_lang)
    parse_details = getattr(prompt, "parse_details", None)
    excluded = getattr(prompt, "EXCLUDE_IDS", set())
    articles = [
        a for a in load_articles(args.negatives, None, args.lang) if a["article_id"] not in excluded
    ]
    if args.limit:
        articles = articles[: args.limit]

    done = {
        p["article_id"]
        for p in fetch_predictions(name)
        if p["model"] == args.model and p["answer"] != "ERROR"
    }
    todo = [a for a in articles if args.force or a["article_id"] not in done]
    print(
        f"{provider}:{args.model} · prompt={name} · {len(articles)} articles"
        f"{f' ({len(excluded)} example articles excluded)' if excluded else ''} · {len(todo)} to run"
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
            "confidence": None,
            "error": None,
            "details": None,
        }
        try:
            result = ask(article) if ask_article else ask(*prompt.build(article))
            parts = list(result) if isinstance(result, tuple) else [result]
            raw, confidence, extra = (parts + [None, None])[:3]
            row.update(answer=prompt.parse(raw), raw_output=raw.strip(), confidence=confidence)
            if parse_details:
                row["details"] = parse_details(raw)
            if extra:
                row["details"] = {**(row["details"] or {}), **extra}
        except Exception as exc:  # noqa: BLE001
            row.update(answer="ERROR", raw_output=None, error=f"{type(exc).__name__}: {exc}")
        row["created_at"] = datetime.now(timezone.utc)
        with lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**row, "created_at": row["created_at"].isoformat()}, ensure_ascii=False) + "\n")
            upsert_predictions([row])
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, a) for a in todo]
        for i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            detail = r["error"] or r["raw_output"] or ""
            print(f"[{i}/{len(todo)}] {r['article_id']}  expected={r['expected']}  got={r['answer']:8} {detail[:60]}")

    print(f"saved to {out_path.relative_to(ROOT)} and violations.llm_predictions")
