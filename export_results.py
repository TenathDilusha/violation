"""Export the results of the 9 ways (task × prompt) of every model from the database to results/.

Writes articles.csv, summary.csv and one CSV of answers per task. Only the test articles
every task and prompt ran on are included (the same articles as evaluate.py).

Usage:
    python export_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import db
from evaluation import compare as compare_eval
from span_detection.locate import located
from span_detection.review import load_review

OUT_DIR = Path(__file__).resolve().parent / "results"
PROMPTS = ("zero_shot", "few_shot", "per_rule")
TASK_FILES = {"llm": "binary_detection", "rule": "rule_detection", "span": "span_detection"}
LANGUAGES = {"all": None, "en": "en", "si": "si"}


def _main_runs(rows: list[dict]) -> list[dict]:
    return [r for r in rows if compare_eval.split_repeat(r["experiment"])[0] in PROMPTS]


def _write(name: str, rows: list[dict]) -> None:
    with (OUT_DIR / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"results/{name}: {len(rows)} rows")


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def correct_rules() -> dict[str, list[str]]:
    rules = {m["article_id"]: list(m["rule_ids"] or []) for m in db.fetch_manual_annotations()}
    for aid, review in load_review().items():
        rules.setdefault(aid, list(review.get("rules", {})))
    return rules


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    preds = {
        "llm": _main_runs(db.fetch_predictions()),
        "rule": _main_runs(db.fetch_rule_predictions()),
        "span": _main_runs(db.fetch_span_predictions()),
    }
    binary = compare_eval.binary_rows(preds["llm"], preds["rule"], preds["span"])
    articles = compare_eval.compare(binary)["articles"]
    ids = {a["article_id"] for a in articles}
    correct = correct_rules()

    _write("articles.csv", [{
        "article_id": a["article_id"],
        "lang": a["lang"],
        "source": "external" if a["article_id"].startswith("ext-") else "platform",
        "status": a["status"],
        "expected": a["expected"],
        "correct_rules": ";".join(correct.get(a["article_id"], [])) if a["expected"] == "YES" else "",
        "heading": a["heading"],
    } for a in articles])

    for task, name in TASK_FILES.items():
        out = []
        for p in sorted(preds[task], key=lambda p: (p["article_id"], p["experiment"], p["model"])):
            if p["article_id"] not in ids:
                continue
            prompt, run = compare_eval.split_repeat(p["experiment"])
            row = {"article_id": p["article_id"], "lang": p["lang"], "expected": p["expected"],
                   "model": p["model"], "prompt": prompt, "run": run}
            if task == "llm":
                row.update(answer=p["answer"], p_yes=_r(p["confidence"]))
            elif task == "rule":
                row.update(answer="ERROR" if p["rules"] is None else ("YES" if p["rules"] else "NO"),
                           rules=";".join(r["rule_id"] for r in p["rules"] or []))
            else:
                found = None if p["spans"] is None else located(p["spans"])
                row.update(answer="ERROR" if found is None else ("YES" if found else "NO"),
                           rules=";".join(dict.fromkeys(s["rule_id"] for s in found or [])),
                           highlights=json.dumps([{"rule_id": s["rule_id"], "excerpt": s["excerpt"]} for s in found or []],
                                                 ensure_ascii=False))
            out.append(row)
        _write(f"{name}.csv", out)

    summary = []
    for lang_key, lang in LANGUAGES.items():
        sub = [r for r in binary if r["article_id"] in ids and (lang is None or r["lang"] == lang)]
        result = compare_eval.compare(sub)
        for (task, prompt, model), c in sorted(result["cells"].items(), key=lambda kv: (list(TASK_FILES).index(kv[0][0]), PROMPTS.index(kv[0][1]), kv[0][2])):
            named = right = total = 0
            if task != "llm":
                for p in preds[task]:
                    a = p["article_id"]
                    if a not in ids or p["model"] != model or p["expected"] != "YES" or compare_eval.split_repeat(p["experiment"])[0] != prompt:
                        continue
                    if (lang and p["lang"] != lang) or a not in correct:
                        continue
                    items = p["rules"] if task == "rule" else (None if p["spans"] is None else located(p["spans"]))
                    got = {r["rule_id"] for r in items or []}
                    named += len(got)
                    right += len(got & set(correct[a]))
                    total += len(correct[a])
            summary.append({
                "language": lang_key, "task": TASK_FILES[task], "prompt": prompt, "model": model,
                "runs": c["n_repeats"], "articles": result["common"], "violations": result["yes"],
                "tp": _r(c["tp"]), "fp": _r(c["fp"]), "tn": _r(c["tn"]), "fn": _r(c["fn"]), "errors": _r(c["errors"]),
                "precision": _r(c["precision"]), "recall": _r(c["recall"]), "f1": _r(c["f1"]), "f2": _r(c["f2"]),
                "accuracy": _r(c["accuracy"]),
                "rule_precision": _r(right / named) if named else None,
                "rule_recall": _r(right / total) if total else None,
            })
    _write("summary.csv", summary)


if __name__ == "__main__":
    main()
