"""Compare every model × prompt across the three tasks as violation / no-violation classifiers.

- LLM results (violations.llm_predictions): the YES/NO answer.
- Rule detection (violations.rule_predictions): YES when at least one rule is named.
- Highlights (violations.span_predictions): YES when at least one highlight was found
  in the article text or heading.

Every task is scored against the article's review status on the same articles:
those that every task and prompt was run on (few-shot example articles are not).

A prompt can be run several times (`--repeat N`, stored as <prompt>.rN). Each repeat
is scored on its own and a cell holds the mean over the repeats, with the
per-repeat scores kept for the spread.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from llm.metrics import summarize
from span_detection.locate import located

REPEAT = re.compile(r"^(?P<prompt>.+)\.r(?P<n>\d+)$")
MEAN_KEYS = ("tp", "fp", "tn", "fn", "errors", "answered", "accuracy", "precision", "recall", "f1", "f2")

TASKS = [
    ("llm", "LLM results (YES/NO)"),
    ("rule", "Rule detection (≥1 rule)"),
    ("span", "Highlights (≥1 highlight)"),
]
EXPERIMENTS = ["zero_shot", "zero_shot_v3", "zero_shot_si_code", "few_shot", "few_shot_si_code", "per_rule", "per_rule_si_code"]
PROVIDER_ORDER = ["openai", "gemini", "sinllama"]
METRICS = [
    ("recall", "Recall"),
    ("precision", "Precision"),
    ("f1", "F1"),
    ("f2", "F2"),
    ("accuracy", "Accuracy"),
]


def split_repeat(experiment: str) -> tuple[str, int]:
    """("few_shot.r2") -> ("few_shot", 2); a name without a suffix is repeat 1."""
    m = REPEAT.match(experiment)
    return (m["prompt"], int(m["n"])) if m else (experiment, 1)


def binary_rows(llm: list[dict], rule: list[dict], span: list[dict]) -> list[dict]:
    """One row per (task, experiment, repeat, model, article) with answer YES / NO / ERROR."""
    rows = []

    def add(task: str, p: dict, answer: str) -> None:
        experiment, repeat = split_repeat(p["experiment"])
        rows.append({
            "task": task,
            "experiment": experiment,
            "repeat": repeat,
            "model": p["model"],
            "provider": p["provider"],
            "article_id": p["article_id"],
            "heading": p.get("heading"),
            "lang": p.get("lang"),
            "status": p.get("article_status"),
            "expected": p["expected"],
            "answer": answer,
        })

    for p in llm:
        add("llm", p, p["answer"] if p["answer"] in ("YES", "NO") else "ERROR")
    for p in rule:
        add("rule", p, "ERROR" if p["rules"] is None else ("YES" if p["rules"] else "NO"))
    for p in span:
        add("span", p, "ERROR" if p["spans"] is None else ("YES" if located(p["spans"]) else "NO"))
    return rows


def _model_order(model: str, provider: str) -> tuple:
    return (PROVIDER_ORDER.index(provider) if provider in PROVIDER_ORDER else 99, model)


def _mean(repeats: list[dict]) -> dict:
    """Mean of each metric over the repeats. An undefined F1 / F2 (nothing caught) counts as 0;
    an undefined precision (nothing flagged) is left out."""
    out = {**repeats[0], "repeats": repeats, "n_repeats": len(repeats), "range": {}}
    for key in MEAN_KEYS:
        values = [s[key] if s[key] is not None or key == "precision" else 0.0 for s in repeats]
        values = [v for v in values if v is not None]
        out[key] = sum(values) / len(values) if values else None
        out["range"][key] = (min(values), max(values)) if values else None
    return out


def _majority(answers: list[str]) -> str:
    """Most common YES / NO over the repeats (YES on a tie), ERROR when every repeat failed."""
    counts = Counter(a for a in answers if a != "ERROR")
    if not counts:
        return "ERROR"
    return "YES" if counts["YES"] >= counts["NO"] else "NO"


def compare(rows: list[dict]) -> dict:
    """Mean metrics per (task, experiment, model) over its repeats on the common articles, plus per-article answers."""
    combos = {(r["task"], r["experiment"], r["repeat"]) for r in rows}
    per_combo = [{r["article_id"] for r in rows if (r["task"], r["experiment"], r["repeat"]) == c} for c in combos]
    common = set.intersection(*per_combo) if per_combo else set()
    scored = [r for r in rows if r["article_id"] in common]

    models = sorted({(r["model"], r["provider"]) for r in scored}, key=lambda m: _model_order(*m))
    experiments = sorted({r["experiment"] for r in scored}, key=lambda e: EXPERIMENTS.index(e) if e in EXPERIMENTS else 99)
    ran = {(r["experiment"], r["model"]) for r in scored}
    runs = [(e, m) for m, _ in models for e in experiments if (e, m) in ran]

    per_repeat: dict[tuple, list[dict]] = defaultdict(list)
    for task, _ in TASKS:
        task_rows = [
            {**r, "model": f"{r['experiment']}|{r['model']}|{r['repeat']}"} for r in scored if r["task"] == task
        ]
        for s in summarize(task_rows):
            experiment, model, repeat = s["model"].split("|")
            per_repeat[(task, experiment, model)].append({**s, "model": model, "repeat": int(repeat)})
    cells = {k: _mean(sorted(v, key=lambda s: s["repeat"])) for k, v in per_repeat.items()}

    best = {}
    for task, _ in TASKS:
        for key, _ in METRICS:
            values = [c[key] for (t, _, _), c in cells.items() if t == task and c[key] is not None]
            best[(task, key)] = max(values) if values else None
    articles: dict[str, dict] = {}
    by_repeat: dict[tuple, dict] = defaultdict(dict)
    for r in scored:
        articles.setdefault(r["article_id"], {
            "article_id": r["article_id"], "heading": r["heading"], "lang": r["lang"],
            "status": r["status"], "expected": r["expected"],
        })
        by_repeat[(r["article_id"], r["experiment"], r["model"], r["repeat"])][r["task"]] = r["answer"]

    answer_runs: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for (aid, experiment, model, repeat), tasks in sorted(by_repeat.items(), key=lambda x: x[0][3]):
        for task, answer in tasks.items():
            answer_runs[(aid, experiment, model)][task].append(answer)
    by_run = {k: {t: _majority(v) for t, v in tasks.items()} for k, tasks in answer_runs.items()}

    agreement = {}
    for experiment, model in runs:
        answered = [
            a for (aid, e, m, _), a in by_repeat.items()
            if (e, m) == (experiment, model) and len(a) == len(TASKS)
        ]
        same = sum(1 for a in answered if len(set(a.values())) == 1)
        agreement[(experiment, model)] = {"same": same, "n": len(answered), "share": same / len(answered) if answered else None}

    return {
        "common": len(common),
        "yes": sum(1 for a in articles.values() if a["expected"] == "YES"),
        "no": sum(1 for a in articles.values() if a["expected"] == "NO"),
        "models": [m for m, _ in models],
        "experiments": experiments,
        "runs": runs,
        "cells": cells,
        "best": best,
        "agreement": agreement,
        "articles": sorted(articles.values(), key=lambda a: (a["expected"] != "YES", a["status"] or "", a["article_id"])),
        "answers": by_run,
        "answer_runs": {k: dict(v) for k, v in answer_runs.items()},
        "max_repeats": max((c["n_repeats"] for c in cells.values()), default=0),
    }
