"""Voting over one model's runs: an article is a violation when at least k runs say so.

The runs are the (task, prompt) pairs of evaluation/compare.py binary_rows()
(LLM results YES/NO, rule detection ≥1 rule, highlights ≥1 highlight). Only
articles every selected run answered are scored.
"""

from __future__ import annotations

from binary_detection.metrics import BETA

MODEL = "gpt-4.1"
SELECTIONS = [
    ("all", "All 9 runs (3 tasks × 3 prompts)", None),
    ("highlights", "The 3 highlight runs", ("span",)),
    ("few_shot", "The 3 few-shot runs (one per task)", "few_shot"),
]


def _runs(rows: list[dict], key) -> list[tuple[str, str]]:
    runs = sorted({(r["task"], r["experiment"]) for r in rows})
    if key is None:
        return runs
    if isinstance(key, tuple):
        return [x for x in runs if x[0] in key]
    return [x for x in runs if x[1] == key]


def score(votes: list[tuple[str, int]], k: int) -> dict:
    """votes: (expected YES/NO, number of runs saying YES) per article; violation when ≥ k."""
    tp = sum(1 for expected, n in votes if expected == "YES" and n >= k)
    fn = sum(1 for expected, n in votes if expected == "YES" and n < k)
    fp = sum(1 for expected, n in votes if expected == "NO" and n >= k)
    tn = sum(1 for expected, n in votes if expected == "NO" and n < k)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    b2 = BETA * BETA
    f2 = (1 + b2) * precision * recall / (b2 * precision + recall) if precision and recall else None
    return {"k": k, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f2": f2,
            "accuracy": (tp + tn) / len(votes) if votes else None}


def vote_table(rows: list[dict], model: str = MODEL) -> list[dict]:
    """rows: compare.binary_rows() output. One entry per selection with a score for every k."""
    rows = [r for r in rows if r["model"] == model]
    out = []
    for key, label, which in SELECTIONS:
        runs = _runs(rows, which)
        if len(runs) < 2:
            continue
        by_article: dict[str, dict] = {}
        for r in rows:
            if (r["task"], r["experiment"]) in runs:
                a = by_article.setdefault(r["article_id"], {"expected": r["expected"], "answers": {}})
                a["answers"][(r["task"], r["experiment"])] = r["answer"]
        votes = [
            (a["expected"], sum(1 for v in a["answers"].values() if v == "YES"))
            for a in by_article.values() if len(a["answers"]) == len(runs)
        ]
        out.append({
            "key": key, "label": label, "runs": runs, "n": len(votes),
            "yes": sum(1 for e, _ in votes if e == "YES"), "no": sum(1 for e, _ in votes if e == "NO"),
            "scores": [score(votes, k) for k in range(1, len(runs) + 1)],
        })
    return out
