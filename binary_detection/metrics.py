"""Metrics over violations.llm_predictions rows. Positive class = YES (violation)."""

from __future__ import annotations

from collections import defaultdict

# Recall-first: F-beta with beta = 2 (recall weighted 2x precision).
BETA = 2


def f_beta(precision: float | None, recall: float | None, beta: float = BETA) -> float | None:
    if not precision or not recall:
        return None
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def summarize(predictions: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for p in predictions:
        by_model[p["model"]].append(p)

    summary = []
    for model, rows in sorted(by_model.items()):
        tp = fp = tn = fn = errors = 0
        for p in rows:
            if p["answer"] not in ("YES", "NO"):
                errors += 1
            elif p["answer"] == "YES":
                tp += p["expected"] == "YES"
                fp += p["expected"] == "NO"
            else:
                tn += p["expected"] == "NO"
                fn += p["expected"] == "YES"
        answered = tp + fp + tn + fn
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        summary.append({
            "model": model,
            "provider": rows[0]["provider"],
            "langs": sorted({p.get("lang") or "—" for p in rows}),
            "answered": answered,
            "errors": errors,
            "yes": tp + fp,
            "no": tn + fn,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / answered if answered else None,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f_beta(precision, recall),
        })
    return summary
