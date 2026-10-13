"""Confidence scores: P(YES) from gpt-4.1's answer-token logprobs instead of its YES/NO answer.

binary_detection/run_openai.py stores P(YES) / (P(YES) + P(NO)) in llm_predictions.confidence
(per_rule: the highest P(VIOLATION) over the rules). A score is the mean of that
over a prompt's repeats, or over several prompts; an article is a violation when
its score is at least a threshold.

The threshold that maximises F1 on the same articles flatters the result, so
every score is also evaluated leave-one-out: each article is predicted with the
threshold that maximises F1 on all the other articles.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from evaluation.compare import split_repeat
from evaluation.vote import score

MODEL = "gpt-4.1"
PROMPTS = ("zero_shot", "few_shot", "per_rule")
SCORES = [
    ("zero_shot", "Zero-shot P(YES)", ("zero_shot",)),
    ("few_shot", "Few-shot P(YES)", ("few_shot",)),
    ("per_rule", "Per-rule P(YES)", ("per_rule",)),
    ("mean", "Mean P(YES) of the 3 prompts", PROMPTS),
]


def _f1(votes: list[tuple[str, float]], t: float) -> float:
    tp = sum(1 for e, s in votes if e == "YES" and s >= t)
    fp = sum(1 for e, s in votes if e == "NO" and s >= t)
    fn = sum(1 for e, s in votes if e == "YES" and s < t)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _best_threshold(votes: list[tuple[str, float]]) -> float:
    return max(sorted({s for _, s in votes}), key=lambda t: _f1(votes, t))


def _with_f1(s: dict) -> dict:
    p, r = s["precision"], s["recall"]
    return {**s, "f1": 2 * p * r / (p + r) if p and r else None}


def _scored(votes: list[tuple[str, float]], t: float) -> dict:
    return _with_f1(score([(e, 1 if s >= t else 0) for e, s in votes], 1))


def confidence_table(predictions: list[dict], articles: set[str] | None = None, model: str = MODEL) -> list[dict]:
    """predictions: llm_predictions rows including repeats (<prompt>.rN). Only articles
    with a confidence in every repeat of every prompt (and in `articles`, if given) are scored."""
    conf: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    expected, repeats = {}, defaultdict(set)
    for p in predictions:
        prompt, n = split_repeat(p["experiment"])
        if p["model"] != model or prompt not in PROMPTS or p["confidence"] is None:
            continue
        if articles is not None and p["article_id"] not in articles:
            continue
        conf[p["article_id"]][prompt].append(p["confidence"])
        expected[p["article_id"]] = p["expected"]
        repeats[prompt].add(n)
    ids = [a for a in conf if all(len(conf[a][e]) == len(repeats[e]) for e in PROMPTS)]

    out = []
    for key, label, prompts in SCORES:
        votes = [(expected[a], mean(mean(conf[a][e]) for e in prompts)) for a in ids]
        if not votes:
            continue
        loo = []
        for i, (e, s) in enumerate(votes):
            t = _best_threshold(votes[:i] + votes[i + 1:])
            loo.append((e, 1 if s >= t else 0))
        best_t = _best_threshold(votes)
        out.append({
            "key": key, "label": label, "n": len(votes), "repeats": min(len(repeats[e]) for e in prompts),
            "at_half": _scored(votes, 0.5),
            "fitted": {**_scored(votes, best_t), "threshold": best_t},
            "fair": _with_f1(score(loo, 1)),
            "fair_votes": {a: (e, [p]) for a, (e, p) in zip(ids, loo)},
            "scores": {a: s for a, (_, s) in zip(ids, votes)},
        })
    return out
