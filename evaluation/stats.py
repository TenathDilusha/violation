"""Uncertainty for small test sets: bootstrap confidence intervals, paired comparisons,
and Wilson intervals.

An approach is given per article as (expected YES/NO, [prediction 0/1 per repeat]).
A score is computed per repeat and averaged over the repeats (an undefined F1 / F2
counts as 0, as in evaluation/compare.py).

The bootstrap resamples articles with replacement, violations and non-violations
separately (stratified), so every resample keeps the same number of each. It
measures how much a score depends on which articles happen to be in the test set;
the run-to-run spread of gpt-4.1 is reported separately.
"""

from __future__ import annotations

import math
import random
from statistics import mean

from binary_detection.metrics import BETA

B = 1000
SEED = 0
METRICS = ("f2", "f1", "precision", "recall")

Votes = list[tuple[str, list[int]]]


def _counts(pairs: list[tuple[int, int]]) -> tuple[int, int, int]:
    """pairs: (is_violation, predicted) → (tp, fp, fn)."""
    tp = fp = fn = 0
    for y, p in pairs:
        if y and p:
            tp += 1
        elif p:
            fp += 1
        elif y:
            fn += 1
    return tp, fp, fn


def metrics(tp: float, fp: float, fn: float) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    b2 = BETA * BETA
    ok = precision and recall
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if ok else 0.0,
        "f2": (1 + b2) * precision * recall / (b2 * precision + recall) if ok else 0.0,
    }


def score(votes: Votes) -> dict:
    """Mean over repeats of each metric (precision: repeats that flagged something)."""
    n_rep = min(len(p) for _, p in votes)
    per = [metrics(*_counts([(e == "YES", p[r]) for e, p in votes])) for r in range(n_rep)]
    out = {}
    for k in METRICS:
        vals = [m[k] for m in per if m[k] is not None]
        out[k] = mean(vals) if vals else None
    return out


def _resamples(votes: Votes, b: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    pos = [i for i, (e, _) in enumerate(votes) if e == "YES"]
    neg = [i for i, (e, _) in enumerate(votes) if e != "YES"]
    return [rng.choices(pos, k=len(pos)) + rng.choices(neg, k=len(neg)) for _ in range(b)]


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    i = q * (len(values) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


_CACHE: dict[tuple, dict] = {}


def bootstrap(votes: Votes, b: int = B, seed: int = SEED, level: float = 0.95) -> dict:
    """{metric: (low, high)}: the central `level` interval of each metric over b resamples."""
    if not votes:
        return {}
    key = (tuple((e, tuple(p)) for e, p in votes), b, seed, level)
    if key not in _CACHE:
        _CACHE[key] = _bootstrap(votes, b, seed, level)
    return _CACHE[key]


def _bootstrap(votes: Votes, b: int, seed: int, level: float) -> dict:
    samples = {k: [] for k in METRICS}
    for idx in _resamples(votes, b, seed):
        s = score([votes[i] for i in idx])
        for k in METRICS:
            if s[k] is not None:
                samples[k].append(s[k])
    a = (1 - level) / 2
    return {k: (_quantile(v, a), _quantile(v, 1 - a)) if v else None for k, v in samples.items()}


def paired(a: dict[str, tuple[str, list[int]]], b: dict[str, tuple[str, list[int]]], metric: str = "f2",
           n: int = B, seed: int = SEED, level: float = 0.95) -> dict | None:
    """Difference a − b on the articles both have, with the same resamples for both.
    Returns the observed difference, its interval and the share of resamples where a beats b."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    va, vb = [a[i] for i in ids], [b[i] for i in ids]
    diffs = []
    for idx in _resamples(va, n, seed):
        diffs.append((score([va[i] for i in idx])[metric] or 0) - (score([vb[i] for i in idx])[metric] or 0))
    q = (1 - level) / 2
    return {
        "diff": (score(va)[metric] or 0) - (score(vb)[metric] or 0),
        "ci": (_quantile(diffs, q), _quantile(diffs, 1 - q)),
        "a_better": sum(1 for d in diffs if d > 0) / len(diffs),
        "n": len(ids),
        "yes": sum(1 for e, _ in va if e == "YES"),
    }


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson interval of a proportion k / n (k may be a mean over repeats)."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)
