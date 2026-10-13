"""Confidence scores for every task: a number per article instead of a yes/no answer.

Each score turns gpt-4.1's stored answers for one task (YES/NO, rule detection or
highlights; all 3 prompts × 3 runs) into a number, e.g. the share of runs that flag
the article, the highest P(VIOLATION) over the rules in the per-rule prompt, or the
number of rules named. An article is a violation when its score is at least a
threshold, chosen leave-one-out to maximise F1 as in evaluation/confidence.py, so the
threshold never sees the article it decides.

Token probabilities exist for the YES/NO prompts (P(YES), stored as `confidence`) and
for the per-rule prompts of all three tasks (P(VIOLATION) per rule, `details.rule_p`);
the zero-shot and few-shot rule detection and highlight runs didn't request them.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Callable

from evaluation.compare import split_repeat
from evaluation.confidence import _best_threshold, _scored, _with_f1
from evaluation.vote import score
from span_detection.locate import located

MODEL = "gpt-4.1"
PROMPTS = ("zero_shot", "few_shot", "per_rule")
TASKS = ("llm", "rule", "span")

Runs = dict[str, list[dict]]  # prompt -> one dict per run: flag (0/1), p, rule_p, count


def _share(prompts: tuple[str, ...]) -> Callable[[Runs], float | None]:
    def f(runs: Runs) -> float | None:
        flags = [r["flag"] for e in prompts for r in runs.get(e, [])]
        return mean(flags) if flags else None
    return f


def _p(prompt: str) -> Callable[[Runs], float | None]:
    def f(runs: Runs) -> float | None:
        ps = [r["p"] for r in runs.get(prompt, []) if r["p"] is not None]
        return mean(ps) if ps else None
    return f


def _rule_p(combine: Callable[[list[float]], float]) -> Callable[[Runs], float | None]:
    def f(runs: Runs) -> float | None:
        per_run = [combine(sorted(r["rule_p"].values(), reverse=True)) for r in runs.get("per_rule", []) if r["rule_p"]]
        return mean(per_run) if per_run else None
    return f


def _noisy_or(ps: list[float]) -> float:
    out = 1.0
    for p in ps:
        out *= 1 - p
    return 1 - out


def _count(prompt: str) -> Callable[[Runs], float | None]:
    def f(runs: Runs) -> float | None:
        counts = [r["count"] for r in runs.get(prompt, []) if r["count"] is not None]
        return mean(counts) if counts else None
    return f


def _mean_of(*fs: Callable[[Runs], float | None]) -> Callable[[Runs], float | None]:
    def f(runs: Runs) -> float | None:
        vals = [g(runs) for g in fs]
        return mean(vals) if vals and None not in vals else None
    return f


_max_p = _rule_p(lambda ps: ps[0] if ps else 0.0)
_share_all = _share(PROMPTS)
_mean_p = _mean_of(_p("zero_shot"), _p("few_shot"), _p("per_rule"))

# (key, label, how it's made, tasks it exists for, score function)
METHODS = [
    ("share_zero_shot", "Share of zero-shot runs that flag", "agreement", TASKS, _share(("zero_shot",))),
    ("share_few_shot", "Share of few-shot runs that flag", "agreement", TASKS, _share(("few_shot",))),
    ("share_per_rule", "Share of per-rule runs that flag", "agreement", TASKS, _share(("per_rule",))),
    ("share_all", "Share of all 9 runs that flag (3 prompts × 3 runs)", "agreement", TASKS, _share_all),
    ("p_zero_shot", "Zero-shot P(YES)", "probability", ("llm",), _p("zero_shot")),
    ("p_few_shot", "Few-shot P(YES)", "probability", ("llm",), _p("few_shot")),
    ("mean_p", "Mean P(YES) of the 3 prompts", "probability", ("llm",), _mean_p),
    ("max_p", "Per-rule: highest P(VIOLATION) over the rules", "probability", TASKS, _max_p),
    ("noisy_or", "Per-rule: P(at least one rule broken), rules taken as independent", "probability", TASKS, _rule_p(_noisy_or)),
    ("sum_p", "Per-rule: expected number of broken rules (sum of P)", "probability", TASKS, _rule_p(sum)),
    ("top3_p", "Per-rule: mean of the 3 highest P(VIOLATION)", "probability", TASKS, _rule_p(lambda ps: mean(ps[:3]) if ps else 0.0)),
    ("count_zero_shot", "Zero-shot: number of rules named / passages highlighted", "count", ("rule", "span"), _count("zero_shot")),
    ("count_few_shot", "Few-shot: number of rules named / passages highlighted", "count", ("rule", "span"), _count("few_shot")),
    ("share_and_max_p", "Mean of the share of all 9 runs and the per-rule highest P", "combined", TASKS, _mean_of(_share_all, _max_p)),
]


def _all_share(t: dict[str, Runs]) -> float | None:
    flags = [r["flag"] for task in TASKS for e in PROMPTS for r in t[task].get(e, [])]
    return mean(flags) if flags else None


def _all_p(t: dict[str, Runs]) -> float | None:
    vals = [_mean_p(t["llm"]), _max_p(t["rule"]), _max_p(t["span"])]
    return mean(vals) if None not in vals else None


def _all_p_and_share(t: dict[str, Runs]) -> float | None:
    vals = [_all_share(t), _all_p(t)]
    return mean(vals) if None not in vals else None


# Scores over all three tasks: (key, label, how it's made, function of {task: Runs})
ACROSS_TASKS = [
    ("all_share", "Share of all 27 runs that flag (3 tasks × 3 prompts × 3 runs)", "agreement", _all_share),
    ("all_p", "Mean of the YES/NO mean P(YES) and the rule detection and highlights per-rule highest P", "probability", _all_p),
    ("all_p_and_share", "Mean of the two above", "combined", _all_p_and_share),
]


def _runs(llm: list[dict], rule: list[dict], span: list[dict], model: str) -> dict[str, dict[str, Runs]]:
    """{task: {article_id: {prompt: [run, ...]}}} of model's answered runs of the 3 prompts."""
    out: dict[str, dict[str, Runs]] = {t: defaultdict(lambda: defaultdict(list)) for t in TASKS}
    expected = {}
    for task, rows in (("llm", llm), ("rule", rule), ("span", span)):
        for p in rows:
            prompt, _ = split_repeat(p["experiment"])
            if p["model"] != model or prompt not in PROMPTS:
                continue
            details = p.get("details") if isinstance(p.get("details"), dict) else {}
            if task == "llm":
                if p["answer"] not in ("YES", "NO"):
                    continue
                run = {"flag": int(p["answer"] == "YES"), "p": p.get("confidence"), "count": None}
            elif task == "rule":
                if p["rules"] is None:
                    continue
                run = {"flag": int(bool(p["rules"])), "p": None, "count": len(p.get("rule_ids") or [])}
            else:
                if p["spans"] is None:
                    continue
                n = len(located(p["spans"]))
                run = {"flag": int(n > 0), "p": None, "count": n}
            run["rule_p"] = details.get("rule_p") if prompt == "per_rule" else None
            out[task][p["article_id"]][prompt].append(run)
            expected[p["article_id"]] = p["expected"]
    out["_expected"] = expected  # type: ignore[assignment]
    return out


def _evaluate(scores: dict[str, float], expected: dict[str, str]) -> dict:
    ids = sorted(scores)
    votes = [(expected[a], scores[a]) for a in ids]
    loo = []
    for i, (e, s) in enumerate(votes):
        t = _best_threshold(votes[:i] + votes[i + 1:])
        loo.append((e, 1 if s >= t else 0))
    best_t = _best_threshold(votes)
    return {
        "n": len(votes),
        "fitted": {**_scored(votes, best_t), "threshold": best_t},
        "fair": _with_f1(score(loo, 1)),
        "fair_votes": {a: (e, [p]) for a, (e, p) in zip(ids, loo)},
        "scores": dict(zip(ids, (s for _, s in votes))),
    }


def score_table(llm: list[dict], rule: list[dict], span: list[dict], articles: set[str], model: str = MODEL) -> list[dict]:
    """One entry per (method, task) and per across-task method, scored on `articles`
    (each article needs every run of the prompts the method uses)."""
    runs = _runs(llm, rule, span, model)
    expected = runs.pop("_expected")
    out = []
    for key, label, kind, tasks, fn in METHODS:
        for task in tasks:
            scores = {}
            for a in articles:
                if a in runs[task] and a in expected:
                    s = fn(runs[task][a])
                    if s is not None:
                        scores[a] = s
            if len(scores) == len(articles):
                out.append({"key": key, "label": label, "kind": kind, "task": task, **_evaluate(scores, expected)})
    for key, label, kind, fn in ACROSS_TASKS:
        scores = {}
        for a in articles:
            if all(a in runs[t] for t in TASKS):
                s = fn({t: runs[t][a] for t in TASKS})
                if s is not None:
                    scores[a] = s
        if len(scores) == len(articles):
            out.append({"key": key, "label": label, "kind": kind, "task": "all", **_evaluate(scores, expected)})
    return out
