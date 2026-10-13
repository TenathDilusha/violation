"""Viewer for the model results.

Usage:
    python app.py   # → http://127.0.0.1:5002
"""

from __future__ import annotations

import math

from flask import Flask, abort, redirect, render_template, request, url_for
from markupsafe import Markup, escape

import db
from db import OWN_SCHEMA, fetch_article, fetch_manual_annotations, fetch_snapshot_articles
from evaluation import compare as compare_eval
from evaluation import confidence as confidence_eval
from evaluation import scores as scores_eval
from evaluation import stats
from evaluation import vote as vote_eval
from binary_detection.prompts.zero_shot import CODE
from span_detection.locate import locate, located
from span_detection.review import load_review, reviewed_spans

app = Flask(__name__)

RULE_TITLES = {rule_id: rule["title"] for rule_id, rule in CODE.items()}
POSITIVE_VERDICTS = ("VIOLATION",)

SHOWN_EXPERIMENTS = ("zero_shot", "few_shot", "per_rule")
# The same prompts with the Sinhala code of conduct (--code-lang si), Sinhala articles only; /compare only.
SI_CODE_EXPERIMENTS = tuple(f"{e}_si_code" for e in SHOWN_EXPERIMENTS)


# Rule detection prompt revised from zero_shot's failures; a column on /compare only, so the
# "9 runs" of voting and the other pages stay zero_shot / few_shot / per_rule.
COMPARE_EXTRA_RULE = ("zero_shot_v3",)


def _shown(rows: list[dict], repeats: bool = False, si_code: bool = False, extra: tuple = ()) -> list[dict]:
    """Rows of SHOWN_EXPERIMENTS (and SI_CODE_EXPERIMENTS if si_code, and `extra`); repeat 1 only unless repeats (<prompt>.rN rows)."""
    out = []
    for r in rows:
        prompt, n = compare_eval.split_repeat(r["experiment"])
        shown = prompt in SHOWN_EXPERIMENTS or prompt in extra or (si_code and prompt in SI_CODE_EXPERIMENTS)
        if shown and (repeats or n == 1):
            out.append(r)
    return out


def fetch_predictions(experiment: str | None = None, repeats: bool = False, si_code: bool = False) -> list[dict]:
    return _shown(db.fetch_predictions(experiment), repeats, si_code)


def fetch_rule_predictions(experiment: str | None = None, repeats: bool = False, si_code: bool = False,
                           extra: tuple = ()) -> list[dict]:
    return _shown(db.fetch_rule_predictions(experiment), repeats, si_code, extra)


def fetch_span_predictions(
    experiment: str | None = None, article_id: str | None = None, repeats: bool = False, si_code: bool = False
) -> list[dict]:
    return _shown(db.fetch_span_predictions(experiment=experiment, article_id=article_id), repeats, si_code)


def _evidence_spans(content: str, predictions: list[dict]) -> list[dict]:
    """Character spans in content for every positive per-rule evidence quote."""
    spans = []
    for p in predictions:
        for r in (p.get("details") or {}).get("rules", []):
            if r["verdict"] not in POSITIVE_VERDICTS or not r.get("evidence"):
                continue
            label = f"{p['experiment']} · {p['model']}: {r['rule_id']} {r['verdict']}"
            for start, end in locate(content, r["evidence"], min_len=10):
                spans.append({"start": start, "end": end, "labels": [label], "verdict": r["verdict"]})
    return spans


def _highlight(content: str, spans: list[dict]) -> Markup:
    merged: list[dict] = []
    for s in sorted(spans, key=lambda s: (s["start"], -s["end"])):
        if merged and s["start"] < merged[-1]["end"]:
            last = merged[-1]
            last["end"] = max(last["end"], s["end"])
            last["labels"] += s["labels"]
            if s["verdict"] == "VIOLATION":
                last["verdict"] = "VIOLATION"
        else:
            merged.append({**s, "labels": list(s["labels"])})
    parts, pos = [], 0
    for s in merged:
        parts.append(escape(content[pos : s["start"]]))
        parts.append(Markup('<mark class="ev {}" title="{}">{}</mark>').format(
            s["verdict"].lower(), "\n".join(dict.fromkeys(s["labels"])), content[s["start"] : s["end"]]
        ))
        pos = s["end"]
    parts.append(escape(content[pos:]))
    return Markup("").join(parts)


@app.get("/")
def index():
    return redirect(url_for("overview"))


@app.get("/article/<article_id>")
def article(article_id: str):
    articles = fetch_snapshot_articles()
    ids = [a["article_id"] for a in articles]
    predictions = [p for p in fetch_predictions() if p["article_id"] == article_id]
    if article_id in ids:
        i = ids.index(article_id)
        a = articles[i]
        nav = {
            "position": i + 1,
            "total": len(articles),
            "prev_url": url_for("article", article_id=ids[i - 1]) if i > 0 else None,
            "next_url": url_for("article", article_id=ids[i + 1]) if i < len(ids) - 1 else None,
        }
    else:
        a = fetch_article(article_id)
        if a is None:
            abort(404)
        nav = {"position": None, "total": None, "prev_url": None, "next_url": None}

    rule_predictions = [p for p in predictions if (p.get("details") or {}).get("rules")]
    detections = sorted(
        (p for p in fetch_rule_predictions() if p["article_id"] == article_id),
        key=lambda p: (p["model"], RULE_EXPERIMENTS.index(p["experiment"]) if p["experiment"] in RULE_EXPERIMENTS else 99),
    )
    detection_spans = [
        {"experiment": f"rule detection {p['experiment']}", "model": p["model"], "details": {"rules": p["rules"] or []}}
        for p in detections
    ]
    content = a.get("content") or ""
    spans = _evidence_spans(content, rule_predictions + detection_spans)
    return render_template(
        "article.html",
        schema=OWN_SCHEMA,
        a=a,
        predictions=predictions,
        rule_predictions=rule_predictions,
        detections=detections,
        rule_titles=RULE_TITLES,
        content_html=_highlight(content, spans),
        highlighted=len(spans),
        **nav,
    )


RULE_EXPERIMENTS = ["zero_shot", "few_shot", "per_rule"]
LANGUAGES = [("si", "Sinhala"), ("en", "English")]


@app.get("/compare")
def compare_tasks():
    llm_preds = fetch_predictions(repeats=True, si_code=True)
    rule_preds = fetch_rule_predictions(repeats=True, si_code=True, extra=COMPARE_EXTRA_RULE)
    span_preds = fetch_span_predictions(repeats=True, si_code=True)
    all_rows = compare_eval.binary_rows(llm_preds, rule_preds, span_preds)
    langs = sorted({r["lang"] for r in all_rows if r["lang"]})
    lang = request.args.get("lang", "").strip()
    if lang:
        rows = [r for r in all_rows if r["lang"] == lang]
    else:
        rows = [r for r in all_rows if r["experiment"] not in SI_CODE_EXPERIMENTS]
    result = compare_eval.compare(rows)
    main_rows = [r for r in all_rows if r["experiment"] not in SI_CODE_EXPERIMENTS]
    main_common = {a["article_id"] for a in (result if not lang else compare_eval.compare(main_rows))["articles"]}
    score_rows = scores_eval.score_table(llm_preds, rule_preds, span_preds, main_common)
    top = _top_approaches(main_rows, main_common, score_rows=score_rows)

    def scores_in(articles: list[dict]) -> dict[tuple[str, str], dict]:
        """{(score key, task): scores on these articles}, each article judged leave-one-out on all common articles."""
        ids = {a["article_id"] for a in articles}
        out = {}
        for s in score_rows:
            sub = _summary({k: v for k, v in s["fair_votes"].items() if k in ids}, with_ci=False)
            if sub:
                out[(s["key"], s["task"])] = {**sub, "threshold": s["fitted"]["threshold"]}
        return out

    sections = []
    for code, name in LANGUAGES:
        lang_rows = [r for r in all_rows if r["lang"] == code]
        if lang_rows and lang in ("", code):
            sec = compare_eval.compare(lang_rows)
            sections.append({"lang": code, "name": name, "r": sec, "scores": scores_in(sec["articles"])})

    article_lang = {a["article_id"]: a["lang"] for a in compare_eval.compare(main_rows)["articles"]}
    top_by_lang = [
        {**a, "cells": [_summary({k: v for k, v in a["votes"].items() if code is None or article_lang.get(k) == code}, with_ci=False)
                        for code in (None, "si", "en")]}
        for a in top
    ]

    metric = request.args.get("metric", "f2").strip()
    if metric not in dict(compare_eval.METRICS):
        metric = "f2"
    show = request.args.get("show", "").strip()
    answers = result["answers"]
    articles = []
    for a in result["articles"]:
        cells = [answers.get((a["article_id"], e, m), {}) for e, m in result["runs"]]
        given = [c for c in cells if c]
        if show == "disagree" and all(len(set(c.values())) <= 1 for c in given):
            continue
        if show == "wrong" and all(v == a["expected"] for c in given for v in c.values()):
            continue
        if show == "violations" and a["expected"] != "YES":
            continue
        if show == "non_violations" and a["expected"] != "NO":
            continue
        articles.append(a)

    return render_template(
        "compare.html",
        schema=OWN_SCHEMA,
        tasks=compare_eval.TASKS,
        metrics=compare_eval.METRICS,
        metric=metric,
        metric_label=dict(compare_eval.METRICS)[metric],
        r=result,
        sections=sections,
        top=top,
        top_cells={(*a["key"], BEST_MODEL): a["rank"] for a in top},
        scores=scores_in(result["articles"]),
        score_methods=[(k, label, kind, tasks) for k, label, kind, tasks, _ in scores_eval.METHODS],
        across_methods=[(k, label, kind) for k, label, kind, _ in scores_eval.ACROSS_TASKS],
        score_ranks=_score_ranks(top),
        top_by_lang=top_by_lang,
        model=BEST_MODEL,
        articles=articles,
        show=show,
        langs=langs,
        lang=lang,
    )


BEST_EXPERIMENT, BEST_MODEL = "few_shot", "gpt-4.1"
OUTCOMES = {
    "caught": "Caught",
    "missed": "Missed",
    "false_alarm": "False alarm",
    "correct_no": "Correctly clear",
    "error": "Error",
}


def _outcome(expected: str, answer: str) -> str:
    if answer == "ERROR":
        return "error"
    return {("YES", "YES"): "caught", ("YES", "NO"): "missed", ("NO", "YES"): "false_alarm", ("NO", "NO"): "correct_no"}[
        (expected, answer)
    ]


def _correct_codes(expected: str, manual: dict | None, review: dict | None) -> tuple[str, list[str] | None]:
    """(source, rule_ids) of the correct answer; rule_ids is None when nobody listed the codes."""
    if manual is not None:
        return "annotation", [rid for rid in CODE if rid in (manual.get("rule_ids") or [])]
    if review is not None:
        return "review", [rid for rid in CODE if rid in review.get("rules", {})]
    return "status", [] if expected == "NO" else None


def _best_rows(article_id: str | None = None, experiment: str = BEST_EXPERIMENT) -> list[dict]:
    manual = {m["article_id"]: m for m in fetch_manual_annotations(article_id)}
    review = load_review()
    rows = []
    for p in db.fetch_span_predictions(experiment, article_id=article_id):
        if p["model"] != BEST_MODEL:
            continue
        found = None if p["spans"] is None else located(p["spans"])
        answer = "ERROR" if found is None else ("YES" if found else "NO")
        outcome = _outcome(p["expected"], answer)
        source, correct = _correct_codes(p["expected"], manual.get(p["article_id"]), review.get(p["article_id"]))
        detected = [rid for rid in CODE if rid in {s["rule_id"] for s in found or []}]
        known = set(correct or [])
        rows.append({
            "article_id": p["article_id"],
            "heading": p["heading"],
            "lang": p["lang"],
            "status": p["article_status"],
            "expected": p["expected"],
            "answer": answer,
            "outcome": outcome,
            "source": source,
            "correct": correct,
            "detected": detected,
            "matched": [r for r in detected if r in known] if correct is not None else [],
            "missed_codes": [r for r in correct if r not in detected] if correct is not None else [],
            "extra_codes": [r for r in detected if r not in known] if correct is not None else detected,
            "prediction": p,
            "manual": manual.get(p["article_id"]),
            "review": review.get(p["article_id"]),
        })
    order = list(OUTCOMES)
    rows.sort(key=lambda r: (r["expected"] != "YES", order.index(r["outcome"]), r["status"] or "", r["article_id"]))
    return rows


def _run_summary(rows: list[dict]) -> dict:
    s = vote_eval.score([(r["expected"], 1 if r["answer"] == "YES" else 0) for r in rows], 1)
    violations = [r for r in rows if r["expected"] == "YES" and r["correct"] is not None]
    return {
        **s,
        "code_total": sum(len(r["correct"]) for r in violations),
        "code_found": sum(len(r["matched"]) for r in violations),
        "code_extra": sum(len(r["extra_codes"]) for r in violations),
    }


TWO_STEP_APPROACHES = [
    {
        "key": "codes_then_highlights",
        "title": "Codes first, then highlights",
        "steps": [
            ("Step 1 · name the codes",
             "The LLM gets the article and the whole code of conduct and lists the rules the article breaks, "
             "with a reason but no quotes. If it lists none, the article is clear and step 2 is skipped."),
            ("Step 2 · find the words",
             "The LLM gets the article again with only the codes from step 1 and copies the exact words that break "
             "each code. A code it can't find words for is dropped."),
        ],
    },
    {
        "key": "highlights_then_codes",
        "title": "Highlights first, then codes",
        "steps": [
            ("Step 1 · find the words",
             "The LLM gets the article and the whole code of conduct and copies the exact words that break the code, "
             "saying in plain words what is wrong, without naming rules. If it marks nothing, the article is clear "
             "and step 2 is skipped."),
            ("Step 2 · name the codes",
             "The LLM gets the article again with the numbered passages from step 1 and gives each passage the "
             "rule(s) it breaks, or none. A passage with no rule is dropped."),
        ],
    },
]
TWO_STEP_PROMPTS = [("zero_shot", "zero-shot"), ("few_shot", "few-shot")]
TWO_STEP_RUNS = [
    (f"{a['key']}_{p}", f"{a['title']} · {label}", a["key"], p)
    for a in TWO_STEP_APPROACHES for p, label in TWO_STEP_PROMPTS
]
ONE_STEP_RUNS = [
    ("zero_shot", "One step · zero-shot highlights", None, "zero_shot"),
    ("few_shot", "One step · few-shot highlights", None, "few_shot"),
]


def _correct_spans(r: dict, a: dict) -> list[dict]:
    """The correct highlights of a _best_rows() row: {rule_id, part, start, end, excerpt}."""
    texts = {"heading": a.get("heading") or "", "content": a.get("content") or ""}
    correct = []
    if r["source"] == "annotation":
        for s in r["manual"].get("spans") or []:
            correct.append({"rule_id": s["rule_id"], "part": s.get("part") or "content", "start": s["start"], "end": s["end"]})
    elif r["source"] == "review":
        correct = reviewed_spans(a, r["review"])
    for s in correct:
        s["excerpt"] = texts[s["part"]][s["start"]:s["end"]]
    return correct


def _correct_chars(correct: list[dict]) -> set[tuple[str, int]]:
    return {(s["part"], i) for s in correct for i in range(s["start"], s["end"])}


def _detected_chars(spans: list[dict] | None) -> set[tuple[str, int]]:
    return {
        (part, i) for s in spans or [] for part, key in (("heading", "heading_offsets"), ("content", "offsets"))
        for start, end in s.get(key) or [] for i in range(start, end)
    }


def _two_step_rows() -> tuple[dict[str, list[dict]], dict[str, set]]:
    """_best_rows() per run on the articles every run has, and the correct characters of each violation article."""
    by_run = {e: _best_rows(experiment=e) for e, _, _, _ in TWO_STEP_RUNS + ONE_STEP_RUNS}
    by_run = {e: rows for e, rows in by_run.items() if rows}
    common = set.intersection(*({r["article_id"] for r in rows} for rows in by_run.values())) if by_run else set()
    by_run = {e: [r for r in rows if r["article_id"] in common] for e, rows in by_run.items()}
    correct: dict[str, set] = {}
    for r in next(iter(by_run.values()), []):
        if r["expected"] == "YES" and r["source"] in ("annotation", "review"):
            a = fetch_article(r["article_id"])
            if a is not None:
                correct[r["article_id"]] = _correct_chars(_correct_spans(r, a))
    return by_run, correct


def _two_step_score(rows: list[dict], correct: dict[str, set]) -> dict:
    s = _run_summary(rows)
    p, rc = s["precision"], s["recall"]
    inter = covered = marked = 0
    for r in rows:
        chars = correct.get(r["article_id"])
        if not chars:
            continue
        detected = _detected_chars(r["prediction"]["spans"])
        inter += len(chars & detected)
        covered += len(chars)
        marked += len(detected)
    calls = [(r["prediction"].get("details") or {}).get("calls", 1) for r in rows]
    return {
        **s,
        "f1": 2 * p * rc / (p + rc) if p and rc else None,
        "char_recall": inter / covered if covered else None,
        "char_precision": inter / marked if marked else None,
        "calls": sum(calls) / len(calls) if calls else None,
    }


OVERVIEW_TASKS = {"llm": "Binary detection (YES/NO)", "rule": "Rule detection", "span": "Span detection"}
OVERVIEW_PROMPTS = {"zero_shot": "Zero-shot", "few_shot": "Few-shot", "per_rule": "Per-rule"}


def _board_entry(label: str, group: str, s: dict, runs: int = 1, link: str | None = None,
                 votes: list[tuple[str, list[int]]] | None = None, key: tuple | None = None) -> dict:
    """votes (expected, per-repeat predictions per article) adds a bootstrap 95% interval of F2;
    key is the (task, experiment) of _top_approaches()."""
    p, r = s.get("precision"), s.get("recall")
    return {"label": label, "group": group, "f2": s["f2"], "tp": s["tp"], "total": s["tp"] + s["fn"],
            "fp": s["fp"], "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p and r else None,
            "runs": runs, "link": link, "ci": stats.bootstrap(votes)["f2"] if votes else None, "key": key}


@app.get("/overview")
def overview():
    all_rows = compare_eval.binary_rows(
        fetch_predictions(repeats=True, si_code=True),
        fetch_rule_predictions(repeats=True, si_code=True),
        fetch_span_predictions(repeats=True, si_code=True),
    )
    main_rows = [r for r in all_rows if r["experiment"] not in SI_CODE_EXPERIMENTS]
    main = compare_eval.compare(main_rows)
    si = compare_eval.compare([r for r in all_rows if r["lang"] == "si"])
    common = {a["article_id"] for a in main["articles"]}
    board = []

    grid = []
    for t, task_label in OVERVIEW_TASKS.items():
        cells = [main["cells"].get((t, e, BEST_MODEL)) for e in OVERVIEW_PROMPTS]
        grid.append({"label": task_label, "task": t, "cells": cells})
        for (e, prompt_label), c in zip(OVERVIEW_PROMPTS.items(), cells):
            if c:
                board.append(_board_entry(f"{task_label} · {prompt_label}", "One call", c, c["n_repeats"], url_for("compare_tasks"),
                                          list(_votes(main_rows, t, e, common).values()), key=(t, e)))
    gpt_cells = [c for row in grid for c in row["cells"] if c and c["range"].get("f2")]
    spreads = [c["range"]["f2"][1] - c["range"]["f2"][0] for c in gpt_cells]
    other_models = []
    for m in main["models"]:
        if m == BEST_MODEL:
            continue
        best = max(((k, c) for k, c in main["cells"].items() if k[2] == m), key=lambda kc: kc[1]["f2"] or 0, default=None)
        if best:
            (t, e, _), c = best
            other_models.append({"model": m, "task": OVERVIEW_TASKS[t], "prompt": OVERVIEW_PROMPTS.get(e, e), "c": c})

    by_run, correct = _two_step_rows()
    two_step_runs = []
    for e, label, approach, _ in TWO_STEP_RUNS + ONE_STEP_RUNS:
        if e in by_run:
            s = _two_step_score(by_run[e], correct)
            two_step_runs.append({"experiment": e, "label": label, "approach": approach, **s})
            if approach:
                votes = [(r["expected"], [1 if r["answer"] == "YES" else 0]) for r in by_run[e] if r["answer"] != "ERROR"]
                board.append(_board_entry(label, "Two calls", s, 1, votes=votes))

    binary = compare_eval.binary_rows(fetch_predictions(), fetch_rule_predictions(), fetch_span_predictions())
    votes = next((v for v in vote_eval.vote_table(binary) if v["key"] == "all"), None)
    best_vote = max(votes["scores"], key=lambda s: s["f2"] or 0) if votes else None
    if best_vote:
        board.append(_board_entry(f"Vote: violation if ≥ {best_vote['k']} of {len(votes['runs'])} runs say so (threshold picked after seeing the results)",
                                  "Voting", best_vote, 1))

    confidence = confidence_eval.confidence_table(fetch_predictions(repeats=True), common)
    score_rows = _score_rows(common)
    task_scores = [{**s, **s["fair"], "task_label": OVERVIEW_TASKS.get(s["task"], "All 3 tasks")} for s in score_rows]
    task_scores.sort(key=lambda s: -(s["f2"] or 0))
    for c in confidence:
        board.append(_board_entry(f"Confidence: {c['label']}", "Confidence", c["fair"], c["repeats"],
                                  votes=list(c["fair_votes"].values()), key=("confidence", c["key"])))
    for c in _score_candidates(score_rows, {_decisions(c["fair_votes"]) for c in confidence}):
        board.append(_board_entry(c["label"], "Confidence score", {**c, "fn": c["yes"] - c["tp"]}, 3,
                                  votes=list(c["votes"].values()), key=c["key"]))

    versions = _prompt_versions(common, _violation_codes())
    for v in versions[1:]:
        board.append(_board_entry(f"Rule detection · Zero-shot {v['label']}", "Prompt fixed from failures",
                                  {**v, "fn": v["yes"] - v["tp"]}, v["repeats"], votes=list(v["votes"].values()),
                                  key=("rule", v["experiment"])))
    top = _top_approaches(main_rows, common, score_rows=score_rows)
    for a in top:
        a["by_source"] = [_source_recall(a["votes"], external) for external in (False, True)]

    si_pairs = []
    for t, task_label in OVERVIEW_TASKS.items():
        for e, prompt_label in OVERVIEW_PROMPTS.items():
            en_c, si_c = si["cells"].get((t, e, BEST_MODEL)), si["cells"].get((t, f"{e}_si_code", BEST_MODEL))
            if en_c and si_c:
                si_pairs.append({"label": f"{task_label} · {prompt_label}", "en": en_c, "si": si_c})

    board.sort(key=lambda b: -(b["f2"] or 0))
    arts = main["articles"]
    return render_template(
        "overview.html",
        schema=OWN_SCHEMA,
        model=BEST_MODEL,
        main=main,
        langs={code: sum(1 for a in arts if a["lang"] == code) for code, _ in LANGUAGES},
        sets={
            "platform": {code: sum(1 for a in arts if a["expected"] == "YES" and a["lang"] == code
                                   and not a["article_id"].startswith("ext-")) for code, _ in LANGUAGES},
            "external": sum(1 for a in arts if a["expected"] == "YES" and a["article_id"].startswith("ext-")),
            "clean": {code: sum(1 for a in arts if a["expected"] != "YES" and a["lang"] == code) for code, _ in LANGUAGES},
        },
        n_rules=len(CODE),
        groups=sorted({r["group"] for r in CODE.values()}),
        tasks=OVERVIEW_TASKS,
        prompts=OVERVIEW_PROMPTS,
        grid=grid,
        spread_mean=sum(spreads) / len(spreads) if spreads else None,
        spread_max=max(spreads) if spreads else None,
        other_models=other_models,
        two_step_runs=two_step_runs,
        votes=votes,
        best_vote=best_vote,
        confidence=confidence,
        task_scores=task_scores,
        versions=versions,
        top=top,
        top_rank={a["key"]: a["rank"] for a in top},
        si=si,
        si_pairs=si_pairs,
        si_better=sum(1 for p in si_pairs if (p["si"]["f2"] or 0) > (p["en"]["f2"] or 0)),
        board=board,
    )


def _votes(rows: list[dict], task: str, experiment: str, articles: set[str] | None = None,
           model: str = BEST_MODEL) -> dict[str, tuple[str, list[int]]]:
    """{article_id: (expected, [1 if YES else 0 per repeat])} for articles every repeat answered."""
    by: dict[str, dict[int, int]] = {}
    expected = {}
    for r in rows:
        if (r["task"], r["experiment"], r["model"]) != (task, experiment, model) or r["answer"] == "ERROR":
            continue
        if articles is not None and r["article_id"] not in articles:
            continue
        by.setdefault(r["article_id"], {})[r["repeat"]] = 1 if r["answer"] == "YES" else 0
        expected[r["article_id"]] = r["expected"]
    n = max((len(v) for v in by.values()), default=0)
    return {a: (expected[a], [v[k] for k in sorted(v)]) for a, v in by.items() if len(v) == n}


def _scored(label: str, votes: dict[str, tuple[str, list[int]]], **extra) -> dict:
    vs = list(votes.values())
    s = stats.score(vs)
    tp = [sum(1 for e, p in vs if e == "YES" and p[r]) for r in range(len(vs[0][1]))]
    fp = [sum(1 for e, p in vs if e != "YES" and p[r]) for r in range(len(vs[0][1]))]
    per_run = [stats.metrics(t, f, sum(1 for e, _ in vs if e == "YES") - t)["f2"] for t, f in zip(tp, fp)]
    return {"label": label, "votes": votes, **s, "ci": stats.bootstrap(vs), "n": len(vs),
            "yes": sum(1 for e, _ in vs if e == "YES"), "tp": sum(tp) / len(tp), "fp": sum(fp) / len(fp),
            "repeats": len(tp), "run_f2": per_run, **extra}


PROMPT_VERSIONS = [
    ("zero_shot", "Original zero-shot", "the code of conduct and the article"),
    ("zero_shot_v2", "v2: + guide from the failures", "adds a guide on how to apply the rules zero_shot misapplied"),
    ("zero_shot_v3", "v3: v2 with its over-corrections undone", "the v2 guide, with the points that made it miss violations rewritten"),
]


TOP_N = 4


def _decisions(votes: dict[str, tuple[str, list[int]]]) -> tuple:
    return tuple(sorted((a, tuple(p)) for a, (_, p) in votes.items()))


def _score_rows(common: set[str]) -> list[dict]:
    return scores_eval.score_table(fetch_predictions(repeats=True), fetch_rule_predictions(repeats=True),
                                   fetch_span_predictions(repeats=True), common)


def _score_candidates(score_rows: list[dict], skip: set[tuple] = frozenset()) -> list[dict]:
    """The confidence scores of evaluation/scores.py as approaches, key ("score", score key, task). A score whose
    decisions equal an earlier one's, or one of the `skip` decision sets, is left out."""
    seen, out = set(skip), []
    for s in score_rows:
        d = _decisions(s["fair_votes"])
        if d in seen:
            continue
        seen.add(d)
        task_label = OVERVIEW_TASKS.get(s["task"], "All 3 tasks")
        out.append(_scored(f"{task_label} · score: {s['label']}", s["fair_votes"], key=("score", s["key"], s["task"]),
                           kind="Confidence score", score_label=s["label"], score_kind=s["kind"],
                           threshold=s["fitted"]["threshold"]))
    return out


def _top_approaches(main_rows: list[dict], common: set[str], n: int = TOP_N,
                    score_rows: list[dict] | None = None) -> list[dict]:
    """The n best approaches of all kinds by F2, on the same footing: gpt-4.1, the common articles, every repeat
    averaged. Candidates: the 9 one-call formats, the confidence score (threshold chosen leave-one-out), the
    rule detection prompts fixed from failures and the confidence scores of every task (evaluation/scores.py).
    Left out: runs on the earlier article set (two-step, better examples, other prompts) and voting, whose
    threshold was picked after seeing the results.
    Each entry has key (task, experiment), ("confidence", "mean") or ("score", score key, task)."""
    cands = []
    for t, task_label in OVERVIEW_TASKS.items():
        for e, prompt_label in OVERVIEW_PROMPTS.items():
            v = _votes(main_rows, t, e, common)
            if v:
                cands.append(_scored(f"{task_label} · {prompt_label}", v, key=(t, e), kind="One call"))
    conf = {c["key"]: c for c in confidence_eval.confidence_table(fetch_predictions(repeats=True), common)}
    skip = set()
    if "mean" in conf:
        cands.append(_scored("Confidence score (mean P(YES) of the 3 YES/NO prompts)", conf["mean"]["fair_votes"],
                             key=("confidence", "mean"), kind="Confidence score"))
        skip.add(_decisions(conf["mean"]["fair_votes"]))
    raw = compare_eval.binary_rows([], [p for p in db.fetch_rule_predictions() if p["model"] == BEST_MODEL], [])
    for e, label, _ in PROMPT_VERSIONS[1:]:
        v = _votes(raw, "rule", e, common)
        if v:
            cands.append(_scored(f"Rule detection · Zero-shot {label.split(':')[0]}", v, key=("rule", e),
                                 kind="Prompt fixed from failures"))
    cands += _score_candidates(_score_rows(common) if score_rows is None else score_rows, skip)
    cands.sort(key=lambda c: -(c["f2"] or 0))
    return [{**c, "rank": i + 1} for i, c in enumerate(cands[:n])]


def _score_ranks(top: list[dict]) -> dict[tuple[str, str], int]:
    """{(score key, task): rank} of the top approaches that are confidence scores."""
    out = {}
    for a in top:
        if a["key"][0] == "score":
            out[(a["key"][1], a["key"][2])] = a["rank"]
        elif a["key"] == ("confidence", "mean"):
            out[("mean_p", "llm")] = a["rank"]
    return out


def _source_recall(votes: dict[str, tuple[str, list[int]]], external: bool) -> dict | None:
    """Scores on the violation articles from the platform (external=False) or the spreadsheets (external=True)."""
    sub = [v for k, v in votes.items() if v[0] == "YES" and k.startswith("ext-") == external]
    return {**stats.score(sub), "ci": stats.bootstrap(sub), "n": len(sub)} if sub else None


def _summary(votes: dict[str, tuple[str, list[int]]], with_ci: bool = True) -> dict | None:
    """Accuracy, precision, recall, F1 and F2 (means over repeats) with 95% intervals, and the mean counts per run."""
    vs = list(votes.values())
    if not vs:
        return None
    reps = min(len(p) for _, p in vs)
    n, yes = len(vs), sum(1 for e, _ in vs if e == "YES")
    tp = sum(p[r] for r in range(reps) for e, p in vs if e == "YES") / reps
    fp = sum(p[r] for r in range(reps) for e, p in vs if e != "YES") / reps
    correct = tp + (n - yes - fp)
    return {**stats.score(vs), "ci": stats.bootstrap(vs) if with_ci else {}, "accuracy": correct / n,
            "accuracy_ci": stats.wilson(correct, n),
            "n": n, "yes": yes, "no": n - yes, "tp": tp, "fn": yes - tp, "fp": fp, "tn": n - yes - fp, "repeats": reps}


TASK_ABOUT = {
    "llm": "answers one word, YES or NO",
    "rule": "lists the ids of the rules the article breaks; the article counts as a violation when it lists any",
    "span": "copies the exact words that break a rule and names the rule; the article counts as a violation when it highlights anything",
}
PROMPT_ABOUT = {
    "zero_shot": "The prompt holds the code of conduct and the article.",
    "few_shot": "The prompt holds the code of conduct, 3 worked example articles per language with their correct answers, and the article.",
    "per_rule": "The model goes through every rule one by one and gives a verdict for each.",
    "zero_shot_v2": "Zero-shot plus a guide on how to apply the rules it misapplied.",
    "zero_shot_v3": "Zero-shot plus a guide on how to apply the rules it misapplied, with the over-corrections of v2 undone.",
}


@app.get("/best-model")
def best_model():
    main_rows = compare_eval.binary_rows(
        fetch_predictions(repeats=True), fetch_rule_predictions(repeats=True), fetch_span_predictions(repeats=True)
    )
    main = compare_eval.compare(main_rows)
    common = {a["article_id"] for a in main["articles"]}
    top = _top_approaches(main_rows, common)
    if not top:
        abort(404)
    best = top[0]
    task, experiment = best["key"][:2]
    loo = ("An article is a violation when its score passes a threshold; each article is judged with a threshold chosen "
           "on the other articles only (leave-one-out), so the threshold never sees the article it decides.")
    threshold_text = None
    if task == "confidence":
        conf = {c["key"]: c for c in confidence_eval.confidence_table(fetch_predictions(repeats=True), common)}
        t = conf[experiment]["fitted"]["threshold"]
        about = (f"{BEST_MODEL} answers the YES / NO question with the zero-shot, few-shot and per-rule prompts, 3 times each, "
                 f"and reports how likely its answer YES was. The score is the mean of those P(YES) values. {loo}")
        threshold_text = (f"On all articles together the best threshold is {t * 100:.1f}% P(YES): {BEST_MODEL} is very sure "
                          "of itself, often answering NO while still giving YES a few percent, and those articles are often real violations.")
    elif task == "score":
        score_task = best["key"][2]
        runs = {"all_share": 27, "share_all": 9}.get(experiment, 3 if experiment.startswith("share_") else None)
        where = ("YES/NO, rule detection and highlight answers (each with the zero-shot, few-shot and per-rule prompt, 3 runs each)"
                 if score_task == "all" else f"{OVERVIEW_TASKS[score_task].lower()} answers (3 runs per prompt)")
        about = f"A confidence score made from {BEST_MODEL}'s {where}: {best['score_label'][0].lower()}{best['score_label'][1:]}. {loo}"
        t = best["threshold"]
        threshold_text = f"On all articles together the best threshold is {t:.3g}" + (
            f", i.e. at least {math.ceil(t * runs - 1e-9)} of the {runs} runs flag the article." if runs else ".")
    else:
        about = f"{BEST_MODEL} {TASK_ABOUT[task]}. {PROMPT_ABOUT.get(experiment, '')}"
    lang = {a["article_id"]: a["lang"] for a in main["articles"]}
    return render_template(
        "best_model.html",
        schema=OWN_SCHEMA,
        model=BEST_MODEL,
        best=best,
        about=about,
        threshold_text=threshold_text,
        s=_summary(best["votes"]),
        by_lang=[(name, _summary({k: v for k, v in best["votes"].items() if lang.get(k) == code})) for code, name in LANGUAGES],
        by_source=[_source_recall(best["votes"], external) for external in (False, True)],
        runners_up=top[1:],
    )


def _violation_codes(rule_preds: list[dict] | None = None) -> dict[str, list[str]]:
    """Correct rule ids of every violation article with rule detection results and hand-checked codes."""
    manual = {m["article_id"]: m for m in fetch_manual_annotations()}
    review = load_review()
    codes = {}
    for p in rule_preds if rule_preds is not None else fetch_rule_predictions(repeats=True):
        if p["expected"] == "YES" and p["article_id"] not in codes:
            correct = _correct_codes("YES", manual.get(p["article_id"]), review.get(p["article_id"]))[1]
            if correct is not None:
                codes[p["article_id"]] = correct
    return codes


def _prompt_versions(common: set[str], codes: dict[str, list[str]]) -> list[dict]:
    """Rule detection prompt versions on the common articles and the violation articles' codes."""
    raw = [p for p in db.fetch_rule_predictions() if p["model"] == BEST_MODEL]
    rows = compare_eval.binary_rows([], raw, [])
    out, base = [], None
    for e, label, about in PROMPT_VERSIONS:
        votes = _votes(rows, "rule", e, common)
        if not votes:
            continue
        v = _scored(label, votes, about=about, experiment=e)
        named = correct = total = 0
        for p in raw:
            prompt, _ = compare_eval.split_repeat(p["experiment"])
            if prompt == e and p["article_id"] in codes and p["rules"] is not None:
                got, want = set(p["rule_ids"]), set(codes[p["article_id"]])
                named, correct, total = named + len(got), correct + len(got & want), total + len(want)
        v["code_precision"] = correct / named if named else None
        v["code_recall"] = correct / total if total else None
        if base is None:
            base = v
        else:
            v["vs_base"] = {m: stats.paired(votes, base["votes"], m) for m in ("f2", "f1", "precision", "recall")}
        out.append(v)
    return out


if __name__ == "__main__":
    app.run(debug=True, port=5002)
