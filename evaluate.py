"""Print the scores of the 9 ways (task × prompt) for every model, per language, from the database.

Each cell is scored on the articles every task and prompt ran on, as the mean over the
repeats. These are the numbers of the /compare page. --scores adds the confidence scores
of evaluation/scores.py (gpt-4.1, both languages, thresholds chosen leave-one-out), ranked by F2.

Usage:
    python evaluate.py
    python evaluate.py --metric recall --lang si
    python evaluate.py --scores
"""

from __future__ import annotations

import argparse

import db
from evaluation import compare as compare_eval
from evaluation import scores as scores_eval

PROMPTS = ("zero_shot", "few_shot", "per_rule")
SI_CODE = tuple(f"{p}_si_code" for p in PROMPTS)
RULE_EXTRA = ("zero_shot_v3",)
LANGUAGES = {"": "Both languages", "en": "English", "si": "Sinhala"}


def _keep(task: str, experiment: str, si_code: bool) -> bool:
    prompt = compare_eval.split_repeat(experiment)[0]
    return prompt in PROMPTS or (si_code and prompt in SI_CODE) or (task == "rule" and prompt in RULE_EXTRA)


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.0f}"


def print_table(title: str, rows: list[dict], metric: str) -> None:
    r = compare_eval.compare(rows)
    print(f"\n{title}: {r['common']} articles ({r['yes']} violations), {metric.upper()}, mean over repeats")
    models = r["models"]
    print(f"{'Task':<28}{'Prompt':<20}" + "".join(f"{m:>18}" for m in models))
    for task, label in compare_eval.TASKS:
        for e in r["experiments"]:
            cells = [r["cells"].get((task, e, m)) for m in models]
            if any(cells):
                print(f"{label:<28}{e:<20}" + "".join(f"{_pct(c[metric]) if c else '':>18}" for c in cells))


def print_scores(llm: list[dict], rule: list[dict], span: list[dict], articles: set[str], top: int) -> None:
    scored = scores_eval.score_table(llm, rule, span, articles)
    scored.sort(key=lambda s: -(s["fair"]["f2"] or 0))
    print(f"\nConfidence scores (gpt-4.1, {len(articles)} articles, leave-one-out threshold), top {top} by F2")
    print(f"{'Score':<60}{'Task':<8}{'F2':>6}{'P':>6}{'R':>6}")
    for s in scored[:top]:
        f = s["fair"]
        print(f"{s['label'][:58]:<60}{s['task']:<8}{_pct(f['f2']):>6}{_pct(f['precision']):>6}{_pct(f['recall']):>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metric", default="f2", choices=[k for k, _ in compare_eval.METRICS])
    parser.add_argument("--lang", choices=("en", "si"), help="only this language")
    parser.add_argument("--scores", action="store_true", help="also rank the confidence scores")
    parser.add_argument("--top", type=int, default=10, help="number of confidence scores to show")
    args = parser.parse_args()

    llm = [p for p in db.fetch_predictions() if _keep("llm", p["experiment"], True)]
    rule = [p for p in db.fetch_rule_predictions() if _keep("rule", p["experiment"], True)]
    span = [p for p in db.fetch_span_predictions() if _keep("span", p["experiment"], True)]
    rows = compare_eval.binary_rows(llm, rule, span)

    for code, name in LANGUAGES.items():
        if args.lang and code != args.lang:
            continue
        if code:
            print_table(name, [r for r in rows if r["lang"] == code], args.metric)
        else:
            print_table(name, [r for r in rows if _keep(r["task"], r["experiment"], False)], args.metric)

    if args.scores:
        main_rows = [r for r in rows if compare_eval.split_repeat(r["experiment"])[0] in PROMPTS]
        common = {a["article_id"] for a in compare_eval.compare(main_rows)["articles"]}
        main = [[p for p in preds if compare_eval.split_repeat(p["experiment"])[0] in PROMPTS] for preds in (llm, rule, span)]
        print_scores(*main, common, args.top)


if __name__ == "__main__":
    main()
