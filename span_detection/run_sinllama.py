"""Span detection on SinLlama (local GPU), Sinhala articles only.

SinLlama is a base model, so instead of writing excerpts it scores each article
sentence with a YES/NO question on top of a cached article prompt. Each score is
calibrated: a sentence is highlighted when its log-odds of YES are at least
--margin above both the same question for the sentence "N/A" and the article's
median sentence (at most --max-spans sentences per article, highest first).
The median check matters: SinLlama rates almost any real sentence above "N/A".

zero_shot / few_shot: every sentence is asked "Does this sentence break a rule?";
for a highlighted sentence the rule is the most likely rule_id after "Broken rule:".

per_rule: the rules are flagged with the calibrated single-rule prompts
(binary_detection/sinllama.py per_rule); for each flagged rule the sentence that most raises
P(YES) for "Does this sentence break rule X?" is highlighted.

    PYTHONPATH=~/anish_project/.python_pkgs ~/anish_project/venv/bin/python span_detection/run_sinllama.py --prompt zero_shot --negatives 30
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binary_detection.sinllama import SinLlama, add_arguments, logit  # noqa: E402
from span_detection.common import load_prompt, parse_args, run  # noqa: E402
from span_detection.locate import sentences  # noqa: E402


def _add_arguments(parser) -> None:
    add_arguments(parser)
    parser.add_argument("--max-spans", type=int, default=3, help="most sentences highlighted per article")


def main() -> None:
    args = parse_args("sinllama", "sinllama-8b", default_lang="si", add_arguments=_add_arguments)
    args.workers = 1
    prompt = load_prompt(args.prompt)
    sl = SinLlama.from_args(args)

    def score_sentences(prefix: str, question: str, content: str, **fmt) -> tuple[list[dict], float]:
        """Every sentence, highest first, with its log-odds shift vs "N/A" ("shift") and vs the
        article's median sentence ("standout"); also the "N/A" baseline P(YES)."""
        texts = [content[s:e] for s, e in sentences(content)]
        ps = sl.score_continuations(
            prefix, [question.format(sentence=t, **fmt) for t in texts] + [question.format(sentence="N/A", **fmt)]
        )
        base = ps.pop()
        median = statistics.median(logit(p) for p in ps) if ps else 0.0
        scored = [
            {"sentence": t, "p_yes": p, "shift": logit(p) - logit(base), "standout": logit(p) - median,
             "p": 1 / (1 + math.exp(-(logit(p) - logit(base))))}
            for t, p in zip(texts, ps)
        ]
        return sorted(scored, key=lambda x: -x["shift"]), base

    def ask_sentences(article: dict):
        content = article.get("content") or ""
        prefix = prompt.sentence_prefix(article)
        scored, base = score_sentences(prefix, prompt.SENTENCE_QUESTION, content)
        violations = []
        flagged = [x for x in scored if x["shift"] >= args.margin and x["standout"] >= args.margin]
        for s in flagged[: args.max_spans]:
            dist = sl.choose(prefix + prompt.RULE_QUESTION.format(sentence=s["sentence"]), prompt.CANDIDATES)
            rule = max(dist, key=dist.get)
            violations.append({
                "rule_id": rule.strip(),
                "excerpt": s["sentence"],
                "p": s["p"],
                "explanation": (
                    f"P(YES)={s['p_yes']:.3f} vs {base:.3f} for 'N/A' (odds x{math.exp(s['shift']):.2f}), "
                    f"odds x{math.exp(s['standout']):.2f} vs the article's median sentence; rule P={dist[rule]:.2f}"
                ),
            })
        details = {"baseline_p": base, "sentences": [{k: x[k] for k in ("sentence", "p_yes", "p", "standout")} for x in scored]}
        return json.dumps({"violations": violations}, ensure_ascii=False), details

    def ask_per_rule(article: dict):
        content = article.get("content") or ""
        raw, _, details = sl.per_rule(article, prompt, args.margin, quote_evidence=False)
        rules = json.loads(raw)["rules"]
        details["sentences"] = {}
        for rule_id, r in rules.items():
            if r["verdict"] != "VIOLATION":
                continue
            scored, base = score_sentences(
                prompt.rule_sentence_prefix(article, rule_id), prompt.RULE_SENTENCE_QUESTION, content, rule_id=rule_id
            )
            best = scored[0] if scored and scored[0]["shift"] > 0 else None
            r["excerpts"] = [best["sentence"]] if best else []
            if best:
                r["explanation"] += f"; sentence P(YES)={best['p_yes']:.3f} vs {base:.3f} for 'N/A'"
            details["sentences"][rule_id] = [{k: x[k] for k in ("sentence", "p_yes", "p", "standout")} for x in scored[:5]]
        return json.dumps({"rules": rules}, ensure_ascii=False), details

    run("sinllama", args, ask_per_rule if hasattr(prompt, "rule_sentence_prefix") else ask_sentences, ask_article=True)


if __name__ == "__main__":
    main()
