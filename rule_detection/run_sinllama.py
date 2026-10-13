"""Rule detection on SinLlama (local GPU), Sinhala articles only.

zero_shot / few_shot: the prompt ends in "Broken rule:" and every rule_id plus
"none" is scored as a continuation; the most likely candidate is the answer
(so SinLlama names at most one rule). For a rule, SinLlama is asked to quote
the offending sentence, kept only if it matches a sentence of the article.

per_rule: every rule is scored in its own YES/NO prompt, calibrated against an
empty article (see binary_detection/sinllama.py, --margin).

    PYTHONPATH=~/anish_project/.python_pkgs ~/anish_project/venv/bin/python rule_detection/run_sinllama.py --prompt zero_shot --negatives 30
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binary_detection.sinllama import SinLlama, add_arguments, snap_to_sentence  # noqa: E402
from rule_detection.common import load_prompt, parse_args, run  # noqa: E402


def main() -> None:
    args = parse_args("sinllama", "sinllama-8b", default_lang="si", add_arguments=add_arguments)
    args.workers = 1
    prompt = load_prompt(args.prompt)
    sl = SinLlama.from_args(args)

    def ask_per_rule(article: dict):
        raw, _, details = sl.per_rule(article, prompt, args.margin)
        return raw, details

    def ask_choice(article: dict):
        text = prompt.build_choice(article)
        dist = sl.choose(text, prompt.CANDIDATES)
        top = max(dist, key=dist.get)
        rules = []
        if top.strip() != "none":
            quote = sl.quote(text + top + prompt.EVIDENCE_SUFFIX)
            rules.append({
                "rule_id": top.strip(),
                "evidence": snap_to_sentence(quote, article.get("content") or ""),
                "explanation": f"P={dist[top]:.2f} among the 17 rules and none; P(none)={dist[' none']:.2f}",
            })
        raw = json.dumps({"violated_rules": rules}, ensure_ascii=False)
        return raw, {"rule_p": {c.strip(): p for c, p in dist.items()}}

    run("sinllama", args, ask_per_rule if hasattr(prompt, "build_rule") else ask_choice, ask_article=True)


if __name__ == "__main__":
    main()
