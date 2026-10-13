"""Run a prompt on SinLlama (local GPU), Sinhala articles only.

SinLlama is a base (non-chat) Llama-3-8B model, so instead of free generation
the prompt ends with "Answer:" and we compare the next-token probabilities of
YES vs NO. The answer is whichever is more likely; confidence = P(YES).

With --prompt per_rule, every rule is scored in its own prompt (17 passes per
article). SinLlama's raw per-rule P(YES) sits near 0.5 for every rule, so each
score is calibrated against the same rule with an empty ("N/A") article:
calibrated log-odds = logit(P(YES | article)) - logit(P(YES | N/A)).
A rule is VIOLATION when the article raises the odds of YES by at least
exp(--margin) (default 0.5, i.e. x1.65). SinLlama is then asked to quote the
offending sentence; the quote is kept only if it matches a sentence of the
article. Confidence = max calibrated P(YES) over rules.

Needs torch + transformers (+ bitsandbytes for --load-in-4bit) with CUDA.
anish_project's venv has them:

    PYTHONPATH=~/anish_project/.python_pkgs ~/anish_project/venv/bin/python binary_detection/run_sinllama.py
    ... binary_detection/run_sinllama.py --load-in-4bit --limit 3 --force
    ... binary_detection/run_sinllama.py --prompt per_rule --negatives 30
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binary_detection.common import load_prompt, parse_args, run  # noqa: E402
from binary_detection.sinllama import SinLlama, add_arguments  # noqa: E402


def main() -> None:
    args = parse_args("sinllama", "sinllama-8b", default_lang="si", add_arguments=add_arguments)
    args.workers = 1

    prompt = load_prompt(args.prompt)
    rule_mode = hasattr(prompt, "build_rule")
    if getattr(prompt, "JSON_OUTPUT", False) and not rule_mode:
        raise SystemExit(f"SinLlama only scores YES/NO; prompt {args.prompt!r} needs JSON output")

    sl = SinLlama.from_args(args)

    def ask(system: str, user: str) -> tuple[str, float]:
        confidence, mass = sl.score_yes_no(f"{system}\n\n{user}\nAnswer:")
        answer = "YES" if confidence >= 0.5 else "NO"
        return f"{answer} (P(YES)={confidence:.3f}, P(YES)+P(NO)={mass:.3f})", confidence

    if rule_mode:
        run("sinllama", args, lambda article: sl.per_rule(article, prompt, args.margin), ask_article=True)
    else:
        run("sinllama", args, ask)


if __name__ == "__main__":
    main()
