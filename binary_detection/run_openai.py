"""Run a prompt on OpenAI.

Confidence = P(YES) / (P(YES) + P(NO)) from the token logprobs of the answer
token, unless the prompt defines token_confidence() (per_rule: max over rules
of P(VIOLATION) at each verdict token).

Usage (from the project root):
    python binary_detection/run_openai.py                       # zero_shot on violation articles
    python binary_detection/run_openai.py --negatives 30        # plus 30 non-violation articles
    python binary_detection/run_openai.py --prompt per_rule --negatives 30
    python binary_detection/run_openai.py --model gpt-4.1-mini --limit 3 --force
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binary_detection.common import load_prompt, parse_args, run  # noqa: E402


def _word(token: str) -> str:
    return token.strip().strip('"').upper()


def p_yes_from_logprobs(tokens) -> float | None:
    """P(YES) at the last generated token that is YES or NO."""
    for tok in reversed(tokens or []):
        word = _word(tok.token)
        if word not in ("YES", "NO"):
            continue
        p_yes = sum(math.exp(t.logprob) for t in tok.top_logprobs if _word(t.token) == "YES")
        p_no = sum(math.exp(t.logprob) for t in tok.top_logprobs if _word(t.token) == "NO")
        if p_yes + p_no == 0:
            return 1.0 if word == "YES" else 0.0
        return p_yes / (p_yes + p_no)
    return None


def main() -> None:
    from openai import OpenAI

    args = parse_args("openai", os.getenv("OPENAI_MODEL", "gpt-4.1"))
    prompt = load_prompt(args.prompt)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    extra = {"response_format": {"type": "json_object"}} if getattr(prompt, "JSON_OUTPUT", False) else {}

    token_confidence = getattr(prompt, "token_confidence", None)

    def ask(system: str, user: str) -> tuple:
        resp = client.chat.completions.create(
            model=args.model,
            temperature=0,
            max_tokens=getattr(prompt, "MAX_OUTPUT_TOKENS", 5),
            logprobs=True,
            top_logprobs=20,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **extra,
        )
        choice = resp.choices[0]
        tokens = (choice.logprobs.content if choice.logprobs else None) or []
        raw = choice.message.content or ""
        if token_confidence:
            simple = [(t.token, [(a.token, a.logprob) for a in t.top_logprobs]) for t in tokens]
            confidence, details = token_confidence(simple)
            return raw, confidence, details
        return raw, p_yes_from_logprobs(tokens)

    run("openai", args, ask)


if __name__ == "__main__":
    main()
