"""Span detection on OpenAI.

Usage (from the project root):
    python span_detection/run_openai.py --prompt zero_shot --negatives 30
    python span_detection/run_openai.py --prompt few_shot --negatives 30
    python span_detection/run_openai.py --prompt per_rule --negatives 30
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from span_detection.common import load_prompt, parse_args, run  # noqa: E402


def main() -> None:
    from openai import OpenAI

    args = parse_args("openai", os.getenv("OPENAI_MODEL", "gpt-4.1"))
    prompt = load_prompt(args.prompt)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    token_confidence = getattr(prompt, "token_confidence", None)
    extra = {"logprobs": True, "top_logprobs": 20} if token_confidence else {}

    def ask(system: str, user: str):
        resp = client.chat.completions.create(
            model=args.model,
            temperature=0,
            max_tokens=prompt.MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **extra,
        )
        choice = resp.choices[0]
        raw = choice.message.content or ""
        if token_confidence and choice.logprobs:
            tokens = [(t.token, [(a.token, a.logprob) for a in t.top_logprobs]) for t in choice.logprobs.content or []]
            return raw, token_confidence(tokens)[1]
        return raw

    run("openai", args, ask)


if __name__ == "__main__":
    main()
