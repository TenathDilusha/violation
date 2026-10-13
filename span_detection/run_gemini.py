"""Span detection on Gemini.

Usage (from the project root):
    python span_detection/run_gemini.py --prompt zero_shot --negatives 30
    python span_detection/run_gemini.py --prompt few_shot --negatives 30
    python span_detection/run_gemini.py --prompt per_rule --negatives 30
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from span_detection.common import parse_args, run  # noqa: E402


def main() -> None:
    from google import genai
    from google.genai import types

    args = parse_args("gemini", os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def ask(system: str, user: str) -> str:
        resp = client.models.generate_content(
            model=args.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                response_mime_type="application/json",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return resp.text or ""

    run("gemini", args, ask)


if __name__ == "__main__":
    main()
