"""Run a prompt on Gemini.

Usage (from the project root):
    python binary_detection/run_gemini.py                       # zero_shot on violation articles
    python binary_detection/run_gemini.py --negatives 30        # plus 30 non-violation articles
    python binary_detection/run_gemini.py --prompt per_rule --negatives 30
    python binary_detection/run_gemini.py --model gemini-3.5-pro --limit 3 --force
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binary_detection.common import load_prompt, parse_args, run  # noqa: E402


def main() -> None:
    from google import genai
    from google.genai import types

    args = parse_args("gemini", os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    prompt = load_prompt(args.prompt)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    extra = {"response_mime_type": "application/json"} if getattr(prompt, "JSON_OUTPUT", False) else {}

    def ask(system: str, user: str) -> str:
        resp = client.models.generate_content(
            model=args.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                **extra,
            ),
        )
        return resp.text or ""

    run("gemini", args, ask)


if __name__ == "__main__":
    main()
