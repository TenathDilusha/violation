"""Few-shot span detection: zero-shot instructions + 3 fixed examples in the article's language.

Examples live in few_shot_examples.json (2 violations + 1 non-violation per
language, the same articles as rule_detection) with short exact excerpts. The
example articles are excluded from evaluation (EXCLUDE_IDS).

Base models see the same examples as sentence questions: the sentence holding
each excerpt is answered YES (and its rule is given), the last other sentence NO.
"""

from __future__ import annotations

import json
from pathlib import Path

from span_detection.locate import locate, sentences
from span_detection.prompts import zero_shot

NAME = "few_shot"
JSON_OUTPUT = zero_shot.JSON_OUTPUT
MAX_OUTPUT_TOKENS = zero_shot.MAX_OUTPUT_TOKENS
RULE_IDS = zero_shot.RULE_IDS
CANDIDATES = zero_shot.CANDIDATES
SENTENCE_QUESTION = zero_shot.SENTENCE_QUESTION
RULE_QUESTION = zero_shot.RULE_QUESTION
parse = zero_shot.parse

EXAMPLES = json.loads(
    (Path(__file__).resolve().parent / "few_shot_examples.json").read_text(encoding="utf-8")
)
EXCLUDE_IDS = {ex["article_id"] for examples in EXAMPLES.values() for ex in examples}


def _chat_examples(lang: str) -> str:
    return "\n\n".join(
        f"--- EXAMPLE {i} ---\n{zero_shot.USER.format(heading=ex['heading'], content=ex['content'])}\n"
        f"{json.dumps({'violations': ex['violations']}, ensure_ascii=False)}"
        for i, ex in enumerate(EXAMPLES[lang], start=1)
    )


def _sentence_examples(lang: str) -> str:
    parts = []
    for i, ex in enumerate(EXAMPLES[lang], start=1):
        content = ex["content"].strip()
        bounds = sentences(content)
        qa, marked = [], set()
        for v in ex["violations"]:
            for start, _ in locate(content, v["excerpt"]):
                s, e = next(b for b in bounds if b[0] <= start < b[1])
                marked.add((s, e))
                sentence = content[s:e]
                qa.append(f"{SENTENCE_QUESTION.format(sentence=sentence)} YES")
                qa.append(f"{RULE_QUESTION.format(sentence=sentence)} {v['rule_id']}")
        other = next((b for b in reversed(bounds) if b not in marked), None)
        if other:
            qa.append(f"{SENTENCE_QUESTION.format(sentence=content[other[0]:other[1]])} NO")
        block = zero_shot.article_block({"heading": ex["heading"], "content": content})
        parts.append(f"--- EXAMPLE {i} ---\n{block}\n\n" + "\n\n".join(qa))
    return "\n\n".join(parts)


SYSTEM = {
    lang: f"{zero_shot.SYSTEM}\n\nEXAMPLES of articles with the correct reply:\n\n{_chat_examples(lang)}"
    for lang in EXAMPLES
}
SENTENCE_CONTEXT = {
    lang: f"{zero_shot.CONTEXT}\n\nEXAMPLES of articles with correct answers:\n\n{_sentence_examples(lang)}"
    for lang in EXAMPLES
}


def _lang(article: dict) -> str:
    return article.get("lang") if article.get("lang") in EXAMPLES else "en"


def build(article: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt); examples match the article's language (default en)."""
    return SYSTEM[_lang(article)], zero_shot.USER.format(**zero_shot._fields(article))


def sentence_prefix(article: dict) -> str:
    return f"{SENTENCE_CONTEXT[_lang(article)]}\n\n--- NOW THIS ARTICLE ---\n{zero_shot.article_block(article)}\n\n"
