"""Prompts for the two-step pipelines (gpt-4.1, JSON output).

Codes then highlights:
  step 1 CODES      article → the rules it breaks (no excerpts)
  step 2 LOCATE     article + those rules → the exact words that break each one
Highlights then codes:
  step 1 PASSAGES   article → the exact words that break the code (no rule ids)
  step 2 ASSIGN     article + those passages → the rule(s) each passage breaks, or none

Every step sees the full code of conduct. The few-shot variants add examples in
the article's language built from span_detection/prompts/few_shot_examples.json
(the same 3 articles per language as the one-step few_shot prompt; they are
excluded from evaluation).
"""

from __future__ import annotations

import json

from binary_detection.prompts.zero_shot import CODE, DEFINITIONS
from span_detection.prompts import few_shot as span_few_shot
from span_detection.prompts.zero_shot import EXCERPT_RULES, USER, _fields

EXAMPLES = span_few_shot.EXAMPLES
EXCLUDE_IDS = span_few_shot.EXCLUDE_IDS

# Step-2 examples of a passage that breaks no rule (routine reporting), so ASSIGN learns to reject.
CLEAN_PASSAGES = {
    "si": {
        "excerpt": "මෙසේ රිමාන්ඩ් බාරයට පත් කරන ලද්දේ නින්දානේ චුට්ටේ නොහොත් අමිත චතුරංග සමරනායක නමැත්තෙකි.",
        "problem": "Names the accused.",
        "explanation": "Naming an accused person in a court report is routine reporting; no rule applies.",
    },
    "en": {
        "excerpt": "Ajinkya Rahane has announced his retirement from all forms of cricket",
        "problem": "Names a person.",
        "explanation": "Names a public figure in his public role; no rule applies.",
    },
}

CONTEXT = f"""You are a media ethics reviewer for Sri Lankan newspapers.
Articles may be in English or Sinhala.

CODE OF CONDUCT (JSON, keyed by rule_id):
{json.dumps(CODE, ensure_ascii=False, indent=1)}

DEFINITIONS of the exception names used in the rules:
{json.dumps(DEFINITIONS, ensure_ascii=False, indent=1)}

A rule is broken when all of its violation_conditions hold and none of its exceptions apply."""

CODES_SYSTEM = f"""{CONTEXT}

Task: decide which rules of the code the article breaks. Do not quote the article; only name the rules.

Reply with one JSON object and nothing else:
{{"broken_rules": [{{"rule_id": "<rule_id>", "reason": "one short sentence in English"}}]}}
List every broken rule once. If the article breaks no rule, reply {{"broken_rules": []}}."""

LOCATE_SYSTEM = f"""{CONTEXT}

Task: another reviewer decided that the article breaks the rules listed under RULES FOUND BROKEN.
For each of those rules, find the exact places in the article that break it. Only use the listed rules.
If you cannot find any words in the article that break a listed rule, leave that rule out.

{EXCERPT_RULES}

Reply with one JSON object and nothing else:
{{"highlights": [{{"rule_id": "<one of the listed rule_ids>",
                  "excerpt": "exact words from the article",
                  "explanation": "one short sentence in English"}}]}}"""

PASSAGES_SYSTEM = f"""{CONTEXT}

Task: mark the exact places where the article breaks the code. Do not name the rules; only give the
words and say in plain words what is wrong with them.

{EXCERPT_RULES}

Reply with one JSON object and nothing else:
{{"passages": [{{"excerpt": "exact words from the article", "problem": "one short sentence in English"}}]}}
If the article breaks no rule, reply {{"passages": []}}."""

ASSIGN_SYSTEM = f"""{CONTEXT}

Task: another reviewer marked the numbered PASSAGES of the article as possibly breaking the code.
For each passage decide which rules it breaks, reading it in the context of the whole article.
A passage may break one rule, several rules or none: give an empty list when it breaks no rule.

Reply with one JSON object and nothing else:
{{"passages": [{{"index": <passage number>, "rule_ids": ["<rule_id>", ...], "explanation": "one short sentence in English"}}]}}
Include every passage number once."""

LOCATE_USER = """{article}

RULES FOUND BROKEN:
{rules}

Find the words that break each of these rules. Reply with the JSON object."""

ASSIGN_USER = """{article}

PASSAGES:
{passages}

Which rules does each passage break? Reply with the JSON object."""


def _article(article: dict) -> str:
    return USER.format(**_fields(article)).rsplit("\n\n", 1)[0]


def _rule_list(rule_ids: list[str]) -> str:
    return "\n".join(f"- {rid}: {CODE[rid]['title']}" for rid in rule_ids)


def _passage_list(passages: list[dict]) -> str:
    return "\n".join(f"{i}. \"{p['excerpt']}\" ({p['problem']})" for i, p in enumerate(passages, start=1))


def _example_rules(ex: dict) -> list[dict]:
    seen: dict[str, str] = {}
    for v in ex["violations"]:
        seen.setdefault(v["rule_id"], v["explanation"])
    return [{"rule_id": rid, "reason": reason} for rid, reason in seen.items()]


def _with_examples(system: str, blocks: list[str]) -> str:
    body = "\n\n".join(f"--- EXAMPLE {i} ---\n{b}" for i, b in enumerate(blocks, start=1))
    return f"{system}\n\nEXAMPLES with the correct reply:\n\n{body}"


def _codes_examples(lang: str) -> list[str]:
    return [
        f"{USER.format(heading=ex['heading'], content=ex['content'])}\n"
        f"{json.dumps({'broken_rules': _example_rules(ex)}, ensure_ascii=False)}"
        for ex in EXAMPLES[lang]
    ]


def _locate_examples(lang: str) -> list[str]:
    return [
        LOCATE_USER.format(article=_article(ex), rules=_rule_list([r["rule_id"] for r in _example_rules(ex)]))
        + "\n" + json.dumps({"highlights": ex["violations"]}, ensure_ascii=False)
        for ex in EXAMPLES[lang] if ex["violations"]
    ]


def _passages_examples(lang: str) -> list[str]:
    return [
        f"{USER.format(heading=ex['heading'], content=ex['content'])}\n"
        + json.dumps({"passages": [{"excerpt": v["excerpt"], "problem": v["explanation"]} for v in ex["violations"]]},
                     ensure_ascii=False)
        for ex in EXAMPLES[lang]
    ]


def _assign_examples(lang: str) -> list[str]:
    blocks = []
    for ex in EXAMPLES[lang]:
        if ex["violations"]:
            passages = [{"excerpt": v["excerpt"], "problem": v["explanation"]} for v in ex["violations"]]
            answer = [{"index": i, "rule_ids": [v["rule_id"]], "explanation": v["explanation"]}
                      for i, v in enumerate(ex["violations"], start=1)]
        else:
            clean = CLEAN_PASSAGES[lang]
            assert clean["excerpt"] in ex["content"]
            passages = [clean]
            answer = [{"index": 1, "rule_ids": [], "explanation": clean["explanation"]}]
        blocks.append(
            ASSIGN_USER.format(article=_article(ex), passages=_passage_list(passages))
            + "\n" + json.dumps({"passages": answer}, ensure_ascii=False)
        )
    return blocks


def _systems(system: str, examples) -> dict[tuple[bool, str], str]:
    return {
        (few, lang): _with_examples(system, examples(lang)) if few else system
        for few in (False, True) for lang in EXAMPLES
    }


SYSTEMS = {
    "codes": _systems(CODES_SYSTEM, _codes_examples),
    "locate": _systems(LOCATE_SYSTEM, _locate_examples),
    "passages": _systems(PASSAGES_SYSTEM, _passages_examples),
    "assign": _systems(ASSIGN_SYSTEM, _assign_examples),
}


def lang_of(article: dict) -> str:
    return article.get("lang") if article.get("lang") in EXAMPLES else "en"


def codes_messages(article: dict, few: bool) -> tuple[str, str]:
    return SYSTEMS["codes"][(few, lang_of(article))], USER.format(**_fields(article))


def locate_messages(article: dict, rule_ids: list[str], few: bool) -> tuple[str, str]:
    return SYSTEMS["locate"][(few, lang_of(article))], LOCATE_USER.format(article=_article(article), rules=_rule_list(rule_ids))


def passages_messages(article: dict, few: bool) -> tuple[str, str]:
    return SYSTEMS["passages"][(few, lang_of(article))], USER.format(**_fields(article))


def assign_messages(article: dict, passages: list[dict], few: bool) -> tuple[str, str]:
    return SYSTEMS["assign"][(few, lang_of(article))], ASSIGN_USER.format(article=_article(article), passages=_passage_list(passages))
