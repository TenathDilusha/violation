"""Per-rule checklist prompt: a verdict + evidence quote for every rule.

The model returns JSON:
    {"rules": {"<rule_id>": {"verdict": "VIOLATION|NO_VIOLATION",
                             "evidence": "<sentence copied from the article>",
                             "explanation": "<one sentence>"}, ...},
     "violation": "YES|NO"}

The article answer is YES when any rule is VIOLATION.

With token logprobs (OpenAI), each rule gets P(VIOLATION) from its verdict
token, and the article confidence is the maximum over rules.

Base models that can't write JSON (SinLlama) use build_rule(): one prompt per
rule ending in "Answer:", scored on YES vs NO; the runner assembles the same JSON.
"""

from __future__ import annotations

import json
import math
import re
from bisect import bisect_right

from binary_detection.prompts.zero_shot import CODE, DEFINITIONS

NAME = "per_rule"
JSON_OUTPUT = True
MAX_OUTPUT_TOKENS = 4000

RULE_IDS = list(CODE)
VERDICTS = ("VIOLATION", "NO_VIOLATION")
POSITIVE_VERDICTS = ("VIOLATION",)

CONTEXT = f"""You are a media ethics reviewer for Sri Lankan newspapers.
Check a news article against EVERY rule of the Code of Professional Practice below.
Articles may be in English or Sinhala.

CODE OF CONDUCT (JSON, keyed by rule_id):
{json.dumps(CODE, ensure_ascii=False, indent=1)}

DEFINITIONS of the exception names used in the rules:
{json.dumps(DEFINITIONS, ensure_ascii=False, indent=1)}"""

# Verdict definitions and evidence rules, shared with span_detection/prompts/per_rule.py.
VERDICT_GUIDE = """For each rule choose one verdict:
- VIOLATION: the article's own text shows that all violation_conditions hold and no exception applies.
- NO_VIOLATION: everything else, including when the rule's situation is not in the article.

Evidence rules (apply them before choosing VIOLATION):
1. You must be able to quote a sentence from the article that shows the problematic content itself
   (the name of a victim or relative, the method detail, the private information, the loaded wording).
   If you cannot quote such a sentence, the verdict is NO_VIOLATION.
2. Missing information is never evidence. Do not flag a rule because the article does not state something
   (an age, a source, consent, verification, public interest). Hypotheticals such as "if the accused were
   under 16" or "if the victim did not consent" are not allowed.
3. The rule's situation must actually be in the article: 6.2.i needs a reported sex crime; 6.2.ii needs an
   accused person the article shows is a child; 7.4 needs an identifiable child; 6.5 needs a suicide;
   6.1.b / 6.1.c need a described method; 6.2.iii needs a relative of an accused or convicted person.
4. For 2.2.a / 2.2.b / 2.2.c: attribution to police, courts, officials, named people, institutions,
   experts or the paper's correspondents counts as a credible, relevant source. Routine reporting of
   arrests, court orders, statements or events is not unsourced. Flag only claims presented as fact with
   no attribution at all, or sources the article itself describes as unreliable or unconnected.
5. For 2.5: flag only when the reporter's own opinion or speculation is presented as fact; quoted
   opinions, headlines summarising the story and ordinary descriptive words do not count.
6. For 7.1: naming people in their public, official or legal role (accused persons, officials, public
   figures, witnesses) is not by itself an intrusion; flag only private details such as home address,
   health, family life or private messages of an identifiable person."""

SYSTEM = f"""{CONTEXT}

{VERDICT_GUIDE}

Reply with one JSON object and nothing else, in this shape:
{{
  "rules": {{
    "<rule_id>": {{"verdict": "VIOLATION | NO_VIOLATION",
                  "evidence": "the sentence from the article that shows it, copied exactly in the article's language; empty for NO_VIOLATION",
                  "explanation": "one short sentence in English; may be empty for NO_VIOLATION"}}
  }},
  "violation": "YES or NO"
}}
Include every rule_id exactly once, in this order: {", ".join(RULE_IDS)}.
"violation" is YES if any rule is VIOLATION, otherwise NO."""

USER = """ARTICLE
Heading: {heading}

{content}

Check this article against every rule and reply with the JSON object."""


def build(article: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one article."""
    user = USER.format(
        heading=(article.get("heading") or "").strip(),
        content=(article.get("content") or "").strip(),
    )
    return SYSTEM, user


RULE_PROMPT = """You are a media ethics reviewer for Sri Lankan newspapers.
Decide whether the news article below breaks ONE rule of the Code of Professional Practice.
The article may be in English or Sinhala.

ARTICLE
Heading: {heading}

{content}

RULE {rule_id} (JSON):
{rule}
{definitions}
The rule is broken only when the article's own text shows all of its violation_conditions and none of its
exceptions apply. Missing information is not evidence.

Does this article break rule {rule_id} ({title})? Answer YES or NO.
Answer:"""

EVIDENCE_SUFFIX = " YES\nThe sentence from the article that breaks the rule:"


def build_rule(article: dict, rule_id: str) -> str:
    """Single-rule prompt for YES/NO next-token scoring."""
    rule = CODE[rule_id]
    definitions = {name: DEFINITIONS[name] for name in rule.get("exceptions", []) if name in DEFINITIONS}
    return RULE_PROMPT.format(
        heading=(article.get("heading") or "").strip(),
        content=(article.get("content") or "").strip(),
        rule_id=rule_id,
        title=rule["title"],
        rule=json.dumps(rule, ensure_ascii=False, indent=1),
        definitions=f"DEFINITIONS:\n{json.dumps(definitions, ensure_ascii=False, indent=1)}\n" if definitions else "",
    )


def _load(text: str) -> dict | None:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_details(text: str) -> dict | None:
    """{"rules": [{rule_id, verdict, evidence, explanation}, ...], "model_answer": "YES|NO"}."""
    data = _load(text)
    if data is None:
        return None
    rules = data.get("rules") or {}
    if isinstance(rules, list):
        rules = {r.get("rule_id"): r for r in rules if isinstance(r, dict)}
    out = []
    for rule_id, r in rules.items():
        if not isinstance(r, dict):
            continue
        verdict = str(r.get("verdict") or "").strip().upper()
        out.append({
            "rule_id": str(rule_id),
            "verdict": verdict if verdict in VERDICTS else "UNPARSED",
            "evidence": (r.get("evidence") or "").strip(),
            "explanation": (r.get("explanation") or "").strip(),
        })
    model_answer = str(data.get("violation") or "").strip().upper()
    return {"rules": out, "model_answer": model_answer if model_answer in ("YES", "NO") else None}


_VERDICT_START = re.compile(r'"([0-9][^"]*)"\s*:\s*\{\s*"verdict"\s*:\s*"')


def _verdict_side(value_prefix: str) -> bool | None:
    """True if the prefix can only start a positive verdict, False if only NO_VIOLATION."""
    part = value_prefix.upper()
    if not part:
        return None
    matches = {v for v in VERDICTS if v.startswith(part) or part.startswith(v)}
    if not matches:
        return None
    if matches <= set(POSITIVE_VERDICTS):
        return True
    if matches == {"NO_VIOLATION"}:
        return False
    return None


def token_confidence(tokens: list[tuple[str, list[tuple[str, float]]]]) -> tuple[float | None, dict]:
    """tokens = [(token, [(alternative_token, logprob), ...]), ...] as generated.

    Returns (max over rules of P(positive verdict), {"rule_p": {rule_id: p}}).
    """
    text, starts = "", []
    for tok, _ in tokens:
        starts.append(len(text))
        text += tok
    rule_p = {}
    for m in _VERDICT_START.finditer(text):
        idx = bisect_right(starts, m.end()) - 1
        offset = m.end() - starts[idx]
        if offset >= len(tokens[idx][0]):
            idx, offset = idx + 1, 0
        if idx >= len(tokens):
            continue
        head = tokens[idx][0][:offset]
        pos = neg = 0.0
        for alt, logprob in tokens[idx][1]:
            if not alt.startswith(head):
                continue
            side = _verdict_side(alt[offset:])
            if side is True:
                pos += math.exp(logprob)
            elif side is False:
                neg += math.exp(logprob)
        if pos + neg > 0:
            rule_p[m.group(1)] = pos / (pos + neg)
    confidence = max(rule_p.values()) if rule_p else None
    return confidence, {"rule_p": rule_p}


def parse(text: str) -> str:
    details = parse_details(text)
    if details is None:
        return "UNPARSED"
    if details["rules"]:
        return "YES" if any(r["verdict"] in POSITIVE_VERDICTS for r in details["rules"]) else "NO"
    return details["model_answer"] or "UNPARSED"
