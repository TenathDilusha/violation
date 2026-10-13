"""Zero-shot rule detection plus guidance written from zero_shot's failures (2026-10-07).

Identical to zero_shot (same code, same JSON reply) except for the GUIDANCE block
between the code and the reply format. Each point restates a rule's
violation_conditions, exceptions or scope for a mistake gpt-4.1 made with zero_shot:
attributed claims flagged as 2.2.a, opinion pieces and attributed reports flagged
as 2.5, headlines that state allegations as fact missed (2.5), flippant or lurid
treatment folded into 6.1.b (6.1.a missed), involved relatives flagged under
6.2.iii, names and towns flagged as 7.1 while home addresses and private
relationships were missed.
"""

from __future__ import annotations

from rule_detection.prompts import zero_shot

NAME = "zero_shot_v2"
JSON_OUTPUT = zero_shot.JSON_OUTPUT
MAX_OUTPUT_TOKENS = zero_shot.MAX_OUTPUT_TOKENS
parse = zero_shot.parse

GUIDANCE = """HOW TO APPLY THE RULES (read before answering):

Sourcing (2.2.a, 2.2.b, 2.2.c)
- A claim is sourced when the article attributes it to anyone: a named or unnamed official, the police, a
  court, a ministry, a report, a survey, a news agency, "sources", or "it is reported" (in Sinhala e.g.
  "වාර්තා වේ", "සඳහන්ය", "ආරංචි මාර්ග සඳහන් කළේය", "පොලිසිය කියයි"). Vague attribution is weak sourcing,
  not a breach of 2.2.a.
- Routine figures and background (official statistics, amounts in a government decision, events anyone can
  check) are not significant claims that need a source.
- Flag 2.2.a only for a significant claim or allegation, usually of wrongdoing by identifiable people or
  groups, that the paper states in its own voice with no attribution at all.
- 2.2.b: flag when the only source for significant claims is a person or group whose identity, role or
  standing the article does not explain.

Comment, conjecture and fact (2.5)
- Opinion in quotation marks or attributed ("X said", "according to", "it is reported that") is the
  source's, not the paper's.
- Editorials, opinion columns and letters may give judgements and recommendations ("poor management",
  "must be reformed"); that is not 2.5. In them, flag only unproven allegations of wrongdoing by people
  stated as established fact.
- Compare the headline with the body. Flag 2.5 when the headline or the paper's own words state as fact what
  the body only reports as alleged, suspected or attributed: calling a suspect a killer or criminal, stating
  guilt, a motive, or a "main cause" that no source states.

Shocking issues (6.1.a, 6.1.b, 6.1.c, 6.5)
- 6.1.b is the method of the killing: the weapon, injuries or actions, even in one phrase (stabbed, throat
  cut, beaten with an iron bar), headline included. The formal wording of a death sentence ("hanged by the
  neck until dead") is not the method of the murder.
- 6.1.a is the tone and needless distress, separate from the method; both often apply to the same article.
  Flag slang, flippant or mocking headlines about a death or a sentence, lurid words in the paper's own voice
  ("cruelly", "went to hell"), and gruesome detail of the body or the scene (stench, decomposition, the body
  being dragged or dumped).
- 6.1.c: detail that could serve as instructions or glamorise the crime; a seized quantity alone is weak.
- 6.5: only when the death is reported as a suicide or attempted suicide.

Identifying people in crime reports (6.2.i, 6.2.ii, 6.2.iii)
- 6.2.iii: the relative must be identifiable and uninvolved. A relative who is also a suspect, accused,
  arrested or detained, who helped the suspect, or who is the victim, is not covered. Famous relatives count
  ("the son of former Minister X"). A convicted person's own job or workplace is not a relative.
- 6.2.i only when a sex crime is reported; 6.2.ii only for an accused person under 16.

Privacy (7.1, 7.4, 7.5)
- Naming a victim or suspect with age, occupation and town or village is normal reporting, not 7.1.
- Flag 7.1 for private details of an identifiable person: a street-level home address or the exact place
  they lodge, health (illness, addiction), a private or romantic relationship, family problems; unless the
  person or their family gave the detail themselves (quoted) or the public interest requires it.
- 7.4 only for children (under 18); 7.5 only for patients or people held in an institution, not its staff.

Identity details (6.4.i, 6.4.ii): only race, colour, caste, religion, sex, sexual orientation, illness or
disability. Age, occupation, nationality and place of residence are not covered.

Check the heading as well as the text, and list every broken rule: an article often breaks two."""

SYSTEM = zero_shot.SYSTEM.replace(
    "\n\nReply with one JSON object and nothing else:",
    f"\n\n{GUIDANCE}\n\nReply with one JSON object and nothing else:",
    1,
)
assert GUIDANCE in SYSTEM


def build(article: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt)."""
    return SYSTEM, zero_shot.USER.format(**zero_shot._fields(article))
