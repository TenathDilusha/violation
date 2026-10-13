# Detecting Code of Conduct Violations in Sri Lankan News with LLMs

Slide content, from the beginning of the project to where it stands now. Each slide's title is its main point, and the body is the evidence for it.
All numbers come from the project database (`violations` schema) and the viewer (`app.py`).
In the tables, **P / R / F1 / F2** means precision / recall / F1 / F2, in %, as the mean of 3 runs unless stated.

**Flow:** goal → code of conduct → dataset → the 9 approaches and 3 models → how they are evaluated → results for English, then Sinhala → why some approaches work and others don't → Sinhala vs English code → other things tried → confidence scores → top 4 per language → lessons and limits → future work

---



## Slide 1 — Can an LLM check Sri Lankan news against the Editors' Guild code, in Sinhala and English?

**The goal:** for every article, answer three questions automatically.


| Question                | Output                                  |
| ----------------------- | --------------------------------------- |
| Does it break the code? | Violation / no violation                |
| Which rules?            | Rule ids, e.g. `2.2.a`, `6.1.b`         |
| Which words?            | The exact passages that break each rule |


- **Reference:** the Editors' Guild of Sri Lanka **Code of Professional Practice**.
- **Languages:** **Sinhala** and **English**.
- **What was tried:**
  - 3 models: **gpt-4.1**, Gemini 3.5 Flash and SinLlama 8B;
  - **9 ways of asking:** 3 tasks × 3 prompt styles;
  - many improvements on top.
- **The result:** about **90% of violations caught** at about 86–91% precision, using confidence scores built from gpt-4.1's answers.

---



## Slide 2 — The first step was turning the code into rules a model can judge from the article text alone

**What I built:** `code_of_conduct/code_of_conduct.json`, with **17 rules in 5 groups**.


| Group                | Rules                                                                                                            |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Accuracy             | 2.2.a unsourced claims · 2.2.b non-credible source · 2.2.c irrelevant source · 2.5 mixing comment and fact       |
| Harm and sensitivity | 6.1.a sensational treatment · 6.1.b murder-method detail · 6.1.c crime-method detail · 6.5 suicide-method detail |
| Crime identification | 6.2.i naming sex-crime victims · 6.2.ii naming young accused · 6.2.iii identifying relatives of the accused      |
| Discrimination       | 6.3 communal discord · 6.4.i pejorative references · 6.4.ii irrelevant identity details                          |
| Privacy              | 7.1 private and family life · 7.4 children · 7.5 people in hospitals and institutions                            |


- Each rule records its **obligation**, the **conditions** for a breach, its **exceptions** (public interest, consent …) and **evidence cues**. A rule is broken only when all its conditions hold and no exception applies.
- **Left out:** sections 2.1, 2.3, 2.4, 3, 4, 5, 7.2, 7.3, 8 and 9, because they need fact-checking, photos or newsroom knowledge.
- **Two labels only:** `VIOLATION` / `NO_VIOLATION`. A third label, "INDETERMINATE", was tried and removed. Missing information (e.g. consent) is never treated as evidence.
- **A Sinhala translation** (`code_of_conduct_si.json`) with the same rule ids, for testing Sinhala rules on Sinhala articles (Slide 9).

---



## Slide 3 — We built the dataset by manually annotating each article at three levels: violation or not, which rule, and which words

**Annotation, in three levels per article**


| Level            | Question the annotator answers                    | Example                                                       |
| ---------------- | ------------------------------------------------- | ------------------------------------------------------------- |
| 1. Violation?    | Does the article break the code of conduct?       | `VIOLATION` / `NO_VIOLATION`                                  |
| 2. Violated rule | Which of the 17 rules does it break?              | `6.1.b`, `2.5`                                                |
| 3. Violated part | Which exact words in the article break that rule? | The sentences describing how a murder was carried out → 6.1.b |


**The annotated dataset: 106 articles**


|                                     | Sinhala | English | Total   |
| ----------------------------------- | ------- | ------- | ------- |
| Violations (all 3 levels annotated) | 15      | 32      | **47**  |
| Clean articles                      | 29      | 30      | **59**  |
| **Total**                           | **44**  | **62**  | **106** |


- **Every level is tested:** level 1 tests detection, level 2 tests the rules named, and level 3 tests the highlighted words.
- **Sources of violations:**
  - articles reviewed on the platform (19);
  - violation spreadsheets from The Island, Daily Mirror, Daily News and Ceylon Today (28), each mapped to a rule, with the words marked and confirmed by hand.
- **Same test set every time:** every approach sees the same 106 articles.

---



## Slide 4 — Each article was tested 9 ways (3 tasks × 3 prompt styles) on 3 models

**3 tasks: how much the model must say**


| Task               | The model returns                 | Counts as "violation" when               |
| ------------------ | --------------------------------- | ---------------------------------------- |
| Violated? (YES/NO) | `YES` / `NO`                      | it says YES                              |
| Rule detection     | `["2.2.a", "6.4.ii"]`             | it names at least one rule               |
| Highlights         | `{"rule": "2.5", "excerpt": "…"}` | it highlights words found in the article |


**3 prompt styles: how the model is guided**


| Prompt    | What the prompt holds                                                            |
| --------- | -------------------------------------------------------------------------------- |
| Zero-shot | The code of conduct and the article                                              |
| Few-shot  | The same, plus 3 worked example articles per language with their correct answers |
| Per-rule  | The model goes through all 17 rules one by one and gives a verdict for each      |


**3 models**


| Model                     | Type                                     | Runs |
| ------------------------- | ---------------------------------------- | ---- |
| **gpt-4.1** (OpenAI)      | Large, general-purpose, via API          |      |
| Gemini 3.5 Flash (Google) | Fast, cheaper model, via API             |      |
| SinLlama 8B               | Small open Sinhala model, on a local GPU |      |


---



## Slide 5 — We compare the 9 ways by F2 on the same articles, because a missed violation costs more than a false alarm

**The metrics** (a "positive" is an article flagged as a violation)


| Metric        | Meaning                                                     | Why it matters here                                                                                                                             |
| ------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Precision** | Of the flagged articles, the share that are real violations | Low precision means reviewers waste time on false alarms                                                                                        |
| **Recall**    | Of the real violations, the share that get flagged          | Low recall means violations are published unnoticed                                                                                             |
| **F1**        | Balance of precision and recall (harmonic mean)             | One number when both matter equally                                                                                                             |
| **F2**        | Like F1, but recall counts **twice** as much                | **Main score:** a reviewer can dismiss a false alarm, but a missed violation is never seen again                                                |
| Accuracy      | Share of all articles judged correctly                      | Reported, but not used to rank: flagging nothing scores 56% here, and about 99% on the real platform, where under 1% of articles are violations |


**How the 9 ways are compared fairly**

- **Same articles:** every way is scored on the same common articles, per model.
- **3 runs each:** gpt-4.1 gives different answers to the same prompt (3 F2 points apart on average, up to 6), so every number is the mean of 3 runs.
- **Ranked by F2,** with a 95% interval from a bootstrap over articles (1,000 resamples). Two approaches whose intervals overlap are treated as **tied**.
- **Languages separately:** English and Sinhala behave very differently (next two slides).
- **Rules and words too:** besides the violation decision, the named rules (level 2) and highlighted words (level 3) are checked against the annotation.

---



## Slide 6 — In English, rule detection with few-shot examples is the best of the 9 ways, because F2 rewards catching violations

**62 English articles (32 violations, 30 clean). F2, mean of 3 runs.**


| Task · prompt                 | gpt-4.1 | Gemini 3.5 Flash* |
| ----------------------------- | ------- | ----------------- |
| YES/NO · zero-shot            | 65      | 0                 |
| YES/NO · few-shot             | 84      | 26                |
| YES/NO · per-rule             | 58      | 0                 |
| Rule detection · zero-shot    | 87      | 0                 |
| **Rule detection · few-shot** | **89**  | 26                |
| Rule detection · per-rule     | 60      | 0                 |
| Highlights · zero-shot        | 85      | 28                |
| Highlights · few-shot         | 84      | 28                |
| Highlights · per-rule         | 76      | 0                 |


\*Gemini ran out of credits and answered only 34 of the 62 English articles (4 of the 32 violations), so its English scores can't be compared.

**Why F2 picks the winner**
- **F2 penalises missed violations:**
  - YES/NO zero-shot has 95% precision but only **65 F2**, because it misses 40% of the violations (13 of 32 per run).
  - A reviewer would never see those articles.
- **F2 makes the right trade-off where F1 doesn't:**
  - F1 would pick rule detection zero-shot (F1 90 vs 89): it has fewer false alarms (1.0 vs 3.3 per run) but catches fewer violations (27.0 vs 28.3 per run).
  - **F2 picks few-shot (89 vs 87):** catching more violations is worth a couple of extra false alarms, which a reviewer dismisses in seconds.
- **In English, precision barely varies:** it is 90–96 in every format, so F2 is effectively ranking the 9 ways by how many violations they catch.

---



## Slide 7 — In Sinhala, gpt-4.1 reaches F2 86 with rule detection, while the other two models stay far behind

**44 Sinhala articles (15 violations, 29 clean). F2, mean of 3 runs (Gemini: 1 run).**


| Task · prompt                  | gpt-4.1 | Gemini 3.5 Flash | SinLlama 8B |
| ------------------------------ | ------- | ---------------- | ----------- |
| YES/NO · zero-shot             | 58      | 0                | 22          |
| YES/NO · few-shot              | 72      | 24               | 36          |
| YES/NO · per-rule              | 58      | 16               | 49          |
| **Rule detection · zero-shot** | **86**  | 29               | 72 †        |
| Rule detection · few-shot      | 83      | 36               | 43          |
| Rule detection · per-rule      | 64      | 23               | 49          |
| Highlights · zero-shot         | 84      | 35               | 24          |
| Highlights · few-shot          | 77      | **50**           | 58          |
| Highlights · per-rule          | 70      | 23               | 49          |


† SinLlama flagged **all 44** articles, including all 29 clean ones.

- **gpt-4.1 is the only model that works:**
  - best F2 is **86** (rule detection, zero-shot), close to its English best of 89;
  - rule detection and highlights (77–86) beat YES/NO (58–72), as in English.
- **Gemini 3.5 Flash:** best F2 is only **50** (highlights, few-shot), and YES/NO zero-shot scores **0**, because it never says YES.
- **SinLlama 8B:**
  - its best F2 of 72 comes only from flagging every article;
  - its other formats score 22–58;
  - all three per-rule formats score exactly 49, the same answers regardless of the article.

---



## Slide 8 — How much the model is asked to say decides whether it commits to a violation

**1. YES/NO is too cautious: high precision, low recall**

- **Evidence:** precision 89–96, recall 53–81 (English and Sinhala).
- **Reason:** one word must carry the whole accusation, and the model only says YES when it is sure. It often "leans YES" but answers NO: its P(YES) is just a few percent on many real violations (Slide 12).

**2. Rule detection and highlights make the model search: high recall**

- **Evidence:** recall 81–93 in both languages.
- **Reason:** "list any rules that are broken" turns the task into a search for evidence. Once it finds a matching passage, it commits.
- **Cost:** in Sinhala, the same search over-matches (7–10 false alarms per run).

**3. Per-rule is too strict: it misses the most**

- **Evidence:** recall 53–72, the lowest (or tied lowest) of the three styles in every task.
- **Reason:** 17 separate checks, each needing all of a rule's conditions, so the model says "no" to every single one. The signal is still there, though: the highest per-rule P(VIOLATION) raises rule detection F2 from 62 to 86.

**4. Few-shot examples lower the bar**

- **Evidence:**
  - YES/NO recall: English 60 → 81, Sinhala 53 → 69;
  - Sinhala rule detection false alarms: 7.7 → 10.3 per run.
- **Reason:** the examples help when the model is too cautious, and add false alarms when it already over-flags.

**5. Model size and training matter more than speaking Sinhala**

- **Evidence:** gpt-4.1 F2 86 in Sinhala vs Gemini Flash 50 and SinLlama 72 (SinLlama's 72 comes from flagging everything).
- **Reason:** SinLlama is not instruction-tuned, so it can't apply a long rulebook. Judging the code needs a large model that follows instructions.

**6. Why Sinhala precision is lower (possible reasons, not yet tested)**

- **The model:** its understanding of Sinhala is weaker.
- **Writing style:** emotive Sinhala wording may trigger 6.1.a and 2.5.
- **The clean articles:** the Sinhala ones were human-reviewed and may be harder cases than the English ones, which were assumed clean.

---



## Slide 9 — We translated the code of conduct into Sinhala, but Sinhala articles are judged better with the English code

**gpt-4.1 on the 44 Sinhala articles, same prompts, only the code's language changed. F2, mean of 3 runs.**


| Task · prompt              | English code | Sinhala code | Change  |
| -------------------------- | ------------ | ------------ | ------- |
| YES/NO · zero-shot         | 58           | 42           | −16     |
| YES/NO · few-shot          | 72           | 58           | −14     |
| YES/NO · per-rule          | 58           | 24           | **−34** |
| Rule detection · zero-shot | **86**       | **86**       | 0       |
| Rule detection · few-shot  | 83           | 73           | −10     |
| Rule detection · per-rule  | 64           | 34           | **−30** |
| Highlights · zero-shot     | 84           | 78           | −6      |
| Highlights · few-shot      | 77           | 67           | −10     |
| Highlights · per-rule      | 70           | 59           | −11     |
| **Average of the 9**       | **72**       | **58**       | **−14** |


- **Worse in 8 of 9 formats, and never better:** the average F2 falls from 72 to 58.
- **The best format is unchanged:** rule detection zero-shot scores 86 with either code, so a Sinhala code brings no gain anywhere.
- **SinLlama (1 run with the Sinhala code):** no consistent change in F2. It went up in 5 formats, down in 3 and stayed the same in 1.

**Why the English code works better**

1. **Twice the tokens for the same rules:** 3,419 tokens in English vs **7,535** in Sinhala (gpt-4.1's tokenizer). The same content becomes a much longer, more fragmented prompt.
2. **The model was trained mostly on English:** terms like "public interest", "verification" and "sensational treatment" are well established in English. Their Sinhala translations are rarer and less exact.
3. **Mixed languages:** the instructions, output format and rule ids stay in English, so the Sinhala code adds a language switch inside the prompt.
4. **The per-rule formats lose the most (−30, −34):** they check each rule's conditions word by word, so any loss of exact meaning in the translated rule text hurts them most. Rule detection zero-shot, which only names rule ids, is unchanged.
5. **Article language isn't the problem:** with the English code, the best Sinhala F2 (86) is close to the best English F2 (89).

**Decision:** keep the code in English for every article.

---



## Slide 10 — Prompt tweaks traded false alarms for recall, but a guide written from the model's own mistakes made it more precise

**Prompt tweaks** (single runs on an earlier article set, so not used in the final ranking)


| Idea                                                 | F2                 | Effect                                       |
| ---------------------------------------------------- | ------------------ | -------------------------------------------- |
| More few-shot examples (2.5, 7.4, attributed claims) | Highlights 81 → 86 | +5 caught, **+10 false alarms**              |
| Text explaining the 2.2.a / 2.5 boundary             | Highlights 86      | +3 caught, +2 false alarms                   |
| Reason first, then answer                            | Rule detection 79  | 14 false alarms                              |
| A second call re-checks every finding                | Rule detection 68  | removed false alarms **and** real violations |


**Fixing from failures:** I read every rule-detection mistake and wrote "how to apply the rules" guidance:

- attributed claims are sourced (2.2.a);
- tone is 6.1.a, method is 6.1.b;
- a name and town is not a privacy breach (7.1);
- and similar points for the other common mistakes.


| Rule detection zero-shot (106 articles, 3 runs) | F2  | P      | R   | False alarms / run | Named rules correct |
| ----------------------------------------------- | --- | ------ | --- | ------------------ | ------------------- |
| Original                                        | 86  | 83     | 87  | 8.7                | 67%                 |
| v2: + guide                                     | 79  | 92     | 77  | 3.3                | 74%                 |
| **v3: v2 with over-corrections undone**         | 85  | **88** | 84  | **5.7**            | **74%**             |


- **v3** cuts a third of the false alarms and names the right rule more often, for about the same F2.

---



## Slide 11 — Splitting the work into two steps and combining runs both helped, each in its own way

**Two steps: rules first, then words** (single runs, earlier article set with 48 violations)


| Approach                                | F2  | Correct offending words highlighted |
| --------------------------------------- | --- | ----------------------------------- |
| **Rules first, then words** (zero-shot) | 81  | **53%**                             |
| Words first, then rules (few-shot)      | 80  | 37%                                 |
| One call: highlights (zero-shot)        | 84  | 48%                                 |
| One call: highlights (few-shot)         | 81  | 35%                                 |


- Detection stays about the same, but telling the model which rules to look for helps it **find the exact words**, the third goal of the project.

**Voting: combining the 9 ways**

- Flagging an article when **at least 1 of the 9 ways** flags it gave **F2 91**.
- But that threshold was picked after seeing the results. This led to the fair version: **confidence scores with leave-one-out thresholds** (next slide).

---



## Slide 12 — The model knows more than it says: confidence scores from its existing answers were the biggest gain

**The idea:** gpt-4.1 often answers NO while giving YES a small probability, and those articles are often real violations. So score each article, and choose the threshold leave-one-out, so it never sees the article it decides.


| Same answers, used as a score                 | As plain answers, F2 | As a score, F2      |
| --------------------------------------------- | -------------------- | ------------------- |
| YES/NO zero-shot: P(YES)                      | 63                   | **86**              |
| YES/NO few-shot: P(YES)                       | 80                   | **86**              |
| Rule detection per-rule: highest P(VIOLATION) | 62                   | **86**              |
| **Mean P(YES) of the 3 YES/NO prompts**       | —                    | **90** (P 91, R 89) |


- **The best threshold is just 1.1% P(YES):** the model is over-confident in its NO answers.
- **37 scores were built for all 3 tasks:**
  - agreement: the share of runs that flag;
  - probability: P(YES) and per-rule P(VIOLATION);
  - count: how many rules or passages are named;
  - mixes of these, also across all 3 tasks.
- **No new API calls:** every score uses answers already stored.
- **Why it works:** a score keeps "how sure" instead of rounding it to YES/NO, and combining runs averages out the 3–6 point run-to-run noise.

---



## Slide 13 — The top 4 approaches differ by language, but sharing all 27 runs is near the top in both

**Ranked by F2 among all candidates** (the 9 ways, mean P(YES), the improved prompts and the 37 scores; gpt-4.1)

**English: 62 articles (32 violations, 30 clean)**


| #   | Approach                                       | F2 (95% interval) | P   | R   | F1  | Caught | False alarms |
| --- | ---------------------------------------------- | ----------------- | --- | --- | --- | ------ | ------------ |
| 1   | **Share of all 27 runs that flag** (≥ 8 of 27) | **92** (82–99)    | 97  | 91  | 94  | 29/32  | 1/30         |
| 2   | Mean P(YES) of the 3 YES/NO prompts            | 91 (82–99)        | 94  | 91  | 92  | 29/32  | 2/30         |
| 3   | YES/NO few-shot P(YES)                         | 91 (82–99)        | 94  | 91  | 92  | 29/32  | 2/30         |
| 4   | Rule detection few-shot, ≥ 2 of 3 runs flag    | 91 (81–98)        | 91  | 91  | 91  | 29/32  | 3/30         |


**Sinhala: 44 articles (15 violations, 29 clean)**


| #   | Approach                                           | F2 (95% interval) | P   | R   | F1  | Caught | False alarms |
| --- | -------------------------------------------------- | ----------------- | --- | --- | --- | ------ | ------------ |
| 1   | **Mix of the 27-run share and mean P(YES)**        | **90** (77–97)    | 78  | 93  | 85  | 14/15  | 4/29         |
| 2   | Rule detection zero-shot, all 3 runs flag          | 89 (76–97)        | 74  | 93  | 82  | 14/15  | 5/29         |
| 3   | Share of all 27 runs that flag (≥ 8 of 27)         | 88 (74–96)        | 70  | 93  | 80  | 14/15  | 6/29         |
| 4   | Rule detection zero-shot, by number of rules named | 88 (75–96)        | 70  | 93  | 80  | 14/15  | 6/29         |


**Why these approaches win**

- **All 8 are scores or combinations, not single answers:** single runs are noisy, and combining them is steadier.
- **English:** single answers already have high precision, so the winners are the ones that **recover recall**: P(YES) catches the "leaning YES" articles.
- **Sinhala:** single answers already have high recall, so the winners are the ones that **cut false alarms**: they require agreement (all 3 runs) or mix in P(YES).
- **Over both languages (106 articles),** the share of all 27 runs is #1 with F2 90 (83–96), P 86, R 91, and 43 of 47 caught with 7 false alarms.
  - It also catches **93%** of the spreadsheet violations, which were added after the prompts were written.
- **The top 4 are tied in each language:** their intervals overlap, so none is clearly better. Mean P(YES) is the cheaper choice (9 calls per article instead of 27).

---



## Slide 14 — LLMs are good at "is something wrong?" but weaker at "which rule?", and the evidence is still small

**What works**

- **Detection:** about 90% recall at 86–91% precision with confidence scores (97% precision in English).
- **Concrete rules:** murder-method detail (6.1.b) is found almost every time.
- **Rules in English work for Sinhala articles.**

**What doesn't (yet)**

- **Judgement rules:** opinion presented as fact (2.5), sensational treatment (6.1.a) and privacy (7.1) are flagged but often under the wrong rule. The named rules are correct only 67–74% of the time.
- **Unsourced claims (2.2.a)** cause most false alarms: routine attributed reporting gets flagged.
- **Sinhala false alarms:** precision is 70–78 even for the top approaches.
- **The smaller models:** Gemini 3.5 Flash is too cautious, and SinLlama 8B can't follow the rules.

**Limits of the evidence**

- Only **47 violations** (15 Sinhala), so the 95% intervals are 13–15 points wide overall and up to 30 in Sinhala.
- The English clean articles are **assumed** clean, not reviewed.
- Gemini is incomplete in English (out of credits).
- The top approaches were chosen on the same articles they are scored on, which flatters the winner slightly.

---



## Slide 15 — Next steps: more reviewed data, cheaper deployment, and a reviewer workflow built on the scores

**Data**

- Review more **Sinhala violations** and a set of **English clean articles**, to narrow the intervals and test without assumptions.
- Add **Tamil** (1,380 reviewed clean Tamil articles already exist).

**Models and methods**

- **Gemini:** finish its English run (needs credits) and test more models, including cheaper ones.
- **Per-language thresholds:** a stricter one for Sinhala to cut its false alarms.
- **More probability scores:** record token probabilities for zero-shot and few-shot rule detection and highlights.
- **Training a classifier:** use the LLM's labels to train BERT / XLM-R, and compare with the Ruhuna SVM + TF-IDF baseline (91% accuracy on Sinhala headlines).

**Deployment**

- **Triage:** rank unreviewed articles by confidence score so reviewers see the likely violations first.
- **Two-step highlights:** show reviewers the suggested rules and highlighted words, and record their corrections as new labels.
- **Cost choice:** mean P(YES) needs 9 calls per article, the share of 27 runs needs 27, and they give about the same F2.

