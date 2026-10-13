# Detecting Code of Conduct Violations in News Articles with LLMs

Checks Sinhala and English news articles against the Editors' Guild of Sri Lanka Code of Professional Practice (17 rules). For each article it answers three questions:

- Is it a violation?
- Which rules does it break?
- Which words break them?

Three models were tested (gpt-4.1, Gemini 3.5 Flash, SinLlama 8B), each in **9 ways** (3 tasks × 3 prompt styles). Best result: F2 90 (precision 86, recall 91) on 106 articles.

- Slides: `[presentation/violations_presentation.pdf](presentation/violations_presentation.pdf)`
- Database tables: `[DATABASE.md](DATABASE.md)`



## Repository layout

```
├── evaluate.py             # print the scores of every model × task × prompt
├── export_results.py       # export the answers and scores of the 9 methods to results/
├── results/                # exported answers and scores (see results/README.md)
├── app.py                  # web viewer of the results (http://127.0.0.1:5002)
├── db.py                   # database access
├── run_all.sh              # run the 9 ways for one model
├── run_missing.sh          # run every method on articles that have no answer yet
├── code_of_conduct/        # the 17 rules as JSON (English and Sinhala)
├── binary_detection/       # task 1: violation? YES / NO
├── rule_detection/         # task 2: which rules are broken
├── span_detection/         # task 3: highlight the words that break a rule
├── two_step/               # two calls per article (rules, then words; or words, then rules)
├── evaluation/             # metrics, intervals, voting, confidence scores (see evaluation/README.md)
├── templates/              # HTML of the viewer
└── presentation/           # slides
```

Each task folder has the same layout:

- `prompts/`: one module per prompt;
- `run_openai.py`, `run_gemini.py`, `run_sinllama.py`;
- `common.py`: the shared runner.

Results are stored in the database. Each new run also leaves a local copy in `<task>/outputs/`, which is not committed.

## Step 1: Set up

```bash
cd violations
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY and GEMINI_API_KEY
```

**Database (PostgreSQL):** `db.py` reads `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER` and `DB_PASSWORD` from the file named by `ENV_FILE`. The default is `/home/dilusha/anish_project/.env`.

```bash
export ENV_FILE=/path/to/db.env
```

**SinLlama (optional):** needs a CUDA GPU with `torch` and `transformers`. Point to the model with `--model-dir` or `SINLLAMA_MODEL_DIR`.

## Step 2: The data

The shared database already holds everything in the `violations` schema:

- the 106 test articles;
- the correct rules and highlights;
- every model result.

Nothing needs to be rebuilt. To view or score the existing results, go to Step 5.

**To test more articles**, insert them into the database, then run Step 3:

- **Where:** `violations.violation_articles` (status `REVIEWED_VIOLATION`) or `violations.non_violation_articles` (status `REVIEWED_NON_VIOLATION`).
- **Columns:** `article_id`, `heading`, `content`, `lang` (`en` / `si`) and `status`.
- **Optional:** add their correct rules and words to `violations.manual_annotations`.

See `[DATABASE.md](DATABASE.md)` for the table details. Runners skip articles that already have an answer, so only the new ones are run.

New clean articles are used only if they fall within the first N per language taken by `--negatives N`. To include them, raise N.

**To use other rules,** edit `code_of_conduct/code_of_conduct.json`.

## Step 3: Run the 9 ways

**The three tasks:**


| Task           | Folder              | The model returns                   | Counts as a violation when      |
| -------------- | ------------------- | ----------------------------------- | ------------------------------- |
| Violated?      | `binary_detection/` | `YES` / `NO`                        | it says YES                     |
| Rule detection | `rule_detection/`   | broken rule ids + evidence          | it names at least one rule      |
| Highlights     | `span_detection/`   | rule id + the exact offending words | the words are found in the text |


**The three prompt styles:**


| Prompt      | What the prompt holds                                       |
| ----------- | ----------------------------------------------------------- |
| `zero_shot` | The code of conduct and the article                         |
| `few_shot`  | The same, plus 3 example articles with answers per language |
| `per_rule`  | A verdict and evidence for each of the 17 rules, one by one |


**Commands** (gpt-4.1):


| #   | Way                        | Command                                                                             |
| --- | -------------------------- | ----------------------------------------------------------------------------------- |
| 1   | Violated? · zero-shot      | `.venv/bin/python binary_detection/run_openai.py --prompt zero_shot --negatives 30` |
| 2   | Violated? · few-shot       | `.venv/bin/python binary_detection/run_openai.py --prompt few_shot --negatives 30`  |
| 3   | Violated? · per-rule       | `.venv/bin/python binary_detection/run_openai.py --prompt per_rule --negatives 30`  |
| 4   | Rule detection · zero-shot | `.venv/bin/python rule_detection/run_openai.py --prompt zero_shot --negatives 30`   |
| 5   | Rule detection · few-shot  | `.venv/bin/python rule_detection/run_openai.py --prompt few_shot --negatives 30`    |
| 6   | Rule detection · per-rule  | `.venv/bin/python rule_detection/run_openai.py --prompt per_rule --negatives 30`    |
| 7   | Highlights · zero-shot     | `.venv/bin/python span_detection/run_openai.py --prompt zero_shot --negatives 30`   |
| 8   | Highlights · few-shot      | `.venv/bin/python span_detection/run_openai.py --prompt few_shot --negatives 30`    |
| 9   | Highlights · per-rule      | `.venv/bin/python span_detection/run_openai.py --prompt per_rule --negatives 30`    |


**Other models:**

- **Gemini:** use `run_gemini.py`.
- **SinLlama:** use `run_sinllama.py`, run with a Python that has `torch`. It runs on Sinhala articles only.

**All 9 at once, 3 times each** (the results use the mean of 3 runs):

```bash
./run_all.sh openai --force                
./run_all.sh gemini --force  
./run_all.sh sinllama --force  
```

**Options** (every runner):


| Option           | Meaning                                            |
| ---------------- | -------------------------------------------------- |
| `--negatives N`  | Add the first N clean articles per language        |
| `--lang en       | si`                                                |
| `--repeat N`     | Store the run as repeat N                          |
| `--code-lang si` | Use the Sinhala code of conduct (with `--lang si`) |
| `--model NAME`   | Another model of the same provider                 |
| `--limit N`      | Only the first N articles                          |
| `--force`        | Re-run articles that already have an answer        |
| `--workers N`    | Parallel calls (default 4)                         |


Finished articles are skipped, so a stopped run can simply be started again.

## Step 4: Other methods


| Method                                         | Command                                                                              |
| ---------------------------------------------- | ------------------------------------------------------------------------------------ |
| Sinhala code of conduct (Sinhala articles)     | `./run_all.sh openai --code-lang si --lang si`                                       |
| Rule detection revised from its mistakes       | `.venv/bin/python rule_detection/run_openai.py --prompt zero_shot_v3 --negatives 30` |
| Two steps (rules then words; words then rules) | `.venv/bin/python two_step/run.py --negatives 30`                                    |
| Everything missing for newly added articles    | `./run_missing.sh openai`                                                            |


Voting and the confidence scores need no new runs: they are computed from the stored answers.

## Step 5: Evaluate

```bash
.venv/bin/python evaluate.py                     # F2 of every model × task × prompt, both languages + per language
.venv/bin/python evaluate.py --metric recall     # precision, recall, f1, f2 or accuracy
.venv/bin/python evaluate.py --lang si
.venv/bin/python evaluate.py --scores            # also rank the 37 confidence scores
.venv/bin/python export_results.py               # answers and scores as CSV → results/
.venv/bin/python app.py                          # viewer → http://127.0.0.1:5002
```


| Page            | Shows                                                                   |
| --------------- | ----------------------------------------------------------------------- |
| `/overview`     | The whole project step by step, the top 4 approaches, and a leaderboard |
| `/compare`      | Every model × task × prompt, per language, with per-article answers     |
| `/best-model`   | The best approach with 95% intervals                                    |
| `/article/<id>` | One article with every run's answer and highlighted evidence            |


**How it is scored:**

- **Main score:** F2, which counts recall twice as much as precision, because missing a violation is worse than a false alarm.
- **Other metrics:** precision, recall, F1 and accuracy.
- **Same articles:** every approach is scored on the same articles, and each number is the mean of 3 runs.
- **95% intervals:** bootstrap over articles.
- **Confidence scores:** P(YES), the share of runs that flag an article, and similar. The threshold is chosen leave-one-out.
- **Rules and highlights:** checked against `manual_annotations`.



## Results


| Language | Best of the 9 ways (gpt-4.1)    | Best overall                                   |
| -------- | ------------------------------- | ---------------------------------------------- |
| English  | Rule detection few-shot, F2 89  | Share of all 27 runs that flag, F2 92          |
| Sinhala  | Rule detection zero-shot, F2 86 | Mix of the 27-run share and mean P(YES), F2 90 |


- **Sinhala code of conduct:** the average F2 drops from 72 to 58 with it, so the code is kept in English.
- **Other models:** Gemini 3.5 Flash reaches a best F2 of 50. SinLlama 8B reaches 72 only by flagging every article.

