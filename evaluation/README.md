# Evaluation

Scores the stored model answers. The modules make no API calls: everything is computed from the database.

## How an answer is scored

Each task's answer becomes a YES / NO decision on the article:

| Task             | Counts as YES (violation) when            |
| ---------------- | ----------------------------------------- |
| Binary detection | the model says YES                        |
| Rule detection   | it names at least one rule                |
| Span detection   | at least one highlight is found in the text |

That decision is compared with the article's correct answer, which comes from its review status.

**Metrics:**

| Metric    | Meaning                                                    |
| --------- | ---------------------------------------------------------- |
| Precision | Of the flagged articles, the share that are violations     |
| Recall    | Of the violations, the share that were flagged             |
| F1        | Balance of precision and recall                            |
| **F2**    | Like F1, but recall counts twice; the **main score**        |
| Accuracy  | Share of all articles judged correctly                     |

**Fair comparison:**

- **Same articles:** every method is scored on the articles all methods answered.
- **Mean of runs:** each method runs 3 times and the scores are averaged.
- **95% intervals:** a bootstrap over articles. Methods whose intervals overlap are treated as tied.

## Modules

| File            | What it does                                                                                       |
| --------------- | -------------------------------------------------------------------------------------------------- |
| `compare.py`    | Turns every answer into YES / NO, finds the common articles, and gives precision, recall, F1, F2 and accuracy per model × task × prompt as the mean over runs |
| `stats.py`      | Bootstrap 95% intervals, paired comparison of two methods, and Wilson intervals for accuracy        |
| `vote.py`       | Voting: an article is a violation when at least k of the runs flag it, for every k                 |
| `confidence.py` | P(YES) scores: the model's probability of YES instead of its answer, with a leave-one-out threshold |
| `scores.py`     | 37 confidence scores over all 3 tasks (see below)                                                   |

## Confidence scores

Instead of a single YES / NO, each article gets a number, and the article is a violation when that number reaches a threshold. The threshold is chosen **leave-one-out**: each article is judged with the threshold that works best (maximises F1) on all the *other* articles, so the score never sees the article it decides.

`scores.py` builds scores of four kinds:

| Kind        | Examples                                                               |
| ----------- | ---------------------------------------------------------------------- |
| Agreement   | Share of a prompt's 3 runs that flag the article; share of all 9 or 27 runs |
| Probability | P(YES) of the YES/NO prompts; highest per-rule P(VIOLATION)           |
| Count       | Number of rules named or passages highlighted                          |
| Combined    | Averages of the above, also across the 3 tasks                         |

**Best scores** (gpt-4.1, 106 articles):

| Score                                  | F2 | Precision | Recall |
| -------------------------------------- | -- | --------- | ------ |
| Share of all 27 runs that flag         | 90 | 86        | 91     |
| Mean P(YES) of the 3 YES/NO prompts    | 90 | 91        | 89     |

## How to run

```bash
.venv/bin/python evaluate.py                    # F2 of every model × task × prompt, per language
.venv/bin/python evaluate.py --metric recall    # precision, recall, f1, f2 or accuracy
.venv/bin/python evaluate.py --scores           # also rank the confidence scores
.venv/bin/python export_results.py              # answers and scores as CSV → results/
.venv/bin/python app.py                         # viewer: /overview, /compare, /best-model
```
