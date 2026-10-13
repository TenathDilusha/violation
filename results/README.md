# Results of the 9 methods

These files hold the answers of three LLMs to the 9 methods (3 tasks × 3 prompts) on the 106 test articles, and their scores. They were exported from the database with `python export_results.py`.

## Data

**106 articles** (`articles.csv`):

|          | Violations | Clean | Total |
| -------- | ---------- | ----- | ----- |
| Sinhala  | 15         | 29    | 44    |
| English  | 32         | 30    | 62    |

- **Violation sources:**
  - 19 were reviewed on the SLPI platform;
  - 28 English ones were collected from newspaper spreadsheets (The Island, Daily Mirror, Daily News, Ceylon Today).
- **Clean articles:**
  - Sinhala ones were reviewed as clean;
  - English ones are random unreviewed articles assumed to be clean.
- **Correct rules:** `correct_rules` holds the rules each violation breaks, annotated by hand. 46 of the 47 violations have them.

**Models:**

| Model            | Runs per method | Articles                                                       |
| ---------------- | --------------- | -------------------------------------------------------------- |
| gpt-4.1          | 3               | all 106                                                        |
| SinLlama 8B      | 3               | 44 Sinhala only                                                |
| Gemini 3.5 Flash | 1               | all 106, but 84 answers are `ERROR` (it ran out of credits)    |

**The 9 methods:**

- **Tasks:**
  - binary detection: is it a violation? YES / NO;
  - rule detection: which rules are broken;
  - span detection: highlight the offending words.
- **Prompts:**
  - `zero_shot`: the code of conduct and the article;
  - `few_shot`: the same, plus 3 example articles;
  - `per_rule`: a verdict for each of the 17 rules.

## Files

| File                   | One row per                         | Columns                                                                                           |
| ---------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------- |
| `articles.csv`         | article                             | `article_id`, `lang`, `source` (platform / external), `status`, `expected` (YES / NO), `correct_rules`, `heading` |
| `binary_detection.csv` | article × model × prompt × run      | `answer` (YES / NO / ERROR), `p_yes`: the model's probability of YES (gpt-4.1 and SinLlama)       |
| `rule_detection.csv`   | article × model × prompt × run      | `answer` (YES if any rule is named), `rules` (the rules named, `;`-separated)                     |
| `span_detection.csv`   | article × model × prompt × run      | `answer` (YES if any highlight is found in the text), `rules`, `highlights` (JSON: rule and excerpt) |
| `summary.csv`          | language × task × prompt × model    | Mean over the runs: `tp`, `fp`, `tn`, `fn`, `precision`, `recall`, `f1`, `f2`, `accuracy`; also `rule_precision` and `rule_recall` |

**How `summary.csv` is calculated:**

- **Article-level scores:** an article counts as a violation when the `answer` is YES, compared with `expected`.
- **`language`:** `all`, `en` or `si`.
- **`rule_precision`:** the share of named rules that are correct, on violation articles.
- **`rule_recall`:** the share of correct rules that were named, on violation articles.

## Main results (F2, gpt-4.1)

| Task             | Zero-shot | Few-shot | Per-rule |
| ---------------- | --------- | -------- | -------- |
| Binary detection | 63        | 80       | 58       |
| Rule detection   | 86        | **87**   | 62       |
| Span detection   | 85        | 81       | 74       |

Best per language:

- **English:** rule detection few-shot, F2 89.
- **Sinhala:** rule detection zero-shot, F2 86.

The named rules are correct about 62–73% of the time.
