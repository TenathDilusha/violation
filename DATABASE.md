# Database tables

All project tables are in the PostgreSQL schema **`violations`** of the shared database. The articles were copied once from the SLPI platform (`slpi_prod_v2`), which this code no longer reads.

## Articles

| Table                    | What it holds                                                                  |
| ------------------------ | ------------------------------------------------------------------------------ |
| `violation_articles`     | The violation articles the models are tested on (platform + confirmed external) |
| `non_violation_articles` | The clean articles; `--negatives N` takes the first N per language (ordered by `md5(article_id)`) |
| `external_candidates`    | 28 English violations collected outside the platform (The Island, Daily Mirror, Daily News, Ceylon Today) |
| `external_review`        | Decision per external article (`confirmed` / `rejected`); only confirmed ones are in `violation_articles` |
| `article_review_status`  | Copy of the platform's review statuses (for reference)                          |
| `articles` (view)        | Every test article once: the three article tables combined                     |

**Article columns:** `article_id`, `heading`, `content`, `lang` (`en` / `si`) and `status`. The external tables also have `source`, `url`, `outlet`, `date_posted`, `keyword`, `clause` and `excerpt`.

**Status decides the correct answer:**

- `YES` (violation): `REVIEWED_VIOLATION`, `DECIDED_CRITICAL`, `DECIDED_NON_CRITICAL`, `EXTERNAL_VIOLATION`;
- `NO` (clean): any other status, e.g. `REVIEWED_NON_VIOLATION`, or `ASSUMED_NON_VIOLATION` for unreviewed English articles.

## Ground truth

| Table                | What it holds                                                                  |
| -------------------- | ------------------------------------------------------------------------------ |
| `manual_annotations` | The correct rules and offending words of a violation article, one row per article |

**Columns:**

| Column         | Content                                                                          |
| -------------- | -------------------------------------------------------------------------------- |
| `article_id`   | The article                                                                       |
| `verdict`      | `VIOLATION` / `NO_VIOLATION`                                                      |
| `rule_ids`     | The broken rules, e.g. `{6.1.b,2.5}`                                              |
| `spans`        | `[{rule_id, part, start, end, excerpt}]`; `part` is `content` or `heading`, and `start`/`end` are character offsets in it |
| `notes`        | Free text                                                                         |
| `annotated_by` | Who made the annotation                                                           |

**Articles without a row:** a violation article with no row here takes its correct rules and words from `span_detection/reviewed_suggestions.json`.

## Model results

One row per (article, experiment, model). A re-run overwrites that row. `results/` has a CSV export of the 9 methods.

| Table              | Task                                | Main columns                                                                                  |
| ------------------ | ----------------------------------- | --------------------------------------------------------------------------------------------- |
| `llm_predictions`  | Violated? (`binary_detection/`)     | `answer` (YES / NO / ERROR), `confidence` = P(YES), `details` (per-rule verdicts and evidence) |
| `rule_predictions` | Rule detection (`rule_detection/`)  | `rules` = `[{rule_id, verdict, evidence, explanation, p}]`, `rule_ids`                        |
| `span_predictions` | Highlights (`span_detection/`, `two_step/`) | `spans` = `[{rule_id, excerpt, explanation, offsets, heading_offsets, p}]`, `n_spans`, `n_located` |

**Columns in all three:** `article_id`, `experiment`, `provider` (`openai` / `gemini` / `sinllama`), `model`, `expected` (YES / NO from the status), `article_status`, `raw_output`, `error` and `created_at`.

**Failed runs:** `rules` or `spans` is NULL when a run failed or its output could not be parsed. `error` then says why.

**Experiment names:**

| Name                                      | Meaning                                         |
| ----------------------------------------- | ----------------------------------------------- |
| `zero_shot`, `few_shot`, `per_rule`       | The prompt, run 1                               |
| `<prompt>.r2`, `<prompt>.r3`              | Runs 2 and 3 (`--repeat`)                       |
| `<prompt>_si_code`                        | With the Sinhala code of conduct (`--code-lang si`) |
| `zero_shot_v2`, `zero_shot_v3`            | Revised rule detection prompts                  |
| `codes_then_highlights_<prompt>`, `highlights_then_codes_<prompt>` | Two-step pipelines (in `span_predictions`) |
