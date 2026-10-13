CREATE SCHEMA IF NOT EXISTS violations;

-- Snapshot of <source schema>.article_review_status (data/article_review_status.csv)
CREATE TABLE IF NOT EXISTS violations.article_review_status (
    article_id   text PRIMARY KEY,
    status       varchar,
    reviewed_by  varchar,
    reviewed_at  timestamp,
    decided_by   varchar,
    decided_at   timestamp,
    created_at   timestamp,
    updated_at   timestamp
);

-- Snapshot of violation-status articles (data/violation_articles.csv)
CREATE TABLE IF NOT EXISTS violations.violation_articles (
    article_id  text PRIMARY KEY,
    heading     text,
    content     text,
    lang        varchar,
    status      varchar
);

-- Externally collected violations (data/external_violations.csv, from import_external.py), not in raw_articles.
-- external_review holds a decision for each (never truncated); load_to_db.py copies the confirmed ones
-- into violation_articles with status EXTERNAL_VIOLATION.
CREATE TABLE IF NOT EXISTS violations.external_candidates (
    article_id  text PRIMARY KEY,
    heading     text,
    content     text,
    lang        varchar,
    status      varchar,
    source      varchar,
    url         text,
    outlet      varchar,
    date_posted date,
    keyword     varchar,
    clause      varchar,
    excerpt     text
);
CREATE TABLE IF NOT EXISTS violations.external_review (
    article_id  text PRIMARY KEY,
    decision    varchar NOT NULL,          -- confirmed | rejected
    rule_ids    text[] NOT NULL DEFAULT '{}',
    notes       text,
    decided_at  timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE violations.violation_articles ADD COLUMN IF NOT EXISTS source      varchar;
ALTER TABLE violations.violation_articles ADD COLUMN IF NOT EXISTS url         text;
ALTER TABLE violations.violation_articles ADD COLUMN IF NOT EXISTS outlet      varchar;
ALTER TABLE violations.violation_articles ADD COLUMN IF NOT EXISTS date_posted date;
ALTER TABLE violations.violation_articles ADD COLUMN IF NOT EXISTS keyword     varchar;
ALTER TABLE violations.violation_articles ADD COLUMN IF NOT EXISTS clause      varchar;
ALTER TABLE violations.violation_articles ADD COLUMN IF NOT EXISTS excerpt     text;

-- Fixed sample of non-violation articles used as negatives (data/non_violation_articles.csv):
-- REVIEWED_NON_VIOLATION for Sinhala, ASSUMED_NON_VIOLATION (random unreviewed) for English
CREATE TABLE IF NOT EXISTS violations.non_violation_articles (
    article_id  text PRIMARY KEY,
    heading     text,
    content     text,
    lang        varchar,
    status      varchar
);

-- One row per (article, experiment/prompt, model); re-runs overwrite.
CREATE TABLE IF NOT EXISTS violations.llm_predictions (
    id             bigserial PRIMARY KEY,
    article_id     text NOT NULL,
    experiment     varchar NOT NULL,
    provider       varchar NOT NULL,
    model          varchar NOT NULL,
    answer         varchar NOT NULL,
    confidence     double precision,
    expected       varchar,
    article_status varchar,
    raw_output     text,
    error          text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (article_id, experiment, model)
);

-- P(YES) for models that expose token probabilities (SinLlama, OpenAI logprobs)
ALTER TABLE violations.llm_predictions ADD COLUMN IF NOT EXISTS confidence double precision;
-- Structured output, e.g. per-rule verdicts with evidence
ALTER TABLE violations.llm_predictions ADD COLUMN IF NOT EXISTS details jsonb;

CREATE INDEX IF NOT EXISTS llm_predictions_experiment_idx
    ON violations.llm_predictions (experiment, model);

-- Rule detection (rule_detection/): which rules each model says an article breaks.
-- rules = [{rule_id, verdict, evidence, explanation, p}] for flagged rules only;
-- NULL when the run failed or the output could not be parsed (see error).
CREATE TABLE IF NOT EXISTS violations.rule_predictions (
    id             bigserial PRIMARY KEY,
    article_id     text NOT NULL,
    experiment     varchar NOT NULL,
    provider       varchar NOT NULL,
    model          varchar NOT NULL,
    rules          jsonb,
    rule_ids       text[],
    expected       varchar,
    article_status varchar,
    raw_output     text,
    error          text,
    details        jsonb,
    created_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (article_id, experiment, model)
);

-- Hand-made ground truth (correct rule codes and highlights). One row per article.
-- verdict: VIOLATION (rule_ids ticked) | NO_VIOLATION (explicitly marked clean).
-- spans: [{rule_id, part: content|heading, start, end, excerpt}]; start/end are character
-- offsets into raw_articles.content (or .heading).
CREATE TABLE IF NOT EXISTS violations.manual_annotations (
    article_id   text PRIMARY KEY,
    verdict      varchar NOT NULL,
    rule_ids     text[] NOT NULL DEFAULT '{}',
    spans        jsonb NOT NULL DEFAULT '[]',
    notes        text,
    annotated_by varchar,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

-- Where in the article each LLM says the code is broken (span_detection/).
-- spans: [{rule_id, verdict, excerpt, explanation, start, end, p}]; start/end are character
-- offsets into raw_articles.content, null when the excerpt is not in the article.
CREATE TABLE IF NOT EXISTS violations.span_predictions (
    id             bigserial PRIMARY KEY,
    article_id     text NOT NULL,
    experiment     varchar NOT NULL,
    provider       varchar NOT NULL,
    model          varchar NOT NULL,
    spans          jsonb,
    n_spans        integer,
    n_located      integer,
    expected       varchar,
    article_status varchar,
    raw_output     text,
    error          text,
    details        jsonb,
    created_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (article_id, experiment, model)
);
