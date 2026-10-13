"""Postgres access for the violations project.

Connection settings (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD) are read
from the anish_project .env file. Override with ENV_FILE / DB_SCHEMA env vars.

DB_SCHEMA (default slpi_prod_v2) is the source data; our own tables live in the
`violations` schema (see sql/violations_schema.sql).
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor, execute_values

ENV_FILE = Path(os.getenv("ENV_FILE", "/home/dilusha/anish_project/.env"))
load_dotenv(ENV_FILE)

SCHEMA = os.getenv("DB_SCHEMA", "slpi_prod_v2")
OWN_SCHEMA = "violations"

VIOLATION_STATUSES = (
    "REVIEWED_VIOLATION",
    "DECIDED_CRITICAL",
    "DECIDED_NON_CRITICAL",
    "EXTERNAL_VIOLATION",
)

EXCLUDED_LANGS = ("ta",)


def expected_answer(status: str | None) -> str:
    return "YES" if status in VIOLATION_STATUSES else "NO"


def connect():
    password = (os.getenv("DB_PASSWORD") or "").strip().strip("'\"")
    return psycopg2.connect(
        host=(os.getenv("DB_HOST") or "127.0.0.1").strip(),
        port=int((os.getenv("DB_PORT") or "5432").strip()),
        dbname=(os.getenv("DB_NAME") or "slpi").strip(),
        user=(os.getenv("DB_USER") or "").strip(),
        password=password,
    )


def _fetch_all(query: sql.Composable, params: tuple = ()) -> list[dict]:
    conn = connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


# --- source schema (slpi_prod_v2) -------------------------------------------

def fetch_review_status() -> list[dict]:
    query = sql.SQL(
        "SELECT * FROM {schema}.article_review_status ORDER BY article_id ASC"
    ).format(schema=sql.Identifier(SCHEMA))
    return _fetch_all(query)


def fetch_violation_articles() -> list[dict]:
    query = sql.SQL(
        """
        SELECT ars.article_id, ra.heading, ra.content, ra.lang, ars.status
        FROM {schema}.article_review_status AS ars
        LEFT JOIN {schema}.raw_articles AS ra ON ra.id = ars.article_id
        WHERE ars.status IN %s
          AND coalesce(ra.lang, '') NOT IN %s
        ORDER BY ars.article_id ASC
        """
    ).format(schema=sql.Identifier(SCHEMA))
    return _fetch_all(query, (VIOLATION_STATUSES, EXCLUDED_LANGS))


def fetch_non_violation_sample(n: int) -> list[dict]:
    """Deterministic sample of REVIEWED_NON_VIOLATION articles (same n → same rows)."""
    query = sql.SQL(
        """
        SELECT ars.article_id, ra.heading, ra.content, ra.lang, ars.status
        FROM {schema}.article_review_status AS ars
        JOIN {schema}.raw_articles AS ra ON ra.id = ars.article_id
        WHERE ars.status = 'REVIEWED_NON_VIOLATION'
          AND coalesce(ra.content, '') <> ''
          AND coalesce(ra.lang, '') NOT IN %s
        ORDER BY md5(ars.article_id)
        LIMIT %s
        """
    ).format(schema=sql.Identifier(SCHEMA))
    return _fetch_all(query, (EXCLUDED_LANGS, n))


ASSUMED_NON_VIOLATION = "ASSUMED_NON_VIOLATION"


def fetch_assumed_non_violation_sample(
    n: int, lang: str, exclude: tuple[str, ...] = (), min_chars: int = 300, max_chars: int = 8000
) -> list[dict]:
    """Deterministic random sample of articles nobody has reviewed, labelled ASSUMED_NON_VIOLATION.

    For languages without reviewed non-violations (English). The label is an
    assumption: a random unreviewed article can still break the code.
    """
    query = sql.SQL(
        """
        SELECT ra.id AS article_id, ra.heading, ra.content, ra.lang, %s AS status
        FROM {schema}.raw_articles AS ra
        WHERE ra.lang = %s
          AND length(coalesce(ra.content, '')) BETWEEN %s AND %s
          AND ra.id <> ALL(%s)
          AND NOT EXISTS (SELECT 1 FROM {schema}.article_review_status AS ars WHERE ars.article_id = ra.id)
        ORDER BY md5(ra.id)
        LIMIT %s
        """
    ).format(schema=sql.Identifier(SCHEMA))
    return _fetch_all(query, (ASSUMED_NON_VIOLATION, lang, min_chars, max_chars, list(exclude), n))


def fetch_article(article_id: str) -> dict | None:
    """Any raw article; status from the review table, else from the non-violation snapshot.
    External violations (not in raw_articles) come from external_candidates."""
    query = sql.SQL(
        """
        SELECT ra.id AS article_id, ra.heading, ra.content, ra.lang,
               coalesce(ars.status, nv.status) AS status
        FROM {schema}.raw_articles AS ra
        LEFT JOIN {schema}.article_review_status AS ars ON ars.article_id = ra.id
        LEFT JOIN {own}.non_violation_articles AS nv ON nv.article_id = ra.id
        WHERE ra.id = %s
        UNION ALL
        SELECT c.article_id, c.heading, c.content, c.lang, c.status
        FROM {own}.external_candidates AS c
        WHERE c.article_id = %s
        """
    ).format(schema=sql.Identifier(SCHEMA), own=sql.Identifier(OWN_SCHEMA))
    rows = _fetch_all(query, (article_id, article_id))
    return rows[0] if rows else None


# --- violations schema -------------------------------------------------------

def fetch_snapshot_articles() -> list[dict]:
    """Violation articles from the violations.violation_articles snapshot (with text)."""
    query = sql.SQL(
        """
        SELECT article_id, heading, content, lang, status
        FROM {own}.violation_articles
        WHERE coalesce(content, '') <> ''
        ORDER BY article_id
        """
    ).format(own=sql.Identifier(OWN_SCHEMA))
    return _fetch_all(query)


def fetch_snapshot_negatives(n: int) -> list[dict]:
    """First n articles per language of the violations.non_violation_articles snapshot (fixed order)."""
    query = sql.SQL(
        """
        SELECT article_id, heading, content, lang, status
        FROM (
            SELECT *, row_number() OVER (PARTITION BY lang ORDER BY md5(article_id)) AS rank
            FROM {own}.non_violation_articles
            WHERE coalesce(content, '') <> ''
        ) AS ranked
        WHERE rank <= %s
        ORDER BY lang, md5(article_id)
        """
    ).format(own=sql.Identifier(OWN_SCHEMA))
    return _fetch_all(query, (n,))


PREDICTION_COLUMNS = (
    "article_id", "experiment", "provider", "model", "answer", "confidence",
    "expected", "article_status", "raw_output", "error", "details", "created_at",
)


def upsert_predictions(rows: list[dict]) -> None:
    if not rows:
        return
    query = sql.SQL(
        """
        INSERT INTO {own}.llm_predictions ({cols}) VALUES %s
        ON CONFLICT (article_id, experiment, model) DO UPDATE SET
            provider = EXCLUDED.provider,
            answer = EXCLUDED.answer,
            confidence = EXCLUDED.confidence,
            expected = EXCLUDED.expected,
            article_status = EXCLUDED.article_status,
            raw_output = EXCLUDED.raw_output,
            error = EXCLUDED.error,
            details = EXCLUDED.details,
            created_at = EXCLUDED.created_at
        """
    ).format(
        own=sql.Identifier(OWN_SCHEMA),
        cols=sql.SQL(", ").join(map(sql.Identifier, PREDICTION_COLUMNS)),
    )
    values = [
        tuple(Json(r[c]) if c == "details" and r.get(c) is not None else r.get(c) for c in PREDICTION_COLUMNS)
        for r in rows
    ]
    conn = connect()
    try:
        with conn.cursor() as cur:
            execute_values(cur, query.as_string(cur), values)
        conn.commit()
    finally:
        conn.close()


def fetch_predictions(experiment: str | None = None) -> list[dict]:
    query = sql.SQL(
        """
        SELECT p.article_id, p.experiment, p.provider, p.model, p.answer,
               p.confidence, p.expected, p.article_status, p.raw_output, p.error, p.details, p.created_at,
               coalesce(ra.heading, ec.heading) AS heading, coalesce(ra.lang, ec.lang) AS lang
        FROM {own}.llm_predictions AS p
        LEFT JOIN {schema}.raw_articles AS ra ON ra.id = p.article_id
        LEFT JOIN {own}.external_candidates AS ec ON ec.article_id = p.article_id
        WHERE (%s::text IS NULL OR p.experiment = %s)
        ORDER BY p.article_id, p.model
        """
    ).format(own=sql.Identifier(OWN_SCHEMA), schema=sql.Identifier(SCHEMA))
    return _fetch_all(query, (experiment, experiment))


RULE_PREDICTION_COLUMNS = (
    "article_id", "experiment", "provider", "model", "rules", "rule_ids",
    "expected", "article_status", "raw_output", "error", "details", "created_at",
)


def _upsert_run_rows(table: str, columns: tuple, json_columns: tuple, rows: list[dict]) -> None:
    """Upsert one row per (article_id, experiment, model) into a violations.* table."""
    if not rows:
        return
    updates = [c for c in columns if c not in ("article_id", "experiment", "model")]
    query = sql.SQL(
        "INSERT INTO {own}.{table} ({cols}) VALUES %s "
        "ON CONFLICT (article_id, experiment, model) DO UPDATE SET {updates}"
    ).format(
        own=sql.Identifier(OWN_SCHEMA),
        table=sql.Identifier(table),
        cols=sql.SQL(", ").join(map(sql.Identifier, columns)),
        updates=sql.SQL(", ").join(
            sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(c)) for c in updates
        ),
    )
    values = [
        tuple(Json(r[c]) if c in json_columns and r.get(c) is not None else r.get(c)
              for c in columns)
        for r in rows
    ]
    conn = connect()
    try:
        with conn.cursor() as cur:
            execute_values(cur, query.as_string(cur), values)
        conn.commit()
    finally:
        conn.close()


def upsert_rule_predictions(rows: list[dict]) -> None:
    _upsert_run_rows("rule_predictions", RULE_PREDICTION_COLUMNS, ("rules", "details"), rows)


SPAN_PREDICTION_COLUMNS = (
    "article_id", "experiment", "provider", "model", "spans", "n_spans", "n_located",
    "expected", "article_status", "raw_output", "error", "details", "created_at",
)


def upsert_span_predictions(rows: list[dict]) -> None:
    _upsert_run_rows("span_predictions", SPAN_PREDICTION_COLUMNS, ("spans", "details"), rows)


def fetch_span_predictions(experiment: str | None = None, article_id: str | None = None) -> list[dict]:
    query = sql.SQL(
        """
        SELECT p.article_id, p.experiment, p.provider, p.model, p.spans, p.n_spans, p.n_located,
               p.expected, p.article_status, p.raw_output, p.error, p.details, p.created_at,
               coalesce(ra.heading, ec.heading) AS heading, coalesce(ra.lang, ec.lang) AS lang,
               length(coalesce(ra.content, ec.content)) AS content_length
        FROM {own}.span_predictions AS p
        LEFT JOIN {schema}.raw_articles AS ra ON ra.id = p.article_id
        LEFT JOIN {own}.external_candidates AS ec ON ec.article_id = p.article_id
        WHERE (%s::text IS NULL OR p.experiment = %s)
          AND (%s::text IS NULL OR p.article_id = %s)
        ORDER BY p.article_id, p.model
        """
    ).format(own=sql.Identifier(OWN_SCHEMA), schema=sql.Identifier(SCHEMA))
    try:
        return _fetch_all(query, (experiment, experiment, article_id, article_id))
    except psycopg2.errors.UndefinedTable:
        return []


EXTERNAL_COLUMNS = ("article_id", "heading", "content", "lang", "status",
                    "source", "url", "outlet", "date_posted", "keyword", "clause", "excerpt")


def confirmed_external_insert() -> sql.Composed:
    """INSERT the confirmed external candidates into violation_articles."""
    return sql.SQL(
        "INSERT INTO {own}.violation_articles ({cols}) "
        "SELECT {c_cols} FROM {own}.external_candidates AS c "
        "JOIN {own}.external_review AS r USING (article_id) "
        "WHERE r.decision = 'confirmed' "
        "ON CONFLICT (article_id) DO NOTHING"
    ).format(own=sql.Identifier(OWN_SCHEMA),
             cols=sql.SQL(", ").join(map(sql.Identifier, EXTERNAL_COLUMNS)),
             c_cols=sql.SQL(", ").join(sql.Identifier("c", c) for c in EXTERNAL_COLUMNS))


def fetch_manual_annotations(article_id: str | None = None) -> list[dict]:
    query = sql.SQL(
        """
        SELECT m.*, coalesce(ra.heading, ec.heading) AS heading, coalesce(ra.lang, ec.lang) AS lang
        FROM {own}.manual_annotations AS m
        LEFT JOIN {schema}.raw_articles AS ra ON ra.id = m.article_id
        LEFT JOIN {own}.external_candidates AS ec ON ec.article_id = m.article_id
        WHERE (%s::text IS NULL OR m.article_id = %s)
        ORDER BY m.updated_at DESC
        """
    ).format(own=sql.Identifier(OWN_SCHEMA), schema=sql.Identifier(SCHEMA))
    try:
        return _fetch_all(query, (article_id, article_id))
    except psycopg2.errors.UndefinedTable:
        return []


def fetch_rule_predictions(experiment: str | None = None) -> list[dict]:
    query = sql.SQL(
        """
        SELECT p.article_id, p.experiment, p.provider, p.model, p.rules, p.rule_ids,
               p.expected, p.article_status, p.raw_output, p.error, p.details, p.created_at,
               coalesce(ra.heading, ec.heading) AS heading, coalesce(ra.lang, ec.lang) AS lang
        FROM {own}.rule_predictions AS p
        LEFT JOIN {schema}.raw_articles AS ra ON ra.id = p.article_id
        LEFT JOIN {own}.external_candidates AS ec ON ec.article_id = p.article_id
        WHERE (%s::text IS NULL OR p.experiment = %s)
        ORDER BY p.article_id, p.model
        """
    ).format(own=sql.Identifier(OWN_SCHEMA), schema=sql.Identifier(SCHEMA))
    try:
        return _fetch_all(query, (experiment, experiment))
    except psycopg2.errors.UndefinedTable:
        return []
