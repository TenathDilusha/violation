"""Create the `violations` schema and load the extracted CSVs into it.

Run extract.py (and import_external.py for the external violations) first.
Reloading replaces the snapshot tables' contents and re-adds the external violations confirmed
in violations.external_review. external_review, manual_annotations and the prediction tables
are never truncated.

Usage:
    python load_to_db.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from psycopg2 import sql
from psycopg2.extras import execute_values

from db import OWN_SCHEMA, confirmed_external_insert, connect

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SCHEMA_SQL = ROOT / "sql" / "violations_schema.sql"

CSV_TABLES = {
    "article_review_status": [DATA_DIR / "article_review_status.csv"],
    "violation_articles": [DATA_DIR / "violation_articles.csv"],
    "non_violation_articles": [DATA_DIR / "non_violation_articles.csv"],
    "external_candidates": [DATA_DIR / "external_violations.csv"],
}


def read_csv(path: Path) -> tuple[list[str], list[tuple]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = [tuple(v if v != "" else None for v in row.values()) for row in reader]
    return columns, rows


def main() -> None:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
            for table, paths in CSV_TABLES.items():
                target = sql.Identifier(OWN_SCHEMA, table)
                cur.execute(sql.SQL("TRUNCATE {}").format(target))
                for path in paths:
                    if not path.exists():
                        continue
                    columns, rows = read_csv(path)
                    insert = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
                        target, sql.SQL(", ").join(map(sql.Identifier, columns))
                    )
                    execute_values(cur, insert.as_string(cur), rows)
                    print(f"{OWN_SCHEMA}.{table}: {len(rows)} rows from {path.name}")
            cur.execute(confirmed_external_insert())
            print(f"{OWN_SCHEMA}.violation_articles: {cur.rowcount} confirmed external violations")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
