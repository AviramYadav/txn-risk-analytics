"""SQLite warehouse: schema creation, raw loads and named-query helpers."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript((SQL_DIR / "schema.sql").read_text())


def load_raw(conn: sqlite3.Connection, data: dict[str, pd.DataFrame]) -> None:
    """Dimensions go straight in; transactions land in staging, unvalidated."""
    data["customers"].to_sql("dim_customer", conn, if_exists="append", index=False)
    data["merchants"].to_sql("dim_merchant", conn, if_exists="append", index=False)
    data["transactions"].to_sql("stg_transactions", conn, if_exists="append", index=False, chunksize=50_000)
    conn.commit()


def run_script(conn: sqlite3.Connection, name: str) -> None:
    conn.executescript((SQL_DIR / name).read_text())
    conn.commit()


def named_queries(name: str = "analysis.sql") -> dict[str, str]:
    """Split a .sql file into {query_name: sql} on '-- name: <query_name>' markers."""
    text = (SQL_DIR / name).read_text()
    chunks = re.split(r"^--\s*name:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    return {chunks[i]: chunks[i + 1].strip().rstrip(";") for i in range(1, len(chunks), 2)}


def run_named_queries(conn: sqlite3.Connection, name: str = "analysis.sql") -> dict[str, pd.DataFrame]:
    return {q: pd.read_sql_query(sql, conn) for q, sql in named_queries(name).items()}
