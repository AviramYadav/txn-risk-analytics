"""Rule-based data-quality checks, quarantine and reconciliation.

Every rule is a SQL predicate that is TRUE for a *bad* staging row. The same
predicate drives the scorecard (how many rows fail) and the quarantine (which
rows are held back from the clean fact table), so the two can never disagree.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class Rule:
    name: str
    dimension: str
    predicate: str
    threshold: float = 0.999  # minimum acceptable pass rate


RULES = [
    Rule("txn_id_unique", "uniqueness",
         "rowid NOT IN (SELECT MIN(rowid) FROM stg_transactions GROUP BY txn_id)", 1.0),
    Rule("required_fields_present", "completeness",
         "txn_id IS NULL OR customer_id IS NULL OR merchant_id IS NULL OR ts IS NULL OR is_fraud IS NULL"),
    Rule("amount_not_null", "completeness", "amount IS NULL"),
    Rule("amount_positive", "validity", "amount <= 0"),
    Rule("channel_in_domain", "validity", "channel IS NULL OR channel NOT IN ('POS', 'ECOM', 'ATM')"),
    Rule("ts_parseable", "validity", "datetime(ts) IS NULL"),
    Rule("ts_not_in_future", "timeliness", "datetime(ts) > datetime(:as_of)"),
    Rule("merchant_exists", "referential_integrity",
         "merchant_id NOT IN (SELECT merchant_id FROM dim_merchant)", 1.0),
    Rule("customer_exists", "referential_integrity",
         "customer_id NOT IN (SELECT customer_id FROM dim_customer)", 1.0),
]

FACT_COLUMNS = "txn_id, customer_id, merchant_id, ts, amount, channel, device_id, is_fraud, fraud_type"


def run_checks(conn: sqlite3.Connection, as_of: str, rules=RULES) -> pd.DataFrame:
    """Score every rule against staging and persist the results to dq_results."""
    total = conn.execute("SELECT COUNT(*) FROM stg_transactions").fetchone()[0]
    run_at = datetime.now().isoformat(timespec="seconds")
    rows = []
    for r in rules:
        failed = conn.execute(f"SELECT COUNT(*) FROM stg_transactions WHERE {r.predicate}",
                              {"as_of": as_of}).fetchone()[0]
        pass_rate = 1 - failed / total if total else 1.0
        rows.append({"rule": r.name, "dimension": r.dimension, "failed_rows": failed, "total_rows": total,
                     "pass_rate": round(pass_rate, 6), "threshold": r.threshold,
                     "status": "PASS" if pass_rate >= r.threshold else "FAIL"})
    scorecard = pd.DataFrame(rows)
    scorecard.assign(run_at=run_at).to_sql("dq_results", conn, if_exists="append", index=False)
    conn.commit()
    return scorecard


def quarantine_and_promote(conn: sqlite3.Connection, as_of: str, rules=RULES) -> None:
    """Hold back every failing row (with each reason) and promote the rest to fct_transactions."""
    conn.execute("DELETE FROM dq_quarantine")
    conn.execute("DELETE FROM fct_transactions")
    for r in rules:
        conn.execute(
            f"INSERT INTO dq_quarantine (stg_rowid, txn_id, rule, dimension) "
            f"SELECT rowid, txn_id, :rule, :dimension FROM stg_transactions WHERE {r.predicate}",
            {"rule": r.name, "dimension": r.dimension, "as_of": as_of},
        )
    conn.execute(
        f"INSERT INTO fct_transactions ({FACT_COLUMNS}) "
        f"SELECT {FACT_COLUMNS} FROM stg_transactions "
        f"WHERE rowid NOT IN (SELECT stg_rowid FROM dq_quarantine)"
    )
    conn.commit()


def reconcile(conn: sqlite3.Connection) -> dict:
    """Row-count and control-total reconciliation: staged = promoted + quarantined."""
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    staged = q("SELECT COUNT(*) FROM stg_transactions")
    promoted = q("SELECT COUNT(*) FROM fct_transactions")
    quarantined = q("SELECT COUNT(DISTINCT stg_rowid) FROM dq_quarantine")
    staged_amt = q("SELECT COALESCE(SUM(amount), 0) FROM stg_transactions")
    promoted_amt = q("SELECT COALESCE(SUM(amount), 0) FROM fct_transactions")
    quarantined_amt = q("SELECT COALESCE(SUM(amount), 0) FROM stg_transactions "
                        "WHERE rowid IN (SELECT stg_rowid FROM dq_quarantine)")
    return {
        "staged_rows": staged,
        "promoted_rows": promoted,
        "quarantined_rows": quarantined,
        "rows_balanced": staged == promoted + quarantined,
        "staged_amount": round(staged_amt, 2),
        "promoted_amount": round(promoted_amt, 2),
        "quarantined_amount": round(quarantined_amt, 2),
        "amount_balanced": abs(staged_amt - promoted_amt - quarantined_amt) < 0.01,
    }
