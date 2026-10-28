"""Flask dashboard over the warehouse: overview, alert queue, drill-downs, DQ and a read-only SQL console."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from flask import Flask, abort, render_template, request, send_from_directory

from .quality import RULES
from .warehouse import named_queries

MAX_ROWS = 500
OUTCOMES = {
    "caught": "s.flagged = 1 AND f.is_fraud = 1",
    "false_alarm": "s.flagged = 1 AND f.is_fraud = 0",
    "missed": "s.flagged = 0 AND f.is_fraud = 1",
}


def _readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _deny_writes(action, *_):
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def console_connection(db_path: Path) -> sqlite3.Connection:
    """Separate connection for user SQL: opened read-only and with an authorizer that denies anything but reads."""
    conn = _readonly(db_path)
    conn.set_authorizer(_deny_writes)
    return conn


def run_user_sql(conn: sqlite3.Connection, sql: str, max_seconds: float = 5.0):
    """Execute one read-only statement with a row cap and a time limit.

    `conn` must come from console_connection(), which carries the write-denying authorizer.
    """
    statement = sql.strip().rstrip(";")
    if not statement:
        return [], [], None
    if ";" in statement:
        return [], [], "Run one statement at a time."
    deadline = time.monotonic() + max_seconds
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cur = conn.execute(statement)
        rows = cur.fetchmany(MAX_ROWS + 1)
        cols = [d[0] for d in cur.description or []]
        return cols, [tuple(r) for r in rows], None
    except sqlite3.Error as e:
        msg = str(e)
        if "interrupted" in msg:
            msg = f"Query stopped after {max_seconds:.0f}s."
        elif "not authorized" in msg:
            msg = "This console is read-only: only SELECT / WITH queries are allowed."
        return [], [], msg
    finally:
        conn.set_progress_handler(None, 0)


def create_app(db_path: str | Path = "data/warehouse.db", reports: str | Path = "reports") -> Flask:
    app = Flask(__name__)
    db_path, reports = Path(db_path), Path(reports).resolve()
    conn = _readonly(db_path)
    console = console_connection(db_path)
    q = lambda sql, *p: [dict(r) for r in conn.execute(sql, p)]  # noqa: E731
    one = lambda sql, *p: conn.execute(sql, p).fetchone()  # noqa: E731

    def metrics():
        path = reports / "metrics.json"
        return json.loads(path.read_text()) if path.exists() else {}

    @app.get("/reports/<path:name>")
    def report_file(name):
        if not name.endswith(".png"):
            abort(404)
        return send_from_directory(reports, name)

    @app.get("/")
    def overview():
        m = metrics()
        scorecard = q("SELECT rule, dimension, failed_rows, total_rows, pass_rate, threshold, status FROM dq_results "
                      "WHERE run_at = (SELECT MAX(run_at) FROM dq_results)")
        return render_template("overview.html", m=m, scorecard=scorecard,
                               champion=m.get("metrics", {}).get(m.get("champion"), {}),
                               by_type=q(named_queries()["fraud_by_type"]),
                               by_channel=q(named_queries()["fraud_by_channel"]))

    @app.get("/alerts")
    def alerts():
        outcome = request.args.get("outcome", "")
        sql = """SELECT f.txn_id, f.customer_id, f.ts, f.amount, f.channel, f.is_fraud, f.fraud_type,
                        m.category, m.risk_tier, s.score, s.flagged
                 FROM model_scores s
                 JOIN fct_transactions f ON f.txn_id = s.txn_id
                 JOIN dim_merchant m ON m.merchant_id = f.merchant_id
                 WHERE """ + OUTCOMES.get(outcome, "s.flagged = 1") + " ORDER BY s.score DESC LIMIT 400"
        counts = {k: one(f"SELECT COUNT(*) FROM model_scores s JOIN fct_transactions f USING (txn_id) WHERE {v}")[0]
                  for k, v in OUTCOMES.items()}
        window = one("SELECT MIN(f.ts), MAX(f.ts), COUNT(*) FROM model_scores s JOIN fct_transactions f USING (txn_id)")
        return render_template("alerts.html", rows=q(sql), outcome=outcome, counts=counts, window=window,
                               threshold=metrics().get("threshold"))

    @app.get("/txn/<txn_id>")
    def txn(txn_id):
        t = one("""SELECT f.*, m.category, m.risk_tier, m.city AS merchant_city, c.home_city, c.segment,
                          s.score, s.flagged
                   FROM fct_transactions f JOIN dim_merchant m USING (merchant_id) JOIN dim_customer c USING (customer_id)
                   LEFT JOIN model_scores s USING (txn_id) WHERE f.txn_id = ?""", txn_id)
        if t is None:
            abort(404)
        feats = one("SELECT * FROM features WHERE txn_id = ?", txn_id)
        around = q("""SELECT f.txn_id, f.ts, f.amount, f.channel, f.device_id, f.is_fraud, m.category, m.city, s.score
                      FROM fct_transactions f JOIN dim_merchant m USING (merchant_id) LEFT JOIN model_scores s USING (txn_id)
                      WHERE f.customer_id = ? AND f.ts BETWEEN datetime(?, '-1 day') AND datetime(?, '+6 hours')
                      ORDER BY f.ts""", t["customer_id"], t["ts"], t["ts"])
        return render_template("txn.html", t=dict(t), feats=dict(feats) if feats else {}, around=around,
                               threshold=metrics().get("threshold"))

    @app.get("/customer/<customer_id>")
    def customer(customer_id):
        c = one("SELECT * FROM dim_customer WHERE customer_id = ?", customer_id)
        if c is None:
            abort(404)
        stats = one("""SELECT COUNT(*) n, ROUND(SUM(amount)) spend, ROUND(AVG(amount)) avg_ticket, SUM(is_fraud) frauds,
                              MIN(ts) first_ts, MAX(ts) last_ts FROM fct_transactions WHERE customer_id = ?""", customer_id)
        rows = q("""SELECT f.txn_id, f.ts, f.amount, f.channel, f.device_id, f.is_fraud, f.fraud_type, m.category, m.city, s.score
                    FROM fct_transactions f JOIN dim_merchant m USING (merchant_id) LEFT JOIN model_scores s USING (txn_id)
                    WHERE f.customer_id = ? ORDER BY f.ts DESC LIMIT 300""", customer_id)
        return render_template("customer.html", c=dict(c), stats=dict(stats), rows=rows)

    @app.get("/quality")
    def quality():
        rule = request.args.get("rule", "")
        scorecard = q("SELECT * FROM dq_results WHERE run_at = (SELECT MAX(run_at) FROM dq_results)")
        recon = metrics().get("reconciliation", {})
        sample = []
        if rule:
            sample = q("""SELECT s.* FROM dq_quarantine d JOIN stg_transactions s ON s.rowid = d.stg_rowid
                          WHERE d.rule = ? LIMIT 50""", rule)
        return render_template("quality.html", scorecard=scorecard, recon=recon, rule=rule, sample=sample,
                               rules={r.name: r for r in RULES})

    @app.route("/sql", methods=["GET", "POST"])
    def sql_console():
        examples = named_queries()
        sql = request.form.get("sql") or examples.get(request.args.get("example", ""), "") or examples["fraud_by_channel"]
        cols, rows, error = run_user_sql(console, sql) if request.method == "POST" or request.args.get("example") else ([], [], None)
        tables = [r["name"] for r in q("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        return render_template("sql.html", sql=sql, cols=cols, rows=rows[:MAX_ROWS], truncated=len(rows) > MAX_ROWS,
                               error=error, examples=examples, tables=tables)

    return app
