import sqlite3

import pandas as pd
import pytest

from txnrisk import quality, warehouse
from txnrisk.generate import GeneratorConfig, generate
from txnrisk.model import make_matrix, time_split

SMALL = GeneratorConfig(n_customers=300, n_merchants=200, days=40, defect_rate=0.01, seed=7)
AS_OF = "2025-02-10 00:00:00"


@pytest.fixture(scope="module")
def data():
    return generate(SMALL)


@pytest.fixture()
def conn(data):
    c = warehouse.connect(":memory:")
    warehouse.create_schema(c)
    warehouse.load_raw(c, data)
    yield c
    c.close()


def test_generator_is_reproducible(data):
    again = generate(SMALL)
    pd.testing.assert_frame_equal(data["transactions"], again["transactions"])


def test_generator_produces_labelled_fraud(data):
    txns = data["transactions"]
    assert txns["is_fraud"].sum() > 0
    assert set(txns.loc[txns.is_fraud == 1, "fraud_type"]) <= {"account_takeover", "card_testing", "skimming"}


def test_every_injected_defect_is_caught(conn, data):
    per_type = int(data["defects"]["injected_rows"].iloc[0])
    scorecard = quality.run_checks(conn, AS_OF).set_index("rule")["failed_rows"]
    for rule in ["txn_id_unique", "amount_not_null", "amount_positive",
                 "channel_in_domain", "ts_not_in_future", "merchant_exists"]:
        assert scorecard[rule] == per_type, rule
    assert scorecard["customer_exists"] == 0


def test_quarantine_reconciles_and_fact_is_clean(conn):
    quality.quarantine_and_promote(conn, AS_OF)
    recon = quality.reconcile(conn)
    assert recon["rows_balanced"] and recon["amount_balanced"]
    bad = conn.execute("""SELECT COUNT(*) FROM fct_transactions
                          WHERE amount IS NULL OR amount <= 0 OR channel NOT IN ('POS','ECOM','ATM')
                             OR merchant_id NOT IN (SELECT merchant_id FROM dim_merchant)""").fetchone()[0]
    assert bad == 0


def test_feature_windows_on_hand_built_history():
    c = warehouse.connect(":memory:")
    warehouse.create_schema(c)
    c.execute("INSERT INTO dim_customer VALUES ('C1', 'mass', 'Pune', '2024-01-01', 100000)")
    c.executemany("INSERT INTO dim_merchant VALUES (?, ?, ?, ?)",
                  [("M1", "grocery", "low", "Pune"), ("M2", "electronics", "high", "ONLINE")])
    c.executemany(
        "INSERT INTO fct_transactions VALUES (?, 'C1', ?, ?, ?, ?, ?, ?, NULL)",
        [("T1", "M1", "2025-01-01 10:00:00", 100.0, "POS", None, 0),
         ("T2", "M2", "2025-01-01 10:30:00", 900.0, "ECOM", "D1", 0),
         ("T3", "M2", "2025-01-01 10:45:00", 800.0, "ECOM", "D1", 0),
         ("T4", "M2", "2025-01-02 09:00:00", 700.0, "ECOM", "D2", 1)],
    )
    warehouse.run_script(c, "features.sql")
    f = pd.read_sql_query("SELECT * FROM features ORDER BY txn_id", c).set_index("txn_id")

    assert f.loc["T3", "txn_count_1h"] == 3          # T1, T2, T3 all within the hour
    assert f.loc["T4", "txn_count_1h"] == 1
    assert f.loc["T4", "txn_count_24h"] == 4         # T1 is 23h earlier, so still inside the window
    assert f.loc["T3", "secs_since_prev"] == 15 * 60
    assert f.loc["T3", "amount_to_avg_ratio"] == pytest.approx(800 / 500)
    assert list(f["is_new_device"]) == [0, 1, 0, 1]  # POS has no device; D1 first seen at T2; D2 at T4
    c.close()


def test_time_split_is_ordered_and_disjoint():
    ts = pd.date_range("2025-01-01", periods=90 * 24, freq="h")
    df = pd.DataFrame({"ts": ts, "is_fraud": 0})
    train, valid, test = time_split(df)
    assert train["ts"].max() < valid["ts"].min()
    assert valid["ts"].max() < test["ts"].min()
    assert train["ts"].min() >= pd.Timestamp("2025-01-15")  # warm-up excluded


def test_matrix_has_no_missing_values(conn):
    quality.quarantine_and_promote(conn, AS_OF)
    warehouse.run_script(conn, "features.sql")
    f = pd.read_sql_query("SELECT * FROM features", conn, parse_dates=["ts"])
    assert not make_matrix(f).isna().any().any()


def test_sql_console_is_read_only(tmp_path):
    from txnrisk.web import console_connection, run_user_sql

    db = tmp_path / "w.db"
    c = warehouse.connect(db)
    warehouse.create_schema(c)
    c.execute("INSERT INTO dim_customer VALUES ('C1', 'mass', 'Pune', '2024-01-01', 100000)")
    c.commit()
    c.close()
    console = console_connection(db)
    cols, rows, err = run_user_sql(console, "SELECT customer_id FROM dim_customer")
    assert (cols, rows, err) == (["customer_id"], [("C1",)], None)
    for bad in ["DELETE FROM dim_customer", "DROP TABLE dim_customer", "PRAGMA writable_schema = 1",
                "ATTACH 'other.db' AS o", "SELECT 1; DELETE FROM dim_customer"]:
        assert run_user_sql(console, bad)[2] is not None, bad
    assert warehouse.connect(db).execute("SELECT COUNT(*) FROM dim_customer").fetchone()[0] == 1
