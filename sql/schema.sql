-- Star schema for the transaction-risk warehouse (SQLite).

DROP TABLE IF EXISTS model_scores;
DROP TABLE IF EXISTS features;
DROP TABLE IF EXISTS dq_results;
DROP TABLE IF EXISTS dq_quarantine;
DROP TABLE IF EXISTS fct_transactions;
DROP TABLE IF EXISTS stg_transactions;
DROP TABLE IF EXISTS dim_merchant;
DROP TABLE IF EXISTS dim_customer;

CREATE TABLE dim_customer (
    customer_id   TEXT PRIMARY KEY,
    segment       TEXT NOT NULL,
    home_city     TEXT NOT NULL,
    signup_date   TEXT NOT NULL,
    credit_limit  REAL NOT NULL
);

CREATE TABLE dim_merchant (
    merchant_id   TEXT PRIMARY KEY,
    category      TEXT NOT NULL,
    risk_tier     TEXT NOT NULL CHECK (risk_tier IN ('low', 'medium', 'high')),
    city          TEXT NOT NULL
);

-- Landing table: deliberately unconstrained so bad records arrive and can be measured.
CREATE TABLE stg_transactions (
    txn_id        TEXT,
    customer_id   TEXT,
    merchant_id   TEXT,
    ts            TEXT,
    amount        REAL,
    channel       TEXT,
    device_id     TEXT,
    is_fraud      INTEGER,
    fraud_type    TEXT
);

-- Clean fact table: only rows that pass every data-quality rule are promoted here.
CREATE TABLE fct_transactions (
    txn_id        TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES dim_customer (customer_id),
    merchant_id   TEXT NOT NULL REFERENCES dim_merchant (merchant_id),
    ts            TEXT NOT NULL,
    amount        REAL NOT NULL CHECK (amount > 0),
    channel       TEXT NOT NULL CHECK (channel IN ('POS', 'ECOM', 'ATM')),
    device_id     TEXT,
    is_fraud      INTEGER NOT NULL CHECK (is_fraud IN (0, 1)),
    fraud_type    TEXT
);
CREATE INDEX idx_fct_customer_ts ON fct_transactions (customer_id, ts);

CREATE TABLE dq_quarantine (
    stg_rowid     INTEGER NOT NULL,
    txn_id        TEXT,
    rule          TEXT NOT NULL,
    dimension     TEXT NOT NULL
);

CREATE TABLE dq_results (
    run_at        TEXT NOT NULL,
    rule          TEXT NOT NULL,
    dimension     TEXT NOT NULL,
    failed_rows   INTEGER NOT NULL,
    total_rows    INTEGER NOT NULL,
    pass_rate     REAL NOT NULL,
    threshold     REAL NOT NULL,
    status        TEXT NOT NULL
);
