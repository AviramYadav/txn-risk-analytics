"""End-to-end pipeline: generate -> load -> data quality -> features -> model -> report."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from . import quality, warehouse
from .generate import GeneratorConfig, generate
from .model import train_and_evaluate
from .report import write_report

log = logging.getLogger("txnrisk")


def run(cfg: GeneratorConfig, db_path: Path, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Generating synthetic data (%d customers, %d days)", cfg.n_customers, cfg.days)
    data = generate(cfg)
    as_of = (pd.Timestamp(cfg.start_date) + pd.Timedelta(days=cfg.days)).strftime("%Y-%m-%d %H:%M:%S")

    log.info("Loading %d staged transactions into %s", len(data["transactions"]), db_path)
    conn = warehouse.connect(db_path)
    warehouse.create_schema(conn)
    warehouse.load_raw(conn, data)

    log.info("Running %d data-quality rules", len(quality.RULES))
    scorecard = quality.run_checks(conn, as_of)
    quality.quarantine_and_promote(conn, as_of)
    recon = quality.reconcile(conn)
    if not (recon["rows_balanced"] and recon["amount_balanced"]):
        raise RuntimeError(f"Reconciliation failed: {recon}")

    log.info("Building features in SQL")
    warehouse.run_script(conn, "features.sql")
    features = pd.read_sql_query("SELECT * FROM features", conn, parse_dates=["ts"])

    log.info("Training models on %d rows", len(features))
    training = train_and_evaluate(features, seed=cfg.seed)
    training["test_scores"].to_sql("model_scores", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scores_txn ON model_scores (txn_id)")
    conn.commit()
    mi = warehouse.run_named_queries(conn)

    volumes = {"customers": len(data["customers"]), "merchants": len(data["merchants"]),
               "staged_txns": recon["staged_rows"], "clean_txns": recon["promoted_rows"],
               "fraud_txns": int(features["is_fraud"].sum()),
               "fraud_rate_pct": round(100 * features["is_fraud"].mean(), 3)}
    report = write_report(out_dir, volumes=volumes, defects=data["defects"], scorecard=scorecard,
                          recon=recon, training=training, mi=mi)

    champ = training["champion"]
    joblib.dump({"model": champ.model, "threshold": champ.threshold,
                 "feature_columns": training["feature_columns"]}, out_dir / "champion_model.joblib")
    summary = {"volumes": volumes, "reconciliation": recon,
               "dq_failed_rules": scorecard.loc[scorecard.status == "FAIL", "rule"].tolist(),
               "champion": champ.name, "threshold": champ.threshold,
               "metrics": {name: r.metrics for name, r in training["results"].items()},
               "baseline_accuracy": training["baseline_accuracy"],
               "runtime_seconds": round(time.perf_counter() - t0, 1)}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    conn.close()
    log.info("Done in %.1fs - report at %s", summary["runtime_seconds"], report)
    return summary


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["serve"]:
        return serve(argv[1:])
    p = argparse.ArgumentParser(description="Transaction risk analytics pipeline (use 'serve' to start the dashboard)")
    p.add_argument("--customers", type=int, default=5000)
    p.add_argument("--merchants", type=int, default=800)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--fraud-rate", type=float, default=0.03, help="share of customers hit by fraud")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--db", type=Path, default=Path("data/warehouse.db"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = GeneratorConfig(n_customers=args.customers, n_merchants=args.merchants, days=args.days,
                          fraud_customer_rate=args.fraud_rate, seed=args.seed)
    summary = run(cfg, args.db, args.out)
    champ = summary["metrics"][summary["champion"]]
    print(f"\nChampion: {summary['champion']}  PR-AUC {champ['pr_auc']:.3f}  ROC-AUC {champ['roc_auc']:.3f}  "
          f"precision {champ['precision']:.1%}  recall {champ['recall']:.1%}")


def serve(argv) -> None:
    p = argparse.ArgumentParser(prog="python -m txnrisk serve", description="Start the analytics dashboard")
    p.add_argument("--port", type=int, default=5003)
    p.add_argument("--db", type=Path, default=Path("data/warehouse.db"))
    p.add_argument("--reports", type=Path, default=Path("reports"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if not args.db.exists():
        log.info("No warehouse yet - running the pipeline once (about 30 seconds)")
        run(GeneratorConfig(), args.db, args.reports)
    from .web import create_app
    create_app(args.db, args.reports).run(port=args.port, debug=False)
