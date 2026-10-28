"""Fraud model training and evaluation.

Key choices:
* Time-based split (train on the past, test on the future) - a random split
  leaks future behaviour into training and overstates performance.
* PR-AUC / precision / recall rather than accuracy - with ~0.4% fraud,
  a model that never flags anything is already 99.6% "accurate".
* The alert threshold is tuned on a separate validation window, never on test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CATEGORICAL = ["channel", "category", "risk_tier"]
NUMERIC = ["amount", "hour", "is_night", "is_out_of_city", "txn_count_1h", "txn_count_24h",
           "amount_sum_24h", "amount_to_avg_ratio", "hist_txn_count", "secs_since_prev", "is_new_device"]


@dataclass
class ModelResult:
    name: str
    model: object
    threshold: float
    metrics: dict
    pr_curve: tuple = field(repr=False, default=None)


def make_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(df[CATEGORICAL + NUMERIC], columns=CATEGORICAL, dtype=float)
    for col in ["amount", "amount_sum_24h", "secs_since_prev", "amount_to_avg_ratio"]:
        X[f"log_{col}"] = np.log1p(X[col].clip(lower=0))
    return X


def time_split(df: pd.DataFrame, warmup_days=14, valid_days=10, test_days=25):
    """Warm-up (history only) | train | validation | test, in calendar order."""
    day = (df["ts"] - df["ts"].min().normalize()).dt.days
    last = day.max()
    test_start = last - test_days + 1
    valid_start = test_start - valid_days
    return (df[(day >= warmup_days) & (day < valid_start)],
            df[(day >= valid_start) & (day < test_start)],
            df[day >= test_start])


def _best_f1_threshold(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def _evaluate(y_true, proba, threshold, amounts, n_days) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fraud_value = amounts[y_true == 1].sum()
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "alerts_per_day": (tp + fp) / n_days,
        "fraud_value_caught_pct": 100 * amounts[(y_true == 1) & (pred == 1)].sum() / fraud_value if fraud_value else 0.0,
    }


def candidate_models(seed: int = 42) -> dict:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed),
    }


def train_and_evaluate(features: pd.DataFrame, seed: int = 42) -> dict:
    X_all = make_matrix(features)
    train, valid, test = time_split(features)
    X_tr, X_va, X_te = X_all.loc[train.index], X_all.loc[valid.index], X_all.loc[test.index]
    y_tr, y_va, y_te = train["is_fraud"].to_numpy(), valid["is_fraud"].to_numpy(), test["is_fraud"].to_numpy()
    test_days = test["ts"].dt.normalize().nunique()

    results, probas = {}, {}
    for name, model in candidate_models(seed).items():
        model.fit(X_tr, y_tr)
        threshold = _best_f1_threshold(y_va, model.predict_proba(X_va)[:, 1])
        proba = model.predict_proba(X_te)[:, 1]
        metrics = _evaluate(y_te, proba, threshold, test["amount"].to_numpy(), test_days)
        precision, recall, _ = precision_recall_curve(y_te, proba)
        results[name] = ModelResult(name, model, threshold, metrics, (precision, recall))
        probas[name] = proba

    champion = max(results.values(), key=lambda r: r.metrics["pr_auc"])
    return {
        "results": results,
        "champion": champion,
        "test_scores": pd.DataFrame({"txn_id": test["txn_id"].to_numpy(), "score": probas[champion.name],
                                     "flagged": (probas[champion.name] >= champion.threshold).astype(int)}),
        "importance": _importance(champion.model, X_te, y_te, seed),
        "split": {"train_rows": len(train), "valid_rows": len(valid), "test_rows": len(test),
                  "train_fraud": int(y_tr.sum()), "valid_fraud": int(y_va.sum()), "test_fraud": int(y_te.sum()),
                  "test_period": f"{test['ts'].min():%Y-%m-%d} to {test['ts'].max():%Y-%m-%d}"},
        "baseline_accuracy": 1 - y_te.mean(),
        "feature_columns": list(X_all.columns),
    }


def _importance(model, X_te, y_te, seed, max_legit=20_000) -> pd.DataFrame:
    """Permutation importance (drop in PR-AUC) on all test fraud plus a sample of legit rows."""
    rng = np.random.default_rng(seed)
    legit = np.flatnonzero(y_te == 0)
    keep = np.concatenate([np.flatnonzero(y_te == 1), rng.choice(legit, min(max_legit, len(legit)), replace=False)])
    imp = permutation_importance(model, X_te.iloc[keep], y_te[keep], scoring="average_precision",
                                 n_repeats=5, random_state=seed, n_jobs=1)
    return (pd.DataFrame({"feature": X_te.columns, "importance": imp.importances_mean})
            .sort_values("importance", ascending=False).reset_index(drop=True))
