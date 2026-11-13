"""
training.py — the eight data science steps from slides 5 and 6, as code.

  1 data extraction      from the feature store, point-in-time correct
  2 data analysis        schema and distribution of what arrived
  3 data preparation     split, impute, scale
  4 model training       with hyperparameters
  5 model evaluation     metrics on a held-out set
  6 model validation     is it better than a baseline? only then is it a candidate
  7 model serving        serving.py
  8 model monitoring     monitoring.py

The split is by TIME, not at random. The data is a panel: the same member
appears at several month ends, and their rows are not independent. A random
split puts September's row for a member in training and October's in test,
and the model scores well by recognising the member rather than by learning
churn. Splitting on the snapshot date is the only honest option.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import registry
from .feature_store import get_training_data

ALGORITHMS = {
    "gradient_boosting": (GradientBoostingClassifier,
                          {"n_estimators": 200, "max_depth": 3,
                           "learning_rate": 0.05, "random_state": 42}),
    "random_forest": (RandomForestClassifier,
                      {"n_estimators": 300, "max_depth": 8, "random_state": 42,
                       "class_weight": "balanced", "n_jobs": -1}),
    "logistic_regression": (LogisticRegression,
                            {"max_iter": 2000, "class_weight": "balanced",
                             "random_state": 42}),
}


# --------------------------------------------------------------------------
# Step 2 — data analysis
# --------------------------------------------------------------------------
def analyse(df: pd.DataFrame, features: list[str], target: str) -> dict:
    """What arrived, and is it the shape the model expects?"""
    X = df[features]
    report = {
        "rows": len(df),
        "features": len(features),
        "target_rate": float(df[target].mean()),
        "missing_overall": float(X.isna().mean().mean()),
        "constant_features": [c for c in features if X[c].nunique(dropna=True) <= 1],
        "high_missing": [c for c in features if X[c].isna().mean() > 0.5],
    }
    corr = X.corr(numeric_only=True).abs()
    pairs = [(a, b, round(corr.loc[a, b], 3))
             for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
             if corr.loc[a, b] > 0.95]
    report["near_duplicate_pairs"] = pairs
    return report


def feature_statistics(X: pd.DataFrame) -> pd.DataFrame:
    """
    The distribution parameters the chapter names on slide 14 — mean, median,
    standard deviation, IQR, kurtosis — stored at training time so that drift
    can later be measured against what the model actually learned.

    Slide 14's warning is the reason both central tendency AND spread are
    kept: two distributions can share a mean and have nothing else in common.
    """
    rows = {}
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows[c] = {
            "mean": float(s.mean()), "median": float(s.median()),
            "std": float(s.std()), "iqr": float(q3 - q1),
            "p01": float(s.quantile(0.01)), "p99": float(s.quantile(0.99)),
            "kurtosis": float(stats.kurtosis(s)) if len(s) > 3 else 0.0,
            "skew": float(stats.skew(s)) if len(s) > 2 else 0.0,
            "missing_rate": float(X[c].isna().mean()),
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------
# Step 3 — data preparation
# --------------------------------------------------------------------------
@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    boundaries: dict = field(default_factory=dict)


def time_split(df: pd.DataFrame, features: list[str], target: str,
               *, timestamp: str = "ts", train_frac: float = 0.6,
               val_frac: float = 0.2) -> Splits:
    snaps = sorted(df[timestamp].unique())
    n = len(snaps)
    i_train, i_val = int(n * train_frac), int(n * (train_frac + val_frac))
    tr, va = snaps[:i_train], snaps[i_train:i_val]
    te = snaps[i_val:]

    m_tr = df[timestamp].isin(tr)
    m_va = df[timestamp].isin(va)
    m_te = df[timestamp].isin(te)
    return Splits(
        df.loc[m_tr, features], df.loc[m_tr, target],
        df.loc[m_va, features], df.loc[m_va, target],
        df.loc[m_te, features], df.loc[m_te, target],
        {"train": [str(pd.Timestamp(tr[0]).date()), str(pd.Timestamp(tr[-1]).date())],
         "val": [str(pd.Timestamp(va[0]).date()), str(pd.Timestamp(va[-1]).date())],
         "test": [str(pd.Timestamp(te[0]).date()), str(pd.Timestamp(te[-1]).date())]})


def build_pipeline(algorithm: str = "gradient_boosting", **overrides) -> Pipeline:
    """
    Imputation and scaling live INSIDE the pipeline, never as a separate step
    applied to the whole frame first. Fitting an imputer on all the data
    before splitting leaks test-set statistics into training — subtly, and
    with no error message.
    """
    cls, params = ALGORITHMS[algorithm]
    params = {**params, **overrides}
    return Pipeline([
        # slide 26: KNN imputation rather than a blunt mean
        ("impute", KNNImputer(n_neighbors=5)),
        # slide 29: z-score normalisation
        ("scale", StandardScaler()),
        ("model", cls(**params)),
    ]), params


# --------------------------------------------------------------------------
# Steps 4 and 5 — training and evaluation
# --------------------------------------------------------------------------
def evaluate(model, X: pd.DataFrame, y: pd.Series, *, top_pct: float = 0.20) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Ranking quality matters more than the 0.5 threshold here: the product
    # acts on "who should we contact this week", which is a top-N decision.
    k = max(1, int(len(proba) * top_pct))
    top_idx = np.argsort(proba)[-k:]
    recall_at_k = float(y.iloc[top_idx].sum() / max(1, y.sum()))

    return {
        "roc_auc": float(roc_auc_score(y, proba)) if y.nunique() > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, proba)) if y.nunique() > 1 else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, proba)),
        f"recall_at_top_{int(top_pct*100)}pct": recall_at_k,
        "positive_rate": float(y.mean()),
        "n": int(len(y)),
    }


def baseline_metrics(y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                     *, top_pct: float = 0.20) -> dict:
    """
    The thing a new model has to beat. Predicting the training base rate for
    everyone is the honest floor: it has no skill at all, so any model that
    cannot beat it is not worth deploying.
    """
    p = float(y_train.mean())
    proba = np.full(len(y_test), p)
    k = max(1, int(len(proba) * top_pct))
    return {
        "roc_auc": 0.5,
        "pr_auc": float(y_test.mean()),
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "brier": float(brier_score_loss(y_test, proba)),
        f"recall_at_top_{int(top_pct*100)}pct": float(k / max(1, len(proba))),
        "strategy": "predict the training base rate for every member",
    }


# --------------------------------------------------------------------------
# Step 6 — model validation
# --------------------------------------------------------------------------
def validate(metrics: dict, baseline: dict, *, min_roc_auc: float = 0.60,
             min_lift_over_baseline: float = 0.05,
             importances: pd.DataFrame | None = None,
             suspicious_auc: float = 0.98,
             dominant_feature_share: float = 0.50) -> dict:
    """
    Slide 6: "the model is confirmed to be adequate for deployment — that its
    predictive performance is better than a certain baseline."

    The first three gates are that comparison. The last two are LEAKAGE
    SENTINELS, and they exist because of a real failure in this project: the
    first churn model scored ROC-AUC 0.995 with one feature holding 85% of
    the importance, and the feature turned out to be the label under another
    name. Both numbers were visible and neither was being checked.

    A model can be rejected for being too good. That is not a paradox — an
    implausible score is evidence about the data, not about the model.
    """
    checks = [
        {"gate": "roc_auc above floor",
         "value": round(metrics["roc_auc"], 4), "threshold": min_roc_auc,
         "passed": metrics["roc_auc"] >= min_roc_auc},
        {"gate": "beats baseline on roc_auc",
         "value": round(metrics["roc_auc"] - baseline["roc_auc"], 4),
         "threshold": min_lift_over_baseline,
         "passed": metrics["roc_auc"] - baseline["roc_auc"] >= min_lift_over_baseline},
        {"gate": "better calibrated than baseline (lower Brier)",
         "value": round(metrics["brier"], 4), "threshold": round(baseline["brier"], 4),
         "passed": metrics["brier"] <= baseline["brier"]},
        {"gate": "roc_auc not implausibly high (leakage sentinel)",
         "value": round(metrics["roc_auc"], 4), "threshold": suspicious_auc,
         "passed": metrics["roc_auc"] < suspicious_auc},
    ]
    if importances is not None and len(importances):
        top = float(importances["importance"].iloc[0])
        total = float(importances["importance"].sum())
        share = top / total if total else 0.0
        checks.append({
            "gate": f"no single feature dominates ({importances['feature'].iloc[0]})",
            "value": round(share, 4), "threshold": dominant_feature_share,
            "passed": share < dominant_feature_share})

    df = pd.DataFrame(checks)
    return {"passed": bool(df.passed.all()), "checks": df}


# --------------------------------------------------------------------------
# The whole training pipeline
# --------------------------------------------------------------------------
def train_pipeline(labels: pd.DataFrame, feature_sets: list, *,
                   algorithm: str = "gradient_boosting",
                   model_name: str = "lowrox_churn",
                   run_id: str = "", verbose: bool = True,
                   register_model: bool = True, calibrate: bool = True,
                   calibration_method: str = "sigmoid", **overrides) -> dict:
    # 1 extraction — point-in-time correct, straight from the feature store
    data = get_training_data(labels, feature_sets, verbose=False)
    features = [f for fs in feature_sets for f in fs.features]
    target = "is_churn"

    # 2 analysis
    report = analyse(data, features, target)
    usable = [f for f in features
              if f not in report["constant_features"] and f not in report["high_missing"]]

    # 3 preparation
    sp = time_split(data, usable, target)

    # 4 training
    pipe, params = build_pipeline(algorithm, **overrides)
    pipe.fit(sp.X_train, sp.y_train)
    m_val_raw = evaluate(pipe, sp.X_val, sp.y_val)

    # 4b calibration
    #
    # Gradient boosting on a 9% positive rate ranks well and is badly
    # calibrated: it beat the baseline comfortably on ROC-AUC while being
    # WORSE on Brier score, which the validation gate caught. Ranking is
    # enough to decide who to contact first; it is not enough to answer "how
    # much revenue is at risk", and the business question here is the second
    # kind. So the scores are calibrated on the validation split — which is
    # held out from training, and is the natural use for a split that
    # otherwise only watches.
    served = pipe
    if calibrate and sp.y_val.nunique() > 1:
        served = CalibratedClassifierCV(
            FrozenEstimator(pipe), method=calibration_method)
        served.fit(sp.X_val, sp.y_val)

    # 5 evaluation
    m_val = evaluate(served, sp.X_val, sp.y_val)
    m_test = evaluate(served, sp.X_test, sp.y_test)
    base = baseline_metrics(sp.y_train, sp.X_test, sp.y_test)

    # 6 validation — importances are computed first so the leakage sentinel
    #   has something to inspect
    imp = feature_importance(pipe, usable, top=len(usable))
    verdict = validate(m_test, base, importances=imp)

    stats_df = feature_statistics(sp.X_train)

    mv = None
    if register_model and verdict["passed"]:
        mv = registry.register(
            served, name=model_name, metrics=m_test, baseline_metrics=base,
            features=usable, feature_sets=[fs.name for fs in feature_sets],
            training_rows=len(sp.X_train), training_window=sp.boundaries["train"],
            algorithm=algorithm, hyperparameters=params,
            feature_stats=stats_df, run_id=run_id)

    if verbose:
        print(f"ALGORITHM {algorithm}")
        print(f"  rows {len(data)}  features {len(usable)} of {len(features)} usable")
        if report["constant_features"]:
            print(f"  dropped constant: {report['constant_features']}")
        if report["high_missing"]:
            print(f"  dropped >50% missing: {report['high_missing']}")
        print(f"  split by time  train {sp.boundaries['train']}  "
              f"val {sp.boundaries['val']}  test {sp.boundaries['test']}")
        print(f"  train {len(sp.X_train)} / val {len(sp.X_val)} / test {len(sp.X_test)}")
        if calibrate:
            print(f"  calibrated on the validation split ({calibration_method}): "
                  f"Brier {m_val_raw['brier']:.4f} -> {m_val['brier']:.4f}")
        print(f"\n  test  roc_auc {m_test['roc_auc']:.4f}  pr_auc {m_test['pr_auc']:.4f}  "
              f"recall@top20% {m_test['recall_at_top_20pct']:.4f}")
        print(f"  base  roc_auc {base['roc_auc']:.4f}  pr_auc {base['pr_auc']:.4f}")
        print("\n  VALIDATION GATES")
        print(verdict["checks"].to_string(index=False))
        print(f"\n  verdict: {'PASSED — registered as a candidate' if verdict['passed'] else 'FAILED — not registered'}")
        if mv:
            print(f"  registered {mv.name} {mv.version} (stage={mv.stage})")

    return {"model": served, "raw_model": pipe, "splits": sp, "metrics_test": m_test,
            "metrics_val": m_val, "baseline": base, "validation": verdict,
            "analysis": report, "feature_stats": stats_df, "importances": imp,
            "model_version": mv, "features": usable, "data": data}


def feature_importance(model, features: list[str], top: int = 15) -> pd.DataFrame:
    est = model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        imp = np.abs(est.coef_[0])
    else:
        return pd.DataFrame()
    return (pd.DataFrame({"feature": features, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(top).reset_index(drop=True))
