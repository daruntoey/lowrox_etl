"""
ci.py — stage 2 of slide 12, built around slide 4's distinction.

    "CI is no longer only about testing and validating code and components,
     but also testing and validating data, data schemas, and models."

So there are four suites, and a normal software CI only has the first:

    CODE     the components import, and their contracts hold
    DATA     the values are plausible and the labels are usable
    SCHEMA   the features the model expects still exist, with the right types
    MODEL    the artefact loads, predicts, and is not quietly leaking

Any failure fails the build. That is the point of a gate — a check that can
be ignored is documentation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

RESULTS: list[dict] = []


def _check(suite: str, name: str, passed: bool, detail: str = "") -> bool:
    RESULTS.append({"suite": suite, "check": name,
                    "result": "PASS" if passed else "FAIL", "detail": detail})
    return passed


# --------------------------------------------------------------------------
def test_code(modules: dict) -> bool:
    """Do the components exist and expose what the pipeline calls?"""
    ok = True
    contracts = {
        "feature_store": ["materialise", "get_training_data", "get_online_features"],
        "features": ["ALL_FEATURE_SETS", "build_labels", "snapshot_dates"],
        "training": ["train_pipeline", "evaluate", "validate", "feature_statistics"],
        "registry": ["register", "promote", "production", "load"],
        "serving": ["batch_score", "OnlinePredictor", "check_training_serving_skew"],
        "monitoring": ["monitor", "check_data_drift", "check_concept_drift"],
    }
    for mod_name, attrs in contracts.items():
        mod = modules.get(mod_name)
        missing = [a for a in attrs if mod is None or not hasattr(mod, a)]
        ok &= _check("code", f"{mod_name} exposes its interface", not missing,
                     f"missing {missing}" if missing else f"{len(attrs)} symbols")
    return ok


# --------------------------------------------------------------------------
def test_data(training: pd.DataFrame, features: list[str], target: str) -> bool:
    """Is the data fit to train on? Slide 4's 'validating data'."""
    ok = True
    ok &= _check("data", "enough rows to train", len(training) >= 500,
                 f"{len(training)} rows")
    rate = training[target].mean()
    ok &= _check("data", "label is not degenerate", 0.005 < rate < 0.995,
                 f"positive rate {rate:.2%}")
    ok &= _check("data", "no duplicate (entity, timestamp) rows",
                 not training.duplicated(subset=["user_id", "ts"]).any(),
                 f"{int(training.duplicated(subset=['user_id','ts']).sum())} duplicates")

    allna = [f for f in features if training[f].isna().all()]
    ok &= _check("data", "no entirely empty feature", not allna, f"{allna}")

    inf = [f for f in features
           if np.isinf(pd.to_numeric(training[f], errors="coerce").fillna(0)).any()]
    ok &= _check("data", "no infinities", not inf, f"{inf}")

    # A feature that correlates almost perfectly with the target is the
    # signature of leakage, and it is cheap to look for before training
    # rather than after a suspicious score.
    suspicious = []
    for f in features:
        s = pd.to_numeric(training[f], errors="coerce")
        if s.notna().sum() > 10 and s.nunique() > 1:
            c = abs(s.corr(training[target]))
            if pd.notna(c) and c > 0.95:
                suspicious.append((f, round(float(c), 3)))
    ok &= _check("data", "no feature almost perfectly correlated with the target",
                 not suspicious, f"{suspicious}")
    return ok


# --------------------------------------------------------------------------
def test_schema(training: pd.DataFrame, feature_sets: list) -> bool:
    """Slide 4's 'validating data schemas'."""
    ok = True
    for fs in feature_sets:
        missing = [f for f in fs.features if f not in training.columns]
        ok &= _check("schema", f"{fs.name} contributes all its features",
                     not missing, f"missing {missing}" if missing else
                     f"{len(fs.features)} features present")
    for col in ("user_id", "ts", "is_churn"):
        ok &= _check("schema", f"key column '{col}' present",
                     col in training.columns, "")
    numeric_bad = [f for fs in feature_sets for f in fs.features
                   if f in training.columns
                   and not pd.api.types.is_numeric_dtype(training[f])]
    ok &= _check("schema", "every feature is numeric at the model boundary",
                 not numeric_bad, f"{numeric_bad}")

    # The time index must sit at the store's resolution on BOTH sides of the
    # point-in-time join. When it does not, merge_asof raises MergeError and
    # no training set can be built at all — a schema fault, caught here with
    # a name on it rather than as a traceback six cells later.
    from .feature_store import STORE_TIME_UNIT, read_offline
    want = f"datetime64[{STORE_TIME_UNIT}, UTC]"
    units = {"training.ts": str(training["ts"].dtype)} if "ts" in training else {}
    for fs in feature_sets:
        hist = read_offline(fs)
        if not hist.empty and fs.timestamp in hist.columns:
            units[f"{fs.name}.{fs.timestamp}"] = str(hist[fs.timestamp].dtype)
    wrong = {k: v for k, v in units.items() if v != want}
    ok &= _check("schema", f"time index is {want} everywhere", not wrong,
                 f"{wrong}" if wrong else f"{len(units)} checked")
    return ok


# --------------------------------------------------------------------------
def test_model(result: dict, *, min_roc_auc: float = 0.60) -> bool:
    """Slide 4's 'validating models'."""
    ok = True
    m = result["metrics_test"]
    ok &= _check("model", "beats the floor on ROC-AUC",
                 m["roc_auc"] >= min_roc_auc, f"{m['roc_auc']:.4f}")
    ok &= _check("model", "beats the baseline",
                 m["roc_auc"] > result["baseline"]["roc_auc"],
                 f"{m['roc_auc']:.4f} vs {result['baseline']['roc_auc']:.4f}")
    ok &= _check("model", "all validation gates passed",
                 bool(result["validation"]["passed"]),
                 f"{int(result['validation']['checks'].passed.sum())}"
                 f"/{len(result['validation']['checks'])} gates")

    # The artefact has to survive the round trip it will make in production.
    model = result["model"]
    X = result["splits"].X_test.head(20)
    try:
        p = model.predict_proba(X)[:, 1]
        round_trip = bool(np.all((p >= 0) & (p <= 1)) and len(p) == len(X))
    except Exception as exc:
        round_trip, p = False, str(exc)
    ok &= _check("model", "predicts valid probabilities on held-out rows",
                 round_trip, f"n={len(X)}")

    # A model that returns the same number for everyone passes every metric
    # gate on a sufficiently imbalanced problem and is useless.
    spread = float(np.std(model.predict_proba(result["splits"].X_test)[:, 1]))
    ok &= _check("model", "predictions are not constant", spread > 1e-3,
                 f"std {spread:.4f}")
    return ok


# --------------------------------------------------------------------------
def run_ci(modules: dict, result: dict, feature_sets: list,
           verbose: bool = True) -> dict:
    RESULTS.clear()
    t0 = time.time()
    training = result["data"]
    features = result["features"]

    test_code(modules)
    test_data(training, features, "is_churn")
    test_schema(training, feature_sets)
    test_model(result)

    df = pd.DataFrame(RESULTS)
    passed = bool((df.result == "PASS").all())
    if verbose:
        print("STAGE 2 — PIPELINE CONTINUOUS INTEGRATION")
        for suite, grp in df.groupby("suite", sort=False):
            n_ok = (grp.result == "PASS").sum()
            print(f"\n  {suite.upper()}  {n_ok}/{len(grp)} passed")
            for _, r in grp.iterrows():
                mark = " " if r.result == "PASS" else "!"
                print(f"   {mark} [{r.result}] {r.check}"
                      + (f"  — {r.detail}" if r.detail else ""))
        print(f"\n  BUILD {'PASSED' if passed else 'FAILED'} "
              f"({(df.result=='PASS').sum()}/{len(df)} checks, {time.time()-t0:.1f}s)")
    return {"passed": passed, "results": df}
