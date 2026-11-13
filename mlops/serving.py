"""
serving.py — step 7 of slide 6, in the two shapes the chapter lists.

    batch      score everyone on a schedule, write to Gold, feed the dashboard
    online     score one member on request, from the online feature store

Both read features through feature_store, so a served prediction uses the
same feature definitions the model was trained on. That is the property the
feature store exists to guarantee, and it is worth testing rather than
assuming — see `check_training_serving_skew` at the bottom.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from . import registry
from .feature_store import get_online_features, get_training_data

PREDICTIONS_PATH = "/content/predictions"


def set_predictions_path(path: str) -> str:
    global PREDICTIONS_PATH
    PREDICTIONS_PATH = str(path)
    return PREDICTIONS_PATH


# --------------------------------------------------------------------------
def batch_score(model_name: str, feature_sets: list, entities: list, *,
                version: str | None = None, run_id: str = "",
                high_pct: float = 0.05, medium_pct: float = 0.20) -> pd.DataFrame:
    """
    Score a population from the ONLINE store and persist the result.

    The scored table is what the dashboard reads; nothing downstream re-runs
    the model, so the dashboard and the retention campaign cannot disagree
    about who is at risk.
    """
    mv = (registry.load_meta(model_name, version) if version
          else registry.production(model_name))
    if mv is None:
        raise RuntimeError(f"no production model for '{model_name}' — promote one first")
    model = registry.load(mv.name, mv.version)

    X = get_online_features(entities, feature_sets)
    missing = [f for f in mv.features if f not in X.columns]
    for f in missing:
        X[f] = pd.NA
    proba = model.predict_proba(X[mv.features])[:, 1]

    out = pd.DataFrame({
        "user_id": X["user_id"].values,
        "churn_probability": proba,
        "model_name": mv.name,
        "model_version": mv.version,
        "scored_at": datetime.now(timezone.utc),
        "run_id": run_id,
    })
    # Bands by PERCENTILE of the scored population, not by a fixed
    # probability cut. A retention team has a weekly capacity, not a
    # probability threshold, and a calibrated model on a 9% base rate rarely
    # produces anything above 0.5 — fixed cuts put everyone in "low" and the
    # dashboard says there is nothing to do.
    hi = out["churn_probability"].quantile(1 - high_pct)
    md = out["churn_probability"].quantile(1 - medium_pct)
    out["risk_band"] = np.select(
        [out["churn_probability"] >= hi, out["churn_probability"] >= md],
        ["high", "medium"], default="low")
    if missing:
        out["_features_imputed"] = ",".join(missing)

    write_deltalake(PREDICTIONS_PATH, pa.Table.from_pandas(out, preserve_index=False),
                    mode="overwrite", schema_mode="overwrite")
    return out


def read_predictions() -> pd.DataFrame:
    p = Path(PREDICTIONS_PATH)
    return DeltaTable(str(p)).to_pandas() if (p / "_delta_log").exists() else pd.DataFrame()


# --------------------------------------------------------------------------
class OnlinePredictor:
    """
    The REST-API shape from slide 6, without the HTTP. Loads the production
    model once and answers one member at a time out of the online store.
    """

    def __init__(self, model_name: str, feature_sets: list):
        self.mv = registry.production(model_name)
        if self.mv is None:
            raise RuntimeError(f"no production model for '{model_name}'")
        self.model = registry.load(self.mv.name, self.mv.version)
        self.feature_sets = feature_sets

    def predict(self, user_id: int) -> dict:
        X = get_online_features([user_id], self.feature_sets)
        if X.empty:
            return {"user_id": user_id, "error": "no features in the online store"}
        for f in self.mv.features:
            if f not in X.columns:
                X[f] = pd.NA
        p = float(self.model.predict_proba(X[self.mv.features])[:, 1][0])
        return {"user_id": user_id, "churn_probability": round(p, 4),
                "risk_band": "high" if p >= 0.5 else "medium" if p >= 0.2 else "low",
                "model": f"{self.mv.name}:{self.mv.version}",
                "served_at": datetime.now(timezone.utc).isoformat()}


# --------------------------------------------------------------------------
def check_training_serving_skew(model_name: str, feature_sets: list,
                                labels: pd.DataFrame, n: int = 200) -> dict:
    """
    The test the feature store is supposed to make pass.

    Take the most recent label rows, fetch their features down the TRAINING
    path (point-in-time join against the offline store) and down the SERVING
    path (latest row from the online store), and compare. For members whose
    latest offline row IS the most recent snapshot, the two must agree
    exactly. Any disagreement means the two paths have drifted apart, which
    is the failure mode that makes a model behave differently in production
    than in the notebook.
    """
    mv = registry.production(model_name) or registry.load_meta(
        model_name, registry.versions(model_name)[-1].version)
    latest_ts = labels["ts"].max()
    recent = labels[labels["ts"] == latest_ts].head(n)
    if recent.empty:
        return {"compared": 0, "mismatches": 0, "verdict": "no rows to compare"}

    offline = get_training_data(recent, feature_sets, verbose=False)
    online = get_online_features(recent["user_id"].tolist(), feature_sets)

    feats = [f for f in mv.features if f in offline.columns and f in online.columns]
    a = offline[["user_id", *feats]].set_index("user_id").sort_index()
    b = online[["user_id", *feats]].set_index("user_id").sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    diff = (a.fillna(-999_999) - b.fillna(-999_999)).abs()
    mismatched = (diff > 1e-6).any(axis=1).sum()
    worst = (diff > 1e-6).sum().sort_values(ascending=False).head(5)

    return {"compared": int(len(common)), "features": len(feats),
            "mismatches": int(mismatched),
            "verdict": ("training and serving agree" if mismatched == 0
                        else "SKEW — the two paths disagree"),
            "worst_features": worst[worst > 0].to_dict()}
