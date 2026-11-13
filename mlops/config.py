"""Where the MLOps artefacts live. One place, read from the environment."""
from __future__ import annotations
import os
from pathlib import Path

LAKE_ROOT = Path(os.environ.get("LOWROX_LAKE_ROOT", "/content/lowrox_lake"))
MLOPS_ROOT = Path(os.environ.get("LOWROX_MLOPS_ROOT", str(LAKE_ROOT.parent / "mlops_store")))

FEATURE_STORE = MLOPS_ROOT / "feature_store"
MODEL_REGISTRY = MLOPS_ROOT / "model_registry"
ML_METADATA = MLOPS_ROOT / "ml_metadata"
PREDICTIONS = MLOPS_ROOT / "predictions"
DASHBOARD = Path(os.environ.get("LOWROX_DASHBOARD", str(MLOPS_ROOT / "lowrox_dashboard.html")))

MODEL_NAME = "lowrox_churn"
SNAPSHOT_START = os.environ.get("LOWROX_SNAPSHOT_START", "2026-02-01")
SNAPSHOT_END = os.environ.get("LOWROX_SNAPSHOT_END", "2026-10-31")


def wire():
    """Point every module at these paths. Called once per run."""
    from . import feature_store, registry, orchestrator, serving
    feature_store.set_root(FEATURE_STORE)
    registry.set_root(MODEL_REGISTRY)
    orchestrator.set_metadata_root(ML_METADATA)
    serving.set_predictions_path(str(PREDICTIONS))
    return {"feature_store": FEATURE_STORE, "model_registry": MODEL_REGISTRY,
            "ml_metadata": ML_METADATA, "predictions": PREDICTIONS,
            "dashboard": DASHBOARD}
