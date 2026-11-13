"""
orchestrator.py — MLOps levels 0, 1 and 2 as runnable stages.

Slide 7 (level 0) is a manual, script-driven process with no CI, no CD and no
active monitoring. Slide 9 (level 1) automates the ML pipeline itself and adds
a feature store, an ML metadata store and a trigger. Slide 11 (level 2) wraps
the whole thing in CI/CD so that a change to the pipeline's SOURCE is built,
tested and deployed automatically.

Slide 12 names the six level-2 stages, and they are the functions below:

  1 development and experiment     stage_experiment
  2 pipeline continuous integration stage_ci          (ci.py)
  3 pipeline continuous delivery    stage_cd_pipeline
  4 automated triggering            stage_continuous_training
  5 model continuous delivery       stage_cd_model
  6 monitoring                      stage_monitoring

The ML metadata store under the pipeline in both diagrams is `run_log.jsonl`:
every stage writes what it did, with the run id, so an answer to "why is this
model in production" exists after the fact.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

METADATA_ROOT = Path("/content/ml_metadata")


def set_metadata_root(path: str | Path) -> Path:
    global METADATA_ROOT
    METADATA_ROOT = Path(path)
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    return METADATA_ROOT


def new_run_id(prefix: str = "mlrun") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:4]}"


# --------------------------------------------------------------------------
# ML metadata store
# --------------------------------------------------------------------------
def log_step(run_id: str, stage: str, status: str, **payload) -> dict:
    rec = {"run_id": run_id, "stage": stage, "status": status,
           "at": datetime.now(timezone.utc).isoformat(), **payload}
    path = METADATA_ROOT / "run_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec


def read_metadata(run_id: str | None = None) -> pd.DataFrame:
    path = METADATA_ROOT / "run_log.jsonl"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_json(path, lines=True)
    return df[df.run_id == run_id] if run_id else df


def lineage(run_id: str) -> pd.DataFrame:
    """Which stage produced what, in order — the answer to 'why this model'."""
    df = read_metadata(run_id)
    if df.empty:
        return df
    cols = [c for c in ("stage", "status", "at", "detail", "artifact") if c in df.columns]
    return df[cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Triggers — the "Trigger" box in the level 1 and 2 diagrams
# --------------------------------------------------------------------------
@dataclass
class Trigger:
    kind: str          # 'schedule' | 'new_data' | 'drift' | 'performance' | 'manual'
    fired: bool
    reason: str


def evaluate_triggers(*, last_trained: datetime | None, now: datetime,
                      new_rows_since: int, monitoring: dict | None,
                      schedule_days: int = 30,
                      min_new_rows: int = 500) -> list[Trigger]:
    """
    Slide 12 stage 4: the pipeline is executed in production on a schedule OR
    in response to a trigger. Four triggers, in the order they tend to matter.
    """
    out = []
    age = (now - last_trained).days if last_trained else 10_000
    out.append(Trigger("schedule", age >= schedule_days,
                       f"{age}d since the last training run (threshold {schedule_days}d)"))
    out.append(Trigger("new_data", new_rows_since >= min_new_rows,
                       f"{new_rows_since} new labelled rows (threshold {min_new_rows})"))
    if monitoring:
        drift = [f for f in monitoring["findings"]
                 if f.severity == "alert" and "drift" in f.check or
                 f.severity == "alert" and "shift" in f.check]
        perf = [f for f in monitoring["findings"]
                if f.severity == "alert" and "concept" in f.check]
        out.append(Trigger("drift", bool(drift),
                           "; ".join(f.check for f in drift) or "no drift alert"))
        out.append(Trigger("performance", bool(perf),
                           "; ".join(f.detail for f in perf) or "performance holding"))
    return out


# --------------------------------------------------------------------------
# The six stages of slide 12
# --------------------------------------------------------------------------
def stage_experiment(run_id: str, *, labels, feature_sets, algorithms: list[str],
                     train_fn, verbose: bool = True) -> dict:
    """
    Stage 1. Try several algorithms, keep the best by validation ROC-AUC.
    In the diagram this is the "Orchestrated experiment" box, and its output
    is source code — here, the chosen configuration.
    """
    results = {}
    for algo in algorithms:
        r = train_fn(labels, feature_sets, algorithm=algo, run_id=run_id,
                     verbose=False, register_model=False)
        results[algo] = r
    table = pd.DataFrame([{
        "algorithm": a,
        "val_roc_auc": round(r["metrics_val"]["roc_auc"], 4),
        "test_roc_auc": round(r["metrics_test"]["roc_auc"], 4),
        "test_pr_auc": round(r["metrics_test"]["pr_auc"], 4),
        "brier": round(r["metrics_test"]["brier"], 4),
        "gates_passed": bool(r["validation"]["passed"]),
    } for a, r in results.items()]).sort_values("val_roc_auc", ascending=False)

    # chosen on VALIDATION, never on test — the test set is spent once
    best = table.iloc[0]["algorithm"]
    log_step(run_id, "1_experiment", "ok", detail=f"best={best}",
             candidates=len(algorithms))
    if verbose:
        print("STAGE 1 — DEVELOPMENT AND EXPERIMENT")
        print(table.to_string(index=False))
        print(f"\nchosen on validation ROC-AUC: {best}")
    return {"best_algorithm": best, "table": table, "results": results}


def stage_cd_pipeline(run_id: str, *, package_version: str,
                      verbose: bool = True) -> dict:
    """
    Stage 3. Deploy the pipeline artefact built by CI to the target
    environment. The thing being deployed is the PIPELINE, not a model —
    that distinction is what separates level 2 from level 1.
    """
    manifest = {
        "pipeline_version": package_version,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "components": ["feature_store", "features", "training",
                       "registry", "serving", "monitoring"],
    }
    (METADATA_ROOT / "deployed_pipeline.json").write_text(
        json.dumps(manifest, indent=2))
    log_step(run_id, "3_cd_pipeline", "ok", artifact=package_version)
    if verbose:
        print("STAGE 3 — PIPELINE CONTINUOUS DELIVERY")
        print(f"  deployed pipeline {package_version} with "
              f"{len(manifest['components'])} components")
    return manifest


def stage_continuous_training(run_id: str, *, labels, feature_sets, algorithm,
                              train_fn, verbose: bool = True) -> dict:
    """
    Stage 4. The automated pipeline runs in production and pushes a trained
    model to the registry. This is the CT in "CI/CD/CT".
    """
    r = train_fn(labels, feature_sets, algorithm=algorithm, run_id=run_id,
                 verbose=verbose, register_model=True)
    mv = r["model_version"]
    log_step(run_id, "4_continuous_training",
             "ok" if mv else "rejected",
             artifact=mv.version if mv else None,
             detail=f"roc_auc={r['metrics_test']['roc_auc']:.4f}")
    return r


def stage_cd_model(run_id: str, *, model_name: str, version: str,
                   registry_mod, min_improvement: float = 0.0,
                   verbose: bool = True) -> dict:
    """
    Stage 5. Serve the trained model as a prediction service — but only after
    it beats the incumbent. Automatic retraining without this comparison is
    automatic degradation waiting for a bad week of data.
    """
    res = registry_mod.promote(model_name, version, min_improvement=min_improvement)
    log_step(run_id, "5_cd_model", "ok" if res["promoted"] else "held",
             artifact=version, detail=res["reason"])
    if verbose:
        print("STAGE 5 — MODEL CONTINUOUS DELIVERY")
        print(f"  {'PROMOTED' if res['promoted'] else 'HELD IN STAGING'}: {res['reason']}")
    return res


def stage_monitoring(run_id: str, *, monitor_fn, verbose: bool = True, **kw) -> dict:
    """
    Stage 6. Collect statistics on live data; the output is a trigger back to
    stage 4, which is the loop that closes both diagrams.
    """
    rep = monitor_fn(verbose=verbose, **kw)
    log_step(run_id, "6_monitoring",
             "alert" if rep["retrain_triggered"] else "ok",
             detail=rep["trigger_reason"] or "no alerts")
    return rep
