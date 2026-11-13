"""
registry.py — the model registry from the Chapter 7 diagrams.

Every level-0/1/2 diagram in the chapter has a "Model registry" between the
pipeline and the serving layer. What it has to hold, to be worth having:

  * the artefact itself, versioned
  * the metrics it was accepted on
  * WHICH features and which feature-set versions produced it
  * the training-time feature distribution, so drift can be measured against
    the state of the world the model actually learned (slide 14)
  * a promotion state, so "the model in production" is a fact rather than a
    recollection

Promotion is deliberately a separate act from training. A pipeline that
trains and deploys in one motion has no place to put the question "is this
new model actually better than the one already serving?"
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REGISTRY_ROOT = Path("/content/model_registry")


def set_root(path: str | Path) -> Path:
    global REGISTRY_ROOT
    REGISTRY_ROOT = Path(path)
    REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
    return REGISTRY_ROOT


@dataclass
class ModelVersion:
    name: str
    version: str
    created_at: str
    stage: str                       # 'staging' | 'production' | 'archived'
    metrics: dict
    baseline_metrics: dict
    features: list
    feature_sets: list
    training_rows: int
    training_window: list            # [first_ts, last_ts]
    algorithm: str
    hyperparameters: dict
    run_id: str = ""
    promoted_at: str = ""
    notes: str = ""

    @property
    def dir(self) -> Path:
        return REGISTRY_ROOT / self.name / self.version


def _meta_path(name: str, version: str) -> Path:
    return REGISTRY_ROOT / name / version / "meta.json"


def register(model, *, name: str, metrics: dict, baseline_metrics: dict,
             features: list, feature_sets: list, training_rows: int,
             training_window: list, algorithm: str, hyperparameters: dict,
             feature_stats: pd.DataFrame, run_id: str = "",
             notes: str = "") -> ModelVersion:
    """Write a new version. Always lands in 'staging' — never straight to production."""
    version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%S")
    mv = ModelVersion(
        name=name, version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        stage="staging", metrics=metrics, baseline_metrics=baseline_metrics,
        features=list(features), feature_sets=list(feature_sets),
        training_rows=training_rows, training_window=[str(t) for t in training_window],
        algorithm=algorithm, hyperparameters=hyperparameters,
        run_id=run_id, notes=notes)

    mv.dir.mkdir(parents=True, exist_ok=True)
    with (mv.dir / "model.pkl").open("wb") as fh:
        pickle.dump(model, fh)
    # The distribution the model learned from. Drift is measured against
    # THIS, not against last week's data.
    feature_stats.to_json(mv.dir / "training_feature_stats.json", orient="index")
    (mv.dir / "meta.json").write_text(json.dumps(asdict(mv), indent=2, default=str))
    return mv


def load(name: str, version: str):
    with (REGISTRY_ROOT / name / version / "model.pkl").open("rb") as fh:
        return pickle.load(fh)


def load_meta(name: str, version: str) -> ModelVersion:
    return ModelVersion(**json.loads(_meta_path(name, version).read_text()))


def training_stats(name: str, version: str) -> pd.DataFrame:
    p = REGISTRY_ROOT / name / version / "training_feature_stats.json"
    return pd.read_json(p, orient="index") if p.exists() else pd.DataFrame()


def versions(name: str) -> list[ModelVersion]:
    root = REGISTRY_ROOT / name
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if (d / "meta.json").exists():
            out.append(ModelVersion(**json.loads((d / "meta.json").read_text())))
    return out


def production(name: str) -> ModelVersion | None:
    for mv in versions(name):
        if mv.stage == "production":
            return mv
    return None


def promote(name: str, version: str, *, metric: str = "roc_auc",
            min_improvement: float = 0.0, force: bool = False) -> dict:
    """
    Move a staged version to production — but only if it beats what is
    already serving. This gate is the difference between continuous training
    and continuous degradation: an automatically retrained model is not
    automatically a better model, and without the comparison a bad week of
    data quietly replaces a good model.
    """
    candidate = load_meta(name, version)
    current = production(name)

    if current is None:
        decision, reason = True, "no model in production yet"
    else:
        new, old = candidate.metrics.get(metric, 0), current.metrics.get(metric, 0)
        decision = new >= old + min_improvement
        reason = (f"{metric} {new:.4f} vs production {old:.4f} "
                  f"(needs +{min_improvement:.4f})")
    if force and not decision:
        decision, reason = True, reason + " — FORCED"

    if decision:
        if current is not None:
            current.stage = "archived"
            _meta_path(name, current.version).write_text(
                json.dumps(asdict(current), indent=2, default=str))
        candidate.stage = "production"
        candidate.promoted_at = datetime.now(timezone.utc).isoformat()
        _meta_path(name, version).write_text(
            json.dumps(asdict(candidate), indent=2, default=str))

    return {"promoted": decision, "version": version,
            "previous": current.version if current else None, "reason": reason}


def table(name: str) -> pd.DataFrame:
    rows = []
    for mv in versions(name):
        rows.append({
            "version": mv.version, "stage": mv.stage, "algorithm": mv.algorithm,
            "roc_auc": round(mv.metrics.get("roc_auc", float("nan")), 4),
            "pr_auc": round(mv.metrics.get("pr_auc", float("nan")), 4),
            "recall@top20%": round(mv.metrics.get("recall_at_top_20pct", float("nan")), 4),
            "rows": mv.training_rows, "created": mv.created_at[:19],
        })
    return pd.DataFrame(rows)
