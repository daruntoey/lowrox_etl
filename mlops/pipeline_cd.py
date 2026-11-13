"""
CD entrypoint, called by .github/workflows/cd.yml.

One stage per invocation so each shows as its own step in the workflow log
and can be re-run on its own.
"""
from __future__ import annotations
import argparse, sys, warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
import pandas as pd

from . import (config, dashboard, feature_store, features, monitoring,
               orchestrator, registry, serving, training)
from .sources import load_sources


def _context(run_id: str):
    config.wire()
    snaps = features.snapshot_dates(config.SNAPSHOT_START, config.SNAPSHOT_END)
    src = load_sources(config.LAKE_ROOT, snaps)
    return snaps, src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["deploy-pipeline", "train", "promote", "monitor"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or orchestrator.new_run_id("cd")
    config.wire()

    if args.stage == "deploy-pipeline":
        from . import __version__
        orchestrator.stage_cd_pipeline(run_id, package_version=__version__)
        return 0

    snaps, src = _context(run_id)

    if args.stage == "train":
        for fs in features.ALL_FEATURE_SETS:
            feature_store.materialise(fs, src, run_id=run_id)
        labels = features.build_labels(src)
        r = orchestrator.stage_continuous_training(
            run_id, labels=labels, feature_sets=features.ALL_FEATURE_SETS,
            algorithm="gradient_boosting", train_fn=training.train_pipeline)
        return 0 if r["model_version"] else 1

    if args.stage == "promote":
        vs = registry.versions(config.MODEL_NAME)
        staged = [v for v in vs if v.stage == "staging"]
        if not staged:
            print("nothing in staging to promote")
            return 0
        res = orchestrator.stage_cd_model(
            run_id, model_name=config.MODEL_NAME, version=staged[-1].version,
            registry_mod=registry)
        if args.force and not res["promoted"]:
            registry.promote(config.MODEL_NAME, staged[-1].version, force=True)
        return 0

    if args.stage == "monitor":
        labels = features.build_labels(src)
        users = labels.user_id.drop_duplicates().tolist()
        preds = serving.batch_score(config.MODEL_NAME, features.ALL_FEATURE_SETS,
                                    users, run_id=run_id)
        mv = registry.production(config.MODEL_NAME)
        stats = registry.training_stats(mv.name, mv.version)
        live = feature_store.get_training_data(
            labels[labels.ts == labels.ts.max()], features.ALL_FEATURE_SETS)
        rep = orchestrator.stage_monitoring(
            run_id, monitor_fn=monitoring.monitor,
            training_stats=stats, training_sample=stats,
            training_positive_rate=float(labels.is_churn.mean()),
            training_auc=mv.metrics["roc_auc"],
            live_features=live, live_labels=live["is_churn"],
            live_auc=None, feature_names=mv.features)
        dashboard.build(predictions=preds, model_version=mv, monitoring=rep,
                        feature_catalogue=feature_store.REGISTRY.catalogue(),
                        labels=labels, registry_table=registry.table(mv.name),
                        lineage=orchestrator.lineage(run_id),
                        out_path=str(config.DASHBOARD))
        print(f"dashboard written to {config.DASHBOARD}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
