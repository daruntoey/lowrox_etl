"""
CI entrypoint, called by .github/workflows/ci.yml.

Materialises the feature store, trains once, and runs the four validation
suites from ci.py. Exits non-zero on any failure so the build goes red.
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd

from . import (ci, config, feature_store, features, monitoring, registry,
               serving, training)
from .sources import load_sources


REQUIRED_SILVER = ("ms_users", "txn_workout_summaries", "txn_user_subscriptions")


def _require_lake() -> None:
    """
    Say plainly that the lake is not there, before pandas says it with a
    KeyError thirty frames deep.

    A CI runner has no one watching it, so the first thing it reports has to
    be the actual cause. Without this, a lake at the wrong path surfaces as
    `KeyError: None of ['id', 'tier', 'created_at'] are in the [columns]`
    from inside a feature definition, which reads like a schema bug in code
    that is fine.
    """
    missing = [t for t in REQUIRED_SILVER
               if not (config.LAKE_ROOT / "silver" / t / "_delta_log").exists()]
    if not missing:
        return
    raise SystemExit(
        f"\nNo Silver data at {config.LAKE_ROOT}/silver\n"
        f"  missing: {', '.join(missing)}\n\n"
        f"The feature store is built from Silver, so there is nothing to\n"
        f"materialise. Either point LOWROX_LAKE_ROOT at a lake that has been\n"
        f"loaded, or populate one first:\n\n"
        f"  python3 -c \"from mlops import demo_data; "
        f"demo_data.ensure_training_data('{config.LAKE_ROOT}')\"\n")


def main() -> int:
    config.wire()
    _require_lake()
    snaps = features.snapshot_dates(config.SNAPSHOT_START, config.SNAPSHOT_END)
    src = load_sources(config.LAKE_ROOT, snaps)
    for fs in features.ALL_FEATURE_SETS:
        feature_store.materialise(fs, src, run_id="ci")
    labels = features.build_labels(src)

    result = training.train_pipeline(labels, features.ALL_FEATURE_SETS,
                                     run_id="ci", verbose=False,
                                     register_model=False)
    modules = {"feature_store": feature_store, "features": features,
               "training": training, "registry": registry,
               "serving": serving, "monitoring": monitoring}
    report = ci.run_ci(modules, result, features.ALL_FEATURE_SETS, verbose=True)

    out = os.environ.get("RUNNER_TEMP", "/tmp")
    report["results"].to_csv(f"{out}/ci_report.csv", index=False)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
