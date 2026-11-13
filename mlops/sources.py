"""Where the feature pipelines read from: Silver and Gold, never the OLTP."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from deltalake import DeltaTable


def load_sources(lake_root: str | Path, snapshots) -> dict:
    """
    Silver is the trust boundary, so features are built from it rather than
    from the source database — a feature must never be computed from a row
    that failed the quality gate.
    """
    L = Path(lake_root)
    out = {"snapshots": snapshots}
    for name in ("ms_users", "ms_user_demographics", "ms_subscription_plans",
                 "txn_user_subscriptions", "txn_workout_summaries",
                 "txn_notifications", "txn_event_participants",
                 "txn_group_comments", "txn_payments", "txn_events",
                 "ms_device_connections"):
        p = L / "silver" / name
        out[name] = (DeltaTable(str(p)).to_pandas()
                     if (p / "_delta_log").exists() else pd.DataFrame())
    return out
