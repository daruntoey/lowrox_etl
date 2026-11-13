"""
demo_data.py — enough data to train on, without needing a database.

The ETL notebook reads the real Supabase project, which currently holds 50
members. That is the right size for proving a pipeline and far too small to
train a churn model: nine monthly snapshots over 50 members is a few hundred
label rows at a 9% positive rate, so the test split would contain a handful of
churn events and every metric would be noise.

This module runs the same calibrated generator at a larger scale and writes
the result straight into the Silver zone, skipping PostgreSQL entirely — which
also means the MLOps notebook runs on Colab with nothing installed but Python.

The data is synthetic either way. What changes is only whether there is enough
of it for the model's numbers to mean anything.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
from deltalake import write_deltalake

SILVER_TABLES = [
    "ms_users", "ms_user_demographics", "ms_area", "ms_subscription_plans",
    "ms_device_connections", "txn_events", "txn_workout_summaries",
    "txn_user_history_level", "txn_user_goals", "txn_user_constraints",
    "txn_event_participants", "txn_group_comments", "txn_training_plans",
    "txn_user_subscriptions", "txn_payments", "txn_notifications",
]

TIMESTAMP_COLS = {"created_at", "updated_at", "started_at", "sent_at",
                  "opened_at", "clicked_at", "delivered_at", "dismissed_at",
                  "joined_at", "paid_at", "cancelled_at", "current_period_end",
                  "scheduled_at", "last_synced_at", "achieved_at", "target_date"}


def generate(scale: int = 40, project_dir: str | Path = ".",
             verbose: bool = True) -> Path:
    """Run the calibrated generator and return the directory of CSVs."""
    project_dir = Path(project_dir)
    gen = project_dir / "generate.py"
    if not gen.exists():
        raise FileNotFoundError(f"generate.py not found at {gen}")
    # The generator writes into out/csv and out/source_mock and does not
    # create them. They are gitignored, so a fresh clone has neither.
    for d in ("out/csv", "out/source_mock"):
        (project_dir / d).mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(gen), "--scale", str(scale)],
                       cwd=project_dir, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"generator failed:\n{r.stderr[-2000:]}")
    out = project_dir / "out" / "csv"
    if verbose:
        print(f"generated {scale * 50} rows per table into {out}")
    return out


def load_to_silver(csv_dir: str | Path, lake_root: str | Path,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Write the CSVs into the Silver zone in the shape the feature pipelines
    expect — including pace_sec_per_km, which Silver derives and the feature
    code reads.
    """
    csv_dir, lake_root = Path(csv_dir), Path(lake_root)
    rows = []
    for name in SILVER_TABLES:
        f = csv_dir / f"{name}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for c in df.columns:
            if c in TIMESTAMP_COLS:
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        if name == "txn_workout_summaries" and "pace_sec_per_km" not in df.columns:
            df["pace_sec_per_km"] = (
                df["duration_sec"] / df["distance_km"].replace(0, pd.NA)
            ).astype("Float64").round(1)
        path = lake_root / "silver" / name
        write_deltalake(path, pa.Table.from_pandas(df, preserve_index=False),
                        mode="overwrite", schema_mode="overwrite")
        rows.append({"table": name, "rows": len(df)})
    out = pd.DataFrame(rows)
    if verbose:
        print(f"wrote {out.rows.sum():,} rows across {len(out)} Silver tables "
              f"to {lake_root}/silver")
    return out


def ensure_training_data(lake_root: str | Path, project_dir: str | Path = ".",
                         min_users: int = 1000, scale: int = 40,
                         verbose: bool = True) -> dict:
    """
    Use whatever is already in Silver if there is enough of it; otherwise
    generate. Returns what it decided and why, because "where did this data
    come from" should never be a guess.
    """
    from deltalake import DeltaTable
    users_path = Path(lake_root) / "silver" / "ms_users"
    existing = 0
    if (users_path / "_delta_log").exists():
        existing = len(DeltaTable(str(users_path)).to_pandas())

    if existing >= min_users:
        if verbose:
            print(f"Silver already holds {existing:,} members — using it as is.")
        return {"source": "existing silver", "users": existing, "generated": False}

    if verbose:
        print(f"Silver holds {existing:,} members, which is below the {min_users:,} "
              f"needed for\nthe model's metrics to mean anything. Generating a larger "
              f"synthetic set.\nThe pipeline is identical either way; only the sample "
              f"size changes.\n")
    csvs = generate(scale=scale, project_dir=project_dir, verbose=verbose)
    summary = load_to_silver(csvs, lake_root, verbose=verbose)
    return {"source": f"generated at scale {scale}", "generated": True,
            "users": int(summary.loc[summary.table == "ms_users", "rows"].iloc[0]),
            "summary": summary}
