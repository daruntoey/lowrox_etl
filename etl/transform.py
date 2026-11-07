"""
transform.py — single responsibility: turn raw Bronze into trustworthy Silver.

Silver is the trust boundary. A row reaches it only if it is
  deduplicated   one row per business key, the newest version wins
  conformed      types, timezones and codes are normalised
  valid          it passed every data-quality rule for its table

A row that fails goes to the dead-letter queue with the rule that rejected it
and its full lineage. It is never silently dropped and never silently fixed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from .config import SILVER, RunConfig, SourceTable
from .errors import DeadLetterQueue, ErrorLog
from .faults import FaultInjector
from .lineage import LineageRegistry
from .metrics import MetricsCollector


# --------------------------------------------------------------------------
# Data-quality rules
# --------------------------------------------------------------------------
# Each rule is (name, predicate) where predicate returns True for GOOD rows.
# Rules are declarative and table-scoped so adding one never means editing the
# transform engine.
def _between(col, lo, hi):
    return lambda df: df[col].isna() | df[col].between(lo, hi)


def _not_null(col):
    return lambda df: df[col].notna()


def _positive(col):
    return lambda df: df[col].isna() | (df[col] > 0)


# Activities where distance is the point. A `functional` (gym / strength)
# session legitimately covers zero distance, so applying a distance floor to
# it rejects perfectly good rows — a rule that is wrong in the strict
# direction is still a wrong rule.
DISTANCE_ACTIVITIES = {"running", "cycling", "swimming", "walking", "hyrox_prep"}


def _distance_plausible(df):
    is_distance = df["activity_type"].isin(DISTANCE_ACTIVITIES)
    ok_distance = df["distance_km"].between(0.5, 40)
    ok_other = df["distance_km"].notna() & (df["distance_km"] >= 0) & (df["distance_km"] <= 40)
    return (is_distance & ok_distance) | (~is_distance & ok_other)


DQ_RULES: dict[str, list[tuple]] = {
    "txn_workout_summaries": [
        ("user_id_present",   _not_null("user_id")),
        ("distance_plausible", _distance_plausible),
        ("duration_plausible", _between("duration_sec", 60, 20_000)),
        ("heart_rate_plausible", _between("avg_heart_rate", 30, 220)),
    ],
    "txn_payments": [
        ("user_id_present", _not_null("user_id")),
        ("amount_positive", _positive("amount")),
        ("status_known",    lambda df: df["status"].isin(
            ["succeeded", "failed", "refunded", "pending"])),
    ],
    "ms_users": [
        ("username_present", _not_null("username")),
        ("tier_known", lambda df: df["tier"].isin(["free", "pro", "elite"])),
    ],
    "txn_user_subscriptions": [
        ("user_id_present", _not_null("user_id")),
        ("plan_id_present", _not_null("plan_id")),
    ],
    "txn_event_participants": [
        ("user_id_present",  _not_null("user_id")),
        ("event_id_present", _not_null("event_id")),
    ],
    "txn_notifications": [
        ("user_id_present", _not_null("user_id")),
    ],
}


def apply_dq(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (good_rows, rejected_rows_with_reason)."""
    rules = DQ_RULES.get(table, [])
    if not rules or df.empty:
        return df, df.iloc[0:0].assign(_dq_rule=pd.Series(dtype="object"))

    verdicts, reasons = pd.Series(True, index=df.index), pd.Series("", index=df.index)
    for name, pred in rules:
        try:
            ok = pred(df).fillna(False)
        except KeyError:
            # The column is missing — schema drift. Treat as a failed rule
            # rather than crashing, so the run can continue and alert.
            ok = pd.Series(False, index=df.index)
            name = f"{name}:column_missing"
        failed = ~ok & verdicts
        reasons.loc[failed] = name
        verdicts &= ok

    good = df[verdicts].copy()
    bad = df[~verdicts].copy()
    bad["_dq_rule"] = reasons[~verdicts]
    return good, bad


# --------------------------------------------------------------------------
# Conformance
# --------------------------------------------------------------------------
def conform(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Normalise codes and derive the columns Gold expects."""
    out = df.copy()
    for col in out.columns:
        if col.startswith("_"):
            continue
        if out[col].dtype == object and col in ("tier", "status", "activity_type",
                                                "workout_type", "gender", "provider",
                                                "channel", "send_status", "role",
                                                "payment_method", "source"):
            out[col] = out[col].astype("string").str.strip().str.lower()

    if table == "txn_workout_summaries":
        # pace is derived once, here, so every consumer computes it the same way
        out["pace_sec_per_km"] = (
            out["duration_sec"] / out["distance_km"].replace(0, pd.NA)
        ).astype("Float64").round(1)

    out["_silver_at"] = datetime.now(timezone.utc)
    return out


# --------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------
def deduplicate(df: pd.DataFrame, tbl: SourceTable) -> tuple[pd.DataFrame, int]:
    """
    Bronze is append-only, so the same business key can appear many times:
    an update in the source, or a replayed batch. Silver keeps exactly one —
    the most recently extracted. Returns (deduped, rows_removed).
    """
    if df.empty:
        return df, 0
    before = len(df)
    out = (df.sort_values(["_extracted_at", "_lineage_id"])
             .drop_duplicates(subset=list(tbl.pk), keep="last")
             .reset_index(drop=True))
    return out, before - len(out)


# --------------------------------------------------------------------------
# Bronze -> Silver for one table
# --------------------------------------------------------------------------
def transform_table(tbl: SourceTable, cfg: RunConfig, run_id: str,
                    log: ErrorLog, metrics: MetricsCollector,
                    lineage: LineageRegistry, dlq: DeadLetterQueue) -> dict:
    bronze_path = tbl.bronze_path
    if not Path(bronze_path, "_delta_log").exists():
        return {"table": tbl.name, "rows": 0, "rejected": 0}

    with metrics.step("silver", tbl.name) as m:
        m.load_type = cfg.load_type
        FaultInjector.check("transform.crash", scope=tbl.name)

        bt = DeltaTable(bronze_path)
        bronze_version = bt.version()

        # Incremental: read only the rows this run put into Bronze.
        df = bt.to_pandas()
        if cfg.load_type == "incremental":
            df = df[df["_run_id"] == run_id]
        m.rows_in = len(df)
        if df.empty:
            return {"table": tbl.name, "rows": 0, "rejected": 0}

        df, dupes = deduplicate(df, tbl)
        good, bad = apply_dq(df, tbl.name)
        good = conform(good, tbl.name)

        if len(bad):
            dlq.put(bad, stage=f"silver:{tbl.name}", reason="dq_rule_failed")
            m.rows_rejected = len(bad)

        if good.empty:
            log.warn(f"{tbl.name}: every row rejected by DQ")
            return {"table": tbl.name, "rows": 0, "rejected": len(bad)}

        good["_bronze_version"] = bronze_version
        good["_silver_run_id"] = run_id

        silver_path = (str(SILVER / "restricted" / tbl.name) if tbl.sensitive
                       else tbl.silver_path)

        arrow = pa.Table.from_pandas(good, preserve_index=False)
        if Path(silver_path, "_delta_log").exists():
            st = DeltaTable(silver_path)
            pred = " AND ".join(f"t.{c} = s.{c}" for c in tbl.pk)
            (st.merge(source=arrow, predicate=pred,
                      source_alias="s", target_alias="t")
               .when_matched_update_all()
               .when_not_matched_insert_all()
               .execute())
            version = DeltaTable(silver_path).version()
            op = "merge"
        else:
            write_deltalake(silver_path, arrow, mode="overwrite", schema_mode="overwrite")
            version = DeltaTable(silver_path).version()
            op = "overwrite"

        m.rows_out = len(good)
        m.bytes_out = arrow.nbytes

        lineage.record(
            load_type=cfg.load_type,
            source_object=f"bronze.{tbl.name}", source_version=str(bronze_version),
            target_object=f"silver.{tbl.name}", target_version=str(version),
            operation=op, rows_in=m.rows_in, rows_out=m.rows_out, rows_rejected=len(bad),
        )
        note = f" (deduped {dupes})" if dupes else ""
        log.info(f"{tbl.name}: {m.rows_out} rows -> silver v{version}{note}"
                 + (f", {len(bad)} quarantined" if len(bad) else ""))
        return {"table": tbl.name, "rows": m.rows_out, "rejected": len(bad)}


def transform_all(cfg: RunConfig, run_id: str, log: ErrorLog,
                  metrics: MetricsCollector, lineage: LineageRegistry,
                  dlq: DeadLetterQueue) -> pd.DataFrame:
    log.stage = "transform:silver"
    out = []
    for t in cfg.selected():
        try:
            out.append(transform_table(t, cfg, run_id, log, metrics, lineage, dlq))
        except Exception as exc:
            log.error(f"{t.name}: transform failed — {type(exc).__name__}: {exc}")
            dlq.put(pd.DataFrame([{"table": t.name, "error": str(exc)}]),
                    stage="silver", reason="transform_exception")
            if cfg.fail_fast:
                raise
    return pd.DataFrame(out or [{"table": None, "rows": 0, "rejected": 0}])
