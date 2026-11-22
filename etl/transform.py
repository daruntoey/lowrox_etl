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
import pyarrow.compute as pc
from deltalake import DeltaTable, write_deltalake

from .config import SILVER, RunConfig, SourceTable
from .errors import DeadLetterQueue, ErrorLog
from .extract import read_source_keys
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

# The web app (aom-thanya/lowrox, utils/leveling.ts isPlausibleEffort) accepts a
# logged run of 0.5-100 km at no faster than 2:30 min/km. The quality gate must
# not reject what the product itself accepts, so it uses the same bounds;
# a ride legitimately goes further.
MAX_KM = {"cycling": 300.0}
DEFAULT_MAX_KM = 100.0
MIN_RUN_PACE_SEC_PER_KM = 150.0
# The one bound stricter than the app: slower than 30 min/km is a stopwatch left
# running, not a run. The app saves such a row; the gate quarantines it (DLQ),
# so it is kept and explained rather than lost.
MAX_RUN_PACE_SEC_PER_KM = 1800.0


def _distance_plausible(df):
    is_distance = df["activity_type"].isin(DISTANCE_ACTIVITIES)
    max_km = df["activity_type"].map(MAX_KM).fillna(DEFAULT_MAX_KM)
    ok_distance = (df["distance_km"] >= 0.5) & (df["distance_km"] <= max_km)
    ok_other = df["distance_km"].notna() & (df["distance_km"] >= 0) & (df["distance_km"] <= max_km)
    return (is_distance & ok_distance) | (~is_distance & ok_other)


def _meeting_point_valid(df):
    """Both coordinates or neither, inside WGS84 range (sql/08_party_meeting_point.sql).
    A source from before 08 has no such columns: nothing to check."""
    if "meeting_lat" not in df.columns or "meeting_lng" not in df.columns:
        return pd.Series(True, index=df.index)
    lat = pd.to_numeric(df["meeting_lat"], errors="coerce")
    lng = pd.to_numeric(df["meeting_lng"], errors="coerce")
    neither = lat.isna() & lng.isna()
    both = lat.between(-90, 90) & lng.between(-180, 180)
    return neither | both


def _run_pace_plausible(df):
    """2:30-30:00 min/km. Faster is beyond world-record pace (the app refuses it
    too); slower is a forgotten stopwatch."""
    is_run = df["activity_type"] == "running"
    pace = df["duration_sec"] / df["distance_km"].where(df["distance_km"] > 0)
    return ~is_run | pace.isna() | pace.between(MIN_RUN_PACE_SEC_PER_KM, MAX_RUN_PACE_SEC_PER_KM)


DQ_RULES: dict[str, list[tuple]] = {
    "txn_workout_summaries": [
        ("user_id_present",   _not_null("user_id")),
        ("distance_plausible", _distance_plausible),
        ("run_pace_plausible", _run_pace_plausible),
        # a day; a 100 km run at the app's slowest accepted pace still fits
        ("duration_plausible", _between("duration_sec", 60, 86_400)),
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
    # Values the web app writes (supabase 04_frontend_integration.sql CHECKs).
    "txn_events": [
        ("owner_present", _not_null("event_owner")),
        ("status_known", lambda df: df["status"].str.lower().isin(
            ["draft", "scheduled", "full", "completed", "cancelled"])),
        ("visibility_known", lambda df: df["visibility"].isna()
            | df["visibility"].str.lower().isin(["public", "private"])),
        ("meeting_point_valid", _meeting_point_valid),
    ],
    "txn_plan_checkoffs": [
        ("user_id_present", _not_null("user_id")),
        ("session_present", _not_null("session_id")),
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
                                                "payment_method", "source",
                                                "target_level", "visibility",
                                                "party_style"):
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
def silver_path_for(tbl: SourceTable) -> str:
    return str(SILVER / "restricted" / tbl.name) if tbl.sensitive else tbl.silver_path


def drop_deleted(tbl: SourceTable, cfg: RunConfig, run_id: str, log: ErrorLog,
                 lineage: LineageRegistry) -> int:
    """
    Remove Silver rows whose key the source no longer has.

    The web app hard-deletes in these tables (config: hard_deletes), and an
    incremental extract never sees a row that is gone. The extract snapshots
    the source keys each run; anything in Silver outside that snapshot was
    deleted in the source. Bronze keeps every version it ever read, so the
    deletion is still auditable there.
    """
    if not tbl.hard_deletes:
        return 0
    keys = read_source_keys(tbl.name, run_id)
    path = silver_path_for(tbl)
    if keys is None or not Path(path, "_delta_log").exists():
        return 0
    # Arrow end to end, so the rewrite keeps every column type exactly.
    current = DeltaTable(path).to_pyarrow_table()
    present = pc.is_in(pc.cast(current["_src_pk"], pa.string()),
                       value_set=pa.array(sorted(keys), type=pa.string()))
    n = len(current) - pc.sum(present).as_py() if len(current) else 0
    if not n:
        return 0
    write_deltalake(path, current.filter(present), mode="overwrite")
    version = DeltaTable(path).version()
    lineage.record(
        load_type=cfg.load_type,
        source_object=f"oltp.{tbl.name}", source_version="key_snapshot",
        target_object=f"silver.{tbl.name}", target_version=str(version),
        operation="delete", rows_in=len(current), rows_out=len(current) - n, rows_rejected=0,
    )
    log.info(f"{tbl.name}: {n} row(s) deleted in the source -> removed from silver v{version}")
    return n


def transform_table(tbl: SourceTable, cfg: RunConfig, run_id: str,
                    log: ErrorLog, metrics: MetricsCollector,
                    lineage: LineageRegistry, dlq: DeadLetterQueue) -> dict:
    result = _transform_rows(tbl, cfg, run_id, log, metrics, lineage, dlq)
    result["deleted"] = drop_deleted(tbl, cfg, run_id, log, lineage)
    return result


def _transform_rows(tbl: SourceTable, cfg: RunConfig, run_id: str,
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

        silver_path = silver_path_for(tbl)

        arrow = pa.Table.from_pandas(good, preserve_index=False)
        if Path(silver_path, "_delta_log").exists():
            st = DeltaTable(silver_path)
            pred = " AND ".join(f"t.{c} = s.{c}" for c in tbl.pk)
            # merge_schema: a column added in Supabase (e.g. 06's line_user_id)
            # must reach Silver, not be silently dropped by the merge.
            (st.merge(source=arrow, predicate=pred,
                      source_alias="s", target_alias="t", merge_schema=True)
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
