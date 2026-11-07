"""
load.py — single responsibility: build the Gold star schema from Silver.

Rules enforced here, all of them from the warehouse design:
  * surrogate keys are stable across runs; a fact never changes its dimension
    key because a later run reloaded the dimension
  * every dimension is pre-seeded with the special rows
        0  = Not applicable      -1 = Unknown      -2 = Invalid
    so a fact foreign key is NEVER null
  * dim_user is SCD type 2; the other dimensions here are type 1
  * email never reaches Gold — only a one-way token derived from it
  * every fact row keeps _src_lineage_id, so it can be walked back to the
    exact Bronze row and from there to the source primary key
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from .config import (GOLD, SILVER, KEY_INVALID, KEY_NOT_APPLICABLE, KEY_UNKNOWN,
                     RunConfig)
from .errors import DeadLetterQueue, ErrorLog, TargetUnavailable, retry
from .faults import FaultInjector
from .lineage import LineageRegistry
from .metrics import MetricsCollector

SPECIAL = {
    KEY_NOT_APPLICABLE: "Not applicable",
    KEY_UNKNOWN: "Unknown",
    KEY_INVALID: "Invalid",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def read_silver(name: str) -> pd.DataFrame:
    p = SILVER / name
    if not Path(p, "_delta_log").exists():
        return pd.DataFrame()
    return DeltaTable(str(p)).to_pandas()


def gold_path(name: str) -> str:
    return str(GOLD / name)


def read_gold(name: str) -> pd.DataFrame:
    p = gold_path(name)
    if not Path(p, "_delta_log").exists():
        return pd.DataFrame()
    return DeltaTable(p).to_pandas()


def write_gold(name: str, df: pd.DataFrame, log: ErrorLog | None = None) -> int:
    """
    Write one Gold table. A target that is briefly unavailable is retried,
    because object storage and warehouse endpoints fail transiently far more
    often than they fail permanently; a target that stays down raises, and
    the orchestrator turns that into a critical alert.
    """
    def _attempt():
        FaultInjector.check("load.target_unavailable", scope=name)
        try:
            write_deltalake(gold_path(name),
                            pa.Table.from_pandas(df, preserve_index=False),
                            mode="overwrite", schema_mode="overwrite")
        except OSError as exc:
            raise TargetUnavailable(str(exc)) from exc
        return DeltaTable(gold_path(name)).version()

    version, attempts = retry(_attempt, attempts=3, base_delay=0.2,
                              logger=log, label=f"load {name}")
    if attempts > 1 and log:
        log.info(f"{name}: write succeeded on attempt {attempts}")
    return version


def email_token(value: str | None) -> str:
    """One-way token. Joins still work; the address does not travel."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return hashlib.sha256(str(value).strip().lower().encode()).hexdigest()[:32]


def _seed_rows(key_col: str, attrs: dict[str, str]) -> pd.DataFrame:
    """The three special dimension rows every dimension starts life with."""
    rows = []
    for k, label in SPECIAL.items():
        r = {key_col: k}
        r.update({c: label for c in attrs})
        rows.append(r)
    return pd.DataFrame(rows)


def assign_keys(new: pd.DataFrame, existing: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Give unseen rows the next surrogate keys, leaving existing keys alone."""
    start = int(existing[key_col].max()) + 1 if len(existing) else 1
    start = max(start, 1)
    out = new.copy()
    out[key_col] = range(start, start + len(out))
    return out


_FAR_FUTURE = pd.Timestamp.max.tz_localize("UTC")


def lookup_scd2(fact: pd.DataFrame, nk_col: str, dim: pd.DataFrame, dim_nk: str,
                dim_key: str, when: pd.Series, *, absent_key: int = KEY_UNKNOWN) -> pd.Series:
    """
    Point-in-time lookup against a type 2 dimension.

    A fact must carry the dimension version that was in force WHEN THE FACT
    HAPPENED, not whichever version is current at load time. Resolving
    against is_current instead would silently re-point every historical fact
    at the newest version each run, which destroys the exact history that
    SCD type 2 exists to preserve: a workout recorded while the member was on
    the free tier would start reporting as an elite workout the moment they
    upgraded.
    """
    if dim.empty:
        return pd.Series(KEY_UNKNOWN, index=fact.index, dtype="int64")

    d = dim[[dim_nk, dim_key, "valid_from", "valid_to"]].copy()
    d["valid_from"] = pd.to_datetime(d["valid_from"], errors="coerce", utc=True)
    # '9999-12-31' overflows a nanosecond timestamp, so the open end is capped
    d["valid_to"] = pd.to_datetime(d["valid_to"], errors="coerce", utc=True).fillna(_FAR_FUTURE)
    d["valid_from"] = d["valid_from"].fillna(pd.Timestamp.min.tz_localize("UTC"))

    probe = pd.DataFrame({
        "_row": range(len(fact)),
        "_nk": fact[nk_col].to_numpy(),
        "_at": pd.to_datetime(when, utc=True, errors="coerce").to_numpy(),
    })
    j = probe.merge(d, left_on="_nk", right_on=dim_nk, how="left")
    hit = j[(j["_at"] >= j["valid_from"]) & (j["_at"] < j["valid_to"])]
    hit = hit.drop_duplicates(subset=["_row"], keep="first").set_index("_row")[dim_key]

    keys = pd.Series(absent_key, index=range(len(fact)), dtype="int64")
    keys.loc[hit.index] = hit.astype("int64").to_numpy()
    nk_null = pd.Series(fact[nk_col].isna().to_numpy(), index=range(len(fact)))
    unmatched = ~pd.Series(range(len(fact))).isin(hit.index) & ~nk_null
    keys[unmatched.to_numpy()] = KEY_UNKNOWN
    keys[nk_null.to_numpy()] = absent_key
    keys.index = fact.index
    return keys


def lookup(fact: pd.DataFrame, nk_col: str, dim: pd.DataFrame,
           dim_nk: str, dim_key: str, *, absent_key: int = KEY_NOT_APPLICABLE) -> pd.Series:
    """
    Resolve a natural key to a surrogate key.
      value is null            -> absent_key  (0 Not applicable, by default)
      value present, no match  -> -1 Unknown  (the fact arrived before the dimension)
    Never returns null.
    """
    if dim.empty:
        return pd.Series(KEY_UNKNOWN, index=fact.index, dtype="int64")
    m = dim.set_index(dim_nk)[dim_key]
    vals = fact[nk_col]
    keys = vals.map(m)
    keys = keys.where(vals.notna(), absent_key)          # null NK -> not applicable
    keys = keys.where(~(vals.notna() & keys.isna()), KEY_UNKNOWN)   # no match -> unknown
    return keys.astype("int64")


# --------------------------------------------------------------------------
# Dimensions
# --------------------------------------------------------------------------
def load_dim_day(facts_dates: pd.Series, log: ErrorLog) -> pd.DataFrame:
    existing = read_gold("dim_day")
    dates = pd.to_datetime(facts_dates.dropna(), utc=True).dt.date
    if dates.empty:
        return existing
    wanted = pd.date_range(min(dates), max(dates), freq="D")
    known = set(existing["full_date"].astype(str)) if len(existing) else set()
    new = [d for d in wanted if str(d.date()) not in known]
    if not new and len(existing):
        return existing

    rows = pd.DataFrame({
        "day_key": [int(d.strftime("%Y%m%d")) for d in new],
        "full_date": [str(d.date()) for d in new],
        "day_of_week": [d.strftime("%A") for d in new],
        "is_weekend": [d.weekday() >= 5 for d in new],
        "iso_week": [int(d.isocalendar()[1]) for d in new],
        "month_num": [d.month for d in new],
        "month_name": [d.strftime("%B") for d in new],
        "quarter": [f"Q{(d.month - 1)//3 + 1}" for d in new],
        "year_num": [d.year for d in new],
    })
    if existing.empty:
        seed = pd.DataFrame([
            {"day_key": k, "full_date": "", "day_of_week": v, "is_weekend": False,
             "iso_week": 0, "month_num": 0, "month_name": v, "quarter": v, "year_num": 0}
            for k, v in SPECIAL.items()
        ])
        rows = pd.concat([seed, rows], ignore_index=True)
    out = pd.concat([existing, rows], ignore_index=True) if len(existing) else rows
    write_gold("dim_day", out, log)
    return out


def load_dim_type1(name: str, key_col: str, nk_col: str,
                   source: pd.DataFrame, attr_cols: list[str], log: ErrorLog) -> pd.DataFrame:
    """Generic SCD type 1 dimension: overwrite attributes, keep the key."""
    existing = read_gold(name)
    if existing.empty:
        existing = _seed_rows(key_col, {c: "" for c in [nk_col] + attr_cols})
        existing[nk_col] = [f"__{v}__" for v in SPECIAL.values()]

    if source.empty:
        write_gold(name, existing, log)
        return existing

    src = source[[nk_col] + attr_cols].drop_duplicates(subset=[nk_col]).copy()
    # Dimension attributes are descriptive, and the three special rows carry
    # the words "Unknown" / "Not applicable". Keeping every attribute as text
    # avoids a column that is bool in the real rows and str in the seed rows,
    # which Arrow refuses to write.
    for c in [nk_col] + attr_cols:
        src[c] = src[c].astype(str)
        if c in existing.columns:
            existing[c] = existing[c].astype(str)

    known = set(existing[nk_col])
    fresh = src[~src[nk_col].isin(known)]
    fresh = assign_keys(fresh, existing, key_col)

    # type 1: restate attributes in place for keys we already have
    updated = existing.merge(src, on=nk_col, how="left", suffixes=("", "_new"))
    for c in attr_cols:
        if f"{c}_new" in updated.columns:
            updated[c] = updated[f"{c}_new"].where(updated[f"{c}_new"].notna(), updated[c])
            updated = updated.drop(columns=[f"{c}_new"])

    out = pd.concat([updated, fresh], ignore_index=True)
    write_gold(name, out, log)
    return out


def load_dim_user(log: ErrorLog) -> pd.DataFrame:
    """SCD type 2: a change to tier closes the old version and opens a new one."""
    users = read_silver("ms_users")
    subs = read_silver("txn_user_subscriptions")
    existing = read_gold("dim_user")
    name, key_col, nk = "dim_user", "user_key", "user_id"

    if existing.empty:
        existing = pd.DataFrame([
            {key_col: k, nk: -abs(k) - 1000, "username": v, "email_token": "",
             "tier": v, "referred": False, "valid_from": "1900-01-01",
             "valid_to": "9999-12-31", "is_current": True}
            for k, v in SPECIAL.items()
        ])
    if users.empty:
        write_gold(name, existing, log)
        return existing

    referred = set(subs.loc[subs["referred_by_user_id"].notna(), "user_id"]) if len(subs) else set()
    src = pd.DataFrame({
        nk: users["id"].astype("int64"),
        "username": users["username"].astype(str),
        "email_token": users["email"].map(email_token),
        "tier": users["tier"].astype(str),
        "referred": users["id"].isin(referred),
    })

    today = datetime.now(timezone.utc).date().isoformat()
    cur = existing[existing["is_current"] == True]            # noqa: E712
    merged = src.merge(cur[[nk, key_col, "tier", "username"]], on=nk,
                       how="left", suffixes=("", "_old"))

    brand_new = merged[merged[key_col].isna()].drop(columns=[key_col, "tier_old", "username_old"])
    changed = merged[(merged[key_col].notna()) & (merged["tier"] != merged["tier_old"])]

    out = existing.copy()
    if len(changed):
        out.loc[out[key_col].isin(changed[key_col]) & (out["is_current"] == True),  # noqa: E712
                ["is_current", "valid_to"]] = [False, today]
        reborn = changed.drop(columns=[key_col, "tier_old", "username_old"])
        reborn = assign_keys(reborn, out, key_col)
        reborn["valid_from"], reborn["valid_to"], reborn["is_current"] = today, "9999-12-31", True
        out = pd.concat([out, reborn], ignore_index=True)

    if len(brand_new):
        brand_new = assign_keys(brand_new, out, key_col)
        brand_new["valid_from"], brand_new["valid_to"], brand_new["is_current"] = \
            "1900-01-01", "9999-12-31", True
        out = pd.concat([out, brand_new], ignore_index=True)

    write_gold(name, out, log)
    if len(changed):
        log.info(f"dim_user: {len(changed)} SCD2 version(s) opened")
    return out


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------
def load_fct_workout_session(dims: dict, cfg: RunConfig, run_id: str, log: ErrorLog,
                             metrics: MetricsCollector, lineage: LineageRegistry,
                             dlq: DeadLetterQueue) -> int:
    src = read_silver("txn_workout_summaries")
    with metrics.step("gold", "fct_workout_session") as m:
        m.load_type = cfg.load_type
        m.rows_in = len(src)
        if src.empty:
            return 0

        devices = read_silver("ms_device_connections")
        dev_map = (devices.set_index("id")["provider"].to_dict() if len(devices) else {})

        f = pd.DataFrame(index=src.index)
        f["workout_id"] = src["id"].astype("int64")            # degenerate dimension
        f["day_key"] = pd.to_datetime(src["started_at"], utc=True).dt.strftime("%Y%m%d").astype("int64")
        # point-in-time: the member tier as it was on the day of the session
        f["user_key"] = lookup_scd2(src, "user_id", dims["dim_user"], "user_id",
                                    "user_key", src["started_at"], absent_key=KEY_UNKNOWN)
        f["workout_type_key"] = lookup(src, "activity_type", dims["dim_workout_type"],
                                       "activity_type", "workout_type_key")
        src_prov = src["device_id"].map(dev_map)
        f["device_key"] = lookup(src.assign(_prov=src_prov), "_prov", dims["dim_device"],
                                 "provider", "device_key", absent_key=KEY_NOT_APPLICABLE)
        f["event_key"] = lookup(src, "event_id", dims["dim_event"], "event_id",
                                "event_key", absent_key=KEY_NOT_APPLICABLE)
        # measures
        f["distance_km"] = src["distance_km"].astype("float64")
        f["duration_sec"] = src["duration_sec"].astype("int64")
        f["calories_kcal"] = src["calories_kcal"].fillna(0).astype("int64")
        f["avg_heart_rate"] = src["avg_heart_rate"].fillna(0).astype("int64")
        f["pace_sec_per_km"] = src["pace_sec_per_km"].astype("float64")  # non-additive
        f["session_count"] = 1
        # lineage
        f["_src_lineage_id"] = src["_lineage_id"]
        f["_run_id"] = run_id

        existing = read_gold("fct_workout_session")
        if len(existing):
            out = (pd.concat([existing, f], ignore_index=True)
                     .drop_duplicates(subset=["workout_id"], keep="last"))
        else:
            out = f
        version = write_gold("fct_workout_session", out, log)

        m.rows_out = len(f)
        unknown = int((f["user_key"] == KEY_UNKNOWN).sum())
        lineage.record(
            load_type=cfg.load_type,
            source_object="silver.txn_workout_summaries", source_version="current",
            target_object="gold.fct_workout_session", target_version=str(version),
            operation="merge", rows_in=m.rows_in, rows_out=m.rows_out, rows_rejected=0,
        )
        log.info(f"fct_workout_session: {len(f)} rows -> v{version}, total {len(out)}"
                 + (f", {unknown} unknown user_key" if unknown else ""))
        return len(f)


def load_fct_payment(dims: dict, cfg: RunConfig, run_id: str, log: ErrorLog,
                     metrics: MetricsCollector, lineage: LineageRegistry,
                     dlq: DeadLetterQueue) -> int:
    src = read_silver("txn_payments")
    with metrics.step("gold", "fct_payment_transaction") as m:
        m.load_type = cfg.load_type
        m.rows_in = len(src)
        if src.empty:
            return 0

        subs = read_silver("txn_user_subscriptions")
        plan_of = subs.set_index("id")["plan_id"].to_dict() if len(subs) else {}

        f = pd.DataFrame(index=src.index)
        f["payment_id"] = src["id"].astype("int64")
        f["day_key"] = pd.to_datetime(src["created_at"], utc=True).dt.strftime("%Y%m%d").astype("int64")
        f["user_key"] = lookup_scd2(src, "user_id", dims["dim_user"], "user_id",
                                    "user_key", src["created_at"], absent_key=KEY_UNKNOWN)
        plan_nk = src["subscription_id"].map(plan_of)
        f["plan_key"] = lookup(src.assign(_plan=plan_nk), "_plan", dims["dim_plan"],
                               "plan_id", "plan_key", absent_key=KEY_UNKNOWN)
        f["payment_status"] = src["status"].astype(str)
        f["payment_method"] = src["payment_method"].astype(str)
        f["amount_thb"] = src["amount"].astype("float64")
        f["is_successful"] = (src["status"] == "succeeded").astype("int64")
        f["payment_count"] = 1
        f["_src_lineage_id"] = src["_lineage_id"]
        f["_run_id"] = run_id

        existing = read_gold("fct_payment_transaction")
        out = (pd.concat([existing, f], ignore_index=True)
                 .drop_duplicates(subset=["payment_id"], keep="last")) if len(existing) else f
        version = write_gold("fct_payment_transaction", out, log)

        m.rows_out = len(f)
        lineage.record(
            load_type=cfg.load_type,
            source_object="silver.txn_payments", source_version="current",
            target_object="gold.fct_payment_transaction", target_version=str(version),
            operation="merge", rows_in=m.rows_in, rows_out=m.rows_out, rows_rejected=0,
        )
        log.info(f"fct_payment_transaction: {len(f)} rows -> v{version}, total {len(out)}")
        return len(f)


# --------------------------------------------------------------------------
# Orchestration of the Gold layer
# --------------------------------------------------------------------------
def load_gold(cfg: RunConfig, run_id: str, log: ErrorLog, metrics: MetricsCollector,
              lineage: LineageRegistry, dlq: DeadLetterQueue) -> dict:
    log.stage = "load:gold"

    workouts = read_silver("txn_workout_summaries")
    payments = read_silver("txn_payments")
    events = read_silver("txn_events")
    areas = read_silver("ms_area")
    devices = read_silver("ms_device_connections")
    plans = read_silver("ms_subscription_plans")

    dates = pd.concat([
        workouts["started_at"] if len(workouts) else pd.Series(dtype="object"),
        payments["created_at"] if len(payments) else pd.Series(dtype="object"),
    ], ignore_index=True)

    dims = {}
    with metrics.step("gold", "dimensions") as m:
        dims["dim_day"] = load_dim_day(dates, log)
        dims["dim_user"] = load_dim_user(log)
        dims["dim_workout_type"] = load_dim_type1(
            "dim_workout_type", "workout_type_key", "activity_type",
            workouts[["activity_type"]].assign(
                is_cardio=workouts["activity_type"].isin(["run", "ride", "swim", "walk"])
            ) if len(workouts) else pd.DataFrame(),
            ["is_cardio"], log)
        dims["dim_device"] = load_dim_type1(
            "dim_device", "device_key", "provider",
            devices[["provider"]].assign(is_wearable=True) if len(devices) else pd.DataFrame(),
            ["is_wearable"], log)
        dims["dim_area"] = load_dim_type1(
            "dim_area", "area_key", "area_label",
            areas[["area_label", "area_type"]] if len(areas) else pd.DataFrame(),
            ["area_type"], log)
        dims["dim_event"] = load_dim_type1(
            "dim_event", "event_key", "event_id",
            events.rename(columns={"id": "event_id"})[["event_id", "workout_type", "status"]]
            if len(events) else pd.DataFrame(),
            ["workout_type", "status"], log)
        dims["dim_plan"] = load_dim_type1(
            "dim_plan", "plan_key", "plan_id",
            plans.rename(columns={"id": "plan_id"})[["plan_id", "plan_code", "tier"]]
            if len(plans) else pd.DataFrame(),
            ["plan_code", "tier"], log)
        m.rows_out = sum(len(d) for d in dims.values())
        log.info(f"dimensions: {m.rows_out} rows across {len(dims)} tables")

    n_workout = load_fct_workout_session(dims, cfg, run_id, log, metrics, lineage, dlq)
    n_payment = load_fct_payment(dims, cfg, run_id, log, metrics, lineage, dlq)
    return {"dimensions": {k: len(v) for k, v in dims.items()},
            "fct_workout_session": n_workout, "fct_payment_transaction": n_payment}
