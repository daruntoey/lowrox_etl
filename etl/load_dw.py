"""
load_dw.py — the parts of the warehouse design that load.py did not build yet.

Design (Data warehouse schema slide):
  dimensions  dim_day, dim_user (SCD2), dim_user_profile (mini-dimension),
              dim_area, dim_workout_type, dim_device, dim_workout_info (junk)
              + dim_plan and dim_event, which the bus matrix needs
  facts       fct_workout_session        one row per validated workout   transaction
              fct_event_participation    one row per user joining event  transaction
              fct_subscription_payment   one row per payment attempt     transaction
              fct_notification_delivery  one row per notification        accumulating snapshot
              fct_user_fitness_weekly    one row per user per week       periodic snapshot

Every fact foreign key is a surrogate key and never null (0 Not applicable,
-1 Unknown, -2 Invalid), exactly like load.py.
"""
from __future__ import annotations

import pandas as pd

from .config import KEY_NOT_APPLICABLE, KEY_UNKNOWN
from .load import lookup, lookup_scd2, read_gold, read_silver, write_gold, load_dim_type1

BKK = "Asia/Bangkok"


# --------------------------------------------------------------------------
# dim_user_profile — mini-dimension of the member attributes that change
# often enough to make dim_user explode if they were type 2 there.
# One row per combination: age band × gender × fitness level, resolved
# AS OF the moment of each fact.
# --------------------------------------------------------------------------
AGE_BANDS = [(0, 18, "under 18"), (18, 25, "18-24"), (25, 35, "25-34"), (35, 45, "35-44"),
             (45, 55, "45-54"), (55, 200, "55+")]


def _age_band(age: float) -> str:
    if pd.isna(age):
        return "unknown"
    for lo, hi, label in AGE_BANDS:
        if lo <= age < hi:
            return label
    return "unknown"


def profile_codes(user_ids: pd.Series, when: pd.Series) -> pd.Series:
    """profile_code ('25-34|female|intermediate') for each (user, moment)."""
    if user_ids.empty:
        return pd.Series(dtype="object")
    demo = read_silver("ms_user_demographics")
    levels = read_silver("txn_user_history_level")

    at = pd.to_datetime(when, utc=True, errors="coerce")
    at = at.fillna(pd.Timestamp.now(tz="UTC")).astype("datetime64[ns, UTC]")
    probe = pd.DataFrame({"_row": range(len(user_ids)),
                          "user_id": pd.to_numeric(user_ids, errors="coerce").to_numpy(),
                          "at": at.to_numpy()})

    if len(demo):
        d = demo[["user_id", "birth_date", "gender"]].copy()
        d["user_id"] = pd.to_numeric(d["user_id"], errors="coerce")
        probe = probe.merge(d, on="user_id", how="left")
    else:
        probe["birth_date"], probe["gender"] = None, None
    birth = pd.to_datetime(probe["birth_date"], errors="coerce")
    age = (probe["at"].dt.tz_convert(None) - birth).dt.days / 365.25
    probe["age_band"] = age.map(_age_band)
    g = probe["gender"].astype("string").str.strip().str.lower()
    probe["gender_n"] = g.where(g.isin(["male", "female"]), other="other").where(g.notna(), "unknown")

    probe["level"] = "unknown"
    if len(levels):
        lv = levels[["user_id", "fitness_level", "recorded_at"]].dropna(subset=["recorded_at"]).copy()
        lv["user_id"] = pd.to_numeric(lv["user_id"], errors="coerce")
        lv["recorded_at"] = pd.to_datetime(lv["recorded_at"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
        lv = lv.dropna(subset=["user_id", "recorded_at"]).sort_values("recorded_at")
        p = probe.dropna(subset=["user_id"]).sort_values("at")
        if len(p) and len(lv):
            p = p.assign(at=p["at"].astype("datetime64[ns, UTC]"), user_id=p["user_id"].astype("float64"))
            lv = lv.assign(user_id=lv["user_id"].astype("float64"))
            asof = pd.merge_asof(p[["_row", "user_id", "at"]], lv, left_on="at",
                                 right_on="recorded_at", by="user_id", direction="backward")
            # before the first assessment, use the first level we know of
            first = lv.drop_duplicates("user_id").set_index("user_id")["fitness_level"]
            asof["fitness_level"] = asof["fitness_level"].fillna(asof["user_id"].map(first))
            probe = probe.merge(asof[["_row", "fitness_level"]], on="_row", how="left")
            probe["level"] = probe["fitness_level"].fillna("unknown").astype(str).str.lower()

    probe = probe.sort_values("_row")
    codes = probe["age_band"] + "|" + probe["gender_n"] + "|" + probe["level"]
    codes = codes.where(probe["user_id"].notna(), None)
    return pd.Series(codes.to_numpy(), index=user_ids.index)


def load_dim_user_profile(codes: pd.Series, log) -> pd.DataFrame:
    c = pd.Series(codes.dropna().unique(), dtype="object")
    src = pd.DataFrame({"profile_code": c})
    if len(src):
        parts = src["profile_code"].str.split("|", expand=True)
        src["age_band"], src["gender"], src["fitness_level"] = parts[0], parts[1], parts[2]
    return load_dim_type1("dim_user_profile", "user_profile_key", "profile_code",
                          src, ["age_band", "gender", "fitness_level"], log)


# --------------------------------------------------------------------------
# dim_workout_info — junk dimension: the low-cardinality flags of a workout
# --------------------------------------------------------------------------
def workout_info_codes(w: pd.DataFrame) -> pd.DataFrame:
    src = w["source"].fillna("unknown").astype(str).str.lower()
    entry = src.where(src.eq("manual"), "device sync")
    out = pd.DataFrame({
        "entry_method": entry,
        "has_heart_rate": w["avg_heart_rate"].notna() & (pd.to_numeric(w["avg_heart_rate"], errors="coerce") > 0),
        "has_calories": w["calories_kcal"].notna() & (pd.to_numeric(w["calories_kcal"], errors="coerce") > 0),
        "is_event_session": w["event_id"].notna(),
    }, index=w.index)
    yn = lambda s: s.map({True: "Y", False: "N"})          # noqa: E731
    out["workout_info_code"] = (out["entry_method"] + "|HR" + yn(out["has_heart_rate"])
                                + "|CAL" + yn(out["has_calories"]) + "|EVT" + yn(out["is_event_session"]))
    return out


def load_dim_workout_info(info: pd.DataFrame, log) -> pd.DataFrame:
    src = info.drop_duplicates("workout_info_code") if len(info) else pd.DataFrame()
    return load_dim_type1("dim_workout_info", "workout_info_key", "workout_info_code", src,
                          ["entry_method", "has_heart_rate", "has_calories", "is_event_session"], log)


# --------------------------------------------------------------------------
# facts
# --------------------------------------------------------------------------
def _day_key(ts: pd.Series) -> pd.Series:
    # UTC calendar date, the same rule load.py uses for fct_workout_session
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    return t.dt.strftime("%Y%m%d").fillna(str(KEY_UNKNOWN)).astype("int64")


def _area_label_of_event(events: pd.DataFrame, areas: pd.DataFrame) -> dict:
    if events.empty or areas.empty:
        return {}
    label = areas.set_index("id")["area_label"].astype(str).to_dict()
    return events.set_index("id")["area_id"].map(label).dropna().to_dict()


def build_workout_extras(dims: dict, w: pd.DataFrame, events: pd.DataFrame,
                         areas: pd.DataFrame, profile: pd.Series, info: pd.DataFrame) -> pd.DataFrame:
    """The design's extra columns on fct_workout_session, keyed by workout_id."""
    area_of = _area_label_of_event(events, areas)
    x = pd.DataFrame({"workout_id": w["id"].astype("int64")}, index=w.index)
    x["user_profile_key"] = lookup(pd.DataFrame({"c": profile}), "c", dims["dim_user_profile"],
                                   "profile_code", "user_profile_key", absent_key=KEY_UNKNOWN)
    x["area_key"] = lookup(pd.DataFrame({"a": w["event_id"].map(area_of)}), "a", dims["dim_area"],
                           "area_label", "area_key", absent_key=KEY_NOT_APPLICABLE)
    x["workout_info_key"] = lookup(info, "workout_info_code", dims["dim_workout_info"],
                                   "workout_info_code", "workout_info_key", absent_key=KEY_UNKNOWN)
    x["external_activity_id"] = w["external_activity_id"].astype("string").fillna("")
    hr = pd.to_numeric(w["avg_heart_rate"], errors="coerce").fillna(0)
    x["heart_beat_count"] = (hr * pd.to_numeric(w["duration_sec"], errors="coerce").fillna(0) / 60).round().astype("int64")
    return x


def load_fct_event_participation(dims: dict, run_id: str, log, profile: pd.Series) -> int:
    p = read_silver("txn_event_participants")
    if p.empty:
        return 0
    events = read_silver("txn_events")
    areas = read_silver("ms_area")
    ev = events.set_index("id") if len(events) else pd.DataFrame()
    area_of = _area_label_of_event(events, areas)
    joined_at = pd.to_datetime(p["joined_at"].fillna(p["created_at"]), utc=True, errors="coerce")

    f = pd.DataFrame(index=p.index)
    f["participation_id"] = p["id"].astype("int64")                      # degenerate
    f["day_key"] = _day_key(joined_at)
    f["user_key"] = lookup_scd2(p, "user_id", dims["dim_user"], "user_id", "user_key",
                                joined_at, absent_key=KEY_UNKNOWN)
    f["user_profile_key"] = lookup(pd.DataFrame({"c": profile}), "c", dims["dim_user_profile"],
                                   "profile_code", "user_profile_key", absent_key=KEY_UNKNOWN)
    f["area_key"] = lookup(pd.DataFrame({"a": p["event_id"].map(area_of)}), "a", dims["dim_area"],
                           "area_label", "area_key", absent_key=KEY_NOT_APPLICABLE)
    wt = p["event_id"].map(ev["workout_type"]) if len(ev) else pd.Series(None, index=p.index)
    f["workout_type_key"] = lookup(pd.DataFrame({"t": wt.str.lower() if len(ev) else wt}), "t",
                                   dims["dim_workout_type"], "activity_type", "workout_type_key")
    f["event_key"] = lookup(p, "event_id", dims["dim_event"], "event_id", "event_key",
                            absent_key=KEY_NOT_APPLICABLE)
    f["role"] = p["role"].astype(str)
    f["participation_status"] = p["status"].astype(str)
    f["joined_count"] = 1
    f["attended_count"] = (p["status"] == "attended").astype("int64")
    f["no_show_count"] = (p["status"] == "no_show").astype("int64")
    f["cancelled_count"] = (p["status"] == "cancelled").astype("int64")
    sched = pd.to_datetime(p["event_id"].map(ev["scheduled_at"]), utc=True, errors="coerce") if len(ev) else None
    lead = ((sched - joined_at).dt.total_seconds() / 3600) if sched is not None else pd.Series(0.0, index=p.index)
    f["lead_time_hours"] = lead.clip(lower=0).fillna(0).round(2).astype("float64")
    f["_src_lineage_id"] = p.get("_lineage_id")
    f["_run_id"] = run_id
    # a member who left (web leave_event deletes the row) is gone from Silver: full rebuild
    write_gold("fct_event_participation", f, log)
    log.info(f"fct_event_participation: {len(f)} rows")
    return len(f)


def load_fct_subscription_payment(dims: dict, run_id: str, log, profile: pd.Series) -> int:
    src = read_silver("txn_payments")
    if src.empty:
        return 0
    subs = read_silver("txn_user_subscriptions")
    plan_of = subs.set_index("id")["plan_id"].to_dict() if len(subs) else {}
    at = pd.to_datetime(src["created_at"], utc=True, errors="coerce")
    ok = src["status"].eq("succeeded")
    amount = pd.to_numeric(src["amount"], errors="coerce").fillna(0)

    f = pd.DataFrame(index=src.index)
    f["payment_id"] = src["id"].astype("int64")
    f["day_key"] = _day_key(at)
    f["user_key"] = lookup_scd2(src, "user_id", dims["dim_user"], "user_id", "user_key", at,
                                absent_key=KEY_UNKNOWN)
    f["user_profile_key"] = lookup(pd.DataFrame({"c": profile}), "c", dims["dim_user_profile"],
                                   "profile_code", "user_profile_key", absent_key=KEY_UNKNOWN)
    f["plan_key"] = lookup(src.assign(_plan=src["subscription_id"].map(plan_of)), "_plan",
                           dims["dim_plan"], "plan_id", "plan_key", absent_key=KEY_UNKNOWN)
    f["payment_status"] = src["status"].astype(str)
    f["payment_method"] = src["payment_method"].astype(str)
    f["attempt_count"] = 1
    f["succeeded_amount_thb"] = amount.where(ok, 0.0).astype("float64")
    f["failed_amount_thb"] = amount.where(src["status"].eq("failed"), 0.0).astype("float64")
    f["refunded_amount_thb"] = amount.where(src["status"].eq("refunded"), 0.0).astype("float64")
    f["_src_lineage_id"] = src.get("_lineage_id")
    f["_run_id"] = run_id
    write_gold("fct_subscription_payment", f, log)
    log.info(f"fct_subscription_payment: {len(f)} rows")
    return len(f)


def load_fct_notification_delivery(dims: dict, run_id: str, log, profile: pd.Series) -> int:
    """
    Accumulating snapshot: one row per notification, its milestones filled in
    as they happen (queued → sent → delivered → opened → clicked). Rebuilt
    every run, so a later open updates the same row.
    LINE pushes have no delivery or read receipts: their delivered / opened
    milestones stay empty by design.
    """
    n = read_silver("txn_notifications")
    if n.empty:
        return 0
    created = pd.to_datetime(n["created_at"], utc=True, errors="coerce")
    ms = {c: pd.to_datetime(n[c], utc=True, errors="coerce")
          for c in ("sent_at", "delivered_at", "opened_at", "clicked_at")}
    sent = n["send_status"].eq("sent") & ms["sent_at"].notna()

    f = pd.DataFrame(index=n.index)
    f["notification_id"] = n["id"].astype("int64")
    f["day_key"] = _day_key(ms["sent_at"].fillna(created))
    f["user_key"] = lookup_scd2(n, "user_id", dims["dim_user"], "user_id", "user_key",
                                ms["sent_at"].fillna(created), absent_key=KEY_UNKNOWN)
    f["user_profile_key"] = lookup(pd.DataFrame({"c": profile}), "c", dims["dim_user_profile"],
                                   "profile_code", "user_profile_key", absent_key=KEY_UNKNOWN)
    f["event_key"] = lookup(n, "event_id", dims["dim_event"], "event_id", "event_key",
                            absent_key=KEY_NOT_APPLICABLE)
    f["channel"] = n["channel"].astype(str)
    f["notification_type"] = n["notification_type"].fillna("").astype(str)
    f["send_status"] = n["send_status"].astype(str)
    for c, t in ms.items():
        f[c] = t
    f["sent_count"] = sent.astype("int64")
    f["delivered_count"] = (sent & ms["delivered_at"].notna()).astype("int64")
    f["opened_count"] = (sent & ms["opened_at"].notna()).astype("int64")
    f["clicked_count"] = (sent & ms["clicked_at"].notna()).astype("int64")
    f["minutes_to_open"] = ((ms["opened_at"] - ms["sent_at"]).dt.total_seconds() / 60).round(1)
    f["_src_lineage_id"] = n.get("_lineage_id")
    f["_run_id"] = run_id
    write_gold("fct_notification_delivery", f, log)
    log.info(f"fct_notification_delivery: {len(f)} rows")
    return len(f)


def weekly_frame(w: pd.DataFrame) -> pd.DataFrame:
    """
    The periodic snapshot's grain: every member who has ever logged a workout,
    for every week from their first workout to the latest week in the data —
    including the weeks with no workout (zeros are what make it a snapshot).
    """
    if w.empty:
        return pd.DataFrame()
    t = pd.to_datetime(w["started_at"], utc=True, errors="coerce").dt.tz_convert(BKK)
    week = (t.dt.normalize() - pd.to_timedelta(t.dt.weekday, unit="D")).dt.date
    hr = pd.to_numeric(w["avg_heart_rate"], errors="coerce").fillna(0)
    dur = pd.to_numeric(w["duration_sec"], errors="coerce").fillna(0)
    base = pd.DataFrame({"user_id": w["user_id"].astype("int64"), "week_start": week,
                         "distance_km": pd.to_numeric(w["distance_km"], errors="coerce").fillna(0),
                         "duration_sec": dur, "beats": (hr * dur / 60).round()})
    agg = base.groupby(["user_id", "week_start"], as_index=False).agg(
        session_count=("distance_km", "size"), total_km=("distance_km", "sum"),
        total_duration_sec=("duration_sec", "sum"), total_beat_count=("beats", "sum"))
    last = max(base["week_start"])
    grid = []
    for uid, first in base.groupby("user_id")["week_start"].min().items():
        for d in pd.date_range(first, last, freq="7D"):
            grid.append((uid, d.date()))
    g = pd.DataFrame(grid, columns=["user_id", "week_start"])
    out = g.merge(agg, on=["user_id", "week_start"], how="left").fillna(
        {"session_count": 0, "total_km": 0.0, "total_duration_sec": 0, "total_beat_count": 0})
    out["week_start_ts"] = pd.to_datetime(out["week_start"]).dt.tz_localize(BKK)
    return out


def load_fct_user_fitness_weekly(dims: dict, run_id: str, log, wk: pd.DataFrame,
                                 profile: pd.Series, area_label: pd.Series) -> int:
    if wk.empty:
        return 0
    f = pd.DataFrame(index=wk.index)
    f["day_key"] = wk["week_start"].map(lambda d: int(d.strftime("%Y%m%d"))).astype("int64")
    f["week_start"] = wk["week_start"].astype(str)
    f["user_key"] = lookup_scd2(wk, "user_id", dims["dim_user"], "user_id", "user_key",
                                wk["week_start_ts"], absent_key=KEY_UNKNOWN)
    f["user_profile_key"] = lookup(pd.DataFrame({"c": profile}), "c", dims["dim_user_profile"],
                                   "profile_code", "user_profile_key", absent_key=KEY_UNKNOWN)
    f["area_key"] = lookup(pd.DataFrame({"a": area_label}), "a", dims["dim_area"],
                           "area_label", "area_key", absent_key=KEY_NOT_APPLICABLE)
    f["session_count"] = wk["session_count"].astype("int64")
    f["total_km"] = wk["total_km"].round(2).astype("float64")
    f["total_duration_sec"] = wk["total_duration_sec"].astype("int64")
    f["total_beat_count"] = wk["total_beat_count"].astype("int64")
    f["is_active_week"] = (wk["session_count"] > 0).astype("int64")
    f["_run_id"] = run_id
    write_gold("fct_user_fitness_weekly", f, log)
    log.info(f"fct_user_fitness_weekly: {len(f)} rows")
    return len(f)


def weekly_area(wk: pd.DataFrame, w: pd.DataFrame, events: pd.DataFrame, areas: pd.DataFrame) -> pd.Series:
    """Area of the member's event-linked workouts that week (most frequent), else none."""
    if wk.empty:
        return pd.Series(dtype="object")
    area_of = _area_label_of_event(events, areas)
    t = pd.to_datetime(w["started_at"], utc=True, errors="coerce").dt.tz_convert(BKK)
    x = pd.DataFrame({"user_id": w["user_id"].astype("int64"),
                      "week_start": (t.dt.normalize() - pd.to_timedelta(t.dt.weekday, unit="D")).dt.date,
                      "area": w["event_id"].map(area_of)}).dropna(subset=["area"])
    if x.empty:
        return pd.Series(None, index=wk.index, dtype="object")
    top = x.groupby(["user_id", "week_start"])["area"].agg(lambda s: s.mode().iloc[0])
    keys = list(zip(wk["user_id"], wk["week_start"]))
    return pd.Series([top.get(k) for k in keys], index=wk.index, dtype="object")
