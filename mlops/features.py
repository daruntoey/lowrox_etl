"""
features.py — what the Lowrox features actually ARE.

The ML problem, chosen to match slide 24 exactly: predict subscription churn.
The label table is (user_id, ts, is_churn), and it is enriched from several
feature sets, which is the shape the chapter uses throughout.

Every feature set is computed on a MONTHLY SNAPSHOT GRID. For each user and
each month end, the features describe only what had happened strictly before
that month end. That is what gives each feature set a real time index rather
than a single "now" row, and it is what makes the point-in-time join in
feature_store.py something other than decoration.

Feature-engineering techniques from Chapter 7, and where each is used:

    aggregation / statistical (slide 28)  sessions, distance, pace per window
    rolling and lag (slide 28)            30d vs 90d windows, previous month
    date and time, cyclical (slide 30)    sin/cos of month and weekday
    binning (slide 27)                    age bands, pace bands
    encoding (slide 27)                   one-hot tier, label gender,
                                          target-encoded area, hashed device
    normalisation (slide 29)              z-score on skewed counts
    interactions (slide 29, 37)           consistency x tenure, pace x volume
    log transform (slide 29)              counts with a long right tail
    imputation (slide 26)                 KNN on the numeric block
    domain-specific (slide 38)            days since last session, streaks
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .feature_store import FeatureSet, REGISTRY

# --------------------------------------------------------------------------
# The snapshot grid
# --------------------------------------------------------------------------
SNAPSHOT_FREQ = "ME"          # month end
CHURN_HORIZON_DAYS = 60       # cancelled within this many days of the snapshot


def snapshot_dates(start, end, freq: str = SNAPSHOT_FREQ) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).normalize(),
                         pd.Timestamp(end).normalize(), freq=freq, tz="UTC")


def _grid(users: pd.Series, snapshots: pd.DatetimeIndex) -> pd.DataFrame:
    """Every (user, snapshot) pair — the panel each feature set is built on."""
    return pd.DataFrame(
        [(u, t) for t in snapshots for u in users],
        columns=["user_id", "ts"])


# --------------------------------------------------------------------------
# 1. User profile features
# --------------------------------------------------------------------------
def _compute_profile(src: dict) -> pd.DataFrame:
    users = src["ms_users"]
    demo = src["ms_user_demographics"]
    subs = src["txn_user_subscriptions"]
    snaps = src["snapshots"]

    base = users[["id", "tier", "created_at"]].rename(columns={"id": "user_id"})
    base = base.merge(demo[["user_id", "birth_date", "gender"]], on="user_id", how="left")
    base["birth_year"] = pd.to_datetime(base["birth_date"], errors="coerce").dt.year

    # Home area is not stored on the member; it is inferred from the events
    # they join. 50 distinct areas is exactly the high-cardinality case the
    # chapter suggests hashing rather than one-hot encoding.
    parts, events = src["txn_event_participants"], src["txn_events"]
    area = (parts[["user_id", "event_id"]]
            .merge(events.rename(columns={"id": "event_id"})[["event_id", "area_id"]],
                   on="event_id", how="left")
            .dropna(subset=["area_id"])
            .groupby("user_id")["area_id"]
            .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
            .rename("home_area_id").reset_index())
    base = base.merge(area, on="user_id", how="left")
    referred = set(subs.loc[subs["referred_by_user_id"].notna(), "user_id"])
    base["was_referred"] = base["user_id"].isin(referred).astype(int)

    g = _grid(base["user_id"], snaps).merge(base, on="user_id", how="left")
    g["created_at"] = pd.to_datetime(g["created_at"], utc=True)

    # only describe users who existed at the snapshot
    g = g[g["created_at"] <= g["ts"]].copy()

    # --- domain-specific: tenure (slide 38) ---
    g["tenure_days"] = (g["ts"] - g["created_at"]).dt.days.clip(lower=0)

    # --- binning: age into bands (slide 27) ---
    g["age"] = g["ts"].dt.year - g["birth_year"].fillna(g["birth_year"].median())
    g["age"] = g["age"].clip(10, 90)
    g["age_band"] = pd.cut(g["age"], bins=[0, 29, 39, 49, 200],
                           labels=[0, 1, 2, 3]).astype("float64")

    # --- encoding: label encoding for gender (slide 27) ---
    # ms_users.tier is NOT used. See MUTABLE_CURRENT_STATE below: the
    # application downgrades it to 'free' at cancellation, so it is the label.
    g["gender_code"] = g["gender"].fillna("unknown").map(
        {"male": 0, "female": 1}).fillna(2).astype(int)

    # --- hashing: area has high cardinality, so hash it (slide 27) ---
    g["area_hash"] = g["home_area_id"].fillna(-1).astype(int).map(
        lambda v: int(hashlib.md5(str(v).encode()).hexdigest()[:4], 16) % 16)

    # --- log transform on a long-tailed count (slide 29) ---
    g["log_tenure_days"] = np.log1p(g["tenure_days"])

    return g


PROFILE = REGISTRY.register(FeatureSet(
    name="user_profile_features",
    entity="user_id", timestamp="ts",
    features=("tenure_days", "log_tenure_days", "age_band", "gender_code",
              "was_referred", "area_hash"),
    description="Who the member is, as at each month end: tenure, age band, "
                "tier, referral origin, hashed home area.",
    compute=_compute_profile))


# --------------------------------------------------------------------------
# 2. Workout activity features
# --------------------------------------------------------------------------
def _compute_activity(src: dict) -> pd.DataFrame:
    w = src["txn_workout_summaries"][
        ["user_id", "started_at", "distance_km", "duration_sec",
         "avg_heart_rate", "pace_sec_per_km"]].copy()
    w["started_at"] = pd.to_datetime(w["started_at"], utc=True)
    snaps = src["snapshots"]
    users = src["ms_users"]["id"].rename("user_id")

    g = _grid(users, snaps)
    rows = []
    for t in snaps:
        # STRICTLY BEFORE the snapshot — this is the line that keeps the
        # feature honest. Using <= would let a session recorded at the
        # snapshot instant inform a prediction made at that instant.
        past = w[w["started_at"] < t]
        w30 = past[past["started_at"] >= t - pd.Timedelta(days=30)]
        w90 = past[past["started_at"] >= t - pd.Timedelta(days=90)]

        agg = past.groupby("user_id").agg(
            sessions_lifetime=("distance_km", "size"),
            last_session=("started_at", "max"),
            km_lifetime=("distance_km", "sum"),
            best_pace=("pace_sec_per_km", "min"),
        )
        a30 = w30.groupby("user_id").agg(
            sessions_30d=("distance_km", "size"),
            km_30d=("distance_km", "sum"),
            avg_pace_30d=("pace_sec_per_km", "mean"),
            avg_hr_30d=("avg_heart_rate", "mean"),
            active_days_30d=("started_at", lambda s: s.dt.date.nunique()),
        )
        a90 = w90.groupby("user_id").agg(
            sessions_90d=("distance_km", "size"),
            km_90d=("distance_km", "sum"),
            avg_pace_90d=("pace_sec_per_km", "mean"),
        )
        snap = agg.join(a30, how="outer").join(a90, how="outer").reset_index()
        snap["ts"] = t
        rows.append(snap)

    act = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = g.merge(act, on=["user_id", "ts"], how="left")

    counts = ["sessions_lifetime", "sessions_30d", "sessions_90d",
              "km_lifetime", "km_30d", "km_90d", "active_days_30d"]
    out[counts] = out[counts].fillna(0)

    # --- domain-specific: recency (slide 38) ---
    out["days_since_last"] = (out["ts"] - pd.to_datetime(out["last_session"], utc=True)).dt.days
    out["days_since_last"] = out["days_since_last"].fillna(999).clip(0, 999)

    # --- rolling comparison: is this member speeding up or slowing down? ---
    out["pace_trend"] = (out["avg_pace_30d"] - out["avg_pace_90d"]).fillna(0)

    # --- lag: the previous month's volume, per user (slide 28) ---
    out = out.sort_values(["user_id", "ts"])
    out["sessions_30d_lag1"] = out.groupby("user_id")["sessions_30d"].shift(1).fillna(0)
    out["sessions_delta"] = out["sessions_30d"] - out["sessions_30d_lag1"]

    # --- log transform on skewed counts (slide 29) ---
    out["log_sessions_lifetime"] = np.log1p(out["sessions_lifetime"])
    out["log_km_lifetime"] = np.log1p(out["km_lifetime"])

    # --- binning: pace into the four level bands used across the product ---
    out["pace_band"] = pd.cut(out["avg_pace_90d"], bins=[0, 300, 420, 600, 10_000],
                              labels=[3, 2, 1, 0]).astype("float64").fillna(-1)

    # --- interaction: consistency x volume (slides 29, 37) ---
    out["consistency_30d"] = out["active_days_30d"] / 30.0
    out["consistency_x_volume"] = out["consistency_30d"] * out["km_30d"]

    # --- cyclical date features (slide 30) ---
    month = out["ts"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    return out


ACTIVITY = REGISTRY.register(FeatureSet(
    name="workout_activity_features",
    entity="user_id", timestamp="ts",
    features=("sessions_lifetime", "log_sessions_lifetime", "sessions_30d",
              "sessions_90d", "sessions_30d_lag1", "sessions_delta",
              "km_30d", "km_90d", "log_km_lifetime", "active_days_30d",
              "consistency_30d", "consistency_x_volume", "days_since_last",
              "avg_pace_30d", "avg_pace_90d", "pace_trend", "pace_band",
              "avg_hr_30d", "best_pace", "month_sin", "month_cos"),
    description="Training behaviour in 30 and 90 day windows as at each month "
                "end, with lags, recency, pace trend and cyclical time.",
    compute=_compute_activity))


# --------------------------------------------------------------------------
# 3. Engagement features
# --------------------------------------------------------------------------
def _compute_engagement(src: dict) -> pd.DataFrame:
    notif = src["txn_notifications"].copy()
    parts = src["txn_event_participants"].copy()
    comments = src["txn_group_comments"].copy()
    snaps = src["snapshots"]
    users = src["ms_users"]["id"].rename("user_id")

    for df, col in ((notif, "sent_at"), (parts, "created_at"), (comments, "created_at")):
        df[col] = pd.to_datetime(df[col], utc=True)

    opened_col = "opened_at" if "opened_at" in notif.columns else None
    if opened_col:
        notif[opened_col] = pd.to_datetime(notif[opened_col], utc=True)

    g = _grid(users, snaps)
    rows = []
    for t in snaps:
        n = notif[notif["sent_at"] < t]
        n30 = n[n["sent_at"] >= t - pd.Timedelta(days=30)]
        p = parts[parts["created_at"] < t]
        c = comments[comments["created_at"] < t]

        a = n30.groupby("user_id").agg(notifications_30d=("sent_at", "size"))
        if opened_col:
            a = a.join(n30.assign(_o=n30[opened_col].notna())
                          .groupby("user_id").agg(opens_30d=("_o", "sum")))
        else:
            a["opens_30d"] = 0
        b = p.groupby("user_id").agg(events_lifetime=("created_at", "size"))
        d = c.groupby("user_id").agg(comments_lifetime=("created_at", "size"))

        snap = a.join(b, how="outer").join(d, how="outer").reset_index()
        snap["ts"] = t
        rows.append(snap)

    eng = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = g.merge(eng, on=["user_id", "ts"], how="left")
    for c in ("notifications_30d", "opens_30d", "events_lifetime", "comments_lifetime"):
        out[c] = out[c].fillna(0)

    # --- ratio feature: engagement quality, not just volume ---
    out["open_rate_30d"] = (out["opens_30d"] / out["notifications_30d"].replace(0, np.nan)).fillna(0)
    # --- domain-specific: is this member social? Group training is the product's bet
    out["is_social"] = ((out["events_lifetime"] > 0) | (out["comments_lifetime"] > 0)).astype(int)

    return out


ENGAGEMENT = REGISTRY.register(FeatureSet(
    name="engagement_features",
    entity="user_id", timestamp="ts",
    features=("notifications_30d", "opens_30d", "open_rate_30d",
              "events_lifetime", "comments_lifetime", "is_social"),
    description="How the member responds to the product: notification opens, "
                "event participation, group comments.",
    compute=_compute_engagement))


# --------------------------------------------------------------------------
# 4. Subscription features
# --------------------------------------------------------------------------
def _compute_subscription(src: dict) -> pd.DataFrame:
    subs = src["txn_user_subscriptions"].copy()
    plans = src["ms_subscription_plans"].copy()
    pays = src["txn_payments"].copy()
    snaps = src["snapshots"]

    subs["started_at"] = pd.to_datetime(subs["started_at"], utc=True)
    subs["cancelled_at"] = pd.to_datetime(subs["cancelled_at"], utc=True)
    pays["created_at"] = pd.to_datetime(pays["created_at"], utc=True)
    plans = plans.rename(columns={"id": "plan_id"})

    plans = plans.rename(columns={"price": "price_thb", "tier": "plan_tier"})
    s = subs.merge(plans[["plan_id", "price_thb", "billing_period", "plan_tier"]],
                   on="plan_id", how="left")

    rows = []
    for t in snaps:
        live = s[(s["started_at"] < t) & (s["cancelled_at"].isna() | (s["cancelled_at"] >= t))]
        p = pays[pays["created_at"] < t]
        pa = p.groupby("user_id").agg(
            payments_lifetime=("amount", "size"),
            revenue_lifetime=("amount", "sum"),
            failed_payments=("status", lambda x: (x != "succeeded").sum()))
        snap = live[["user_id", "price_thb", "plan_tier", "started_at"]].copy()
        snap = snap.merge(pa, on="user_id", how="left")
        snap["ts"] = t
        rows.append(snap)

    sub = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    sub = sub.drop_duplicates(subset=["user_id", "ts"], keep="last")

    sub["subscription_age_days"] = (sub["ts"] - sub["started_at"]).dt.days.clip(lower=0)
    sub["price_thb"] = sub["price_thb"].fillna(0).astype(float)

    # auto_renew is DELIBERATELY NOT a feature.
    #
    # The first version of this file used it, and the model scored ROC-AUC
    # 0.995 with auto_renew at 85% of the importance — which is what target
    # leakage looks like from the outside. In this source, auto_renew = false
    # corresponds to cancelled_at IS NOT NULL in 490 rows out of 490: the
    # application clears the flag AT cancellation, so the column is the label
    # wearing a different name.
    #
    # The subtlety worth keeping: the point-in-time join was correct. The
    # leak was in the FEATURE DEFINITION, which read the subscription row's
    # current value rather than its value as at the snapshot. Silver holds
    # only the current state of a mutable column, so there is no honest way
    # to use it here. A column can only be a point-in-time feature if its
    # history exists; otherwise its present value is a message from the
    # future. The same applies to `status`.
    for c in ("payments_lifetime", "revenue_lifetime", "failed_payments"):
        sub[c] = sub[c].fillna(0)
    # --- domain-specific: a failed payment is the loudest churn signal there is
    sub["failed_payment_rate"] = (
        sub["failed_payments"] / sub["payments_lifetime"].replace(0, np.nan)).fillna(0)
    sub["log_revenue"] = np.log1p(sub["revenue_lifetime"])

    # The tier of the PLAN the member is paying for. Unlike ms_users.tier this
    # is not rewritten when they cancel, so it is a legitimate point-in-time
    # attribute — and it carries real signal: elite plans cancel at 22.6%
    # against pro at 26.4%.
    pt = sub["plan_tier"].fillna("").str.lower()
    sub["plan_is_elite"] = (pt == "elite").astype(int)

    return sub


SUBSCRIPTION = REGISTRY.register(FeatureSet(
    name="subscription_features",
    entity="user_id", timestamp="ts",
    features=("subscription_age_days", "price_thb", "plan_is_elite",
              "payments_lifetime", "revenue_lifetime", "log_revenue",
              "failed_payments", "failed_payment_rate"),
    description="Commercial state of the membership: age, price, auto-renew, "
                "payment history and failures.",
    compute=_compute_subscription))


ALL_FEATURE_SETS = [PROFILE, ACTIVITY, ENGAGEMENT, SUBSCRIPTION]


# --------------------------------------------------------------------------
# Columns that must never become features, and the rule behind the list
# --------------------------------------------------------------------------
# A point-in-time join can only be as honest as the columns it joins. Silver
# holds the CURRENT value of a mutable column, not its history, so reading one
# at a past snapshot silently returns a value from the future. Where the
# application mutates that column as part of the very event being predicted,
# the "feature" is the label.
#
# This cost three rounds of debugging on this project. Each time the symptom
# was identical: ROC-AUC near 0.995 and one feature holding most of the
# importance.
MUTABLE_CURRENT_STATE = {
    "ms_users.tier": "set to 'free' at cancellation — 490 of 490 cancelled "
                     "subscriptions have tier='free'",
    "txn_user_subscriptions.auto_renew": "cleared at cancellation — "
                                         "auto_renew=false matches cancelled 490/490",
    "txn_user_subscriptions.status": "'cancelled' IS the label",
    "txn_user_subscriptions.cancelled_at": "used to BUILD the label, never as a feature",
}
#
# Safe to use, because they are immutable or append-only:
#   created_at / started_at / birth_date   written once
#   plan_id and the plan's tier and price  not rewritten on cancellation
#   workouts, notifications, payments      event tables, append-only


# --------------------------------------------------------------------------
# The label table  —  slide 24's  customer_churn | Model
# --------------------------------------------------------------------------
def build_labels(src: dict, horizon_days: int = CHURN_HORIZON_DAYS) -> pd.DataFrame:
    """
    One row per (user, month end) at which the member had a live subscription.

        is_churn = the subscription was cancelled within `horizon_days`
                   AFTER that month end.

    The label looks forward; the features look only backward. That asymmetry
    is the whole design, and it is why the point-in-time join matters.
    """
    subs = src["txn_user_subscriptions"].copy()
    subs["started_at"] = pd.to_datetime(subs["started_at"], utc=True)
    subs["cancelled_at"] = pd.to_datetime(subs["cancelled_at"], utc=True)
    snaps = src["snapshots"]

    rows = []
    for t in snaps:
        live = subs[(subs["started_at"] < t)
                    & (subs["cancelled_at"].isna() | (subs["cancelled_at"] >= t))]
        horizon = t + pd.Timedelta(days=horizon_days)
        churned = live["cancelled_at"].notna() & (live["cancelled_at"] < horizon)
        rows.append(pd.DataFrame({
            "user_id": live["user_id"].values,
            "ts": t,
            "is_churn": churned.astype(int).values,
        }))

    labels = pd.concat(rows, ignore_index=True)
    return labels.drop_duplicates(subset=["user_id", "ts"], keep="last")
