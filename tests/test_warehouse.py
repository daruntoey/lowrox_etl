"""
WAREHOUSE — does Gold match the design (5 facts, 7 dimensions + plan / event)?

  1. every table in the design exists in Gold
  2. every fact foreign key is filled and points at a real dimension row
     (0 / -1 / -2 special rows included)
  3. each fact has the grain the design says (one row per ...)
  4. the additive facts reconcile with Silver (km, money, notifications,
     attendance) — nothing lost, nothing double counted
  5. the periodic snapshot keeps zero weeks, and its totals equal the
     workout fact's totals
  6. no key whose natural key exists falls back to -1 Unknown

Read-only: works as `python tests/test_warehouse.py [--fresh]` and under pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from deltalake import DeltaTable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl import reset_lake, run_pipeline
from etl.config import GOLD, SILVER

RESULTS: list[dict] = []

DIMS = {
    "dim_day": "day_key", "dim_user": "user_key", "dim_user_profile": "user_profile_key",
    "dim_area": "area_key", "dim_workout_type": "workout_type_key", "dim_device": "device_key",
    "dim_workout_info": "workout_info_key", "dim_plan": "plan_key", "dim_event": "event_key",
}
FACTS = {   # fact -> (grain column, foreign keys)
    "fct_workout_session": ("workout_id", ["day_key", "user_key", "user_profile_key", "workout_type_key",
                                           "area_key", "device_key", "workout_info_key", "event_key"]),
    "fct_event_participation": ("participation_id", ["day_key", "user_key", "user_profile_key",
                                                     "area_key", "workout_type_key", "event_key"]),
    "fct_subscription_payment": ("payment_id", ["day_key", "user_key", "user_profile_key", "plan_key"]),
    "fct_notification_delivery": ("notification_id", ["day_key", "user_key", "user_profile_key", "event_key"]),
    "fct_user_fitness_weekly": (None, ["day_key", "user_key", "user_profile_key", "area_key"]),
}
KEY_DIM = {k: d for d, k in DIMS.items()}


def check(name: str, passed: bool, detail: str = "") -> bool:
    RESULTS.append({"test": name, "result": "PASS" if passed else "FAIL", "detail": detail})
    return passed


def _df(path: Path) -> pd.DataFrame:
    return DeltaTable(str(path)).to_pandas() if (path / "_delta_log").exists() else pd.DataFrame()


def g(name):
    return _df(GOLD / name)


def s(name):
    return _df(SILVER / name)


def test_tables_exist():
    missing = [t for t in list(DIMS) + list(FACTS) if g(t).empty]
    assert check("all 9 dimensions and 5 facts exist in gold", not missing,
                 f"missing: {missing}" if missing else f"{len(DIMS)} dims, {len(FACTS)} facts")


def test_foreign_keys():
    bad = []
    for fact, (_, fks) in FACTS.items():
        f = g(fact)
        for fk in fks:
            dim = g(KEY_DIM[fk])
            if f[fk].isna().any():
                bad.append(f"{fact}.{fk} null")
            orphan = ~f[fk].isin(dim[fk])
            if orphan.any():
                bad.append(f"{fact}.{fk} {int(orphan.sum())} orphan")
    assert check("every fact key is filled and found in its dimension", not bad, "; ".join(bad) or "ok")


def test_known_keys_resolve():
    """A fact whose natural key exists in the dimension must not land on -1 Unknown
    (event_id / plan_id once did: the dimension kept them as text)."""
    unknown = {}
    for fact, (_, fks) in FACTS.items():
        f = g(fact)
        for fk in fks:
            n = int((f[fk] == -1).sum())
            if n:
                unknown[f"{fact}.{fk}"] = n
    assert check("no fact key falls back to -1 Unknown", not unknown,
                 ", ".join(f"{k}={v}" for k, v in unknown.items()) or "0 unknown keys")


def test_grain():
    dupes = []
    for fact, (grain, _) in FACTS.items():
        f = g(fact)
        cols = [grain] if grain else ["user_key", "day_key"]
        if grain is None:   # one row per user per week (by natural user id via user_key version)
            cols = ["week_start", "user_key"]
        if f.duplicated(subset=cols).any():
            dupes.append(fact)
    assert check("each fact has one row per its grain", not dupes, ", ".join(dupes) or "ok")


def test_reconcile():
    w, p, pay, n = s("txn_workout_summaries"), s("txn_event_participants"), s("txn_payments"), s("txn_notifications")
    fw, fp, fs, fn = (g("fct_workout_session"), g("fct_event_participation"),
                      g("fct_subscription_payment"), g("fct_notification_delivery"))
    rows = {
        "workout km": (round(w["distance_km"].sum(), 2), round(fw["distance_km"].sum(), 2)),
        "participations": (len(p), int(fp["joined_count"].sum())),
        "attended": (int((p["status"] == "attended").sum()), int(fp["attended_count"].sum())),
        "succeeded THB": (round(pay.loc[pay["status"] == "succeeded", "amount"].astype(float).sum(), 2),
                          round(fs["succeeded_amount_thb"].sum(), 2)),
        "payment attempts": (len(pay), int(fs["attempt_count"].sum())),
        "notifications sent": (int(((n["send_status"] == "sent") & n["sent_at"].notna()).sum()),
                               int(fn["sent_count"].sum())),
    }
    off = {k: v for k, v in rows.items() if v[0] != v[1]}
    assert check("additive facts reconcile with silver", not off,
                 "; ".join(f"{k} {a}≠{b}" for k, (a, b) in off.items()) or
                 ", ".join(f"{k}={b}" for k, (_, b) in rows.items()))


def test_weekly_snapshot():
    fw, wk = g("fct_workout_session"), g("fct_user_fitness_weekly")
    same = (round(fw["distance_km"].sum(), 2) == round(wk["total_km"].sum(), 2)
            and int(fw["session_count"].sum()) == int(wk["session_count"].sum())
            and int(fw["heart_beat_count"].sum()) == int(wk["total_beat_count"].sum()))
    zero_weeks = int((wk["session_count"] == 0).sum())
    assert check("weekly snapshot totals equal the workout fact, zero weeks kept",
                 same and zero_weeks > 0,
                 f"km={wk['total_km'].sum():.2f}, sessions={int(wk['session_count'].sum())}, zero weeks={zero_weeks}")


def run_all(verbose: bool = True, *, fresh: bool = False) -> pd.DataFrame:
    RESULTS.clear()
    if fresh:
        reset_lake()
        assert run_pipeline("init", echo=False).ok
    for t in (test_tables_exist, test_foreign_keys, test_known_keys_resolve, test_grain, test_reconcile, test_weekly_snapshot):
        try:
            t()
        except AssertionError:
            pass
        except Exception as exc:          # a missing table etc. is a failure, not a crash
            check(t.__name__, False, f"{type(exc).__name__}: {exc}")
    df = pd.DataFrame(RESULTS)
    if verbose:
        print(df.to_string(index=False))
        print(f"\n{(df.result == 'PASS').sum()}/{len(df)} passed")
    return df


if __name__ == "__main__":
    out = run_all(fresh="--fresh" in sys.argv)
    sys.exit(0 if (out.result == "PASS").all() else 1)
