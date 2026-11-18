"""
WEB APP CONTRACT — does the pipeline read what the Lowrox web app writes?

The web app (github.com/aom-thanya/lowrox) writes to the same Supabase tables
this pipeline extracts, through sql/04_frontend_integration.sql and
06_line_login.sql. These checks pin the behaviour that contract needs:

  1. the onboarding form copy (ms_users.onboarding_data, which repeats health
     answers) never enters the lake — PDPA s.26
  2. jsonb and array columns land as JSON text, not Python's repr
  3. the quality gate accepts every run the app accepts, and still rejects
     what the app would refuse
  4. a row the app hard-deletes (leave a party, delete a run, re-save
     onboarding) leaves Silver and Gold on the next incremental run
  5. a column added in Supabase reaches Silver instead of being dropped
  6. the weekly plan table is extracted

Like the other suites it works as `python tests/test_app_contract.py` and
under pytest. Checks 4 and 5 write to the source and clean up after
themselves, so they share test_incremental's guard against real databases.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from deltalake import DeltaTable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl import reset_lake, run_pipeline
from etl.config import BRONZE, GOLD, SILVER, TABLES_BY_NAME
from etl.extract import read_watermarks
from etl.transform import apply_dq
from tests.test_incremental import _exec, _guard_source_writes, _scalar

RESULTS: list[dict] = []
MARK = "app-contract"


def check(name: str, passed: bool, detail: str = "") -> bool:
    RESULTS.append({"test": name, "result": "PASS" if passed else "FAIL", "detail": detail})
    return passed


def _df(path: Path) -> pd.DataFrame:
    return DeltaTable(str(path)).to_pandas() if (path / "_delta_log").exists() else pd.DataFrame()


# --------------------------------------------------------------------------
def test_dq_matches_the_app():
    """The app accepts 0.5-100 km at no faster than 2:30 min/km (isPlausibleEffort).
    The gate additionally quarantines slower than 30 min/km (see transform.py)."""
    rows = pd.DataFrame([
        # activity, km, seconds, expected
        ("running", 42.2, 19_800, True),     # marathon in 5:30 — the app accepts it
        ("running", 100.0, 60_000, True),    # the app's upper bound
        ("running", 5.0, 1_800, True),
        ("running", 84.3, 1_127, False),     # 13 s/km
        ("running", 120.0, 40_000, False),   # over the app's 100 km
        ("running", 0.2, 120, False),        # under 0.5 km
        ("running", 6.09, 21_617, False),    # 59 min/km: a stopwatch left running
        ("cycling", 150.0, 18_000, True),    # a long ride
        ("functional", 0.0, 3_600, True),    # gym session, no distance
        ("swimming", 0.0, 3_600, False),     # a swim with no distance is wrong data
    ], columns=["activity_type", "distance_km", "duration_sec", "expected"])
    df = rows.assign(user_id=1, avg_heart_rate=None)
    good, bad = apply_dq(df.drop(columns="expected"), "txn_workout_summaries")
    accepted = df.index.isin(good.index)
    wrong = rows[accepted != rows["expected"]]
    assert check("quality gate accepts every plausible app run, rejects the implausible",
                 wrong.empty, "all 10 cases as expected" if wrong.empty
                 else wrong.to_dict("records"))


def test_onboarding_form_never_enters_the_lake():
    leaked = [layer for layer, root in (("bronze", BRONZE), ("silver", SILVER))
              if "onboarding_data" in _df(root / "ms_users").columns]
    assert check("ms_users.onboarding_data (health answers) is not extracted",
                 not leaked and "onboarding_data" in TABLES_BY_NAME["ms_users"].exclude_cols,
                 f"found in {leaked}" if leaked else "absent from bronze and silver")


def test_json_columns_are_json():
    bad = []
    for table, col in (("ms_users", "interests"), ("txn_group_comments", "images")):
        frame = _df(BRONZE / table)
        if col not in frame.columns:
            bad.append(f"{table}.{col} missing")
            continue
        for v in frame[col].dropna():
            try:
                json.loads(v)
            except (TypeError, ValueError):
                bad.append(f"{table}.{col}: {v[:40]!r}")
                break
    assert check("jsonb / array columns land as JSON text", not bad,
                 "; ".join(bad) or "ms_users.interests, txn_group_comments.images parse")


def test_plan_checkoffs_extracted():
    ok = (BRONZE / "txn_plan_checkoffs" / "_delta_log").exists() or \
         _scalar("SELECT count(*) FROM txn_plan_checkoffs") == 0
    assert check("txn_plan_checkoffs is in the registry and extracted",
                 "txn_plan_checkoffs" in TABLES_BY_NAME and ok)


# --------------------------------------------------------------------------
def _cleanup():
    _exec(f"DELETE FROM txn_workout_summaries WHERE external_activity_id LIKE '{MARK}-%'")
    _exec("ALTER TABLE ms_users DROP COLUMN IF EXISTS zz_contract_probe")


def test_hard_deletes_and_new_columns():
    """Insert, extract, delete in the source, extract again: Silver and Gold follow."""
    _guard_source_writes()
    _cleanup()
    user = _scalar("SELECT min(id) FROM ms_users")
    # created_at must be past the current watermark, as a live app's clock would
    # be; the incremental suite can leave the mark a few seconds in the future.
    mark = read_watermarks().get("txn_workout_summaries")
    after = f"GREATEST(now(), '{mark}'::timestamptz + interval '1 second')" if mark else "now()"
    _exec(f"""INSERT INTO txn_workout_summaries
                (user_id, activity_type, source, started_at, duration_sec, distance_km,
                 external_activity_id, created_at)
              VALUES ({user}, 'running', 'log', now(), 1800, 5, '{MARK}-1', {after})""")
    wid = _scalar(f"SELECT id FROM txn_workout_summaries WHERE external_activity_id = '{MARK}-1'")
    assert run_pipeline("incremental", echo=False).ok
    arrived = (_df(SILVER / "txn_workout_summaries")["id"] == wid).any() and \
        (_df(GOLD / "fct_workout_session")["workout_id"] == wid).any()

    _exec(f"DELETE FROM txn_workout_summaries WHERE id = {wid}")
    _exec("ALTER TABLE ms_users ADD COLUMN zz_contract_probe text")
    _exec(f"UPDATE ms_users SET zz_contract_probe = 'x', updated_at = now() WHERE id = {user}")
    try:
        assert run_pipeline("incremental", echo=False).ok
        gone = not (_df(SILVER / "txn_workout_summaries")["id"] == wid).any() and \
            not (_df(GOLD / "fct_workout_session")["workout_id"] == wid).any()
        silver_users = _df(SILVER / "ms_users")
        column_arrived = "zz_contract_probe" in silver_users.columns and \
            (silver_users.loc[silver_users["id"] == user, "zz_contract_probe"] == "x").any()
    finally:
        _cleanup()
    check("a new run reaches silver and gold", bool(arrived), f"workout {wid}")
    check("a run deleted in the source leaves silver and gold", bool(gone), f"workout {wid}")
    check("a column added in the source reaches silver", bool(column_arrived),
          "ms_users.zz_contract_probe")
    assert arrived and gone and column_arrived


# --------------------------------------------------------------------------
def run_all(verbose: bool = True, *, fresh: bool = False) -> pd.DataFrame:
    RESULTS.clear()
    if fresh:
        reset_lake()
        assert run_pipeline("init", echo=False).ok
    for t in (test_dq_matches_the_app, test_onboarding_form_never_enters_the_lake,
              test_json_columns_are_json, test_plan_checkoffs_extracted,
              test_hard_deletes_and_new_columns):
        try:
            t()
        except AssertionError:
            pass
    df = pd.DataFrame(RESULTS)
    if verbose:
        print(df.to_string(index=False))
        print(f"\n{(df.result == 'PASS').sum()}/{len(df)} passed")
    return df


if __name__ == "__main__":
    out = run_all(fresh="--fresh" in sys.argv)
    sys.exit(0 if (out.result == "PASS").all() else 1)
