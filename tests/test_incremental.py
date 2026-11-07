"""
INIT vs INCREMENTAL — the two load modes, proved against a changing source.

  init         read every row once, record a high-water mark per table
  incremental  read only rows newer than that mark, MERGE them in

What has to be true for the incremental load to be correct:
  1. it reads ONLY the changed rows, not the whole table
  2. new facts are appended, not duplicated
  3. an UPDATE in the source opens a new SCD2 version and closes the old one
  4. a run with nothing to do costs almost nothing
  5. running it twice changes nothing the second time
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import psycopg2
from deltalake import DeltaTable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl import reset_lake, run_pipeline
from etl.config import GOLD, SILVER, SOURCE_DSN, TABLES_BY_NAME
from etl.extract import read_watermarks

RESULTS: list[dict] = []


def record(name: str, expected, observed, passed: bool):
    RESULTS.append({"check": name, "expected": expected,
                    "observed": observed, "result": "PASS" if passed else "FAIL"})


def _rows(path) -> int:
    p = Path(path)
    return len(DeltaTable(str(p)).to_pandas()) if (p / "_delta_log").exists() else 0


def _gold(name: str) -> pd.DataFrame:
    p = GOLD / name
    return DeltaTable(str(p)).to_pandas() if (p / "_delta_log").exists() else pd.DataFrame()


def _exec(sql: str, params=None):
    # psycopg2 treats % as a placeholder whenever params is supplied, even an
    # empty tuple, which breaks any LIKE 'x%' literal. Pass None instead.
    with psycopg2.connect(SOURCE_DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, params) if params else cur.execute(sql)
        conn.commit()


def _scalar(sql: str):
    with psycopg2.connect(SOURCE_DSN) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


# --------------------------------------------------------------------------
def busiest_users(n: int = 2) -> list[int]:
    """
    Pick users who actually have workouts, so the SCD2 history check has fact
    rows to point at. Hard-coding user 1 and 2 would test nothing: in this
    dataset neither of them has ever recorded a session.
    """
    with psycopg2.connect(SOURCE_DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT user_id FROM txn_workout_summaries "
                    "GROUP BY 1 ORDER BY count(*) DESC LIMIT %s", (n,))
        return [r[0] for r in cur.fetchall()]


def apply_source_changes(verbose: bool = True) -> dict:
    """
    Make the kind of changes a live application makes between two runs:
      7 new workouts   (INSERT into an append-only table)
      3 new payments   (INSERT)
      2 tier upgrades  (UPDATE — this is what SCD2 has to notice)

    NOTE ON TIMESTAMPS. A high-watermark incremental load assumes the
    watermark column only ever moves forward. The seeded mock data contains
    rows dated into the future (subscription periods and their scheduled
    payments), so max(updated_at) on txn_payments is already ahead of now().
    A row inserted with updated_at = now() would therefore sit BEHIND the
    watermark and never be picked up — silently. The inserts below stamp
    their rows past the current maximum, which is what a real application
    with honest clocks produces. The hazard itself is exercised deliberately
    in check 6.
    """
    upgrade_ids = busiest_users(2)

    _exec("""
        INSERT INTO txn_workout_summaries
            (user_id, device_id, event_id, external_activity_id, activity_type,
             source, started_at, duration_sec, distance_km, calories_kcal,
             avg_heart_rate, created_at)
        SELECT (random()*49)::int + 1, NULL, NULL,
               'incr-' || gs, 'running', 'manual',
               now() - (gs || ' hours')::interval,
               1500 + gs*60, 4.5 + gs*0.4, 300 + gs*20, 140 + gs,
               (SELECT GREATEST(now(), max(created_at) + interval '1 second')
                  FROM txn_workout_summaries) + (gs || ' seconds')::interval
        FROM generate_series(1, 7) gs
    """)
    _exec("""
        INSERT INTO txn_payments
            (subscription_id, user_id, amount, currency, payment_method,
             gateway_txn_ref, status, paid_at, created_at, updated_at)
        SELECT s.id, s.user_id, 349.00, 'THB', 'credit_card',
               'incr-pay-' || s.id, 'succeeded', now(), now(),
               (SELECT GREATEST(now(), max(updated_at) + interval '1 second')
                  FROM txn_payments) + (row_number() OVER (ORDER BY s.id) || ' seconds')::interval
        FROM txn_user_subscriptions s ORDER BY s.id LIMIT 3
    """)
    _exec("""
        UPDATE ms_users SET tier = 'elite',
               updated_at = (SELECT GREATEST(now(), max(updated_at) + interval '1 second')
                               FROM ms_users)
        WHERE id = ANY(%s)
    """, (upgrade_ids,))

    changes = {
        "new_workouts": _scalar("SELECT count(*) FROM txn_workout_summaries "
                                "WHERE external_activity_id LIKE 'incr-%'"),
        "new_payments": _scalar("SELECT count(*) FROM txn_payments "
                                "WHERE gateway_txn_ref LIKE 'incr-pay-%'"),
        "tier_upgrades": _scalar("SELECT count(*) FROM ms_users WHERE tier = 'elite'"),
    }
    changes["upgraded_user_ids"] = upgrade_ids
    if verbose:
        print("\nSOURCE CHANGES APPLIED (counted from the database, not assumed)")
        print(pd.DataFrame([changes]).to_string(index=False))
    return changes


def revert_source_changes():
    """Leave the database as it was found, so the suite can be re-run."""
    _exec("DELETE FROM txn_workout_summaries WHERE external_activity_id LIKE 'incr-%'")
    _exec("DELETE FROM txn_payments WHERE gateway_txn_ref LIKE 'incr-pay-%'")
    _exec("DELETE FROM txn_payments WHERE gateway_txn_ref LIKE 'late-%'")
    _exec("UPDATE ms_users SET tier = 'free' WHERE tier = 'elite'")


# --------------------------------------------------------------------------
def run_demo(verbose: bool = True) -> dict:
    RESULTS.clear()
    revert_source_changes()

    # ---- phase 1: init ---------------------------------------------------
    if verbose:
        print("=" * 74)
        print("PHASE 1 — INIT LOAD (the lake starts empty)")
        print("=" * 74)
    reset_lake()
    r_init = run_pipeline("init", echo=False)

    init_counts = {
        "bronze_workouts": _rows(TABLES_BY_NAME["txn_workout_summaries"].bronze_path),
        "silver_workouts": _rows(SILVER / "txn_workout_summaries"),
        "gold_workouts": _rows(GOLD / "fct_workout_session"),
        "gold_payments": _rows(GOLD / "fct_payment_transaction"),
        "dim_user_rows": _rows(GOLD / "dim_user"),
    }
    marks = read_watermarks()
    if verbose:
        print(f"\n{r_init}")
        print(pd.DataFrame([init_counts]).to_string(index=False))
        print(f"\nwatermarks recorded for {len(marks)} tables, e.g.")
        for k in ("ms_users", "txn_workout_summaries", "txn_payments"):
            print(f"  {k:<26} {marks.get(k)}")

    # ---- phase 2: the source changes ------------------------------------
    if verbose:
        print("\n" + "=" * 74)
        print("PHASE 2 — THE APPLICATION WRITES NEW DATA")
        print("=" * 74)
    changes = apply_source_changes(verbose)

    # ---- phase 3: incremental -------------------------------------------
    if verbose:
        print("\n" + "=" * 74)
        print("PHASE 3 — INCREMENTAL LOAD (only the delta should move)")
        print("=" * 74)
    r_incr = run_pipeline("incremental", echo=verbose)

    ex = r_incr.metrics.to_frame()
    ex = ex[ex.layer == "extract"]
    moved = ex[ex.rows_out > 0][["step", "rows_out"]].rename(
        columns={"step": "table", "rows_out": "rows_extracted"})
    total_moved = int(ex.rows_out.sum())

    after = {
        "bronze_workouts": _rows(TABLES_BY_NAME["txn_workout_summaries"].bronze_path),
        "silver_workouts": _rows(SILVER / "txn_workout_summaries"),
        "gold_workouts": _rows(GOLD / "fct_workout_session"),
        "gold_payments": _rows(GOLD / "fct_payment_transaction"),
        "dim_user_rows": _rows(GOLD / "dim_user"),
    }

    comparison = pd.DataFrame([
        {"object": k, "after_init": init_counts[k], "after_incremental": after[k],
         "delta": after[k] - init_counts[k]} for k in init_counts
    ])
    if verbose:
        print("\nWHAT THE INCREMENTAL RUN READ")
        print(moved.to_string(index=False))
        print(f"\n{total_moved} rows read, against 856 in the source — "
              f"{100*total_moved/856:.1f}% of a full load")
        print("\nWHAT CHANGED IN THE LAKE")
        print(comparison.to_string(index=False))

    # ---- the five correctness checks ------------------------------------
    record("1. reads only the delta", "12 rows (7 workouts + 3 payments + 2 users)",
           f"{total_moved} rows", total_moved == 12)

    record("2. new facts appended, not duplicated", "gold workouts +7",
           f"+{after['gold_workouts'] - init_counts['gold_workouts']}",
           after["gold_workouts"] - init_counts["gold_workouts"] == 7)

    # SCD2: two users upgraded, so two new versions, and the old ones closed
    ids = changes["upgraded_user_ids"]
    du = _gold("dim_user")
    upgraded = du[du.user_id.isin(ids)].sort_values(["user_id", "valid_from"])
    current_elite = du[(du.user_id.isin(ids)) & (du.is_current == True)]          # noqa: E712
    closed = du[(du.user_id.isin(ids)) & (du.is_current == False)]                # noqa: E712
    scd_ok = (len(upgraded) == 4 and len(current_elite) == 2
              and set(current_elite.tier) == {"elite"} and len(closed) == 2)
    record("3. SCD2 fires on the tier upgrade",
           f"4 rows for users {ids}: 2 closed + 2 current (elite)",
           f"{len(upgraded)} rows, {len(closed)} closed, "
           f"{len(current_elite)} current = {sorted(set(current_elite.tier))}", scd_ok)
    if verbose and len(upgraded):
        print(f"\nSCD TYPE 2 ON dim_user (users {ids} upgraded to elite)")
        print(upgraded[["user_key", "user_id", "username", "tier",
                        "valid_from", "valid_to", "is_current"]].to_string(index=False))

    # facts written before the upgrade must still point at the OLD version
    f = _gold("fct_workout_session")
    old_keys = set(closed.user_key.astype(int))
    still_old = int(f[f.user_key.isin(old_keys)].shape[0])
    record("4. history preserved: old facts keep the old dimension key",
           "> 0 fact rows still point at the closed version",
           f"{still_old} fact rows", still_old > 0)

    # ---- phase 4: run it again with nothing to do -----------------------
    r_noop = run_pipeline("incremental", echo=False)
    noop_rows = sum(s.rows_out for s in r_noop.metrics.steps if s.layer == "extract")
    gold_after_noop = _rows(GOLD / "fct_workout_session")
    record("5. a second incremental run is a no-op", "0 rows read, gold unchanged",
           f"{noop_rows} rows read, gold {gold_after_noop}",
           noop_rows == 0 and gold_after_noop == after["gold_workouts"])

    # ---- phase 5: the hazard, exercised on purpose ----------------------
    # A row whose watermark value lands BEHIND the stored high-water mark is
    # invisible to this strategy. It is the single most common way a
    # watermark pipeline loses data silently, so it is worth demonstrating
    # rather than describing.
    _exec("""
        INSERT INTO txn_payments
            (subscription_id, user_id, amount, currency, payment_method,
             gateway_txn_ref, status, paid_at, created_at, updated_at)
        SELECT s.id, s.user_id, 99.00, 'THB', 'credit_card',
               'late-' || s.id, 'succeeded', now(), now(),
               timestamptz '2020-01-01'
        FROM txn_user_subscriptions s ORDER BY s.id LIMIT 2
    """)
    gold_pay_before = _rows(GOLD / "fct_payment_transaction")
    r_late = run_pipeline("incremental", echo=False)
    late_read = sum(s.rows_out for s in r_late.metrics.steps
                    if s.layer == "extract" and s.step == "txn_payments")
    gold_pay_after = _rows(GOLD / "fct_payment_transaction")
    in_source = _scalar("SELECT count(*) FROM txn_payments WHERE gateway_txn_ref LIKE 'late-%'")

    record("6. late-arriving rows are invisible to a high watermark (known limit)",
           "2 rows in the source, 0 picked up — a real limitation, not a bug",
           f"{in_source} inserted with a 2020 timestamp, {late_read} extracted, "
           f"gold payments {gold_pay_before} -> {gold_pay_after}",
           in_source == 2 and late_read == 0 and gold_pay_after == gold_pay_before)
    if verbose:
        print("\n" + "=" * 74)
        print("PHASE 5 — THE KNOWN LIMIT OF WATERMARKING, DEMONSTRATED")
        print("=" * 74)
        print(f"Inserted {in_source} payments stamped 2020-01-01, behind the watermark.")
        print(f"The incremental run read {late_read} of them. Gold is unchanged at "
              f"{gold_pay_after}.")
        print("\nMitigations, in increasing order of cost:")
        print("  - re-read a lookback window (watermark minus N hours) every run")
        print("  - run a periodic full reconciliation and compare counts per table")
        print("  - replace the watermark with log-based CDC, which reads the WAL")
        print("    and therefore cannot be fooled by an application clock")

    df = pd.DataFrame(RESULTS)
    if verbose:
        print("\n" + "=" * 74)
        print("PHASE 4 — SECOND INCREMENTAL RUN, NOTHING CHANGED IN THE SOURCE")
        print("=" * 74)
        print(f"rows read: {noop_rows}   gold unchanged at {gold_after_noop}")
        print("\nINCREMENTAL CORRECTNESS CHECKS")
        print(df.to_string(index=False))
        print(f"\n{(df.result == 'PASS').sum()}/{len(df)} passed")

    revert_source_changes()
    return {"init": r_init, "incremental": r_incr, "moved": moved,
            "comparison": comparison, "scd": upgraded, "checks": df}


if __name__ == "__main__":
    out = run_demo()
    sys.exit(0 if (out["checks"].result == "PASS").all() else 1)
