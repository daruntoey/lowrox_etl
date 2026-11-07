"""
ERROR HANDLING TESTING — six live simulations.

Course slide mapping:
  Data Extraction Failure   sim_1_transient_source_failure
                            sim_2_permanent_source_failure_is_isolated
  Transformation Errors     sim_3_bad_data_is_quarantined
                            sim_4_schema_drift_is_detected
  Load Failures             sim_5_target_unavailable
  (plus) Recovery           sim_6_replay_is_idempotent

Nothing is mocked out. Each simulation arms a fault in FaultInjector, runs the
REAL pipeline, and shows what the real retry / DLQ / alerting code did about
it. The pipeline has no idea it is being tested.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl import FaultInjector, reset_lake, run_pipeline
from etl.config import GOLD, SILVER, TABLES_BY_NAME
from etl.errors import DeadLetterQueue
from deltalake import DeltaTable

RESULTS: list[dict] = []
BANNER = "-" * 74


def record(sim: str, expected: str, observed: str, passed: bool):
    RESULTS.append({"simulation": sim, "expected": expected,
                    "observed": observed, "result": "PASS" if passed else "FAIL"})


def _delta_rows(path) -> int:
    p = Path(path)
    return len(DeltaTable(str(p)).to_pandas()) if (p / "_delta_log").exists() else 0


def _fresh_baseline(echo: bool = False):
    FaultInjector.clear()
    reset_lake()
    return run_pipeline("init", echo=echo)


# ==========================================================================
def sim_1_transient_source_failure(verbose: bool = True):
    """
    The source database refuses the first two connections, then recovers.
    Expected: the retry absorbs it, the table loads, the run still succeeds.
    """
    if verbose:
        print(f"\n{BANNER}\nSIMULATION 1 — transient source failure (network blip)\n"
              f"{BANNER}\nFault: ms_users refuses 2 connections, then succeeds.\n")
    FaultInjector.clear()
    reset_lake()
    with FaultInjector.armed("extract.source_unavailable", fail_times=2, scope="ms_users"):
        r = run_pipeline("init", echo=verbose, retry_attempts=3)

    users = _delta_rows(TABLES_BY_NAME["ms_users"].bronze_path)
    attempts = next((s.attempts for s in r.metrics.steps
                     if s.layer == "extract" and s.step == "ms_users"), 0)
    ok = r.ok and users == 50 and attempts == 3
    if verbose:
        print(f"\n  retry attempts used : {attempts} of 3")
        print(f"  ms_users in bronze  : {users}")
        print(f"  run status          : {'SUCCESS' if r.ok else 'FAILED'}")
    record("1. Transient source failure", "retry recovers, 50 rows load",
           f"recovered on attempt {attempts}, {users} rows, run {'ok' if r.ok else 'failed'}", ok)
    assert ok
    return r


def sim_2_permanent_source_failure_is_isolated(verbose: bool = True):
    """
    One table's source stays down for good.
    Expected: that table is quarantined and alerted, the other 17 still load.
    A single bad table must not become an outage for the whole warehouse.
    """
    if verbose:
        print(f"\n{BANNER}\nSIMULATION 2 — permanent source failure, isolation\n"
              f"{BANNER}\nFault: txn_payments never connects (exceeds all retries).\n")
    FaultInjector.clear()
    reset_lake()
    with FaultInjector.armed("extract.source_unavailable", fail_times=99, scope="txn_payments"):
        r = run_pipeline("init", echo=verbose, retry_attempts=3)

    loaded = [s.step for s in r.metrics.steps
              if s.layer == "extract" and s.status == "ok" and s.rows_out > 0]
    payments_rows = _delta_rows(TABLES_BY_NAME["txn_payments"].bronze_path)
    workouts_rows = _delta_rows(TABLES_BY_NAME["txn_workout_summaries"].bronze_path)
    critical = [a for a in r.alerts.alerts if a.severity == "critical"]
    ok = payments_rows == 0 and workouts_rows == 50 and len(loaded) >= 16 and critical
    if verbose:
        print(f"\n  tables that still loaded : {len(loaded)} of 18")
        print(f"  txn_payments in bronze   : {payments_rows} (expected 0)")
        print(f"  txn_workout_summaries    : {workouts_rows} (unaffected)")
        print(f"  critical alerts raised   : {len(critical)}")
    record("2. Permanent source failure", "1 table fails, 17 unaffected, alert raised",
           f"{len(loaded)} tables loaded, payments={payments_rows}, "
           f"{len(critical)} critical alert(s)", bool(ok))
    assert ok
    return r


def sim_3_bad_data_is_quarantined(verbose: bool = True):
    """
    The device feed delivers impossible measures inside otherwise valid rows.
    Expected: the DQ rules catch them, they go to the dead-letter queue with
    the rule that rejected them, and the good rows still reach Gold.
    """
    if verbose:
        print(f"\n{BANNER}\nSIMULATION 3 — corrupt values in the feed\n{BANNER}\n"
              "Fault: 3 workout rows get negative / impossible measures.\n")
    FaultInjector.clear()
    reset_lake()
    with FaultInjector.armed("data.bad_values", fail_times=1, scope="txn_workout_summaries"):
        r = run_pipeline("init", echo=verbose)

    dlq = DeadLetterQueue.read(r.run_id)
    w = dlq[dlq["_dlq_stage"] == "silver:txn_workout_summaries"] if len(dlq) else pd.DataFrame()
    gold = _delta_rows(GOLD / "fct_workout_session")
    ok = len(w) >= 3 and gold > 0 and r.ok
    if verbose:
        print(f"\n  rows quarantined     : {len(w)}")
        print(f"  rows reaching Gold   : {gold}")
        print(f"  run status           : {'SUCCESS' if r.ok else 'FAILED'}")
        if len(w):
            print("\n  dead-letter queue (what was rejected and why):")
            cols = [c for c in ["id", "distance_km", "duration_sec",
                                "avg_heart_rate", "_dq_rule"] if c in w.columns]
            print(w[cols].head(8).to_string(index=False))
    record("3. Corrupt values", "bad rows quarantined with a reason, good rows load",
           f"{len(w)} quarantined, {gold} rows in Gold, run ok", bool(ok))
    assert ok
    return r


def sim_4_schema_drift_is_detected(verbose: bool = True):
    """
    An upstream change removes a column the pipeline depends on.
    Expected: this is NOT retried (retrying cannot grow a column back). The
    rule that needs the column fails, the rows are quarantined with
    'column_missing', and a human is alerted.
    """
    if verbose:
        print(f"\n{BANNER}\nSIMULATION 4 — upstream schema drift\n{BANNER}\n"
              "Fault: the source stops sending distance_km.\n")
    FaultInjector.clear()
    reset_lake()
    with FaultInjector.armed("data.schema_drift", fail_times=1, scope="txn_workout_summaries"):
        r = run_pipeline("init", echo=verbose)

    step = next((s for s in r.metrics.steps
                 if s.layer == "extract" and s.step == "txn_workout_summaries"), None)
    bronze_rows = _delta_rows(TABLES_BY_NAME["txn_workout_summaries"].bronze_path)
    others = _delta_rows(TABLES_BY_NAME["txn_payments"].bronze_path)
    critical = [a for a in r.alerts.alerts if a.severity == "critical"]
    drift_detected = step is not None and "SchemaDrift" in (step.error or "")
    not_retried = step is not None and step.attempts == 1

    ok = drift_detected and not_retried and bronze_rows == 0 and others == 50 and critical
    if verbose:
        print(f"\n  fault detected as      : {step.error.split(':')[0] if step else 'n/a'}")
        print(f"  retry attempts used    : {step.attempts if step else 0} "
              f"(1 is correct — drift is not transient)")
        print(f"  rows written to Bronze : {bronze_rows} "
              f"(0 is correct — a partial table is worse than none)")
        print(f"  other tables unaffected: txn_payments has {others} rows")
        print(f"  critical alerts raised : {len(critical)}")
        print("\n  The alternative would have been to accept the row and backfill")
        print("  distance_km with NULL, which loads cleanly and silently empties")
        print("  every distance measure downstream. Refusing is the safer failure.")
    record("4. Schema drift",
           "detected at the contract check, not retried, table refused, alert raised",
           f"SchemaDriftError on attempt {step.attempts if step else 0}, "
           f"{bronze_rows} rows written, {len(critical)} critical alert(s)", bool(ok))
    assert ok
    return r


def sim_5_target_unavailable(verbose: bool = True):
    """
    The warehouse target rejects writes.
    Expected: transient -> the retry recovers. Permanent -> the run aborts
    with a critical alert rather than leaving a half-built star schema.
    """
    if verbose:
        print(f"\n{BANNER}\nSIMULATION 5 — load target unavailable\n{BANNER}")

    # 5a transient
    FaultInjector.clear()
    reset_lake()
    if verbose:
        print("\n5a. Target fails twice, then recovers.\n")
    with FaultInjector.armed("load.target_unavailable", fail_times=2, scope="dim_user"):
        r1 = run_pipeline("init", echo=verbose)
    gold_a = _delta_rows(GOLD / "fct_workout_session")

    # 5b permanent
    FaultInjector.clear()
    reset_lake()
    if verbose:
        print("\n5b. Target stays down.\n")
    with FaultInjector.armed("load.target_unavailable", fail_times=99, scope="dim_user"):
        r2 = run_pipeline("init", echo=verbose)
    critical = [a for a in r2.alerts.alerts if a.severity == "critical"]

    ok = r1.ok and gold_a > 0 and (not r2.ok) and critical
    if verbose:
        print(f"\n  5a transient : run {'ok' if r1.ok else 'failed'}, "
              f"{gold_a} fact rows written after retry")
        print(f"  5b permanent : run {'ok' if r2.ok else 'aborted'}, "
              f"{len(critical)} critical alert(s)")
    record("5. Load target unavailable",
           "transient recovers; permanent aborts with a critical alert",
           f"transient ok={r1.ok} ({gold_a} rows); permanent ok={r2.ok}, "
           f"{len(critical)} critical", bool(ok))
    assert ok
    return r1, r2


def sim_6_replay_is_idempotent(verbose: bool = True):
    """
    The hardest requirement in any pipeline: running the same batch twice must
    not double-count. Bronze keeps both copies, because Bronze is the
    immutable record of what arrived. Silver deduplicates on the business key,
    so Gold is unchanged.
    """
    if verbose:
        print(f"\n{BANNER}\nSIMULATION 6 — replay / duplicate batch\n{BANNER}\n"
              "Fault: the whole init run is executed a second time.\n")
    FaultInjector.clear()
    reset_lake()
    run_pipeline("init", echo=False)
    bronze_1 = _delta_rows(TABLES_BY_NAME["txn_workout_summaries"].bronze_path)
    silver_1 = _delta_rows(SILVER / "txn_workout_summaries")
    gold_1 = _delta_rows(GOLD / "fct_workout_session")
    revenue_1 = float(DeltaTable(str(GOLD / "fct_payment_transaction"))
                      .to_pandas()["amount_thb"].sum())

    run_pipeline("init", echo=verbose)          # the replay
    bronze_2 = _delta_rows(TABLES_BY_NAME["txn_workout_summaries"].bronze_path)
    silver_2 = _delta_rows(SILVER / "txn_workout_summaries")
    gold_2 = _delta_rows(GOLD / "fct_workout_session")
    revenue_2 = float(DeltaTable(str(GOLD / "fct_payment_transaction"))
                      .to_pandas()["amount_thb"].sum())

    df = pd.DataFrame([
        {"layer": "Bronze (append-only)", "after_run_1": bronze_1, "after_replay": bronze_2,
         "expected": "doubles — Bronze records what arrived"},
        {"layer": "Silver (deduplicated)", "after_run_1": silver_1, "after_replay": silver_2,
         "expected": "unchanged — one row per business key"},
        {"layer": "Gold (facts)", "after_run_1": gold_1, "after_replay": gold_2,
         "expected": "unchanged — no double count"},
        {"layer": "Gold revenue (THB)", "after_run_1": round(revenue_1, 2),
         "after_replay": round(revenue_2, 2), "expected": "identical"},
    ])
    ok = (bronze_2 == 2 * bronze_1 and silver_2 == silver_1
          and gold_2 == gold_1 and abs(revenue_2 - revenue_1) < 0.01)
    if verbose:
        print("\n  REPLAY RESULT")
        print(df.to_string(index=False))
    record("6. Replay idempotency",
           "Bronze doubles, Silver and Gold unchanged, revenue identical",
           f"bronze {bronze_1}->{bronze_2}, silver {silver_1}->{silver_2}, "
           f"gold {gold_1}->{gold_2}, revenue {revenue_1:.0f}->{revenue_2:.0f}", ok)
    assert ok
    return df


# ==========================================================================
def run_all(verbose: bool = True) -> pd.DataFrame:
    RESULTS.clear()
    sim_1_transient_source_failure(verbose)
    sim_2_permanent_source_failure_is_isolated(verbose)
    sim_3_bad_data_is_quarantined(verbose)
    sim_4_schema_drift_is_detected(verbose)
    sim_5_target_unavailable(verbose)
    sim_6_replay_is_idempotent(verbose)
    FaultInjector.clear()
    _fresh_baseline()        # leave the lake in a clean, loaded state

    df = pd.DataFrame(RESULTS)
    if verbose:
        print(f"\n{BANNER}\nERROR HANDLING SIMULATION RESULTS\n{BANNER}")
        print(df.to_string(index=False))
        print(f"\n{(df.result == 'PASS').sum()}/{len(df)} simulations behaved as designed")
    return df


if __name__ == "__main__":
    out = run_all()
    sys.exit(0 if (out.result == "PASS").all() else 1)
