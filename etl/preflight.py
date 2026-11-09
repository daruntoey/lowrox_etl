"""
preflight.py — single responsibility: say what state the lake is actually in.

"fct_workout_session is empty" is true but useless: it does not say whether the
pipeline was never run, ran against an empty source, rejected every row at the
quality gate, or wrote somewhere else entirely. Those four have four different
fixes, and this module tells them apart.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from deltalake import DeltaTable

from .config import BRONZE, GOLD, LAKE_ROOT, SILVER, SOURCE_TABLES, source_dsn


# Which Silver table each Gold fact is built from. Checking the specific
# feeder rather than the Silver total is the difference between "the load
# failed" and "that one table was rejected at the quality gate", which are
# different problems with different fixes.
FEEDS = {
    "fct_workout_session": "txn_workout_summaries",
    "fct_payment_transaction": "txn_payments",
}


def _rows(path) -> int:
    p = Path(path)
    if not (p / "_delta_log").exists():
        return -1                       # table does not exist at all
    try:
        return len(DeltaTable(str(p)).to_pandas())
    except Exception:
        return -2                       # exists but unreadable


def source_counts() -> dict[str, int]:
    """Row count per OLTP table, or {} if the source cannot be reached."""
    import psycopg2
    try:
        with psycopg2.connect(source_dsn(), connect_timeout=20) as c, c.cursor() as cur:
            out = {}
            for t in SOURCE_TABLES:
                cur.execute(f"SELECT count(*) FROM {t.name}")
                out[t.name] = cur.fetchone()[0]
            return out
    except Exception:
        return {}


def lake_status() -> pd.DataFrame:
    """One row per zone, with how many tables and rows are in it."""
    zones = []
    for name, root in (("bronze", BRONZE), ("silver", SILVER), ("gold", GOLD)):
        if not Path(root).exists():
            zones.append({"zone": name, "tables": 0, "rows": 0, "exists": False})
            continue
        tables = [p for p in Path(root).iterdir()
                  if p.is_dir() and (p / "_delta_log").exists()]
        zones.append({"zone": name, "tables": len(tables),
                      "rows": sum(max(0, _rows(p)) for p in tables), "exists": True})
    return pd.DataFrame(zones)


def diagnose(verbose: bool = True) -> dict:
    """Full state report. Returns a dict so a caller can act on it."""
    src = source_counts()
    lake = lake_status()
    state = {
        "lake_root": str(LAKE_ROOT),
        "source_reachable": bool(src),
        "source_rows": sum(src.values()) if src else 0,
        "bronze_rows": int(lake.loc[lake.zone == "bronze", "rows"].iloc[0]),
        "silver_rows": int(lake.loc[lake.zone == "silver", "rows"].iloc[0]),
        "gold_rows": int(lake.loc[lake.zone == "gold", "rows"].iloc[0]),
        "fct_workout_session": _rows(GOLD / "fct_workout_session"),
    }

    if verbose:
        print("=" * 72)
        print("PIPELINE STATE")
        print("=" * 72)
        print(f"lake root        : {state['lake_root']}")
        print(f"source           : {source_dsn().split('@')[-1]}")
        if src:
            print(f"source rows      : {state['source_rows']} across {len(src)} tables")
            empty = [k for k, v in src.items() if v == 0]
            if empty:
                print(f"  EMPTY TABLES   : {', '.join(empty)}")
        else:
            print("source rows      : UNREACHABLE")
        print()
        print(lake.to_string(index=False))
        print()
        print(f"fct_workout_session rows: "
              f"{'table does not exist' if state['fct_workout_session'] == -1 else state['fct_workout_session']}")
        print("=" * 72)
    return state


# --------------------------------------------------------------------------
def require_gold(table: str = "fct_workout_session") -> None:
    """
    Raise a message that names the actual problem and the actual fix.
    Called by the test suites before they assume Gold is populated.
    """
    n = _rows(GOLD / table)
    if n > 0:
        return

    state = diagnose(verbose=False)
    lines = [f"Gold table '{table}' is empty, so this test cannot run.", ""]

    if not state["source_reachable"]:
        lines += [
            "CAUSE: the source database cannot be reached at all.",
            f"       {source_dsn().split('@')[-1]}",
            "FIX:   fix the connection string in the setup cell (Part 0) and",
            "       re-run from there.",
        ]
    elif state["source_rows"] == 0:
        lines += [
            "CAUSE: the source database is reachable but every table is empty.",
            "FIX:   load the seed data into it first. In the Supabase SQL editor,",
            "       run sql/01_schema.sql then sql/02_seed.sql (both are in this",
            "       project), then re-run Part 2.",
        ]
    elif state["bronze_rows"] == 0:
        lines += [
            "CAUSE: the source has data but nothing was extracted — Part 2 (the",
            "       INIT LOAD) has not run in this session, or it failed.",
            "FIX:   run the Part 2 cell. If it prints [COMPLETED WITH ERRORS]",
            "       rather than [SUCCESS], read the errors it lists: the run",
            "       does not raise, it reports.",
            "",
            "       Cells must run in order. Running this cell straight after",
            "       Part 0 always lands here, because the lake is still empty.",
            "       Runtime -> Run all is the reliable way.",
        ]
    elif _rows(SILVER / FEEDS.get(table, "")) <= 0:
        feeder = FEEDS.get(table, "(unknown)")
        lines += [
            f"CAUSE: {state['bronze_rows']} rows reached Bronze, but silver/{feeder}",
            "       — the table this fact is built from — is empty. Every row was",
            "       rejected at the quality gate.",
            "FIX:   look at the dead-letter queue to see which rule rejected them:",
            "           from etl.errors import DeadLetterQueue",
            "           DeadLetterQueue.read()[['_dlq_stage', '_dq_rule']]",
            "       A rule reported as 'column_missing' means the source schema",
            "       does not match what the pipeline expects — check that",
            "       sql/01_schema.sql is the version that created your tables.",
        ]
    else:
        lines += [
            f"CAUSE: Silver holds {state['silver_rows']} rows but the Gold load",
            "       produced nothing. This is a load failure, not a data problem.",
            "FIX:   re-run Part 2 with echo=True and read the load stage:",
            "           run_pipeline('init', echo=True)",
        ]

    lines += ["", f"lake root being read: {state['lake_root']}",
              "(if that is not where Part 2 wrote, LOWROX_LAKE_ROOT changed "
              "between cells)"]
    raise RuntimeError("\n" + "\n".join(lines))


def connect(timeout: int = 20):
    """
    Open a connection, turning a driver error into an explanation.

    The test suites query the source directly, and a raw psycopg2
    OperationalError there is a wall of traceback that says nothing about
    which of the three likely causes it is.
    """
    import psycopg2
    from .errors import AuthenticationError, CircuitBreakerOpen, classify_connection_error
    try:
        return psycopg2.connect(source_dsn(), connect_timeout=timeout)
    except psycopg2.Error as exc:
        err = classify_connection_error(exc)
        host = source_dsn().split("@")[-1]
        if isinstance(err, CircuitBreakerOpen):
            raise RuntimeError(
                f"\nThe pooler is blocking connections from this address.\n"
                f"  host: {host}\n\n"
                "Supabase blocks an IP for 120 seconds after 10 failed\n"
                "authentications in 150 seconds, and each further attempt\n"
                "extends it. Stop, wait two full minutes, and fix the password\n"
                "before trying again.") from None
        if isinstance(err, AuthenticationError):
            raise RuntimeError(
                f"\nThe database rejected the credentials.\n"
                f"  host: {host}\n"
                f"  {str(exc).strip().splitlines()[0]}\n\n"
                "'password authentication failed' means the username and project\n"
                "WERE accepted — a wrong username gives 'Tenant or user not\n"
                "found' instead. So the password itself is wrong.\n\n"
                "  Supabase -> Settings -> Database -> Reset database password\n"
                "  Use the DATABASE password, not your Supabase account password.\n"
                "  Percent-encode it if it contains  # / ? @ : % & or a space.\n\n"
                "Do not retry more than a couple of times: 10 failures in 150s\n"
                "locks this address out for 2 minutes.") from None
        raise RuntimeError(
            f"\nCannot reach the source database.\n  host: {host}\n"
            f"  {str(exc).strip().splitlines()[0]}") from None
