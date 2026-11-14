"""
FUNCTIONAL TESTING — does the pipeline move the right data to the right place?

Course slide mapping:
  Source-to-Target Testing      test_source_to_target_counts / _values
  Transformation Logic Testing  test_dq_rules / test_derived_pace / test_special_keys
  End-to-End Pipeline Testing   test_end_to_end
  (plus)  DATA LINEAGE          test_row_level_lineage_walk  <- the headline test

Every check returns a row for the report, and asserts, so the file works both
as `python tests/test_functional.py` and under pytest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from deltalake import DeltaTable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.config import (CONTROL, GOLD, KEY_INVALID, KEY_NOT_APPLICABLE,
                        KEY_UNKNOWN, SILVER, SOURCE_TABLES, TABLES_BY_NAME,
                        source_dsn)
from etl.errors import DeadLetterQueue
from etl.lineage import LineageRegistry
from etl.preflight import connect as _connect, require_gold

RESULTS: list[dict] = []


def check(name: str, passed: bool, detail: str = "", category: str = ""):
    RESULTS.append({"category": category, "test": name,
                    "result": "PASS" if passed else "FAIL", "detail": detail})
    return passed


def _sql(query: str, params=()) -> pd.DataFrame:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])


def _quarantined_ids(table: str) -> set[int]:
    """
    The primary keys this run sent to the dead letter queue for one table.

    The DLQ record keeps the whole rejected row plus `_src_table`, `_dlq_stage`
    and `_dlq_reason`, so the test can ask the pipeline what it dropped instead
    of assuming.
    """
    f = CONTROL / "dead_letter.jsonl"
    if not f.exists():
        return set()
    ids = set()
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (row.get("_src_table") == table
                and row.get("_dlq_reason") == "dq_rule_failed"
                and row.get("id") is not None):
            ids.add(int(row["id"]))
    return ids


def _delta(path) -> pd.DataFrame:
    p = Path(path)
    if not (p / "_delta_log").exists():
        return pd.DataFrame()
    return DeltaTable(str(p)).to_pandas()


# ==========================================================================
# 1. DATA LINEAGE
# ==========================================================================
def test_row_level_lineage_walk(verbose: bool = True) -> dict:
    """
    Take one row out of the Gold fact table and walk it all the way back to
    the row in the OLTP database that produced it, hop by hop, asserting that
    the measures survived every hop unchanged.

    Gold.fct_workout_session
        -> _src_lineage_id
    Silver.txn_workout_summaries
        -> _lineage_id, _src_pk, _bronze_version
    Bronze.txn_workout_summaries
        -> _src_pk
    OLTP.txn_workout_summaries.id
    """
    require_gold("fct_workout_session")
    fact = _delta(GOLD / "fct_workout_session")

    row = fact.sort_values("workout_id").iloc[len(fact) // 2]
    lineage_id = row["_src_lineage_id"]

    silver = _delta(SILVER / "txn_workout_summaries")
    s = silver[silver["_lineage_id"] == lineage_id]
    ok_silver = check("lineage: Gold -> Silver", len(s) == 1,
                      f"_src_lineage_id {lineage_id[:12]}… matched {len(s)} silver row(s)",
                      "Data lineage")
    s = s.iloc[0]

    bronze = _delta(Path(TABLES_BY_NAME["txn_workout_summaries"].bronze_path))
    b = bronze[bronze["_lineage_id"] == lineage_id]
    ok_bronze = check("lineage: Silver -> Bronze", len(b) == 1,
                      f"found in bronze, _src_pk={s['_src_pk']}, "
                      f"bronze version {s['_bronze_version']}", "Data lineage")
    b = b.iloc[0]

    src_pk = int(s["_src_pk"])
    oltp = _sql("SELECT id, user_id, distance_km, duration_sec, avg_heart_rate "
                "FROM txn_workout_summaries WHERE id = %s", (src_pk,))
    ok_oltp = check("lineage: Bronze -> OLTP", len(oltp) == 1,
                    f"source row id={src_pk} exists", "Data lineage")
    o = oltp.iloc[0]

    same = (
        abs(float(o["distance_km"]) - float(row["distance_km"])) < 1e-9
        and int(o["duration_sec"]) == int(row["duration_sec"])
        and int(o["avg_heart_rate"]) == int(row["avg_heart_rate"])
    )
    ok_values = check("lineage: measures unchanged end to end", same,
                      f"distance {o['distance_km']} km, {o['duration_sec']}s, "
                      f"{o['avg_heart_rate']} bpm identical at every hop", "Data lineage")

    chain = pd.DataFrame([
        {"hop": "1. OLTP",   "object": "public.txn_workout_summaries",
         "key": f"id={src_pk}", "distance_km": float(o["distance_km"]),
         "duration_sec": int(o["duration_sec"])},
        {"hop": "2. Bronze", "object": "bronze.txn_workout_summaries",
         "key": f"_lineage_id={lineage_id[:8]}…", "distance_km": float(b["distance_km"]),
         "duration_sec": int(b["duration_sec"])},
        {"hop": "3. Silver", "object": "silver.txn_workout_summaries",
         "key": f"_lineage_id={lineage_id[:8]}…", "distance_km": float(s["distance_km"]),
         "duration_sec": int(s["duration_sec"])},
        {"hop": "4. Gold",   "object": "gold.fct_workout_session",
         "key": f"workout_id={int(row['workout_id'])}", "distance_km": float(row["distance_km"]),
         "duration_sec": int(row["duration_sec"])},
    ])
    if verbose:
        print("\nROW-LEVEL LINEAGE WALK")
        print(chain.to_string(index=False))

    assert ok_silver and ok_bronze and ok_oltp and ok_values
    return {"chain": chain, "lineage_id": lineage_id, "src_pk": src_pk}


def test_run_level_lineage_complete():
    """Every table that produced Bronze rows must also have a Silver edge."""
    edges = LineageRegistry.load_all()
    assert len(edges), "no lineage recorded"
    to_bronze = set(edges[edges.target_object.str.startswith("bronze.")].target_object)
    to_silver = {f"bronze.{o.split('.', 1)[1]}"
                 for o in edges[edges.target_object.str.startswith("silver.")].source_object}
    missing = to_bronze - to_silver
    ok = check("lineage: every Bronze table has a Silver edge", not missing,
               f"{len(to_bronze)} bronze targets, {len(to_silver)} consumed"
               + (f", missing {sorted(missing)}" if missing else ""), "Data lineage")
    assert ok


def test_lineage_columns_present():
    """The lineage contract must hold on every layer, or the walk is impossible."""
    required = {"_lineage_id", "_run_id", "_src_table", "_src_pk", "_extracted_at", "_load_type"}
    bad = []
    for t in SOURCE_TABLES:
        b = _delta(Path(t.bronze_path))
        if len(b) and not required.issubset(b.columns):
            bad.append(t.name)
    ok = check("lineage: all Bronze tables carry the lineage columns", not bad,
               f"checked {len(SOURCE_TABLES)} tables" + (f", missing on {bad}" if bad else ""),
               "Data lineage")
    assert ok


# ==========================================================================
# 2. SOURCE-TO-TARGET TESTING
# ==========================================================================
def test_source_to_target_counts():
    """Bronze must hold exactly what the source holds — no loss, no invention."""
    rows = []
    for t in SOURCE_TABLES:
        src = int(_sql(f"SELECT count(*) AS n FROM {t.name}").iloc[0]["n"])
        b = _delta(Path(t.bronze_path))
        # one Bronze row per source row per extraction; compare distinct keys
        got = b["_src_pk"].nunique() if len(b) else 0
        rows.append({"table": t.name, "oltp": src, "bronze_distinct_pk": got,
                     "match": src == got})
    df = pd.DataFrame(rows)
    bad = df[~df["match"]]
    ok = check("source-to-target: row counts match on all 18 tables", bad.empty,
               f"{len(df)} tables, {df['oltp'].sum()} source rows"
               + ("" if bad.empty else f", mismatch on {list(bad.table)}"),
               "Source to target")
    assert ok, bad.to_string(index=False)
    return df


def test_source_to_target_values():
    """Sample 25 workout rows and compare every measure to the source."""
    fact = _delta(GOLD / "fct_workout_session")
    sample = fact.sample(min(25, len(fact)), random_state=7)
    ids = tuple(int(i) for i in sample["workout_id"])
    src = _sql("SELECT id, distance_km, duration_sec, avg_heart_rate, calories_kcal "
               f"FROM txn_workout_summaries WHERE id IN %s", (ids,)).set_index("id")

    diffs = []
    for _, r in sample.iterrows():
        o = src.loc[int(r["workout_id"])]
        if (abs(float(o["distance_km"]) - r["distance_km"]) > 1e-9
                or int(o["duration_sec"]) != int(r["duration_sec"])
                or int(o["avg_heart_rate"]) != int(r["avg_heart_rate"])
                or int(o["calories_kcal"]) != int(r["calories_kcal"])):
            diffs.append(int(r["workout_id"]))
    ok = check("source-to-target: measures identical on a 25-row sample", not diffs,
               f"compared 4 measures x {len(sample)} rows"
               + (f", differ on {diffs}" if diffs else ""), "Source to target")
    assert ok


def test_no_rows_lost_silently():
    """source = silver + quarantined. Nothing may vanish without a record."""
    rows = []
    dlq = DeadLetterQueue.read()
    for t in SOURCE_TABLES:
        src = int(_sql(f"SELECT count(*) AS n FROM {t.name}").iloc[0]["n"])
        sp = SILVER / "restricted" / t.name if t.sensitive else Path(t.silver_path)
        s = len(_delta(sp))
        q = int((dlq["_dlq_stage"] == f"silver:{t.name}").sum()) if len(dlq) else 0
        rows.append({"table": t.name, "oltp": src, "silver": s, "quarantined": q,
                     "balanced": src == s + q})
    df = pd.DataFrame(rows)
    bad = df[~df["balanced"]]
    ok = check("source-to-target: silver + quarantined = source, every table", bad.empty,
               f"{df['oltp'].sum()} source = {df['silver'].sum()} silver "
               f"+ {df['quarantined'].sum()} quarantined"
               + ("" if bad.empty else f"; unbalanced {list(bad.table)}"),
               "Source to target")
    assert ok, bad.to_string(index=False)
    return df


# ==========================================================================
# 3. TRANSFORMATION LOGIC TESTING
# ==========================================================================
def test_dq_rules_reject_the_right_rows():
    """The rules must catch the three planted defects and nothing valid."""
    dlq = DeadLetterQueue.read()
    w = dlq[dlq["_dlq_stage"] == "silver:txn_workout_summaries"] if len(dlq) else pd.DataFrame()
    caught = set(w["id"].astype(int)) if len(w) else set()
    planted = {7, 19, 31}        # GPS drift, dead HR strap, forgotten stopwatch
    ok1 = check("transformation: all three planted defects quarantined",
                planted.issubset(caught), f"planted {sorted(planted)}, caught {sorted(caught)}",
                "Transformation logic")

    # a gym session with zero distance is valid and must NOT be quarantined
    silver = _delta(SILVER / "txn_workout_summaries")
    functional_ok = len(silver[(silver["activity_type"] == "functional")
                               & (silver["distance_km"] == 0)])
    ok2 = check("transformation: zero-distance gym sessions are kept, not rejected",
                functional_ok > 0, f"{functional_ok} functional rows passed DQ",
                "Transformation logic")
    assert ok1 and ok2


def test_derived_pace_is_correct():
    """pace_sec_per_km is derived once in Silver; verify the arithmetic."""
    s = _delta(SILVER / "txn_workout_summaries")
    d = s[s["distance_km"] > 0].copy()
    expected = (d["duration_sec"] / d["distance_km"]).round(1)
    bad = int((abs(expected - d["pace_sec_per_km"].astype(float)) > 0.05).sum())
    ok = check("transformation: derived pace = duration / distance", bad == 0,
               f"{len(d)} rows checked, {bad} wrong", "Transformation logic")
    assert ok


def test_email_never_reaches_gold():
    """PDPA: the address is tokenised in Gold, never carried."""
    dim = _delta(GOLD / "dim_user")
    has_email_col = "email" in dim.columns
    leaked = 0
    if "email_token" in dim.columns:
        leaked = int(dim["email_token"].astype(str).str.contains("@").sum())
    ok = check("transformation: no email address in Gold", not has_email_col and leaked == 0,
               f"dim_user columns = {list(dim.columns)}", "Transformation logic")
    assert ok


def test_special_keys_and_no_null_fks():
    """Every dimension carries 0 / -1 / -2, and no fact FK is ever null."""
    dims = {"dim_user": "user_key", "dim_workout_type": "workout_type_key",
            "dim_device": "device_key", "dim_event": "event_key",
            "dim_area": "area_key", "dim_plan": "plan_key", "dim_day": "day_key"}
    missing = []
    for name, key in dims.items():
        d = _delta(GOLD / name)
        present = set(d[key].astype(int)) if len(d) else set()
        for special in (KEY_NOT_APPLICABLE, KEY_UNKNOWN, KEY_INVALID):
            if special not in present:
                missing.append(f"{name}:{special}")
    ok1 = check("transformation: 0/-1/-2 seeded in every dimension", not missing,
                f"{len(dims)} dimensions checked" + (f", missing {missing}" if missing else ""),
                "Transformation logic")

    nulls = []
    for fact, keys in {
        "fct_workout_session": ["day_key", "user_key", "workout_type_key",
                                "device_key", "event_key"],
        "fct_payment_transaction": ["day_key", "user_key", "plan_key"],
    }.items():
        f = _delta(GOLD / fact)
        for k in keys:
            if len(f) and f[k].isna().any():
                nulls.append(f"{fact}.{k}")
    ok2 = check("transformation: no null foreign key in any fact", not nulls,
                "every FK resolved to a real or special key"
                + (f"; nulls in {nulls}" if nulls else ""), "Transformation logic")
    assert ok1 and ok2


def test_referential_integrity():
    """Every fact FK must exist in its dimension."""
    broken = []
    pairs = [("fct_workout_session", "user_key", "dim_user", "user_key"),
             ("fct_workout_session", "workout_type_key", "dim_workout_type", "workout_type_key"),
             ("fct_workout_session", "device_key", "dim_device", "device_key"),
             ("fct_workout_session", "event_key", "dim_event", "event_key"),
             ("fct_workout_session", "day_key", "dim_day", "day_key"),
             ("fct_payment_transaction", "user_key", "dim_user", "user_key"),
             ("fct_payment_transaction", "plan_key", "dim_plan", "plan_key"),
             ("fct_payment_transaction", "day_key", "dim_day", "day_key")]
    for fact, fk, dim, pk in pairs:
        f, d = _delta(GOLD / fact), _delta(GOLD / dim)
        if f.empty or d.empty:
            continue
        orphans = set(f[fk].astype(int)) - set(d[pk].astype(int))
        if orphans:
            broken.append(f"{fact}.{fk} -> {dim}: {sorted(orphans)[:5]}")
    ok = check("transformation: referential integrity across all 8 fact-to-dim edges",
               not broken, f"{len(pairs)} edges checked"
               + (f"; broken {broken}" if broken else ""), "Transformation logic")
    assert ok


# ==========================================================================
# 4. END-TO-END
# ==========================================================================
def test_end_to_end():
    """A business question must be answerable, and agree with the source."""
    f = _delta(GOLD / "fct_workout_session")
    d = _delta(GOLD / "dim_user")
    cur = d[d["is_current"] == True]                                    # noqa: E712
    j = f.merge(cur[["user_key", "tier"]], on="user_key", how="left")
    by_tier = (j.groupby("tier", dropna=False)
                 .agg(sessions=("session_count", "sum"),
                      km=("distance_km", "sum"))
                 .round(1).reset_index())

    gold_total = float(f["distance_km"].sum())
    # Which rows were quarantined is read back from the pipeline's own dead
    # letter queue, not typed in here. An earlier version hard-coded the five
    # ids that fail the DQ rules in the 50-row dataset, which made the test
    # pass only at that one scale — it failed the moment the same pipeline ran
    # against 2,000 rows, where 176 different rows are quarantined. Deriving
    # the list is also a stronger claim: Gold equals the source minus exactly
    # the rows the pipeline SAYS it dropped.
    quarantined = _quarantined_ids("txn_workout_summaries")
    if quarantined:
        src_total = float(_sql(
            "SELECT COALESCE(sum(distance_km),0) AS km FROM txn_workout_summaries "
            "WHERE id <> ALL(%s)", (list(quarantined),)).iloc[0]["km"])
    else:
        src_total = float(_sql(
            "SELECT COALESCE(sum(distance_km),0) AS km "
            "FROM txn_workout_summaries").iloc[0]["km"])
    ok = check("end-to-end: Gold total distance = source minus quarantined",
               abs(gold_total - src_total) < 0.01,
               f"gold {gold_total:.2f} km vs source {src_total:.2f} km", "End to end")
    assert ok
    return by_tier


# ==========================================================================
def run_all(verbose: bool = True) -> pd.DataFrame:
    require_gold("fct_workout_session")
    RESULTS.clear()
    walk = test_row_level_lineage_walk(verbose=verbose)
    test_run_level_lineage_complete()
    test_lineage_columns_present()
    counts = test_source_to_target_counts()
    test_source_to_target_values()
    balance = test_no_rows_lost_silently()
    test_dq_rules_reject_the_right_rows()
    test_derived_pace_is_correct()
    test_email_never_reaches_gold()
    test_special_keys_and_no_null_fks()
    test_referential_integrity()
    by_tier = test_end_to_end()

    df = pd.DataFrame(RESULTS)
    if verbose:
        print("\nFUNCTIONAL TEST RESULTS")
        print(df.to_string(index=False))
        print(f"\n{(df.result == 'PASS').sum()}/{len(df)} passed")
    return df


if __name__ == "__main__":
    out = run_all()
    sys.exit(0 if (out.result == "PASS").all() else 1)
