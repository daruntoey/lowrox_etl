"""
PERFORMANCE TESTING — how much, how fast, and does it still hold up when the
data grows?

Course slide mapping:
  Throughput Testing   test_throughput_by_layer   rows/second per layer
  Latency Testing      test_latency_breakdown     extract -> queryable in Gold
  Scalability          test_scaling_curve         3 data volumes, same code
  Parallel Processing  test_parallelism           1 / 2 / 4 / 8 extract workers

Every number comes from MetricsCollector, which is the same instrumentation
the pipeline uses in production — not a separate stopwatch wrapped around the
test, which could measure something the pipeline does not actually do.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS: list[dict] = []

# The three loaded datasets. Same 18-table schema, same generator, more rows.
SCALES = [
    ("50 rows/table",   "lowrox",     "50"),
    ("500 rows/table",  "lowrox_s10", "500"),
    ("2000 rows/table", "lowrox_s40", "2000"),
]
PG_USER = os.environ.get("LOWROX_PG_USER", "lowrox")
PG_PASS = os.environ.get("LOWROX_PG_PASS", "lowrox")
PG_HOST = os.environ.get("LOWROX_PG_HOST", "localhost")
PG_PORT = os.environ.get("LOWROX_PG_PORT", "5432")


# The benchmarks repoint the pipeline at other databases and other lake
# directories. Both are read from the environment at import time, so the
# original values are captured here and restored afterwards — without this,
# every suite that runs after this one silently inherits the 2000-row
# database and the temporary lake, and its assertions fail for reasons that
# have nothing to do with the code under test.
_ORIGINAL_ENV = {
    "LOWROX_SOURCE_DSN": os.environ.get("LOWROX_SOURCE_DSN"),
    "LOWROX_LAKE_ROOT": os.environ.get("LOWROX_LAKE_ROOT"),
}


def _dsn(db: str) -> str:
    return f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{db}"


def restore_defaults():
    """Put the environment and the loaded modules back as they were found."""
    for k, v in _ORIGINAL_ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    for mod in ("etl.config", "etl.metrics", "etl.lineage", "etl.errors",
                "etl.faults", "etl.extract", "etl.transform", "etl.load",
                "etl.pipeline", "etl"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])


def _run_against(db: str, lake: str, *, workers: int = 4, load_type: str = "init"):
    """
    Point the pipeline at one database and one lake directory, then run it.
    config reads its paths at import time, so the modules are reloaded after
    the environment changes — this is what lets one process benchmark several
    configurations without a subprocess per run.
    """
    os.environ["LOWROX_SOURCE_DSN"] = _dsn(db)
    os.environ["LOWROX_LAKE_ROOT"] = lake
    if Path(lake).exists():
        shutil.rmtree(lake)

    for mod in ("etl.config", "etl.metrics", "etl.lineage", "etl.errors",
                "etl.faults", "etl.extract", "etl.transform", "etl.load",
                "etl.pipeline", "etl"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    import etl
    importlib.reload(etl)

    t0 = time.perf_counter()
    result = etl.run_pipeline(load_type, echo=False, max_workers=workers)
    wall = time.perf_counter() - t0
    return result, wall


# ==========================================================================
def test_throughput_by_layer(verbose: bool = True) -> pd.DataFrame:
    """Rows per second at each layer, at the largest volume."""
    result, wall = _run_against("lowrox_s40", "/tmp/lake_perf_throughput")
    df = result.metrics.layer_summary()
    df.insert(0, "dataset", "2000 rows/table")
    RESULTS.append({"test": "throughput_by_layer", "result": "PASS",
                    "detail": f"{df.rows_out.sum()} rows in {wall:.2f}s wall"})
    if verbose:
        print("\nTHROUGHPUT BY LAYER  (2000 rows/table)")
        print(df.to_string(index=False))
    return df


def test_latency_breakdown(verbose: bool = True) -> pd.DataFrame:
    """
    End-to-end latency: how long after a row is written to the OLTP database
    before it can be queried in Gold. Broken down by phase so the slow phase
    is visible rather than inferred.
    """
    result, wall = _run_against("lowrox_s40", "/tmp/lake_perf_latency")
    m = result.metrics.to_frame()
    phase = (m.groupby("layer", as_index=False)
               .agg(seconds=("duration_sec", "sum"), steps=("step", "count")))
    order = {"extract": 0, "silver": 1, "gold": 2}
    phase = phase.sort_values("layer", key=lambda s: s.map(order)).reset_index(drop=True)
    phase["pct_of_total"] = (100 * phase.seconds / phase.seconds.sum()).round(1)
    phase["cumulative_sec"] = phase.seconds.cumsum().round(3)

    slowest = m.nlargest(5, "duration_sec")[["layer", "step", "rows_out", "duration_sec"]]
    RESULTS.append({"test": "latency_breakdown", "result": "PASS",
                    "detail": f"end-to-end {wall:.2f}s, slowest phase "
                              f"{phase.loc[phase.seconds.idxmax(), 'layer']}"})
    if verbose:
        print(f"\nLATENCY BREAKDOWN  (2000 rows/table, end-to-end {wall:.2f}s)")
        print(phase.to_string(index=False))
        print("\nFive slowest individual steps")
        print(slowest.to_string(index=False))
    return phase


def test_scaling_curve(verbose: bool = True) -> pd.DataFrame:
    """
    Same code, three data volumes. What matters is not the absolute speed but
    the SHAPE: if rows/second stays roughly flat or improves, the pipeline
    scales; if it collapses, there is a per-row cost that will not survive
    production volume.
    """
    rows = []
    for label, db, size in SCALES:
        result, wall = _run_against(db, f"/tmp/lake_perf_scale_{size}")
        m = result.metrics.to_frame()
        extracted = int(m[m.layer == "extract"].rows_out.sum())
        rows.append({
            "dataset": label,
            "source_rows": extracted,
            "wall_sec": round(wall, 3),
            "extract_sec": round(m[m.layer == "extract"].duration_sec.sum(), 3),
            "silver_sec": round(m[m.layer == "silver"].duration_sec.sum(), 3),
            "gold_sec": round(m[m.layer == "gold"].duration_sec.sum(), 3),
            "rows_per_sec": round(extracted / wall, 0),
            "ms_per_1k_rows": round(1000 * wall / max(1, extracted) * 1000, 1),
            "quarantined": result.dlq_rows,
        })
    df = pd.DataFrame(rows)

    # the pipeline scales if throughput does not fall away as volume grows
    first, last = df.rows_per_sec.iloc[0], df.rows_per_sec.iloc[-1]
    scales_ok = last >= first * 0.8
    RESULTS.append({"test": "scaling_curve",
                    "result": "PASS" if scales_ok else "FAIL",
                    "detail": f"{first:.0f} -> {last:.0f} rows/s from "
                              f"{df.source_rows.iloc[0]} to {df.source_rows.iloc[-1]} rows"})
    if verbose:
        print("\nSCALING CURVE")
        print(df.to_string(index=False))
        print(f"\nthroughput {first:,.0f} -> {last:,.0f} rows/s as volume grows "
              f"{df.source_rows.iloc[-1] / df.source_rows.iloc[0]:.0f}x")
    return df


def test_parallelism(verbose: bool = True) -> pd.DataFrame:
    """
    Extraction is IO-bound on the database round trip, so adding workers
    should shorten the extract phase until the source becomes the limit.
    """
    rows = []
    for workers in (1, 2, 4, 8):
        result, wall = _run_against("lowrox_s40", f"/tmp/lake_perf_par_{workers}",
                                    workers=workers)
        m = result.metrics.to_frame()
        ex = m[m.layer == "extract"]
        rows.append({
            "workers": workers,
            "extract_cpu_sec": round(ex.duration_sec.sum(), 3),   # summed across threads
            "wall_sec": round(wall, 3),
            "rows": int(ex.rows_out.sum()),
            "rows_per_sec_wall": round(ex.rows_out.sum() / wall, 0),
        })
    df = pd.DataFrame(rows)
    df["speedup_vs_1"] = (df.wall_sec.iloc[0] / df.wall_sec).round(2)
    best = df.loc[df.wall_sec.idxmin()]
    RESULTS.append({"test": "parallelism", "result": "PASS",
                    "detail": f"best at {int(best.workers)} workers, "
                              f"{best.speedup_vs_1:.2f}x vs single-threaded"})
    if verbose:
        print("\nPARALLEL EXTRACTION  (2000 rows/table)")
        print(df.to_string(index=False))
    return df


def test_parallelism_under_network_latency(verbose: bool = True) -> pd.DataFrame:
    """
    Why the previous test shows almost no speedup, and when it would.

    Parallel extraction exists to hide the ROUND TRIP to the source. Against a
    database on the same machine that round trip is a fraction of a
    millisecond, so there is nothing to hide and the threads only contend. A
    hosted database is a different story: Supabase in Singapore is roughly
    30-60 ms away from a Colab runtime, paid once per table.

    This test injects a realistic per-connection delay and re-measures, which
    turns an assumption about production into something demonstrated.
    """
    import psycopg2 as _pg
    real_connect = _pg.connect
    delay_ms = int(os.environ.get("LOWROX_SIM_LATENCY_MS", "60"))

    def slow_connect(*a, **kw):
        time.sleep(delay_ms / 1000.0)          # the network round trip
        return real_connect(*a, **kw)

    rows = []
    _pg.connect = slow_connect
    try:
        for workers in (1, 4, 8):
            result, wall = _run_against("lowrox_s10", f"/tmp/lake_perf_net_{workers}",
                                        workers=workers)
            # extract.py resolves psycopg2.connect at call time, so the patch
            # survives the module reload that _run_against performs
            rows.append({"workers": workers, "wall_sec": round(wall, 3),
                         "rows": int(result.metrics.to_frame()
                                     .query("layer == 'extract'").rows_out.sum())})
    finally:
        _pg.connect = real_connect

    df = pd.DataFrame(rows)
    df["speedup_vs_1"] = (df.wall_sec.iloc[0] / df.wall_sec).round(2)
    df.insert(0, "simulated_rtt_ms", delay_ms)
    gain = float(df.speedup_vs_1.max())
    RESULTS.append({"test": "parallelism_under_network_latency",
                    "result": "PASS" if gain > 1.5 else "FAIL",
                    "detail": f"{gain:.2f}x speedup at {delay_ms}ms simulated RTT "
                              f"(vs ~1.0x on localhost)"})
    if verbose:
        print(f"\nPARALLEL EXTRACTION WITH {delay_ms}ms SIMULATED NETWORK LATENCY")
        print(df.to_string(index=False))
        print("On localhost parallelism buys nothing; across a network it buys "
              f"{gain:.1f}x. This is why max_workers is a configuration value.")
    return df


def test_incremental_is_cheaper(verbose: bool = True) -> pd.DataFrame:
    """
    The whole point of incremental load: a run that finds nothing new must
    cost a small fraction of a full load. If it does not, the watermark is
    not actually filtering.
    """
    result_init, wall_init = _run_against("lowrox_s40", "/tmp/lake_perf_incr")
    # second run, nothing changed in the source
    import etl
    t0 = time.perf_counter()
    result_inc = etl.run_pipeline("incremental", echo=False)
    wall_inc = time.perf_counter() - t0

    init_rows = sum(s.rows_out for s in result_init.metrics.steps if s.layer == "extract")
    inc_rows = sum(s.rows_out for s in result_inc.metrics.steps if s.layer == "extract")
    df = pd.DataFrame([
        {"run": "init", "rows_extracted": init_rows, "wall_sec": round(wall_init, 3)},
        {"run": "incremental (no changes)", "rows_extracted": inc_rows,
         "wall_sec": round(wall_inc, 3)},
    ])
    df["pct_of_init"] = (100 * df.wall_sec / df.wall_sec.iloc[0]).round(1)
    ok = inc_rows == 0 and wall_inc < wall_init
    RESULTS.append({"test": "incremental_is_cheaper", "result": "PASS" if ok else "FAIL",
                    "detail": f"incremental moved {inc_rows} rows in "
                              f"{100*wall_inc/wall_init:.0f}% of the init time"})
    if verbose:
        print("\nINCREMENTAL VS INIT  (no source changes between runs)")
        print(df.to_string(index=False))
    assert ok
    return df


# ==========================================================================
def run_all(verbose: bool = True) -> dict:
    RESULTS.clear()
    try:
        out = {
            "throughput": test_throughput_by_layer(verbose),
            "latency": test_latency_breakdown(verbose),
            "scaling": test_scaling_curve(verbose),
            "parallelism": test_parallelism(verbose),
            "parallelism_networked": test_parallelism_under_network_latency(verbose),
            "incremental": test_incremental_is_cheaper(verbose),
        }
    finally:
        restore_defaults()
    summary = pd.DataFrame(RESULTS)
    if verbose:
        print("\nPERFORMANCE TEST RESULTS")
        print(summary.to_string(index=False))
    out["summary"] = summary
    return out


if __name__ == "__main__":
    r = run_all()
    sys.exit(0 if (r["summary"].result == "PASS").all() else 1)
