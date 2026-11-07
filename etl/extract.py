"""
extract.py — single responsibility: move rows out of the OLTP database into
Bronze, unchanged.

Bronze is immutable and append-only. Nothing is cleaned, deduplicated or
reshaped here; that is Silver's job. The only columns added are lineage
columns, and they are prefixed with _ so they can never collide with a source
column.

Two load types:
  init         read every row, once, and record the high-water mark
  incremental  read only rows whose watermark column is newer than the mark
"""
from __future__ import annotations

import json
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import psycopg2
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from .config import CONTROL, SOURCE_DSN, RunConfig, SourceTable
from .errors import (DeadLetterQueue, ErrorLog, SchemaDriftError,
                     SourceUnavailable, retry)
from .faults import FaultInjector
from .lineage import LineageRegistry, new_batch_id, stamp
from .metrics import MetricsCollector

# --------------------------------------------------------------------------
# Watermark store — the state that makes incremental load possible
# --------------------------------------------------------------------------
WATERMARK_PATH = Path(CONTROL) / "watermarks.json"

# Extraction runs several tables at once, so every read-modify-write of the
# watermark file has to be serialised. Without this lock two threads can read
# the same state, write back over each other, and leave the file half-written
# for a third thread to parse.
_WM_LOCK = threading.Lock()


def read_watermarks() -> dict[str, str]:
    with _WM_LOCK:
        return _read_watermarks_unlocked()


def _read_watermarks_unlocked() -> dict[str, str]:
    if not WATERMARK_PATH.exists():
        return {}
    try:
        return json.loads(WATERMARK_PATH.read_text() or "{}")
    except json.JSONDecodeError:
        return {}


def write_watermark(table: str, value) -> None:
    with _WM_LOCK:
        marks = _read_watermarks_unlocked()
        marks[table] = value.isoformat() if hasattr(value, "isoformat") else str(value)
        # write to a temp file and rename, so a reader never sees a partial file
        tmp = WATERMARK_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(marks, indent=2, sort_keys=True))
        os.replace(tmp, WATERMARK_PATH)


def reset_watermarks() -> None:
    if WATERMARK_PATH.exists():
        WATERMARK_PATH.unlink()


# --------------------------------------------------------------------------
# Type mapping — an explicit Arrow schema per table keeps Bronze stable even
# when a batch happens to be all-NULL in some column.
# --------------------------------------------------------------------------
_PG_TO_ARROW = {
    "integer": pa.int64(), "bigint": pa.int64(), "smallint": pa.int64(),
    "numeric": pa.float64(), "double precision": pa.float64(), "real": pa.float64(),
    "boolean": pa.bool_(),
    "character varying": pa.string(), "text": pa.string(), "character": pa.string(),
    "uuid": pa.string(), "time without time zone": pa.string(),
    "date": pa.date32(),
    "timestamp with time zone": pa.timestamp("us", tz="UTC"),
    "timestamp without time zone": pa.timestamp("us"),
}

_LINEAGE_SCHEMA = [
    ("_lineage_id", pa.string()), ("_run_id", pa.string()), ("_batch_id", pa.string()),
    ("_src_system", pa.string()), ("_src_table", pa.string()), ("_src_pk", pa.string()),
    ("_extracted_at", pa.timestamp("us", tz="UTC")), ("_load_type", pa.string()),
    ("_ingest_date", pa.string()),
]


def describe_source(conn, table: str) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position",
            (table,),
        )
        return cur.fetchall()


def arrow_schema_for(cols: list[tuple[str, str]]) -> pa.Schema:
    fields = [pa.field(n, _PG_TO_ARROW.get(t, pa.string())) for n, t in cols]
    fields += [pa.field(n, t) for n, t in _LINEAGE_SCHEMA]
    return pa.schema(fields)


def _nullsafe(values, is_float: bool):
    """
    NaT, NaN and None all have to reach Arrow as a plain None. Converting the
    Series wholesale fails on mixed object columns and on NaT in particular
    ("NaTType does not support utcoffset"), so the values are normalised one
    by one before the Arrow array is built.
    """
    out = []
    for v in values:
        if v is None or v is pd.NaT:
            out.append(None)
            continue
        try:
            if pd.isna(v):
                out.append(None)
                continue
        except (TypeError, ValueError):
            pass
        if is_float and isinstance(v, Decimal):
            out.append(float(v))
        else:
            out.append(v)
    return out


def _stringify(values):
    """PostgreSQL `time` and a few other types arrive as Python objects that
    Arrow will not accept into a string column. Render them, keeping None."""
    return [None if v is None else (v if isinstance(v, str) else str(v)) for v in values]


def _coerce(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """Cast a frame to the declared schema, tolerating all-null columns."""
    arrays, names = [], []
    for field in schema:
        if field.name in df.columns:
            vals = _nullsafe(df[field.name].tolist(), pa.types.is_floating(field.type))
            if pa.types.is_string(field.type):
                vals = _stringify(vals)
            arrays.append(pa.array(vals, type=field.type))
        else:
            arrays.append(pa.nulls(len(df), type=field.type))
        names.append(field.name)
    return pa.Table.from_arrays(arrays, names=names).cast(schema)


# --------------------------------------------------------------------------
# One table
# --------------------------------------------------------------------------
def extract_table(tbl: SourceTable, cfg: RunConfig, run_id: str,
                  log: ErrorLog, metrics: MetricsCollector,
                  lineage: LineageRegistry, dlq: DeadLetterQueue) -> dict:
    """Extract one OLTP table into Bronze. Returns a small result dict."""
    batch_id = new_batch_id(run_id, tbl.name)
    marks = read_watermarks()
    since = marks.get(tbl.name) if cfg.load_type == "incremental" else None

    with metrics.step("extract", tbl.name) as m:
        m.load_type = cfg.load_type
        _t0 = time.perf_counter()

        def _read():
            # Fault point 1: the source database is unreachable.
            FaultInjector.check("extract.source_unavailable", scope=tbl.name)
            FaultInjector.check("extract.timeout", scope=tbl.name)
            try:
                conn = psycopg2.connect(SOURCE_DSN, connect_timeout=10)
            except psycopg2.Error as exc:
                raise SourceUnavailable(str(exc)) from exc
            try:
                cols = describe_source(conn, tbl.name)
                colnames = [c for c, _ in cols]
                sql = f'SELECT {", ".join(colnames)} FROM {tbl.name}'
                params: tuple = ()
                if since is not None:
                    sql += f" WHERE {tbl.watermark_col} > %s"
                    params = (since,)
                sql += f" ORDER BY {tbl.watermark_col}"
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                return pd.DataFrame(rows, columns=colnames), cols
            finally:
                conn.close()

        (df, cols), attempts = retry(
            _read, attempts=cfg.retry_attempts, base_delay=cfg.retry_base_delay,
            max_delay=cfg.retry_max_delay, logger=log, label=f"extract {tbl.name}",
        )
        m.attempts = attempts
        m.rows_in = len(df)

        # Fault point 2: the data itself is wrong or the shape changed.
        df = FaultInjector.corrupt("data.bad_values", df, scope=tbl.name)
        df = FaultInjector.corrupt("data.duplicates", df, scope=tbl.name)
        df = FaultInjector.corrupt("data.orphan_keys", df, scope=tbl.name)
        df = FaultInjector.corrupt("data.schema_drift", df, scope=tbl.name)

        # Schema contract check. The Arrow schema below would happily backfill
        # a vanished column with NULLs, which is the worst outcome: the load
        # succeeds and every downstream measure is quietly empty. A column
        # that the source contract declares and the extract did not return is
        # drift, and drift is a non-retryable fault for a human to resolve.
        declared = {c for c, _ in cols}
        missing = declared - set(df.columns)
        if missing:
            raise SchemaDriftError(
                f"{tbl.name}: source contract declares {sorted(missing)} but the "
                f"extract returned {len(df.columns)} columns without them")

        if df.empty:
            m.rows_out = 0
            log.info(f"{tbl.name}: no new rows since {since}")
            return {"table": tbl.name, "rows": 0, "version": None}

        df = stamp(df, run_id=run_id, batch_id=batch_id, src_table=tbl.name,
                   pk_cols=tbl.pk, load_type=cfg.load_type)
        df["_ingest_date"] = datetime.now(timezone.utc).date().isoformat()

        schema = arrow_schema_for(cols)
        table_arrow = _coerce(df, schema)

        write_deltalake(tbl.bronze_path, table_arrow, mode="append",
                        partition_by=["_ingest_date"], schema_mode="merge")

        dt = DeltaTable(tbl.bronze_path)
        version = dt.version()
        m.rows_out = len(df)
        m.bytes_out = table_arrow.nbytes

        # advance the watermark only after a successful write
        wm = df[tbl.watermark_col].max()
        if pd.notna(wm):
            write_watermark(tbl.name, wm)

        lineage.record(
            load_type=cfg.load_type,
            source_object=f"oltp.{tbl.name}", source_version="oltp",
            target_object=f"bronze.{tbl.name}", target_version=str(version),
            operation="append", rows_in=m.rows_in, rows_out=m.rows_out, rows_rejected=0,
        )
        # m.duration_sec is only finalised when the context manager exits, so
        # the rate shown here is measured locally.
        rate = m.rows_out / max(1e-9, time.perf_counter() - _t0)
        log.info(f"{tbl.name}: {m.rows_out} rows -> bronze v{version} ({rate:,.0f} rows/s)")
        return {"table": tbl.name, "rows": m.rows_out, "version": version}


# --------------------------------------------------------------------------
# All tables, in parallel
# --------------------------------------------------------------------------
def extract_all(cfg: RunConfig, run_id: str, log: ErrorLog,
                metrics: MetricsCollector, lineage: LineageRegistry,
                dlq: DeadLetterQueue) -> pd.DataFrame:
    """
    Parallel extraction (course slide: Parallel Extraction). Tables are
    independent of one another at this stage, so they are read concurrently;
    one table failing does not abort the others unless fail_fast is set.
    """
    log.stage = f"extract:{cfg.load_type}"
    results, failures = [], []
    tables = cfg.selected()

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {
            pool.submit(extract_table, t, cfg, run_id, log, metrics, lineage, dlq): t
            for t in tables
        }
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                log.error(f"{t.name}: extraction failed after retries — {type(exc).__name__}: {exc}")
                failures.append({"table": t.name, "error": f"{type(exc).__name__}: {exc}"})
                if cfg.fail_fast:
                    raise

    if failures:
        dlq.put(pd.DataFrame(failures), stage="extract", reason="table_level_failure")

    return pd.DataFrame(results or [{"table": None, "rows": 0, "version": None}])
