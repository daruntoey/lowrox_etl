"""
pipeline.py — single responsibility: sequence the phases and report.

Knows the ORDER of work and nothing about how any phase does its job. That is
what lets extract, transform and load be tested, replaced or re-run on their
own (course slide: Modular Design, clear interfaces between modules).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import BRONZE, CONTROL, GOLD, LAKE_ROOT, SILVER, RunConfig
from .errors import Alerter, DeadLetterQueue, ErrorLog
from .extract import extract_all, read_watermarks, reset_watermarks
from .lineage import LineageRegistry, new_run_id
from .load import load_gold
from .metrics import MetricsCollector
from .transform import transform_all


@dataclass
class RunResult:
    run_id: str
    load_type: str
    ok: bool
    extract: pd.DataFrame
    silver: pd.DataFrame
    gold: dict
    metrics: MetricsCollector
    lineage: LineageRegistry
    log: ErrorLog
    alerts: Alerter
    dlq_rows: int = 0

    def summary(self) -> pd.DataFrame:
        return self.metrics.layer_summary()

    def __str__(self) -> str:
        state = "SUCCESS" if self.ok else "COMPLETED WITH ERRORS"
        return (f"[{state}] {self.run_id} ({self.load_type})  "
                f"{self.metrics.total_seconds:.2f}s  "
                f"dlq={self.dlq_rows}  alerts={len(self.alerts.alerts)}")


def reset_lake(keep_control: bool = False) -> None:
    """Wipe the lake. Used before an init load in the demo, never in production."""
    for p in (BRONZE, SILVER, GOLD):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    if not keep_control:
        if CONTROL.exists():
            shutil.rmtree(CONTROL)
        CONTROL.mkdir(parents=True, exist_ok=True)
    reset_watermarks()


def run_pipeline(load_type: str = "init", *, tables: tuple[str, ...] = (),
                 max_workers: int = 4, fail_fast: bool = False,
                 echo: bool = True, retry_attempts: int = 3) -> RunResult:
    """
    One end-to-end run.

    load_type 'init'        read everything, build the lake from nothing
    load_type 'incremental' read only what changed since the watermark
    """
    cfg = RunConfig(load_type=load_type, tables=tables, max_workers=max_workers,
                    fail_fast=fail_fast, retry_attempts=retry_attempts)
    run_id = new_run_id()

    log = ErrorLog(run_id, echo=echo)
    metrics = MetricsCollector(run_id, load_type)
    lineage = LineageRegistry(run_id)
    alerts = Alerter(run_id)
    dlq = DeadLetterQueue(run_id, log)

    log.stage = "start"
    marks = read_watermarks()
    log.info(f"{load_type} run {run_id}; {len(marks)} watermark(s) on file")

    ok = True
    extract_df = silver_df = pd.DataFrame()
    gold_result: dict = {}

    try:
        extract_df = extract_all(cfg, run_id, log, metrics, lineage, dlq)
        silver_df = transform_all(cfg, run_id, log, metrics, lineage, dlq)
        gold_result = load_gold(cfg, run_id, log, metrics, lineage, dlq)
    except Exception as exc:
        ok = False
        log.stage = "abort"
        log.error(f"run aborted: {type(exc).__name__}: {exc}")
        alerts.fire("critical", f"Pipeline run {run_id} aborted",
                    f"{type(exc).__name__}: {exc}\nload_type={load_type}")

    # -- alerting policy ---------------------------------------------------
    failed_steps = [s for s in metrics.steps if s.status == "failed"]
    if failed_steps:
        ok = False
        alerts.fire("critical", f"{len(failed_steps)} step(s) failed",
                    "\n".join(f"  {s.layer}.{s.step}: {s.error}" for s in failed_steps))
    if dlq.count:
        sev = "critical" if dlq.count > 0.05 * max(1, sum(s.rows_in for s in metrics.steps)) else "warning"
        alerts.fire(sev, f"{dlq.count} row(s) quarantined",
                    f"Inspect {DeadLetterQueue.PATH} and replay once the cause is fixed.")

    metrics.persist()
    log.stage = "done"
    log.info(f"finished in {metrics.total_seconds:.2f}s")

    return RunResult(run_id=run_id, load_type=load_type, ok=ok, extract=extract_df,
                     silver=silver_df, gold=gold_result, metrics=metrics,
                     lineage=lineage, log=log, alerts=alerts, dlq_rows=dlq.count)
