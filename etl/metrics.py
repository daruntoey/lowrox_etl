"""
metrics.py — single responsibility: measure, never decide.

Collects one record per pipeline step: rows in, rows out, rows rejected,
bytes, wall time. Everything the performance test reports is computed from
these records, so the numbers in the report and the numbers the pipeline
acted on are the same numbers.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import CONTROL


@dataclass
class StepMetric:
    run_id: str
    layer: str              # 'extract' | 'bronze' | 'silver' | 'gold'
    step: str               # usually the table name
    load_type: str
    rows_in: int = 0
    rows_out: int = 0
    rows_rejected: int = 0
    bytes_out: int = 0
    started_at: str = ""
    ended_at: str = ""
    duration_sec: float = 0.0
    status: str = "ok"      # 'ok' | 'failed' | 'quarantined'
    attempts: int = 1
    error: str = ""

    @property
    def rows_per_sec(self) -> float:
        return self.rows_out / self.duration_sec if self.duration_sec > 0 else 0.0


class MetricsCollector:
    """Holds every StepMetric for one run and persists them for later analysis."""

    def __init__(self, run_id: str, load_type: str):
        self.run_id = run_id
        self.load_type = load_type
        self.steps: list[StepMetric] = []

    @contextmanager
    def step(self, layer: str, step: str):
        """
        Times one step. The caller mutates the yielded metric to record counts;
        an exception is recorded and re-raised, never swallowed here, because
        deciding what to do about a failure is the error module's job.
        """
        m = StepMetric(
            run_id=self.run_id, layer=layer, step=step, load_type=self.load_type,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        t0 = time.perf_counter()
        try:
            yield m
        except Exception as exc:
            m.status = "failed"
            m.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            m.duration_sec = round(time.perf_counter() - t0, 4)
            m.ended_at = datetime.now(timezone.utc).isoformat()
            self.steps.append(m)

    # -- reporting ---------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        if not self.steps:
            return pd.DataFrame()
        df = pd.DataFrame([asdict(s) for s in self.steps])
        df["rows_per_sec"] = (df["rows_out"] / df["duration_sec"]).replace(
            [float("inf"), float("-inf")], 0
        ).fillna(0).round(0)
        return df

    def layer_summary(self) -> pd.DataFrame:
        df = self.to_frame()
        if df.empty:
            return df
        g = df.groupby("layer", as_index=False).agg(
            steps=("step", "count"),
            rows_out=("rows_out", "sum"),
            rows_rejected=("rows_rejected", "sum"),
            duration_sec=("duration_sec", "sum"),
        )
        g["rows_per_sec"] = (g["rows_out"] / g["duration_sec"]).round(0)
        order = {"extract": 0, "bronze": 1, "silver": 2, "gold": 3}
        return g.sort_values("layer", key=lambda s: s.map(order)).reset_index(drop=True)

    @property
    def total_seconds(self) -> float:
        return round(sum(s.duration_sec for s in self.steps), 4)

    @property
    def rows_landed(self) -> int:
        return sum(s.rows_out for s in self.steps if s.layer == "gold")

    def persist(self) -> Path:
        path = Path(CONTROL) / "run_metrics.jsonl"
        with path.open("a") as fh:
            for s in self.steps:
                fh.write(json.dumps(asdict(s)) + "\n")
        return path
