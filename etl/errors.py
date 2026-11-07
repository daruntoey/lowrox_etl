"""
errors.py — single responsibility: what the pipeline DOES when something breaks.

Implements the four strategies from the course slide:
  Retry mechanism    exponential backoff with jitter, for transient faults only
  Error logging      structured, never stops the pipeline
  Dead-letter queue  failed records are quarantined, not dropped, and are replayable
  Alerting           a pluggable sink; prints here, email/Slack in production
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import CONTROL


# --------------------------------------------------------------------------
# Exception taxonomy — the retry decision depends on the KIND of failure
# --------------------------------------------------------------------------
class PipelineError(Exception):
    """Base. Not retryable unless a subclass says so."""
    retryable = False


class TransientError(PipelineError):
    """Network blip, timeout, pooler busy. Retrying is likely to work."""
    retryable = True


class SourceUnavailable(TransientError):
    pass


class TargetUnavailable(TransientError):
    pass


class SchemaDriftError(PipelineError):
    """The source grew or lost a column. Retrying cannot help; a human must look."""
    retryable = False


class DataQualityError(PipelineError):
    """A row failed a rule. Never retried — the row is quarantined instead."""
    retryable = False


# --------------------------------------------------------------------------
# Retry
# --------------------------------------------------------------------------
def retry(fn, *, attempts: int = 3, base_delay: float = 0.25,
          max_delay: float = 4.0, logger=None, label: str = ""):
    """
    Run fn(), retrying only retryable failures.

    Backoff is exponential with jitter so that several tables retrying at once
    do not resynchronise and hammer the source together.
    Returns (result, attempts_used).
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(), attempt
        except PipelineError as exc:
            last = exc
            if not exc.retryable or attempt == attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random()          # jitter
            if logger:
                logger.warn(f"{label}: {type(exc).__name__} on attempt {attempt}/{attempts}; "
                            f"retrying in {delay:.2f}s")
            time.sleep(delay)
    raise last  # type: ignore[misc]


# --------------------------------------------------------------------------
# Structured error log
# --------------------------------------------------------------------------
@dataclass
class LogEvent:
    run_id: str
    level: str
    stage: str
    message: str
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ErrorLog:
    PATH = Path(CONTROL) / "pipeline_log.jsonl"

    def __init__(self, run_id: str, echo: bool = True):
        self.run_id = run_id
        self.echo = echo
        self.events: list[LogEvent] = []
        self.stage = "init"

    def _emit(self, level: str, message: str):
        ev = LogEvent(self.run_id, level, self.stage, message)
        self.events.append(ev)
        with self.PATH.open("a") as fh:
            fh.write(json.dumps(asdict(ev)) + "\n")
        if self.echo:
            icon = {"INFO": "  ", "WARN": " !", "ERROR": "!!"}.get(level, "  ")
            print(f"{icon} [{level:<5}] {self.stage:<22} {message}")

    def info(self, m): self._emit("INFO", m)
    def warn(self, m): self._emit("WARN", m)
    def error(self, m): self._emit("ERROR", m)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(e) for e in self.events])


# --------------------------------------------------------------------------
# Dead-letter queue
# --------------------------------------------------------------------------
class DeadLetterQueue:
    """
    Rows that cannot be processed are written here with the reason and full
    lineage, so they can be investigated and replayed once the cause is fixed.
    Quarantining instead of failing is what lets one bad row stop being an
    outage for the other 49,999.
    """

    PATH = Path(CONTROL) / "dead_letter.jsonl"

    def __init__(self, run_id: str, logger: ErrorLog | None = None):
        self.run_id = run_id
        self.logger = logger
        self.count = 0

    def put(self, df: pd.DataFrame, *, stage: str, reason: str) -> int:
        if df is None or df.empty:
            return 0
        payload = df.copy()
        payload["_dlq_run_id"] = self.run_id
        payload["_dlq_stage"] = stage
        payload["_dlq_reason"] = reason
        payload["_dlq_at"] = datetime.now(timezone.utc).isoformat()
        with self.PATH.open("a") as fh:
            for rec in payload.to_dict(orient="records"):
                fh.write(json.dumps(rec, default=str) + "\n")
        self.count += len(payload)
        if self.logger:
            self.logger.warn(f"DLQ +{len(payload)} rows ({reason})")
        return len(payload)

    @classmethod
    def read(cls, run_id: str | None = None) -> pd.DataFrame:
        if not cls.PATH.exists():
            return pd.DataFrame()
        df = pd.read_json(cls.PATH, lines=True)
        return df[df._dlq_run_id == run_id] if run_id else df


# --------------------------------------------------------------------------
# Alerting
# --------------------------------------------------------------------------
@dataclass
class Alert:
    severity: str          # 'warning' | 'critical'
    title: str
    detail: str
    run_id: str
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Alerter:
    """
    Collects alerts and hands them to a sink. The default sink prints; swap in
    an email or webhook sink in production without touching pipeline code.
    """

    def __init__(self, run_id: str, sink=None):
        self.run_id = run_id
        self.sink = sink or self._print_sink
        self.alerts: list[Alert] = []

    @staticmethod
    def _print_sink(a: Alert):
        bar = "=" * 72
        print(f"\n{bar}\nALERT [{a.severity.upper()}] {a.title}\n{a.detail}\n{bar}\n")

    def fire(self, severity: str, title: str, detail: str):
        a = Alert(severity, title, detail, self.run_id)
        self.alerts.append(a)
        self.sink(a)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(a) for a in self.alerts])
