"""
faults.py — single responsibility: break things on purpose, deterministically.

This is test scaffolding, not production code, but it lives beside the
pipeline so the error-handling tests exercise the REAL code paths rather than
a mock of them. Nothing here runs unless a fault is explicitly armed.

Usage:
    with FaultInjector.armed("extract.source_unavailable", fail_times=2):
        run_pipeline(...)        # fails twice, the retry succeeds on the 3rd
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

from .errors import (
    SourceUnavailable, TargetUnavailable, SchemaDriftError, TransientError,
)


@dataclass
class Fault:
    name: str
    fail_times: int = 1         # how many calls fail before it starts succeeding
    scope: str | None = None    # restrict to one table; None = any
    triggered: int = 0

    def should_fire(self, scope: str | None) -> bool:
        if self.scope is not None and scope != self.scope:
            return False
        if self.triggered >= self.fail_times:
            return False
        self.triggered += 1
        return True


class FaultInjector:
    """Process-wide registry of armed faults. Thread-safe for parallel extract."""

    _lock = threading.Lock()
    _armed: dict[str, Fault] = {}

    # -- arming ------------------------------------------------------------
    @classmethod
    @contextmanager
    def armed(cls, name: str, *, fail_times: int = 1, scope: str | None = None):
        f = Fault(name=name, fail_times=fail_times, scope=scope)
        with cls._lock:
            cls._armed[name] = f
        try:
            yield f
        finally:
            with cls._lock:
                cls._armed.pop(name, None)

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._armed.clear()

    # -- checking ----------------------------------------------------------
    @classmethod
    def check(cls, name: str, scope: str | None = None):
        """
        Called from pipeline code at the points where a real failure could
        occur. Raises the exception type that the real fault would raise, so
        the retry / DLQ / alert logic sees exactly what it would see in
        production.
        """
        with cls._lock:
            f = cls._armed.get(name)
            fire = f.should_fire(scope) if f else False
        if not fire:
            return
        where = f" [{scope}]" if scope else ""
        raise _EXCEPTIONS[name](f"injected fault '{name}'{where} "
                                f"(call {f.triggered} of {f.fail_times})")

    @classmethod
    def corrupt(cls, name: str, df, scope: str | None = None):
        """
        Data-level faults mutate the frame instead of raising, because that is
        how bad data actually arrives: quietly, in otherwise valid rows.
        """
        with cls._lock:
            f = cls._armed.get(name)
            fire = f.should_fire(scope) if f else False
        if not fire or df is None or df.empty:
            return df
        return _CORRUPTORS[name](df)


# --------------------------------------------------------------------------
# What each named fault does
# --------------------------------------------------------------------------
_EXCEPTIONS = {
    "extract.source_unavailable": SourceUnavailable,
    "extract.timeout":            SourceUnavailable,
    "transform.crash":            TransientError,
    "load.target_unavailable":    TargetUnavailable,
    "extract.schema_drift":       SchemaDriftError,
}


def _corrupt_values(df):
    """Impossible measures in otherwise valid rows — the Silver DQ rules must catch these."""
    out = df.copy()
    n = min(3, len(out))
    idx = out.index[:n]
    if "distance_km" in out.columns:
        out.loc[idx, "distance_km"] = [-5.0, 9999.0, 0.0][:n]
    if "duration_sec" in out.columns:
        out.loc[idx, "duration_sec"] = [0, 86_400 * 3, -1][:n]
    if "amount" in out.columns:
        out.loc[idx, "amount"] = [-1.0, 0.0, -999.0][:n]
    return out


def _drop_column(df):
    """A column disappears from the source — classic upstream schema drift."""
    out = df.copy()
    for cand in ("distance_km", "amount", "tier", "status"):
        if cand in out.columns:
            return out.drop(columns=[cand])
    return out


def _duplicate_rows(df):
    """The source or the CDC connector replays a batch."""
    import pandas as pd
    return pd.concat([df, df.head(max(1, len(df) // 10))], ignore_index=True)


def _null_keys(df):
    """A foreign key arrives null or pointing at nothing."""
    out = df.copy()
    for cand in ("user_id", "device_id", "event_id"):
        if cand in out.columns and len(out) > 2:
            out.loc[out.index[:2], cand] = None
            break
    return out


_CORRUPTORS = {
    "data.bad_values":    _corrupt_values,
    "data.schema_drift":  _drop_column,
    "data.duplicates":    _duplicate_rows,
    "data.orphan_keys":   _null_keys,
}

FAULT_CATALOGUE = {
    **{k: "raises " + v.__name__ for k, v in _EXCEPTIONS.items()},
    "data.bad_values":   "writes impossible measures into otherwise valid rows",
    "data.schema_drift": "removes a column from the extracted frame",
    "data.duplicates":   "replays 10% of the batch as duplicate rows",
    "data.orphan_keys":  "nulls a foreign key on the first two rows",
}
