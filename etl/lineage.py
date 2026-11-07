"""
lineage.py — single responsibility: record where every row came from, and
make the chain walkable afterwards.

Two levels of lineage:

  COARSE (run level)   which run read which source object at which version and
                       wrote which target object at which version.
  FINE  (row level)    every row carries _lineage_id / _src_table / _src_pk,
                       so one Gold row can be traced back to one source row.

The functional test uses the fine level: it picks a fact row, walks Gold ->
Silver -> Bronze -> OLTP, and asserts the measures survived the trip.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import CONTROL, LINEAGE_COLS, SOURCE_SYSTEM


def new_run_id() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"


def new_batch_id(run_id: str, table: str) -> str:
    return f"{run_id}:{table}"


# --------------------------------------------------------------------------
# Row-level lineage
# --------------------------------------------------------------------------
def stamp(df: pd.DataFrame, *, run_id: str, batch_id: str, src_table: str,
          pk_cols: tuple[str, ...], load_type: str) -> pd.DataFrame:
    """
    Attach the lineage columns to an extracted frame.

    _lineage_id is unique per row *version*, not per business key, so a row
    that is extracted twice (an update, or a replayed batch) gets two lineage
    ids and both remain in Bronze. That is what makes Bronze replayable.
    """
    out = df.copy()
    now = datetime.now(timezone.utc)
    out["_lineage_id"] = [uuid.uuid4().hex for _ in range(len(out))]
    out["_run_id"] = run_id
    out["_batch_id"] = batch_id
    out["_src_system"] = SOURCE_SYSTEM
    out["_src_table"] = src_table
    out["_src_pk"] = (
        out[list(pk_cols)].astype(str).agg("|".join, axis=1) if len(out) else pd.Series(dtype="object")
    )
    out["_extracted_at"] = now
    out["_load_type"] = load_type
    return out


def business_cols(df: pd.DataFrame) -> list[str]:
    """The non-lineage columns — i.e. the actual source payload."""
    return [c for c in df.columns if not c.startswith("_")]


# --------------------------------------------------------------------------
# Run-level lineage registry
# --------------------------------------------------------------------------
@dataclass
class LineageEdge:
    run_id: str
    load_type: str
    source_object: str
    source_version: str          # delta version, or 'oltp' for the database
    target_object: str
    target_version: str
    operation: str               # 'append' | 'merge' | 'overwrite'
    rows_in: int
    rows_out: int
    rows_rejected: int
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LineageRegistry:
    """Append-only log of every source -> target edge the pipeline creates."""

    PATH = Path(CONTROL) / "lineage_edges.jsonl"

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.edges: list[LineageEdge] = []

    def record(self, **kw) -> LineageEdge:
        edge = LineageEdge(run_id=self.run_id, **kw)
        self.edges.append(edge)
        with self.PATH.open("a") as fh:
            fh.write(json.dumps(asdict(edge)) + "\n")
        return edge

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(e) for e in self.edges])

    @classmethod
    def load_all(cls) -> pd.DataFrame:
        if not cls.PATH.exists():
            return pd.DataFrame()
        return pd.read_json(cls.PATH, lines=True)

    @classmethod
    def graph_text(cls, run_id: str | None = None) -> str:
        """A readable dependency graph of one run — the lineage diagram, as text."""
        df = cls.load_all()
        if df.empty:
            return "(no lineage recorded)"
        if run_id:
            df = df[df.run_id == run_id]
        lines = []
        for layer_pair, grp in df.groupby(
            df.source_object.str.split(".").str[0] + " -> " + df.target_object.str.split(".").str[0]
        ):
            lines.append(f"\n{layer_pair}")
            for _, r in grp.iterrows():
                arrow = f"  {r.source_object}@{r.source_version} --{r.operation}--> {r.target_object}@{r.target_version}"
                counts = f"in={r.rows_in} out={r.rows_out} rejected={r.rows_rejected}"
                lines.append(f"{arrow:<96} {counts}")
        return "\n".join(lines)
