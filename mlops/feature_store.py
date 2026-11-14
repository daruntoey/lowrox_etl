"""
feature_store.py — the centralised repository of curated features.

Chapter 7, slide 22: a feature store sits between the raw data and the models,
turning raw data into features used for BOTH training and inference, so that
the features are consistent across the two. That consistency is the whole
point: a model trained on one definition of "sessions in the last 30 days" and
served another will fail in production in a way no test catches.

Slide 23 gives the shape of a feature set:

    index (entity)  +  time index (ts)  +  features

Slide 24 gives the join: a label table (user_id, ts, is_churn) is enriched from
several feature sets to produce training data. The enrichment MUST be
point-in-time correct — each label row may only see feature values that existed
at or before its own ts. Doing it with an ordinary join instead leaks the
future into the training set, and the model looks excellent until it is
deployed.

Two stores, one definition:

    OFFLINE   full history, append-only, used to build training sets
    ONLINE    the latest row per entity, used to serve predictions

Both are written from the same transformation, which is what makes
training/serving skew a thing that cannot happen here by accident.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

# --------------------------------------------------------------------------
FEATURE_STORE_ROOT = Path("/content/feature_store")   # overridden by config

# One resolution for every timestamp in the store.
#
# This is not housekeeping — it decides whether the point-in-time join runs at
# all. pandas stores a datetime column at a resolution (ns, us, ms, s), and
# merge_asof refuses to join two keys that disagree, with
#
#     MergeError: incompatible merge keys, must be the same type
#
# while an ordinary merge quietly casts. The two sides disagree by default:
# labels are built in pandas, and on pandas 2.x `date_range` yields
# nanoseconds, while anything read back out of Delta arrives as microseconds,
# because that is what Arrow writes. Nothing in the pipeline is wrong when this
# fires; the two halves simply came from different places.
#
# Microseconds, not nanoseconds, because that is what survives the Delta round
# trip unchanged. Normalising to ns would be undone by the next write, and the
# mismatch would come back on the next read.
#
# Worth noting for the report: this never appeared while the code was developed
# on pandas 3, where `date_range` already returns microseconds, and it failed
# immediately on Colab's pandas 2. A library-version difference is enough to
# turn a working pipeline into a broken one, which is the argument for pinning
# versions (requirements-ml.txt) rather than trusting the environment.
STORE_TIME_UNIT = "us"


def to_store_time(values, unit: str = STORE_TIME_UNIT) -> pd.Series:
    """Parse to UTC and force one resolution, so joins cannot fail on dtype."""
    s = pd.to_datetime(values, utc=True)
    try:
        return s.dt.as_unit(unit)          # pandas >= 2.0
    except AttributeError:                 # pandas 1.x: ns is the only option
        return s


def set_root(path: str | Path) -> Path:
    global FEATURE_STORE_ROOT
    FEATURE_STORE_ROOT = Path(path)
    for sub in ("offline", "online", "_registry"):
        (FEATURE_STORE_ROOT / sub).mkdir(parents=True, exist_ok=True)
    return FEATURE_STORE_ROOT


# --------------------------------------------------------------------------
@dataclass
class FeatureSet:
    """
    One feature set: slide 23's index + time index + features.

    `compute` takes the source frames and returns a frame containing at least
    the entity column, the timestamp column, and the declared features. It is
    the ONLY place a feature is defined, which is what keeps training and
    serving honest.
    """
    name: str
    entity: str                      # the index, e.g. 'user_id'
    timestamp: str                   # the time index, e.g. 'ts'
    features: tuple[str, ...]
    description: str
    compute: Callable[[dict], pd.DataFrame] = field(repr=False, default=None)
    owner: str = "lowrox-data"

    @property
    def offline_path(self) -> str:
        return str(FEATURE_STORE_ROOT / "offline" / self.name)

    @property
    def online_path(self) -> str:
        return str(FEATURE_STORE_ROOT / "online" / self.name)

    def columns(self) -> list[str]:
        return [self.entity, self.timestamp, *self.features]


class FeatureRegistry:
    """
    What features exist, who owns them, what they mean, and when they were
    last materialised. Without it a feature store is just another folder of
    parquet, and the discovery and sharing the chapter describes cannot
    happen.
    """

    def __init__(self):
        self._sets: dict[str, FeatureSet] = {}

    def register(self, fs: FeatureSet) -> FeatureSet:
        if fs.name in self._sets:
            raise ValueError(f"feature set '{fs.name}' is already registered")
        self._sets[fs.name] = fs
        return fs

    def get(self, name: str) -> FeatureSet:
        return self._sets[name]

    def all(self) -> list[FeatureSet]:
        return list(self._sets.values())

    def catalogue(self) -> pd.DataFrame:
        rows = []
        for fs in self._sets.values():
            meta = self._meta(fs)
            rows.append({
                "feature_set": fs.name,
                "index": fs.entity,
                "time_index": fs.timestamp,
                "n_features": len(fs.features),
                "features": ", ".join(fs.features[:4]) + ("…" if len(fs.features) > 4 else ""),
                "rows_offline": meta.get("rows_offline", 0),
                "last_materialised": meta.get("materialised_at", "never")[:19],
                "owner": fs.owner,
            })
        return pd.DataFrame(rows)

    def describe(self, name: str) -> pd.DataFrame:
        fs = self.get(name)
        return pd.DataFrame([{"feature": f, "feature_set": fs.name,
                              "entity": fs.entity, "owner": fs.owner}
                             for f in fs.features])

    # -- materialisation metadata -----------------------------------------
    @staticmethod
    def _meta_path(fs: FeatureSet) -> Path:
        return FEATURE_STORE_ROOT / "_registry" / f"{fs.name}.json"

    @classmethod
    def _meta(cls, fs: FeatureSet) -> dict:
        p = cls._meta_path(fs)
        return json.loads(p.read_text()) if p.exists() else {}

    @classmethod
    def _write_meta(cls, fs: FeatureSet, **kw) -> None:
        p = cls._meta_path(fs)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = cls._meta(fs)
        meta.update(kw)
        p.write_text(json.dumps(meta, indent=2, default=str))


REGISTRY = FeatureRegistry()


# --------------------------------------------------------------------------
# Materialisation
# --------------------------------------------------------------------------
def materialise(fs: FeatureSet, sources: dict, *, run_id: str = "") -> dict:
    """
    Run a feature set's transformation once, and write the result to BOTH
    stores. One computation, two destinations — that is the structural reason
    training and serving cannot disagree.
    """
    df = fs.compute(sources)

    missing = [c for c in fs.columns() if c not in df.columns]
    if missing:
        raise ValueError(f"{fs.name}: compute() did not return {missing}")

    df = df[fs.columns()].copy()
    df[fs.timestamp] = to_store_time(df[fs.timestamp])
    df = df.sort_values([fs.entity, fs.timestamp]).reset_index(drop=True)
    df["_materialised_at"] = datetime.now(timezone.utc)
    df["_run_id"] = run_id

    # OFFLINE: the whole history, because a training set needs to ask what a
    # feature looked like months ago.
    write_deltalake(fs.offline_path, pa.Table.from_pandas(df, preserve_index=False),
                    mode="overwrite", schema_mode="overwrite")

    # ONLINE: one row per entity, the most recent. This is what a prediction
    # request reads, and it has to be a single cheap lookup.
    latest = df.sort_values(fs.timestamp).groupby(fs.entity, as_index=False).last()
    write_deltalake(fs.online_path, pa.Table.from_pandas(latest, preserve_index=False),
                    mode="overwrite", schema_mode="overwrite")

    FeatureRegistry._write_meta(
        fs, materialised_at=datetime.now(timezone.utc).isoformat(),
        rows_offline=len(df), rows_online=len(latest), run_id=run_id,
        features=list(fs.features))
    return {"feature_set": fs.name, "offline_rows": len(df),
            "online_rows": len(latest), "features": len(fs.features)}


def read_offline(fs: FeatureSet) -> pd.DataFrame:
    p = Path(fs.offline_path)
    return DeltaTable(str(p)).to_pandas() if (p / "_delta_log").exists() else pd.DataFrame()


def read_online(fs: FeatureSet, entities: list | None = None) -> pd.DataFrame:
    p = Path(fs.online_path)
    if not (p / "_delta_log").exists():
        return pd.DataFrame()
    df = DeltaTable(str(p)).to_pandas()
    return df[df[fs.entity].isin(entities)] if entities is not None else df


# --------------------------------------------------------------------------
# Point-in-time correct join  (slide 24)
# --------------------------------------------------------------------------
def get_training_data(labels: pd.DataFrame, feature_sets: list[FeatureSet], *,
                      entity: str = "user_id", timestamp: str = "ts",
                      tolerance: pd.Timedelta | None = None,
                      verbose: bool = False) -> pd.DataFrame:
    """
    Enrich a label table with features AS THEY STOOD at each label's timestamp.

    labels: one row per (entity, ts) with the target column, e.g.
            user_id | ts | is_churn      — exactly slide 24's shape.

    The join is merge_asof with direction="backward": for each label row it
    takes the most recent feature row at or before that label's ts, per
    entity. An ordinary join on entity alone would attach TODAY's feature
    values to a label from six months ago — the model would be trained on
    information that did not exist when the prediction would have been made,
    score beautifully offline, and collapse in production. This is the single
    most common way a feature store earns its keep.
    """
    out = labels.copy()
    out[timestamp] = to_store_time(out[timestamp])
    out = out.sort_values(timestamp).reset_index(drop=True)
    report = []

    for fs in feature_sets:
        right = read_offline(fs)
        if right.empty:
            raise ValueError(f"{fs.name} has not been materialised")
        right = right[fs.columns()].copy()
        right[fs.timestamp] = to_store_time(right[fs.timestamp])
        right = right.sort_values(fs.timestamp).reset_index(drop=True)

        before = len(out)
        out = pd.merge_asof(
            out, right,
            left_on=timestamp, right_on=fs.timestamp,
            left_by=entity, right_by=fs.entity,
            direction="backward",            # never look forward in time
            tolerance=tolerance,
            suffixes=("", f"_{fs.name}"),
        )
        assert len(out) == before, "merge_asof must not change the row count"
        matched = out[list(fs.features)].notna().any(axis=1).sum()
        report.append({"feature_set": fs.name, "features_added": len(fs.features),
                       "rows_matched": int(matched),
                       "match_rate": round(100 * matched / max(1, len(out)), 1)})
        # drop the right-hand timestamp; the label's ts is the one that matters
        if fs.timestamp in out.columns and fs.timestamp != timestamp:
            out = out.drop(columns=[fs.timestamp])

    if verbose:
        print("POINT-IN-TIME JOIN")
        print(pd.DataFrame(report).to_string(index=False))
        print(f"\n{len(out)} label rows enriched with "
              f"{sum(len(fs.features) for fs in feature_sets)} features")
    return out


def get_online_features(entities: list, feature_sets: list[FeatureSet], *,
                        entity: str = "user_id") -> pd.DataFrame:
    """
    Serving path. Reads the latest row per entity from the online store — the
    same columns, produced by the same code, as the training path above.
    """
    base = pd.DataFrame({entity: entities})
    for fs in feature_sets:
        right = read_online(fs, entities)
        if right.empty:
            continue
        base = base.merge(right[[fs.entity, *fs.features]],
                          left_on=entity, right_on=fs.entity, how="left")
        if fs.entity != entity and fs.entity in base.columns:
            base = base.drop(columns=[fs.entity])
    return base


def assert_no_future_leakage(training: pd.DataFrame, feature_sets: list[FeatureSet],
                             *, entity: str = "user_id", timestamp: str = "ts") -> dict:
    """
    Prove the join did what it claims: for a sample of rows, no feature value
    used was recorded after the label's timestamp. A test, not a comment.
    """
    checked = leaks = 0
    sample = training.sample(min(200, len(training)), random_state=11)
    for fs in feature_sets:
        hist = read_offline(fs)
        if hist.empty:
            continue
        hist[fs.timestamp] = to_store_time(hist[fs.timestamp])
        for _, row in sample.iterrows():
            rows = hist[(hist[fs.entity] == row[entity])
                        & (hist[fs.timestamp] <= row[timestamp])]
            if rows.empty:
                continue
            checked += 1
            used = rows.sort_values(fs.timestamp).iloc[-1]
            for f in fs.features:
                a, b = row.get(f), used.get(f)
                if pd.notna(a) and pd.notna(b) and not _close(a, b):
                    leaks += 1
                    break
    return {"rows_checked": checked, "leaks_found": leaks,
            "verdict": "no future data used" if leaks == 0 else "LEAKAGE DETECTED"}


def _close(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return a == b
