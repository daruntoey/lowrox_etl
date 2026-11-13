"""
simulate.py — break the model's world on purpose, so the monitoring in
monitoring.py can be watched doing its job rather than described.

The same reasoning as the fault injection in the ETL pipeline: a monitoring
system that has never been shown to fire is a monitoring system nobody has
tested. Each scenario corresponds to one slide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20261113)


def shift_distribution(live: pd.DataFrame, columns: list[str],
                       sds: float = 1.5, stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Slide 14. Move the population without touching the relationship: members
    simply got older, or started training more. p(X) changes; p(y|X) does not.
    """
    out = live.copy()
    for c in columns:
        if c not in out.columns:
            continue
        sd = float(stats.loc[c, "std"]) if stats is not None and c in stats.index \
            else float(pd.to_numeric(out[c], errors="coerce").std() or 1.0)
        out[c] = pd.to_numeric(out[c], errors="coerce") + sds * sd
    return out


def widen_spread(live: pd.DataFrame, columns: list[str], factor: float = 2.5) -> pd.DataFrame:
    """
    Slide 14's explicit warning: the mean can stay exactly where it was while
    the distribution becomes a different distribution. A check that only
    compares means sees nothing here.
    """
    out = live.copy()
    for c in columns:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = s.mean() + (s - s.mean()) * factor
    return out


def shift_labels(labels: pd.Series, multiplier: float = 2.5) -> pd.Series:
    """
    Slide 15. p(y) moves — a price rise, a competitor, a bad app release —
    while the features that describe each member are unchanged.
    """
    out = labels.copy()
    zeros = out[out == 0].index
    n_flip = int(min(len(zeros), out.sum() * (multiplier - 1)))
    if n_flip > 0:
        out.loc[RNG.choice(zeros, n_flip, replace=False)] = 1
    return out


def break_relationship(live: pd.DataFrame, labels: pd.Series,
                       frac: float = 0.5) -> pd.Series:
    """
    Slide 16. Concept drift: the inputs are untouched, but what they MEAN for
    the outcome has changed. Shuffling part of the label against fixed
    features is exactly p(y|X) moving while p(X) stands still.
    """
    out = labels.copy()
    idx = RNG.choice(out.index, int(len(out) * frac), replace=False)
    out.loc[idx] = RNG.permutation(out.loc[idx].values)
    return out


def inject_edge_cases(live: pd.DataFrame, columns: list[str],
                      frac: float = 0.05, magnitude: float = 50) -> pd.DataFrame:
    """
    Slide 17. A device firmware bug reports absurd values; a data feed sends
    a sentinel instead of a null. Uni-dimensional and obvious.
    """
    out = live.copy()
    n = max(1, int(len(out) * frac))
    idx = RNG.choice(out.index, n, replace=False)
    for c in columns:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            out.loc[idx, c] = s.max() * magnitude if s.max() else magnitude
    return out


def inject_multivariate_edge_cases(live: pd.DataFrame, pair: tuple[str, str],
                                   frac: float = 0.05) -> pd.DataFrame:
    """
    Slide 17's second half. Each value stays inside its own normal range, but
    the COMBINATION is one no member has ever produced — here, very high
    volume with near-zero consistency, which no real training pattern creates.
    """
    out = live.copy()
    a, b = pair
    if a not in out.columns or b not in out.columns:
        return out
    n = max(1, int(len(out) * frac))
    idx = RNG.choice(out.index, n, replace=False)
    out.loc[idx, a] = pd.to_numeric(out[a], errors="coerce").quantile(0.99)
    out.loc[idx, b] = pd.to_numeric(out[b], errors="coerce").quantile(0.01)
    return out


def drop_feature_values(live: pd.DataFrame, columns: list[str],
                        frac: float = 0.4) -> pd.DataFrame:
    """
    Slide 19. An upstream integration starts failing quietly. Nothing errors;
    the imputer fills the gap; the model begins predicting from the
    imputation. This is the one that produces no symptom at all until the
    missingness is watched.
    """
    out = live.copy()
    n = max(1, int(len(out) * frac))
    for c in columns:
        if c in out.columns:
            out.loc[RNG.choice(out.index, n, replace=False), c] = np.nan
    return out


SCENARIOS = {
    "healthy": "live data resembles training — nothing should fire",
    "distribution_shift": "slide 14 — the population moved",
    "spread_only": "slide 14 — same mean, different distribution",
    "label_drift": "slide 15 — p(y) moved, features unchanged",
    "concept_drift": "slide 16 — p(y|X) moved, p(X) unchanged",
    "edge_cases": "slide 17 — impossible values, and impossible combinations",
    "missing_as_feature": "slide 19 — a feed went quiet",
}
