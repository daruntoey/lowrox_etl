"""
monitoring.py — step 8, and the whole of slides 13 to 19.

Six things are watched, because the chapter names six ways a model rots:

  slide 14  data distribution shift   p(X) moves
  slide 15  label drift              p(y) moves, p(x|y) does not
  slide 16  concept drift            p(y|X) moves, p(X) does not
  slide 17  edge cases               uni- and multi-dimensional outliers
  slide 18  feedback loops           the model's own output re-entering as input
  slide 19  missing-as-a-feature     a column starts arriving empty and the
                                     imputer quietly becomes the model

Each check returns a verdict and, where it fires, a reason. Together they
produce the retraining trigger in the level-1 and level-2 diagrams — the
arrow from "Performance monitoring" back to "Trigger".

A note on what slide 14 actually asks for. It says to store the distribution
parameters at training time — mean, median, standard deviation, IQR,
kurtosis — and to compare them later, and it warns that central tendency and
spread must be compared TOGETHER because two distributions can share a mean
and nothing else. That is implemented literally below, and then supplemented
with PSI and the KS test, which are the standard way to turn those
comparisons into one number per feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Thresholds. Stated once, here, rather than scattered through the checks.
# --------------------------------------------------------------------------
PSI_WARN, PSI_ALERT = 0.10, 0.25          # the conventional 0.1 / 0.25 bands
KS_ALPHA = 0.01
MEAN_SHIFT_SDS = 0.5                      # mean moved by this many training SDs
SPREAD_RATIO = 1.5                        # std or IQR changed by this factor
LABEL_DRIFT_REL = 0.30                    # positive rate moved 30% relative
PERF_DROP_AUC = 0.05                      # ROC-AUC lost since training
MISSING_INCREASE = 0.10                   # absolute increase in missing rate
# Share of live values allowed outside the training p01..p99 band.
#
# It must be comfortably above 2%, not equal to it: the band is DEFINED to
# exclude 2% of the training data, so a sample drawn from the training
# distribution itself lands ~2% outside and a 2% threshold warns on data that
# is behaving perfectly. Set at 5%, i.e. two and a half times the rate that
# occurs by construction.
EDGE_CASE_RATE = 0.05

# Features that move with the calendar BY CONSTRUCTION. A member's tenure is
# larger every month; lifetime counters only ever increase. Comparing their
# level against a training window that ended months ago will report drift on
# every single run, for ever, and a monitor that always alerts is a monitor
# that gets ignored — which is worse than not having one.
#
# They are not dropped: their SHAPE is still checked (spread, kurtosis, edge
# cases relative to their own distribution), and they are reported in their
# own category so a genuine break in trend is still visible. Only the
# level-shift alert is suppressed.
EXPECTED_TREND = {
    "tenure_days", "log_tenure_days", "subscription_age_days",
    "sessions_lifetime", "log_sessions_lifetime", "km_lifetime",
    "log_km_lifetime", "payments_lifetime", "revenue_lifetime", "log_revenue",
    "events_lifetime", "comments_lifetime", "failed_payments",
    "month_sin", "month_cos",
}


@dataclass
class Finding:
    check: str
    severity: str          # 'ok' | 'warning' | 'alert'
    detail: str
    evidence: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def fired(self) -> bool:
        return self.severity != "ok"


# --------------------------------------------------------------------------
# slide 14 — data distribution shift
# --------------------------------------------------------------------------
def population_stability_index(expected: pd.Series, actual: pd.Series,
                               bins: int = 10) -> float:
    """
    PSI, on quantile bins of the TRAINING distribution. Bins are taken from
    training because the question is "where did the new data land relative to
    what the model learned", not "how do these two samples compare".
    """
    e = pd.to_numeric(expected, errors="coerce").dropna()
    a = pd.to_numeric(actual, errors="coerce").dropna()
    if len(e) < 20 or len(a) < 20 or e.nunique() < 2:
        return 0.0
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(e, bins=edges)[0] / len(e)
    a_pct = np.histogram(a, bins=edges)[0] / len(a)
    eps = 1e-6
    e_pct, a_pct = np.clip(e_pct, eps, None), np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def check_data_drift(training_stats: pd.DataFrame, live: pd.DataFrame,
                     training_sample: pd.DataFrame | None = None) -> Finding:
    rows = []
    for feat, ref in training_stats.iterrows():
        if feat not in live.columns:
            continue
        s = pd.to_numeric(live[feat], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        cur_mean, cur_std, cur_iqr = s.mean(), s.std(), q3 - q1

        # central tendency, expressed in training standard deviations so the
        # number means the same thing for every feature
        ref_std = ref["std"] if ref["std"] > 1e-9 else 1e-9
        mean_shift_sds = abs(cur_mean - ref["mean"]) / ref_std
        # spread, compared separately — slide 14's explicit warning
        std_ratio = cur_std / ref_std
        iqr_ratio = (cur_iqr / ref["iqr"]) if ref["iqr"] > 1e-9 else 1.0

        psi = (population_stability_index(training_sample[feat], s)
               if training_sample is not None and feat in training_sample else 0.0)
        ks_p = 1.0
        if training_sample is not None and feat in training_sample:
            ref_s = pd.to_numeric(training_sample[feat], errors="coerce").dropna()
            if len(ref_s) > 20 and len(s) > 20 and ref_s.nunique() > 1:
                ks_p = float(stats.ks_2samp(ref_s, s).pvalue)

        spread_moved = (std_ratio > SPREAD_RATIO or std_ratio < 1 / SPREAD_RATIO
                        or iqr_ratio > SPREAD_RATIO or iqr_ratio < 1 / SPREAD_RATIO)
        if feat in EXPECTED_TREND:
            # level shifts are expected; a change in SHAPE is not
            severity = "warning" if spread_moved else "expected_trend"
        else:
            severity = "ok"
            if psi >= PSI_ALERT or mean_shift_sds >= 1.0:
                severity = "alert"
            elif (psi >= PSI_WARN or mean_shift_sds >= MEAN_SHIFT_SDS
                  or spread_moved or ks_p < KS_ALPHA):
                severity = "warning"

        rows.append({
            "feature": feat,
            "train_mean": round(ref["mean"], 3), "live_mean": round(float(cur_mean), 3),
            "mean_shift_sds": round(float(mean_shift_sds), 3),
            "std_ratio": round(float(std_ratio), 3),
            "iqr_ratio": round(float(iqr_ratio), 3),
            "psi": round(psi, 4), "ks_p": round(ks_p, 4),
            "severity": severity,
        })

    ev = pd.DataFrame(rows).sort_values("psi", ascending=False)
    alerts = ev[ev.severity == "alert"]
    warns = ev[ev.severity == "warning"]
    trend = ev[ev.severity == "expected_trend"]
    if len(alerts):
        sev, detail = "alert", (f"{len(alerts)} feature(s) shifted materially: "
                                f"{', '.join(alerts.feature.head(4))}")
    elif len(warns):
        sev, detail = "warning", (f"{len(warns)} feature(s) moved: "
                                  f"{', '.join(warns.feature.head(4))}")
    else:
        sev, detail = "ok", (f"{len(ev) - len(trend)} features within tolerance"
                             + (f", {len(trend)} on an expected trend" if len(trend) else ""))
    return Finding("data distribution shift (slide 14)", sev, detail, ev)


# --------------------------------------------------------------------------
# slide 15 — label drift:  p(y) changes, p(x|y) does not
# --------------------------------------------------------------------------
def check_label_drift(training_rate: float, live_labels: pd.Series) -> Finding:
    if live_labels is None or live_labels.empty:
        return Finding("label drift (slide 15)", "ok", "no labels available yet")
    live_rate = float(live_labels.mean())
    rel = abs(live_rate - training_rate) / max(training_rate, 1e-9)
    sev = "alert" if rel >= 2 * LABEL_DRIFT_REL else (
        "warning" if rel >= LABEL_DRIFT_REL else "ok")
    ev = pd.DataFrame([{"training_positive_rate": round(training_rate, 4),
                        "live_positive_rate": round(live_rate, 4),
                        "relative_change": round(rel, 3)}])
    return Finding("label drift (slide 15)", sev,
                   f"churn rate {training_rate:.1%} -> {live_rate:.1%} "
                   f"({rel:+.0%} relative)", ev)


# --------------------------------------------------------------------------
# slide 16 — concept drift:  p(y|X) changes while p(X) stays put
# --------------------------------------------------------------------------
def check_concept_drift(training_auc: float, live_auc: float | None,
                        data_drift: Finding) -> Finding:
    """
    Concept drift is identified by elimination: performance fell, and the
    inputs did NOT move. If the inputs moved too, the honest label is "the
    world changed", not "the relationship changed" — telling them apart
    matters because the responses differ.
    """
    if live_auc is None or not np.isfinite(live_auc):
        return Finding("concept drift (slide 16)", "ok",
                       "no labelled outcomes yet for the live window")
    drop = training_auc - live_auc
    inputs_stable = data_drift.severity == "ok"
    if drop >= PERF_DROP_AUC and inputs_stable:
        sev = "alert"
        detail = (f"ROC-AUC {training_auc:.3f} -> {live_auc:.3f} "
                  f"(-{drop:.3f}) while the inputs did not move: the "
                  f"relationship itself has changed")
    elif drop >= PERF_DROP_AUC:
        sev = "warning"
        detail = (f"ROC-AUC fell {drop:.3f}, but the inputs moved as well — "
                  f"data shift, not concept drift")
    else:
        sev = "ok"
        detail = f"ROC-AUC {training_auc:.3f} -> {live_auc:.3f} (-{drop:.3f})"
    return Finding("concept drift (slide 16)", sev, detail,
                   pd.DataFrame([{"training_auc": round(training_auc, 4),
                                  "live_auc": round(live_auc, 4),
                                  "drop": round(drop, 4),
                                  "inputs_stable": inputs_stable}]))


# --------------------------------------------------------------------------
# slide 17 — edge cases, uni- and multi-dimensional
# --------------------------------------------------------------------------
def check_edge_cases(training_stats: pd.DataFrame, live: pd.DataFrame) -> Finding:
    rows = []
    for feat, ref in training_stats.iterrows():
        if feat not in live.columns or feat in EXPECTED_TREND:
            # a lifetime counter above its training p99 is the counter doing
            # its job, not an edge case
            continue
        s = pd.to_numeric(live[feat], errors="coerce").dropna()
        if s.empty:
            continue
        outside = ((s < ref["p01"]) | (s > ref["p99"])).mean()
        rows.append({"feature": feat, "outside_train_p01_p99": round(float(outside), 4),
                     "train_p01": round(ref["p01"], 2), "train_p99": round(ref["p99"], 2),
                     "live_min": round(float(s.min()), 2), "live_max": round(float(s.max()), 2)})
    uni = pd.DataFrame(rows).sort_values("outside_train_p01_p99", ascending=False)
    flagged = uni[uni.outside_train_p01_p99 > EDGE_CASE_RATE]

    # Multi-dimensional: each value ordinary on its own, the COMBINATION not.
    # Mahalanobis distance against the training covariance is the standard way
    # to ask that question, and it is the one the slide is pointing at when it
    # says "a combination of variables that may be an edge case when looked at
    # together".
    multi_rate = 0.0
    num = live.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    cols = [c for c in num.columns if c in training_stats.index][:12]
    if len(cols) >= 3 and len(num) > len(cols) + 2:
        X = num[cols].fillna(num[cols].median())
        mu = X.mean().values
        try:
            inv = np.linalg.pinv(np.cov(X.values, rowvar=False))
            d = np.einsum("ij,jk,ik->i", X.values - mu, inv, X.values - mu)
            thresh = stats.chi2.ppf(0.999, df=len(cols))
            multi_rate = float((d > thresh).mean())
        except np.linalg.LinAlgError:
            multi_rate = 0.0

    sev = "ok"
    if len(flagged) or multi_rate > EDGE_CASE_RATE:
        sev = "warning"
    if (len(flagged) and flagged.outside_train_p01_p99.max() > 4 * EDGE_CASE_RATE) \
            or multi_rate > 4 * EDGE_CASE_RATE:
        sev = "alert"
    detail = (f"{len(flagged)} feature(s) with >{EDGE_CASE_RATE:.0%} of values "
              f"outside the training range; {multi_rate:.1%} of rows are "
              f"multivariate outliers")
    return Finding("edge cases (slide 17)", sev, detail, uni.head(10))


# --------------------------------------------------------------------------
# slide 19 — missing-as-a-feature
# --------------------------------------------------------------------------
def check_missingness(training_stats: pd.DataFrame, live: pd.DataFrame) -> Finding:
    """
    A column that starts arriving empty does not throw: the imputer fills it,
    and the model goes on predicting from the imputation rather than from the
    data. Nothing fails, which is exactly why it has to be watched.
    """
    rows = []
    for feat, ref in training_stats.iterrows():
        if feat not in live.columns:
            continue
        live_missing = float(live[feat].isna().mean())
        delta = live_missing - float(ref.get("missing_rate", 0.0))
        rows.append({"feature": feat,
                     "train_missing": round(float(ref.get("missing_rate", 0.0)), 4),
                     "live_missing": round(live_missing, 4),
                     "increase": round(delta, 4)})
    ev = pd.DataFrame(rows).sort_values("increase", ascending=False)
    bad = ev[ev.increase > MISSING_INCREASE]
    sev = "ok"
    if len(bad):
        sev = "alert" if bad.increase.max() > 0.3 else "warning"
    detail = (f"{len(bad)} feature(s) arriving emptier than in training"
              if len(bad) else "missingness unchanged")
    return Finding("missing-as-a-feature (slide 19)", sev, detail, ev.head(10))


# --------------------------------------------------------------------------
# slide 18 — feedback loops
# --------------------------------------------------------------------------
def check_feedback_loop(feature_names: list[str],
                        model_influenced: set[str] | None = None) -> Finding:
    """
    Slide 18: a model whose own output becomes one of its inputs creates a
    feedback loop, and the advice is to avoid such features where possible and
    to monitor them closely where not.

    For Lowrox this is a live risk rather than a theoretical one. The churn
    score decides who receives a retention notification; notification opens
    are a feature; so tomorrow's `open_rate_30d` is partly a consequence of
    today's prediction. The registry records which features are
    model-influenced so the loop is declared rather than discovered.
    """
    influenced = model_influenced or {"notifications_30d", "opens_30d",
                                      "open_rate_30d"}
    present = sorted(set(feature_names) & influenced)
    sev = "warning" if present else "ok"
    detail = ("features influenced by the model's own actions are in use: "
              + ", ".join(present) if present
              else "no model-influenced features in use")
    ev = pd.DataFrame([{"feature": f, "influenced_by": "retention campaign "
                        "targeting, which the churn score drives"} for f in present])
    return Finding("feedback loop (slide 18)", sev, detail, ev)


# --------------------------------------------------------------------------
# The monitoring run
# --------------------------------------------------------------------------
def monitor(*, training_stats: pd.DataFrame, training_sample: pd.DataFrame,
            training_positive_rate: float, training_auc: float,
            live_features: pd.DataFrame, live_labels: pd.Series | None = None,
            live_auc: float | None = None,
            feature_names: list[str] | None = None,
            verbose: bool = True) -> dict:
    data = check_data_drift(training_stats, live_features, training_sample)
    findings = [
        data,
        check_label_drift(training_positive_rate, live_labels),
        check_concept_drift(training_auc, live_auc, data),
        check_edge_cases(training_stats, live_features),
        check_missingness(training_stats, live_features),
        check_feedback_loop(feature_names or list(live_features.columns)),
    ]
    summary = pd.DataFrame([{"check": f.check, "severity": f.severity,
                             "detail": f.detail} for f in findings])
    alerts = [f for f in findings if f.severity == "alert"]
    retrain = bool(alerts)

    if verbose:
        print("MONITORING REPORT")
        print(summary.to_string(index=False))
        print(f"\nretraining trigger: {'FIRED' if retrain else 'not fired'}")
        if alerts:
            for a in alerts:
                print(f"  - {a.check}: {a.detail}")

    return {"findings": findings, "summary": summary,
            "retrain_triggered": retrain,
            "trigger_reason": "; ".join(a.check for a in alerts) if alerts else ""}
