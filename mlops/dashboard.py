"""
dashboard.py — the end of the pipeline: Staging -> Dashboard.

Writes one self-contained HTML file. No CDN, no build step, no network: it
opens from the Colab file browser, from a laptop, or from a USB stick, which
matters for a deliverable that has to still work when it is marked.

Four panels, because four audiences read this:

  business    members at risk, revenue exposed, churn by cohort
  model       what is serving, how well, how calibrated
  drift       the six monitoring checks and the retraining trigger
  features    the feature store's own health

The numbers come from the Gold layer and the prediction table, not from
re-running anything — the dashboard and the retention campaign read the same
scored table, so they cannot disagree about who is at risk.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEV_COLOUR = {"ok": "#10b981", "expected_trend": "#64748b",
              "warning": "#f59e0b", "alert": "#ef4444"}


# --------------------------------------------------------------------------
def _sparkline(values, width=240, height=48, colour="#2563eb") -> str:
    v = [float(x) for x in values if pd.notna(x)]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1.0
    step = width / (len(v) - 1)
    pts = " ".join(f"{i*step:.1f},{height - (x-lo)/rng*(height-6) - 3:.1f}"
                   for i, x in enumerate(v))
    return (f'<svg viewBox="0 0 {width} {height}" class="spark" '
            f'preserveAspectRatio="none" role="img">'
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-linejoin="round"/></svg>')


def _bars(labels, values, colour="#2563eb", width=300, height=110) -> str:
    if not len(values):
        return ""
    mx = max(values) or 1
    n = len(values)
    bw = width / n * 0.62
    gap = width / n
    bars, texts = [], []
    for i, (l, v) in enumerate(zip(labels, values)):
        h = (v / mx) * (height - 26)
        x = i * gap + (gap - bw) / 2
        bars.append(f'<rect x="{x:.1f}" y="{height-18-h:.1f}" width="{bw:.1f}" '
                    f'height="{max(h,1):.1f}" rx="2" fill="{colour}"/>')
        texts.append(f'<text x="{x+bw/2:.1f}" y="{height-6}" text-anchor="middle" '
                     f'class="tick">{html.escape(str(l))}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">'
            + "".join(bars) + "".join(texts) + "</svg>")


def _tile(label, value, sub="", tone="") -> str:
    cls = f"tile {tone}".strip()
    return (f'<div class="{cls}"><div class="tile-label">{html.escape(label)}</div>'
            f'<div class="tile-value">{html.escape(str(value))}</div>'
            f'<div class="tile-sub">{html.escape(sub)}</div></div>')


def _table(df: pd.DataFrame, max_rows=12) -> str:
    if df is None or df.empty:
        return '<p class="muted">no rows</p>'
    d = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    body = ""
    for _, r in d.iterrows():
        cells = ""
        for c in d.columns:
            v = r[c]
            txt = html.escape("" if pd.isna(v) else str(v))
            if c == "severity":
                cells += (f'<td><span class="pill" style="background:'
                          f'{SEV_COLOUR.get(str(v), "#64748b")}1a;color:'
                          f'{SEV_COLOUR.get(str(v), "#64748b")}">{txt}</span></td>')
            elif c in ("risk_band",):
                tone = {"high": "#ef4444", "medium": "#f59e0b"}.get(str(v), "#10b981")
                cells += f'<td><span class="pill" style="background:{tone}1a;color:{tone}">{txt}</span></td>'
            else:
                cells += f"<td>{txt}</td>"
        body += f"<tr>{cells}</tr>"
    more = (f'<p class="muted">showing {len(d)} of {len(df)}</p>'
            if len(df) > max_rows else "")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{more}'


# --------------------------------------------------------------------------
def build(*, predictions: pd.DataFrame, model_version, monitoring: dict,
          feature_catalogue: pd.DataFrame, labels: pd.DataFrame,
          registry_table: pd.DataFrame, lineage: pd.DataFrame,
          out_path: str = "/content/lowrox_dashboard.html",
          arpu_thb: float = 349.0) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # --- business -------------------------------------------------------
    n = len(predictions)
    high = predictions[predictions.risk_band == "high"] if n else predictions
    med = predictions[predictions.risk_band == "medium"] if n else predictions
    exposed = float(predictions["churn_probability"].sum() * arpu_thb) if n else 0.0

    by_month = (labels.assign(m=labels.ts.dt.strftime("%Y-%m"))
                .groupby("m", as_index=False)
                .agg(members=("is_churn", "size"), churn=("is_churn", "sum")))
    by_month["rate"] = (by_month.churn / by_month.members * 100).round(1)

    # --- model ----------------------------------------------------------
    mv = model_version
    m = mv.metrics if mv else {}
    base = mv.baseline_metrics if mv else {}
    lift = (m.get("recall_at_top_20pct", 0) / 0.20) if m.get("recall_at_top_20pct") else 0

    # --- drift ----------------------------------------------------------
    summary = monitoring["summary"]
    fired = monitoring["retrain_triggered"]
    drift_ev = next((f.evidence for f in monitoring["findings"]
                     if f.check.startswith("data distribution")), pd.DataFrame())
    if drift_ev is not None and len(drift_ev):
        drift_ev = drift_ev[["feature", "train_mean", "live_mean",
                             "mean_shift_sds", "psi", "severity"]]

    top_risk = (predictions.sort_values("churn_probability", ascending=False)
                .head(10)[["user_id", "churn_probability", "risk_band"]]
                .assign(churn_probability=lambda d: d.churn_probability.round(3))
                if n else pd.DataFrame())

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lowrox MLOps</title>
<style>
  :root {{
    --bg:#f6f7f9; --panel:#fff; --ink:#0f172a; --muted:#64748b;
    --line:#e2e8f0; --accent:#2563eb;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#0b1020; --panel:#121a2e; --ink:#e8eefc; --muted:#94a3b8;
      --line:#223050;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#0b1020; --panel:#121a2e; --ink:#e8eefc; --muted:#94a3b8; --line:#223050;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:24px 16px 56px; }}
  header {{ display:flex; flex-wrap:wrap; gap:12px; align-items:baseline;
    justify-content:space-between; margin-bottom:20px; }}
  h1 {{ font-size:22px; margin:0; letter-spacing:-.01em; }}
  h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.08em;
    color:var(--muted); margin:28px 0 12px; font-weight:600; }}
  .muted {{ color:var(--muted); font-size:12px; margin:8px 0 0; }}
  .grid {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); }}
  .panel {{ background:var(--panel); border:1px solid var(--line);
    border-radius:12px; padding:16px; }}
  .tile {{ background:var(--panel); border:1px solid var(--line);
    border-radius:12px; padding:14px 16px; }}
  .tile.alert {{ border-color:#ef4444; }}
  .tile.warn {{ border-color:#f59e0b; }}
  .tile-label {{ color:var(--muted); font-size:11px; text-transform:uppercase;
    letter-spacing:.06em; }}
  .tile-value {{ font-size:26px; font-weight:650; margin:4px 0 2px;
    font-variant-numeric:tabular-nums; letter-spacing:-.02em;
    overflow-wrap:anywhere; }}
  .tile-sub {{ color:var(--muted); font-size:12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; font-weight:600; color:var(--muted); font-size:11px;
    text-transform:uppercase; letter-spacing:.05em;
    border-bottom:1px solid var(--line); padding:6px 8px; }}
  td {{ padding:6px 8px; border-bottom:1px solid var(--line);
    font-variant-numeric:tabular-nums; }}
  tr:last-child td {{ border-bottom:0; }}
  .pill {{ padding:2px 8px; border-radius:99px; font-size:11px; font-weight:600; }}
  .banner {{ border-radius:12px; padding:12px 16px; margin-bottom:18px;
    border:1px solid; font-weight:500; }}
  .chart, .spark {{ width:100%; height:auto; }}
  .tick {{ font-size:9px; fill:var(--muted); }}
  .two {{ display:grid; gap:12px; grid-template-columns:1fr 1fr; }}
  @media (max-width:760px) {{ .two {{ grid-template-columns:1fr; }} }}
</style></head><body><div class="wrap">

<header>
  <div>
    <h1>Lowrox — churn model operations</h1>
    <p class="muted">Gold layer &rarr; feature store &rarr; model &rarr; predictions &rarr; this page</p>
  </div>
  <div class="muted">generated {now}</div>
</header>

<div class="banner" style="border-color:{'#ef4444' if fired else '#10b981'};
  background:{'#ef444414' if fired else '#10b98114'};
  color:{'#ef4444' if fired else '#10b981'}">
  {'RETRAINING TRIGGER FIRED — ' + html.escape(monitoring['trigger_reason'])
   if fired else 'All monitoring checks within tolerance. No retraining required.'}
</div>

<h2>Business</h2>
<div class="grid">
  {_tile("Members scored", f"{n:,}", "from the online feature store")}
  {_tile("High risk", f"{len(high):,}", f"{len(high)/max(n,1):.1%} of the base", "alert" if len(high) else "")}
  {_tile("Medium risk", f"{len(med):,}", f"{len(med)/max(n,1):.1%} of the base", "warn" if len(med) else "")}
  {_tile("Revenue exposed", f"THB {exposed:,.0f}", f"expected loss at THB {arpu_thb:,.0f} ARPU")}
</div>
<div class="two" style="margin-top:12px">
  <div class="panel">
    <div class="tile-label">Churn rate by month</div>
    {_bars(by_month.m.str[-2:].tolist(), by_month.rate.tolist(), "#2563eb")}
    <p class="muted">percent of live subscriptions cancelling within 60 days</p>
  </div>
  <div class="panel">
    <div class="tile-label">Highest-risk members</div>
    {_table(top_risk, 8)}
  </div>
</div>

<h2>Model</h2>
<div class="grid">
  {_tile("Serving", mv.version if mv else "none", f"{mv.name} · {mv.algorithm}" if mv else "")}
  {_tile("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}", f"baseline {base.get('roc_auc', 0):.3f}")}
  {_tile("PR-AUC", f"{m.get('pr_auc', float('nan')):.3f}", f"baseline {base.get('pr_auc', 0):.3f}")}
  {_tile("Lift at top 20%", f"{lift:.1f}x", f"catches {m.get('recall_at_top_20pct',0):.0%} of churners")}
  {_tile("Brier score", f"{m.get('brier', float('nan')):.4f}", f"baseline {base.get('brier', 0):.4f}")}
</div>
<div class="panel" style="margin-top:12px">
  <div class="tile-label">Model registry</div>
  {_table(registry_table, 6)}
</div>

<h2>Monitoring</h2>
<div class="panel">{_table(summary, 8)}</div>
<div class="panel" style="margin-top:12px">
  <div class="tile-label">Feature drift detail</div>
  {_table(drift_ev, 10)}
  <p class="muted">mean_shift_sds is the move in training standard deviations;
     PSI above 0.25 is a material shift. Features on an expected trend
     (tenure, lifetime counters) are reported but not alerted on.</p>
</div>

<h2>Feature store</h2>
<div class="panel">{_table(feature_catalogue, 8)}</div>

<h2>Pipeline run</h2>
<div class="panel">{_table(lineage, 10)}
  <p class="muted">Written to the ML metadata store by each stage, so
     "why is this model in production" has an answer after the fact.</p>
</div>

<p class="muted" style="margin-top:28px">
  Synthetic data. The pipeline is real and tested; the business relationships
  it surfaces were placed in the generator, so no commercial conclusion should
  be drawn from these figures.
</p>
</div></body></html>"""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(doc, encoding="utf-8")
    return out_path
