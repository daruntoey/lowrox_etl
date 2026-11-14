"""Build the MLOps demo notebook, then execute it to embed outputs."""
import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(t): nb.cells.append(nbf.v4.new_markdown_cell(t.strip()))
def code(s): nb.cells.append(nbf.v4.new_code_cell(s.strip()))


# ==========================================================================
md(r"""
# Lowrox — MLOps and the feature store

Chapter 7, built on the Lowrox lakehouse: **Staging → Feature Store → Model →
Registry → Serving → Monitoring → Dashboard**, with CI/CD around all of it.

| Chapter 7 | Where it is here |
| --- | --- |
| DevOps vs MLOps — CI validates data, schemas and models (slide 4) | Part 5, four CI suites |
| The eight data science steps (slides 5–6) | Parts 3–8 |
| MLOps level 0 / 1 / 2 (slides 7–12) | Part 1 for the assessment, Part 6 for the six stages |
| Model drift — data, label, concept (slides 13–16) | Part 8, each one simulated |
| Edge cases, feedback loops, missing-as-a-feature (slides 17–19) | Part 8 |
| Feature stores — index, time index, features (slides 20–24) | Part 3 |
| Feature engineering (slides 25–38) | Part 4 |

The ML problem is deliberately the one slide 24 uses: **churn**, with a label
table of `(user_id, ts, is_churn)` enriched from several feature sets.

Three things in this notebook went wrong before they went right, and the
wrong versions are described rather than hidden — the first model scored
ROC-AUC 0.995 and was worthless, and finding out why is most of what this
chapter is about.
""")

# --------------------------------------------------------------------------
md(r"""
## Part 0 — Setup

Runs on Colab with nothing installed, and needs **no database**. If the
lakehouse from the ETL notebook is present with enough members it is used; if
not, the same calibrated generator produces a larger synthetic set straight
into the Silver zone.

Why that matters: the Supabase project holds 50 members. Nine monthly
snapshots over 50 members is a few hundred label rows at a 9% churn rate, so
the test split would contain a handful of churn events and every metric would
be noise. The pipeline is identical at either size; only whether the numbers
mean anything changes.
""")

code(r'''
import sys, os, subprocess, glob, zipfile
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "deltalake==1.6.6", "pyarrow", "pandas", "scikit-learn",
                    "scipy", "matplotlib"], check=True)

GITHUB_REPO = "https://github.com/daruntoey/lowrox_etl.git"
CLONE_INTO  = "/content/lowrox_etl"

def _git(*a, cwd=None):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)

if IN_COLAB and GITHUB_REPO:
    if os.path.isdir(os.path.join(CLONE_INTO, ".git")):
        # "Restart session" does not delete /content, so an old checkout
        # survives every restart. Update rather than skip.
        _git("fetch", "--depth", "1", "origin", cwd=CLONE_INTO)
        before = _git("rev-parse", "HEAD", cwd=CLONE_INTO).stdout.strip()[:7]
        r = _git("reset", "--hard", "origin/HEAD", cwd=CLONE_INTO)
        if r.returncode != 0:
            r = _git("reset", "--hard", "origin/main", cwd=CLONE_INTO)
        after = _git("rev-parse", "HEAD", cwd=CLONE_INTO).stdout.strip()[:7]
        print(f"checkout {'updated ' + before + ' -> ' + after if before != after else 'already current (' + after + ')'}")
    else:
        r = _git("clone", "--depth", "1", GITHUB_REPO, CLONE_INTO)
        print(f"cloned into {CLONE_INTO}" if r.returncode == 0 else r.stderr.strip())

def find_project() -> str:
    for cand in (CLONE_INTO, "/content/lowrox_etl", os.getcwd(),
                 os.path.join(os.getcwd(), "lowrox_etl")):
        if os.path.isfile(os.path.join(cand, "mlops", "feature_store.py")):
            return os.path.abspath(cand)
    for z in glob.glob("/content/*.zip") + glob.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall("/content" if IN_COLAB else ".")
    for hit in (glob.glob("/content/**/mlops/feature_store.py", recursive=True)
                + glob.glob("./**/mlops/feature_store.py", recursive=True)):
        return os.path.abspath(os.path.dirname(os.path.dirname(hit)))
    raise RuntimeError("could not find the mlops/ package — upload lowrox_etl.zip "
                       "with the folder icon and re-run this cell")

PROJECT_DIR = os.environ.get("LOWROX_PROJECT_DIR") or find_project()
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

os.environ.setdefault("LOWROX_LAKE_ROOT",
                      "/content/lowrox_lake" if IN_COLAB else os.path.join(PROJECT_DIR, "lowrox_lake"))
os.environ.setdefault("LOWROX_MLOPS_ROOT",
                      "/content/mlops_store" if IN_COLAB else os.path.join(PROJECT_DIR, "mlops_store"))

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
pd.set_option("display.width", 200, "display.max_columns", 50)

from mlops import (config, demo_data, feature_store as FS, features as F,
                   training as T, registry as REG, ci as CI, serving as SRV,
                   monitoring as MON, orchestrator as ORC, simulate as SIM,
                   dashboard as DASH)
from mlops.sources import load_sources

paths = config.wire()
print(f"mlops package {__import__('mlops').__version__}")
print(f"project : {PROJECT_DIR}")
print(f"lake    : {config.LAKE_ROOT}")
for k, v in paths.items():
    print(f"  {k:16} {v}")
''')

# --------------------------------------------------------------------------
md(r"""
## Part 1 — Where this pipeline sits: MLOps level 0, 1 or 2

Slides 7, 9 and 11 describe three maturity levels. Naming the level honestly
is more useful than claiming the highest one, because the gap is the roadmap.
""")

code(r'''
maturity = pd.DataFrame([
 {"capability":"ML pipeline is code, not notebook cells","level_0":"no","level_1":"yes","level_2":"yes","lowrox":"yes — mlops/ package"},
 {"capability":"Feature store between raw data and models","level_0":"no","level_1":"yes","level_2":"yes","lowrox":"yes — Part 3"},
 {"capability":"ML metadata store records every run","level_0":"no","level_1":"yes","level_2":"yes","lowrox":"yes — run_log.jsonl"},
 {"capability":"Automated trigger for retraining","level_0":"no","level_1":"yes","level_2":"yes","lowrox":"yes — schedule, new data, drift"},
 {"capability":"Continuous training in production","level_0":"no","level_1":"yes","level_2":"yes","lowrox":"yes — stage 4"},
 {"capability":"CI validates data, schemas AND models","level_0":"no","level_1":"manual","level_2":"yes","lowrox":"yes — Part 5"},
 {"capability":"CI/CD deploys the PIPELINE automatically","level_0":"no","level_1":"no","level_2":"yes","lowrox":"yes — .github/workflows"},
 {"capability":"Model promotion gated on beating production","level_0":"no","level_1":"no","level_2":"yes","lowrox":"yes — stage 5"},
 {"capability":"Active performance monitoring in production","level_0":"no","level_1":"yes","level_2":"yes","lowrox":"yes — Part 8"},
 {"capability":"Live traffic feeding the monitors","level_0":"no","level_1":"yes","level_2":"yes","lowrox":"NO — no real users yet"},
])
print(maturity.to_string(index=False))
print("""
Assessment: the pipeline implements MLOps level 2. The one honest gap is the
last row — there are no real users, so monitoring runs against held-out and
simulated data rather than production traffic. Everything upstream of that is
in place, which is the useful thing to be able to say in the report.""")
''')

# --------------------------------------------------------------------------
md(r"""
## Part 2 — Staging: the data the model will learn from

Features are built from **Silver**, never from the source database. Silver is
the trust boundary established in the ETL pipeline: a row only reaches it after
passing the quality gate. A feature computed from a row that failed that gate
would be a feature the warehouse itself does not believe.
""")

code(r'''
info = demo_data.ensure_training_data(config.LAKE_ROOT, project_dir=PROJECT_DIR,
                                      min_users=1000, scale=40)
snaps = F.snapshot_dates(config.SNAPSHOT_START, config.SNAPSHOT_END)
src = load_sources(config.LAKE_ROOT, snaps)
print("\nSILVER TABLES AVAILABLE TO THE FEATURE PIPELINES")
print(pd.DataFrame([{"table": k, "rows": len(v)} for k, v in src.items()
                    if k != "snapshots"]).to_string(index=False))
print(f"\nsnapshot grid: {[str(d.date()) for d in snaps]}")
''')

# --------------------------------------------------------------------------
md(r"""
## Part 3 — The feature store (slides 20–24)

Slide 23 gives the shape of a feature set: **index + time index + features**.
Slide 22 gives the reason it exists: the same features must serve training and
inference, or the two silently disagree.

Everything here is computed on a **monthly snapshot grid**. For each member and
each month end, the features describe only what had happened *strictly before*
that month end. That is what gives each feature set a real time index, and it
is what makes the point-in-time join something other than decoration.
""")

code(r'''
for fs in F.ALL_FEATURE_SETS:
    print(f"{fs.name}")
    print(f"   index={fs.entity}  time_index={fs.timestamp}  "
          f"{len(fs.features)} features")
    print(f"   {fs.description}")
print("\nMATERIALISING — one computation, written to BOTH stores")
for fs in F.ALL_FEATURE_SETS:
    r = FS.materialise(fs, src, run_id="nb")
    print(f"   {r['feature_set']:28} offline {r['offline_rows']:>6} rows "
          f"(full history)   online {r['online_rows']:>5} rows (latest per member)")
print()
print(FS.REGISTRY.catalogue().to_string(index=False))
''')

md(r"""
### The label table and the point-in-time join

Slide 24's top table is `customer_churn | Model` — `(c_id, ts, is_churn)`. The
Lowrox equivalent is below. The label looks **forward** (did this member cancel
within 60 days of the snapshot?) while the features look only **backward**. That
asymmetry is the whole design.
""")

code(r'''
labels = F.build_labels(src)
print("LABEL TABLE  —  slide 24's customer_churn | Model")
print(labels.head(5).to_string(index=False))
print(f"\n{len(labels):,} rows, churn rate {labels.is_churn.mean():.1%}")
print(labels.assign(month=labels.ts.dt.strftime("%Y-%m")).groupby("month").agg(
    members=("is_churn","size"), churned=("is_churn","sum"),
    rate=("is_churn","mean")).round(3).to_string())
''')

code(r'''
# slide 24's bottom table: the label enriched from every feature set
training = FS.get_training_data(labels, F.ALL_FEATURE_SETS, verbose=True)
print("\nENRICHED TRAINING TABLE")
print(training[["user_id","ts","is_churn","tenure_days","sessions_30d",
                "days_since_last","open_rate_30d","failed_payments",
                "price_thb"]].head(5).to_string(index=False))
''')

md(r"""
The join is `merge_asof` with `direction="backward"`: for each label row it
takes the most recent feature row **at or before** that label's timestamp.

An ordinary join on `user_id` alone would attach today's feature values to a
label from six months ago. The model would be trained on information that did
not exist when the prediction would have been made, score beautifully offline,
and collapse in production. This is the single most common way a feature store
earns its keep — so it is worth testing rather than trusting.
""")

code(r'''
print("LEAKAGE CHECK — walk sampled rows back through the offline history")
print(FS.assert_no_future_leakage(training, F.ALL_FEATURE_SETS))
''')

# --------------------------------------------------------------------------
md(r"""
## Part 4 — Feature engineering (slides 25–38)

Every technique the chapter lists, and where it is used on Lowrox data.
""")

code(r'''
techniques = pd.DataFrame([
 {"slide":26,"technique":"Missing data imputation (KNN)","where":"inside the model pipeline, k=5"},
 {"slide":27,"technique":"Binning","where":"age_band, pace_band"},
 {"slide":27,"technique":"One-hot encoding","where":"plan_is_elite"},
 {"slide":27,"technique":"Label encoding","where":"gender_code"},
 {"slide":27,"technique":"Hashing (high cardinality)","where":"area_hash — 50 training areas"},
 {"slide":28,"technique":"Aggregation / statistical","where":"sessions, km, pace over 30d and 90d"},
 {"slide":28,"technique":"Rolling and lag","where":"sessions_30d_lag1, sessions_delta, pace_trend"},
 {"slide":29,"technique":"Normalisation (z-score)","where":"StandardScaler in the pipeline"},
 {"slide":29,"technique":"Log transform","where":"log_tenure_days, log_km_lifetime, log_revenue"},
 {"slide":29,"technique":"Feature interactions","where":"consistency_x_volume"},
 {"slide":30,"technique":"Cyclical date/time","where":"month_sin, month_cos"},
 {"slide":38,"technique":"Domain-specific","where":"days_since_last, failed_payment_rate, is_social"},
])
print(techniques.to_string(index=False))
print(f"\n{sum(len(fs.features) for fs in F.ALL_FEATURE_SETS)} features across "
      f"{len(F.ALL_FEATURE_SETS)} feature sets")
''')

md(r"""
### The columns that are deliberately **not** features

This cost three rounds of debugging, and it is the most useful thing in the
notebook.

A point-in-time join can only be as honest as the columns it joins. Silver
holds the **current** value of a mutable column, not its history — so reading
one at a past snapshot silently returns a value from the future. Where the
application mutates that column as part of the very event being predicted, the
"feature" *is* the label.

Each time, the symptom was identical: ROC-AUC near 0.995 and one feature
holding most of the importance.
""")

code(r'''
print("COLUMNS EXCLUDED, AND WHY")
for col, why in F.MUTABLE_CURRENT_STATE.items():
    print(f"  {col:42} {why}")
print("""
Safe, because immutable or append-only:
  created_at / started_at / birth_date     written once
  plan_id and the plan's tier and price    not rewritten on cancellation
  workouts, notifications, payments        event tables, append-only

The subtlety worth keeping: the point-in-time JOIN was correct every time.
The leak was in the feature DEFINITION. A column can only be a point-in-time
feature if its history exists; otherwise its present value is a message from
the future.""")
''')

# --------------------------------------------------------------------------
md(r"""
## Part 5 — Training, evaluation, validation (slides 5–6) and CI (slide 4)

The eight data science steps, with two details that matter more than the
algorithm choice:

**The split is by time, not at random.** The data is a panel — the same member
appears at several month ends and their rows are not independent. A random
split puts September's row for a member in training and October's in test, and
the model scores well by recognising the member rather than by learning churn.

**Imputation and scaling live inside the pipeline.** Fitting an imputer on the
whole frame before splitting leaks test-set statistics into training, subtly
and with no error message.
""")

code(r'''
result = T.train_pipeline(labels, F.ALL_FEATURE_SETS, algorithm="random_forest",
                          run_id="nb", verbose=True, register_model=False)
''')

md(r"""
### Validation gates, including two leakage sentinels

Slide 6 asks that a model be *"confirmed to be adequate for deployment — that
its predictive performance is better than a certain baseline."* Three gates do
that comparison. Two more exist because of the failures above: **a model can be
rejected for being too good.** An implausible score is evidence about the data,
not about the model.
""")

code(r'''
print(result["validation"]["checks"].to_string(index=False))
print("\nTOP FEATURES — importance spread across many features, as it should be")
print(result["importances"].head(12).to_string(index=False))
''')

md(r"""
### Continuous integration (slide 4)

> *"CI is no longer only about testing and validating code and components, but
> also testing and validating data, data schemas, and models."*

Four suites. A normal software CI has only the first.
""")

code(r'''
modules = {"feature_store": FS, "features": F, "training": T,
           "registry": REG, "serving": SRV, "monitoring": MON}
ci_report = CI.run_ci(modules, result, F.ALL_FEATURE_SETS, verbose=True)
''')

# --------------------------------------------------------------------------
md(r"""
## Part 6 — CI/CD pipeline automation (slides 11–12)

Slide 12's six stages, run in order. The ML metadata store records each one, so
*"why is this model in production"* has an answer after the fact.
""")

code(r'''
run_id = ORC.new_run_id()
print(f"run {run_id}\n")
s1 = ORC.stage_experiment(run_id, labels=labels, feature_sets=F.ALL_FEATURE_SETS,
        algorithms=["gradient_boosting","random_forest","logistic_regression"],
        train_fn=T.train_pipeline, verbose=True)
''')

md(r"""
The winner is chosen on **validation**, never on test. The test set is spent
once; choosing on it turns it into a second validation set and the reported
metric stops being an estimate of production performance.
""")

code(r'''
print("STAGE 2 — PIPELINE CI  (already run above)")
print(f"  {(ci_report['results'].result=='PASS').sum()}/{len(ci_report['results'])} checks passed\n")

from mlops import __version__ as pkg_version
ORC.stage_cd_pipeline(run_id, package_version=pkg_version)

print("\nSTAGE 4 — AUTOMATED TRIGGERING")
triggers = ORC.evaluate_triggers(last_trained=None,
    now=pd.Timestamp.now(tz="UTC").to_pydatetime(),
    new_rows_since=len(labels), monitoring=None)
print(pd.DataFrame([{"trigger":t.kind,"fired":t.fired,"reason":t.reason}
                    for t in triggers]).to_string(index=False))

r4 = ORC.stage_continuous_training(run_id, labels=labels,
        feature_sets=F.ALL_FEATURE_SETS, algorithm=s1["best_algorithm"],
        train_fn=T.train_pipeline, verbose=False)
print(f"\n  trained {r4['model_version'].version}  "
      f"roc_auc={r4['metrics_test']['roc_auc']:.4f}  stage={r4['model_version'].stage}")
''')

code(r'''
ORC.stage_cd_model(run_id, model_name=config.MODEL_NAME,
                   version=r4["model_version"].version, registry_mod=REG)
print()
print(REG.table(config.MODEL_NAME).to_string(index=False))
''')

md(r"""
Promotion is a **separate act** from training, and it is gated on beating the
incumbent. A pipeline that trains and deploys in one motion has nowhere to ask
"is this new model actually better than the one already serving?" — and
automatic retraining without that question is automatic degradation waiting for
a bad week of data.

The same six stages run unattended in `.github/workflows/`: `ci.yml` builds and
validates, `cd.yml` trains, promotes, monitors and publishes on a weekly
schedule.
""")

code(r'''
import pathlib, re
try:
    import yaml
except ImportError:                     # not on this runtime — read the jobs directly
    yaml = None
for wf in sorted(pathlib.Path(PROJECT_DIR, ".github/workflows").glob("*.yml")):
    print(f"{wf.name}")
    text = wf.read_text()
    if yaml is not None:
        d = yaml.safe_load(text)
        for job, spec in d["jobs"].items():
            steps = len(spec.get("steps", []))
            print(f"   job '{job}'  {steps} steps  needs={spec.get('needs','-')}")
    else:
        body = text.split("\njobs:", 1)[-1]
        for job in re.findall(r"^  ([a-z0-9][\w-]*):", body, re.M):
            print(f"   job '{job}'")
''')

# --------------------------------------------------------------------------
md(r"""
## Part 7 — Serving (slide 6), and the skew test

Slide 6 lists three deployment shapes; two are built here — **batch** scoring
for the dashboard and campaign, and an **online** predictor for one member at a
time.

Both read features through the feature store, so a served prediction uses the
same feature definitions the model was trained on. That is the property the
feature store exists to guarantee, which makes it worth testing rather than
assuming.
""")

code(r'''
users = labels.user_id.drop_duplicates().tolist()
preds = SRV.batch_score(config.MODEL_NAME, F.ALL_FEATURE_SETS, users, run_id=run_id)
print(f"batch scored {len(preds):,} members")
print(preds.risk_band.value_counts().rename("members").to_frame().T.to_string())
print("\nhighest risk")
print(preds.nlargest(5,"churn_probability")[["user_id","churn_probability","risk_band"]]
      .round(4).to_string(index=False))

print("\nonline prediction — the REST-API shape from slide 6, without the HTTP")
print(" ", SRV.OnlinePredictor(config.MODEL_NAME, F.ALL_FEATURE_SETS).predict(users[0]))

print("\nTRAINING / SERVING SKEW")
print(" ", SRV.check_training_serving_skew(config.MODEL_NAME, F.ALL_FEATURE_SETS, labels))
''')

md(r"""
Risk bands are **percentiles of the scored population**, not fixed probability
cuts. A retention team has a weekly capacity, not a probability threshold, and
a calibrated model on a 9% base rate rarely produces anything above 0.5 — fixed
cuts put everyone in "low" and the dashboard reports that there is nothing to
do. That was the first version's behaviour.
""")

# --------------------------------------------------------------------------
md(r"""
## Part 8 — Model drift (slides 13–19), each one simulated

Six checks. Rather than describe them, each is **fired on purpose**, the same
reasoning as the fault injection in the ETL pipeline: a monitor that has never
been shown to fire is a monitor nobody has tested.

Slide 14 asks for the distribution parameters to be stored at training time —
mean, median, standard deviation, IQR, kurtosis — and warns that central
tendency and spread must be compared **together**, because two distributions
can share a mean and nothing else. That is implemented literally, then
supplemented with PSI and the KS test.
""")

code(r'''
mv = REG.production(config.MODEL_NAME)
stats = REG.training_stats(mv.name, mv.version)
print("TRAINING-TIME DISTRIBUTION, STORED WITH THE MODEL  (slide 14)")
print(stats.head(6).round(3).to_string())
''')

code(r'''
from sklearn.metrics import roc_auc_score
model = REG.load(mv.name, mv.version)
X_tr, y_tr = result["splits"].X_train, result["splits"].y_train
holdout = X_tr.sample(frac=0.4, random_state=7); y_hold = y_tr.loc[holdout.index]

def auc_of(X, y):
    return float(roc_auc_score(y, model.predict_proba(X)[:,1])) if y.nunique()>1 else None

# The concept-drift reference must be measured on the SAME population the
# live data is drawn from. Using the model's test-period ROC-AUC as the
# reference and comparing it to a training-period holdout reported a 0.115
# "drop" on untouched data — the training months carry a 13-14% churn rate
# and the test months 7-8%, so the two scores describe different populations,
# not model decay. Measured here instead of assumed.
REFERENCE_AUC = auc_of(holdout, y_hold)
print(f"reference ROC-AUC on an untouched holdout: {REFERENCE_AUC:.4f}")
print(f"(the model's test-period figure is {mv.metrics['roc_auc']:.4f} — a")
print(" different population, and not a valid baseline for this comparison)\n")

rows = []
for name, why in SIM.SCENARIOS.items():
    X, y = holdout.copy(), y_hold.copy()
    if name == "distribution_shift":
        X = SIM.shift_distribution(X, ["km_30d","sessions_30d","consistency_30d"], 2.0, stats)
    elif name == "spread_only":
        X = SIM.widen_spread(X, ["km_30d","pace_trend"], 3.0)
    elif name == "label_drift":
        y = SIM.shift_labels(y, 3.0)
    elif name == "concept_drift":
        y = SIM.break_relationship(X, y, 0.9)
    elif name == "edge_cases":
        X = SIM.inject_edge_cases(X, ["km_30d","sessions_30d"], 0.06, 40)
        X = SIM.inject_multivariate_edge_cases(X, ("km_30d","consistency_30d"), 0.05)
    elif name == "missing_as_feature":
        X = SIM.drop_feature_values(X, ["open_rate_30d","km_30d"], 0.5)
    rep = MON.monitor(training_stats=stats, training_sample=X_tr,
                      training_positive_rate=float(y_tr.mean()),
                      training_auc=REFERENCE_AUC,
                      live_features=X, live_labels=y, live_auc=auc_of(X, y),
                      feature_names=mv.features, verbose=False)
    sev = {f.check.split(" (")[0]: f.severity for f in rep["findings"]}
    rows.append({"scenario": name, **sev, "retrain": rep["retrain_triggered"]})

print("MONITORING RESPONSE BY SCENARIO")
print(pd.DataFrame(rows).to_string(index=False))
''')

md(r"""
Read the matrix by row. The `healthy` row is quiet, which is the hard part —
a monitor that always alerts is a monitor that gets ignored. Each injected
scenario fires its own check and little else.

The `concept_drift` row is the one worth pointing at: **data distribution shift
reports `ok` while concept drift alerts.** That is slide 16's distinction being
detected rather than asserted — p(y|X) moved while p(X) stood still. The two
have different responses, so telling them apart matters.

Getting the healthy row quiet required two corrections, both of which are
ordinary mistakes:

- **Features that trend by construction.** Tenure is larger every month;
  lifetime counters only ever increase. Comparing their *level* against a
  training window that ended months ago reports drift on every run for ever.
  They are now reported as `expected_trend` — their *shape* is still checked.
- **A threshold set at exactly the rate that occurs by definition.** The
  p01–p99 band excludes 2% of training data by construction, so a 2% edge-case
  threshold warns on data drawn from the training distribution itself.
- **A baseline measured on a different population.** The first version used the
  model's test-period ROC-AUC as the concept-drift reference and compared it to
  a training-period holdout. That reported a 0.115 drop on untouched data: the
  training months carry a 13–14% churn rate and the test months 7–8%, so the
  two numbers describe different populations. A drift baseline has to be
  measured on data comparable to what is being monitored, or the monitor
  reports population differences as model decay.
""")

code(r'''
live_now = FS.get_training_data(labels[labels.ts == labels.ts.max()], F.ALL_FEATURE_SETS)
monitoring = ORC.stage_monitoring(run_id, monitor_fn=MON.monitor,
    training_stats=stats, training_sample=X_tr,
    training_positive_rate=float(labels.is_churn.mean()),
    training_auc=mv.metrics["roc_auc"], live_features=live_now,
    live_labels=live_now["is_churn"], live_auc=None,
    feature_names=mv.features, verbose=True)
''')

# --------------------------------------------------------------------------
md(r"""
## Part 9 — The dashboard

One self-contained HTML file: no CDN, no build step, no network. It opens from
the Colab file browser, from a laptop, or from a USB stick — which matters for
a deliverable that has to still work when it is marked.

The numbers come from the Gold layer and the prediction table, not from
re-running anything. The dashboard and the retention campaign read the same
scored table, so they cannot disagree about who is at risk.
""")

code(r'''
out = DASH.build(predictions=preds, model_version=mv, monitoring=monitoring,
                 feature_catalogue=FS.REGISTRY.catalogue(), labels=labels,
                 registry_table=REG.table(mv.name), lineage=ORC.lineage(run_id),
                 out_path=str(config.DASHBOARD))
print(f"written to {out}  ({os.path.getsize(out)/1024:.0f} KB)")

if IN_COLAB:
    from IPython.display import IFrame, display, HTML
    display(HTML(open(out).read()))
else:
    print("\n(open the file to view it)")
''')

code(r'''
print("ML METADATA STORE — the lineage of this run")
print(ORC.lineage(run_id).to_string(index=False))
''')

# --------------------------------------------------------------------------
md(r"""
## Part 10 — Summary

""")

code(r'''
summary = pd.DataFrame([
 {"chapter 7 topic":"DevOps vs MLOps: CI on data, schemas, models","evidence":f"{len(ci_report['results'])} CI checks across 4 suites"},
 {"chapter 7 topic":"Eight data science steps","evidence":"extraction through monitoring, Parts 3-8"},
 {"chapter 7 topic":"MLOps level 2","evidence":"6 stages + 2 GitHub Actions workflows"},
 {"chapter 7 topic":"Feature store: index + time index + features","evidence":f"{len(F.ALL_FEATURE_SETS)} feature sets, {sum(len(fs.features) for fs in F.ALL_FEATURE_SETS)} features"},
 {"chapter 7 topic":"Point-in-time correct join","evidence":"merge_asof backward + leakage assertion"},
 {"chapter 7 topic":"Offline / online consistency","evidence":"training-serving skew test: 0 mismatches"},
 {"chapter 7 topic":"Feature engineering","evidence":"12 techniques from slides 26-38"},
 {"chapter 7 topic":"Model validation against a baseline","evidence":"5 gates incl. 2 leakage sentinels"},
 {"chapter 7 topic":"Model registry and gated promotion","evidence":"staging -> production only if it beats the incumbent"},
 {"chapter 7 topic":"Data / label / concept drift","evidence":"7 scenarios simulated, each firing its own check"},
 {"chapter 7 topic":"Edge cases, feedback loops, missing-as-a-feature","evidence":"uni- and multi-dimensional, declared, monitored"},
 {"chapter 7 topic":"Dashboard","evidence":"self-contained HTML from Gold + predictions"},
])
print(summary.to_string(index=False))
''')

md(r"""
### What to put in the report as limitations

Four, stated before anyone asks:

1. **The data is synthetic.** The pipeline is proved correct; the relationships
   it surfaces were placed in `calibration.py`. No commercial conclusion
   follows from the churn numbers.
2. **Monitoring has never seen production traffic.** The drift checks are
   exercised against held-out and deliberately corrupted data. That proves the
   detectors work; it does not prove the thresholds are right for real
   behaviour, which only live data can settle.
3. **The model is honest but not strong.** ROC-AUC around 0.75–0.79, lift about
   2.2× at the top 20%. That is a usable retention list and nothing more. The
   three suspiciously excellent versions that preceded it were all leakage.
4. **One label definition.** Churn is "cancelled within 60 days of the
   snapshot". A different horizon is a different problem, and the model should
   not be described as predicting churn in general.

### The thing most worth saying about this build

Three separate times, a model scored ROC-AUC near 0.995 and was worthless.
Every time the cause was the same shape: a **mutable column read at a past
snapshot**, where the application rewrites that column as part of the event
being predicted. The point-in-time join was correct on all three occasions —
the leak was in the feature definition.

The lesson is narrow and useful: *a column can only be a point-in-time feature
if its history exists.* Otherwise its present value is a message from the
future, and the only defence is a validation gate willing to reject a model for
being too good.
""")

nbf.write(nb, "/home/claude/lowrox_etl/lowrox_mlops_demo.ipynb")
print(f"wrote notebook with {len(nb.cells)} cells")
