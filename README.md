# Lowrox ETL/ELT pipeline

OLTP (Supabase) → **Bronze** → **Silver** → **Gold**, on Delta Lake, with the three
test suites the course requires and both load modes.

Run `lowrox_etl_demo.ipynb` — it already has every output embedded, so it can be
read without running anything.

```
mlops/               MLOps and the feature store (Chapter 7)
  feature_store.py   feature sets, offline/online stores, point-in-time joins
  features.py        the Lowrox feature definitions and the churn label
  training.py        the 8 data science steps, validation gates, calibration
  registry.py        model versions, metrics, gated promotion
  serving.py         batch + online scoring, training/serving skew test
  monitoring.py      data / label / concept drift, edge cases, missingness
  orchestrator.py    the 6 MLOps level-2 stages + ML metadata store
  ci.py              CI on code, data, schemas and models
  simulate.py        deliberate drift, so the monitors can be seen firing
  dashboard.py       self-contained HTML dashboard
  demo_data.py       enough data to train on, with no database
.github/workflows/   ci.yml and cd.yml — MLOps level 2
lowrox_mlops_demo.ipynb  the MLOps notebook, executed

etl/                 the ETL pipeline
  config.py          where things live, the 18-table registry
  lineage.py         row-level stamps + the run-level edge registry
  metrics.py         timing and throughput, measured not estimated
  errors.py          retry / error log / dead-letter queue / alerting
  faults.py          deliberate faults, for the simulations only
  extract.py         OLTP -> Bronze (init + incremental, parallel)
  transform.py       Bronze -> Silver (dedupe, DQ, conform)
  load.py            Silver -> Gold (dimensions, facts, SCD2)
  pipeline.py        the order of phases, and nothing else
tests/
  test_functional.py       17 checks: lineage, source-to-target, logic, end-to-end
  test_performance.py      6 checks: throughput, latency, scaling, parallelism
  test_error_handling.py   6 live fault simulations
  test_incremental.py      6 checks: init vs incremental, SCD2, replay
lowrox_etl_demo.ipynb      the whole thing, executed, with outputs
build_notebook.py          regenerates the notebook
```

---

## The three notebooks

| notebook | covers | needs |
| --- | --- | --- |
| **`lowrox_full_demo.ipynb`** | the whole chain, source database to dashboard — Stage A is the ETL, Stage B is MLOps | nothing; it builds its own PostgreSQL source |
| `lowrox_etl_demo.ipynb` | Stage A on its own, against Supabase | a connection string |
| `lowrox_mlops_demo.ipynb` | Stage B on its own | nothing |

`lowrox_full_demo.ipynb` is the one to run and the one to hand in. `SOURCE =
"local"` in its setup cell installs PostgreSQL where the notebook is running
and seeds it from `generate.py` at 50, 500 and 2,000 rows per table (about 20
seconds, `local_source.py`), so there is no password in the notebook and the
model trains on 2,000 members instead of 50. `SOURCE = "supabase"` extracts
from the real project instead.

That combination closed a seam worth describing in the report. While the two
notebooks were separate, the model could not learn from the ETL's output: the
Supabase project holds 50 members, which is about nine churn events in a test
split, so Stage B generated a larger set and wrote it straight into Silver.
Everything worked and the model had never seen the pipeline's own output. The
fix was not a better model — it was a bigger source.

## Where to run it

| | Verdict |
| --- | --- |
| **Google Colab** | **Use this for the submission.** Nothing to install; the executed notebook is the artefact you hand in. |
| Local PC | Better while editing — faster, and the lake persists between sessions. |
| Supabase SQL editor | No. SQL only; no Python, no Delta Lake, no test runner. |

**The one Colab trap.** Colab's disk is wiped on disconnect, and the incremental
load depends on `_control/watermarks.json` surviving between runs. Either run the
notebook top to bottom in one session (the demo does init *and* incremental in
one pass, so this is fine), or mount Drive and point `LOWROX_LAKE_ROOT` at a
Drive folder.

### Colab, step by step

**GitHub is optional.** The setup cell finds the package on its own — it looks
for an `etl/` folder next to the notebook, and failing that unpacks any `.zip`
it finds. Two routes:

**Route A — GitHub (the setup cell is already pointed at it).**

The cell clones `GITHUB_REPO` into `CLONE_INTO` and derives every other path
from those two values, so a folder name can never end up disagreeing with
itself. Change them if your repo is named differently:

```python
GITHUB_REPO = "https://github.com/daruntoey/lowrox_etl.git"
CLONE_INTO  = "/content/lowrox_etl"
```

The repo must be **public** — Colab has no credentials for a private one. If the
clone fails the cell says so and falls through to Route B automatically.

**Route B — upload the zip (no GitHub needed).**

1. [colab.research.google.com](https://colab.research.google.com) → **File → Upload
   notebook** → choose `lowrox_etl_demo.ipynb`.
2. Click the **folder icon** in the left sidebar → upload `lowrox_etl.zip`.
3. Put your Supabase connection string in the setup cell (or add `SUPABASE_DSN`
   to Colab's secrets, the key icon).
4. **Runtime → Run all.**

Set `GITHUB_REPO = None` to skip the clone attempt. The cell finds and unpacks
the zip by itself. Nothing else to edit.

**Pushing to GitHub in the first place**

On Windows, from the folder you keep it in:

```bat
cd C:\Users\1000258745\Documents\lowrox_etl
git init
git add .
git commit -m "Lowrox ETL pipeline with functional, performance and error-handling tests"
git branch -M main
git remote add origin https://github.com/<your-username>/lowrox-etl.git
git push -u origin main
```

Create the repo on github.com **empty** first — no README, no .gitignore, no
licence — or the first push is rejected for having unrelated histories. A
`.gitignore` is already in the folder and keeps `__pycache__/`, `lowrox_lake/`
and `*.zip` out of the repo.

The seed data is synthetic, so a public repo leaks nothing — but keep your
Supabase password out of it regardless.

### If a cell fails in Colab

| Symptom | Cause |
| --- | --- |
| `ModuleNotFoundError: No module named 'test_functional'` | the clone or unzip did not land where `PROJECT_DIR` points. The setup cell now derives both, so re-run it — and **Runtime → Restart session** first, because a stale `sys.path` from an earlier run makes the problem look intermittent. |
| Part 2 prints `[COMPLETED WITH ERRORS]` | the source database was unreachable. `run_pipeline()` catches its own exceptions and returns `ok=False` instead of raising, so the run does not stop — everything after it then fails for reasons that look unrelated. The setup cell now tests the connection first. |
| the connection times out against Supabase | you are using the direct connection, which is IPv6-only. Use the session pooler. |
| `git clone` prints `Repository not found` | the repo is private, or the name is wrong. Make it public or use Route B. |

---

## Connecting to Supabase

Use the **Session pooler**. Not the direct connection.

| | Host | Port | Username | IPv4 |
| --- | --- | --- | --- | --- |
| Direct | `db.<ref>.supabase.co` | 5432 | `postgres` | **No** — IPv6 only on Free/Pro |
| **Session pooler** | `aws-<n>-<region>.pooler.supabase.com` | **5432** | `postgres.<ref>` | Yes |
| Transaction pooler | `aws-<n>-<region>.pooler.supabase.com` | 6543 | `postgres.<ref>` | Yes |

Colab has no IPv6, so the direct connection times out. That failure looks like a
firewall problem and is not one.

Prefer **session** mode over transaction mode: transaction mode discards session
state between transactions — `SET`, temp tables, cursors, `LISTEN` — which an ETL
job uses. Copy the exact string from your project's **Connect** dialog; the `<n>`
is a pooler index, not part of the region name.

```bash
export LOWROX_SOURCE_DSN="postgresql://postgres.<ref>:<password>@aws-<n>-<region>.pooler.supabase.com:5432/postgres"
export LOWROX_LAKE_ROOT="./lowrox_lake"
```

Keep the password out of the notebook you submit. Read it from an environment
variable or Colab's secrets manager.

---

## Running without the notebook

```bash
pip install deltalake==1.6.6 pyarrow psycopg2-binary pandas

python -c "from etl import reset_lake, run_pipeline; reset_lake(); run_pipeline('init')"
python -c "from etl import run_pipeline; run_pipeline('incremental')"

python tests/test_functional.py
python tests/test_incremental.py
python tests/test_error_handling.py
python tests/test_performance.py     # needs the scaled databases, see below
```

`test_performance.py` benchmarks three data volumes and expects two extra
databases seeded from the same generator:

```bash
python generate.py --scale 10 && createdb lowrox_s10   # then load 01+02
python generate.py --scale 40 && createdb lowrox_s40
```

Without them, run the other three suites; they work against the 50-row database
on its own.

---

## What each suite proves

### Functional — 17 checks

**Data lineage** is the one the others lean on. Every Bronze row carries
`_lineage_id`, `_src_table`, `_src_pk`, `_run_id`; Silver and Gold carry them
forward. `test_row_level_lineage_walk` takes one row from `fct_workout_session`
and walks it back through Silver and Bronze to the OLTP row, checking the
measures at each hop:

```
      hop                       object                   key  distance_km  duration_sec
  1. OLTP public.txn_workout_summaries                 id=25         4.78          3058
2. Bronze bronze.txn_workout_summaries _lineage_id=fbacd26d…         4.78          3058
3. Silver silver.txn_workout_summaries _lineage_id=fbacd26d…         4.78          3058
  4. Gold     gold.fct_workout_session         workout_id=25         4.78          3058
```

Then source-to-target (counts and measures on all 18 tables), transformation
logic (DQ rules, derived pace, the 0 / -1 / -2 special keys, referential
integrity, no email in Gold), and end-to-end.

### Performance — 6 checks

Throughput by layer, a latency breakdown by phase, a scaling curve over 856 /
8,056 / 32,056 rows, and parallel extraction at 1 / 2 / 4 / 8 workers.

Two results worth knowing before you read the numbers:

- **Throughput rises with volume** (1,005 → 12,443 rows/s). The fixed cost of a
  run — 18 connections, 18 Delta table creations, 18 commits — dominates at 856
  rows. Quote the large number; the small one measures setup, not work.
- **Parallelism slightly *hurts* locally and helps across a network.** The
  networked test is a controlled comparison — same data, same worker counts,
  measured with and without an injected 200 ms round trip. On localhost there is
  no network wait to hide, so extra workers only contend; once the source is a
  hop away they come out ahead. The gap between the two conditions is the
  finding, not either number alone. This matters because Supabase is a network
  hop away and localhost is not.

All performance figures are single-run measurements on one machine. The shapes
are meaningful; the absolute rates are not transferable.

### Error handling — 6 live simulations

| # | Fault | Behaviour |
| --- | --- | --- |
| 1 | source refuses 2 connections, then recovers | retry absorbs it, run succeeds |
| 2 | one table's source stays down | that table quarantined, other 17 load, alert |
| 3 | impossible measures in the feed | DQ quarantines them, good rows still load |
| 4 | a column disappears upstream | caught at the contract check, **not** retried |
| 5 | the load target rejects writes | transient recovers; permanent aborts loudly |
| 6 | the batch is replayed | Bronze doubles, Silver and Gold unchanged |

Simulation 6 is the one to show a reviewer:

```
                layer  after_run_1  after_replay
 Bronze (append-only)           50           100
Silver (deduplicated)           45            45
         Gold (facts)           45            45
   Gold revenue (THB)       209348        209348
```

### init vs incremental — 6 checks

> **This suite writes to the source database.** It inserts 7 workouts and 3
> payments, upgrades two members to elite, and deletes its own rows at the end.
> Against seeded mock data that is fine; against a project with real members it
> is not, so anything that is not obviously a local database has to be opted
> into with `LOWROX_ALLOW_SOURCE_WRITES=1`. The notebook sets that flag, because
> at the moment your Supabase holds only generated data. Remove it once real
> users exist.


Init reads all 856 rows and records a watermark per table. The source then gains
7 workouts and 3 payments and two members upgrade tier; the incremental run reads
**12 rows — 1.4% of a full load**, opens two SCD2 versions, and leaves 16 existing
fact rows pointing at the closed versions.

---

## Three bugs these tests found

Worth putting in the report, because finding them is what the testing was for.

1. **A DQ rule that was wrong in the strict direction.** `0.5 ≤ distance_km ≤ 40`
   applied to every workout quarantined six valid `functional` gym sessions,
   which cover no distance. Losing good data quietly is as bad as keeping bad
   data. The rule is now activity-aware.
2. **Facts were being re-pointed at the current dimension version.** Resolving
   `user_key` against `is_current` meant a workout recorded on the free tier
   started reporting as an elite workout the moment the member upgraded —
   destroying the exact history SCD type 2 exists to preserve. The lookup is now
   point-in-time.
3. **Schema drift was being silently absorbed.** A vanished column was backfilled
   with NULLs by the Arrow schema, so the load succeeded and every downstream
   measure was empty. Extraction now compares the source contract against what
   the query returned and refuses the table.

## The tests were coupled to one dataset size

Pointing the suites at the 2,000-row seed instead of the 50-row one failed four
assertions, and none of the failures were about the pipeline:

- `test_functional.py` compared Gold's total distance against the source minus
  the ids `[7, 19, 31, 42, 48]` — the five rows that fail the DQ rules in the
  50-row dataset. At 2,000 rows, 176 different rows are quarantined.
- three simulations in `test_error_handling.py` asserted `rows == 50`, which is
  the development dataset's row count rather than anything the simulation
  claims.

Both now derive the expectation instead of stating it. The quarantined ids are
read back out of the pipeline's own dead-letter queue, which is also a stronger
claim than the original — Gold equals the source minus exactly the rows the
pipeline **says** it dropped, rather than minus a list someone typed. The
simulations ask the source for its row count. All four suites now pass at 50,
500 and 2,000 rows per table.

Worth saying plainly: the suites had passed every run for weeks. They were
tested against one dataset, so they were correct about one dataset.

## One limitation to state before anyone asks

A high-watermark incremental load cannot see a row whose timestamp lands *behind*
the stored mark. This is not hypothetical here: the seeded data already contains
future-dated subscription rows, so `max(updated_at)` on `txn_payments` sits ahead
of `now()`. Phase 5 of `test_incremental.py` inserts two rows stamped 2020-01-01
and shows the pipeline read zero of them.

Mitigations, in increasing order of cost: re-read a lookback window every run;
run a periodic full reconciliation; or move to log-based CDC reading the
write-ahead log, which cannot be fooled by an application clock. Only the last
one is a real fix, and it changes extraction only — nothing downstream of Bronze
would have to change.

### Supabase: the circuit breaker

If a run fails with

```
FATAL: (ECIRCUITBREAKER) too many authentication failures,
       new connections are temporarily blocked
```

the password is wrong. Supavisor trips this after **10 authentication failures
in a 150-second window** and blocks the origin IP for **120 seconds**; every
further attempt with the bad credential re-triggers it, so the lockout extends.

Stop running cells, wait two full minutes, then fix the credential:

1. the pooler username is `postgres.<project-ref>`, not plain `postgres`
2. the password is the **database** password, not your Supabase account
   password — Settings → Database → Reset database password
3. percent-encode it if it contains `# / ? @ : %`

The pipeline no longer contributes to this. An authentication failure is
classified as non-retryable and aborts the whole extraction phase on the first
occurrence, so one bad password costs about five connection attempts instead of
fifty-four. That change is in `etl/errors.py` (`classify_connection_error`) and
`etl/extract.py`.


---

# Part 2 — MLOps and the feature store

`lowrox_mlops_demo.ipynb` covers Chapter 7: **Staging → Feature Store → Model →
Registry → Serving → Monitoring → Dashboard**, with CI/CD around all of it.

It runs on Colab with **no database**. If the lakehouse holds enough members it
is used; otherwise the same calibrated generator writes a larger synthetic set
straight into Silver. The Supabase project has 50 members — enough to prove a
pipeline, far too few for a model.

## The feature store

Slide 23's shape — **index + time index + features** — with four feature sets
and 42 features, materialised to two stores from one computation:

| store | contents | used by |
| --- | --- | --- |
| offline | full history, every monthly snapshot | building training sets |
| online | latest row per member | serving predictions |

Training sets are built with `merge_asof(direction="backward")`, so each label
row sees only feature values that existed at or before its own timestamp. An
ordinary join would attach today's values to a six-month-old label, and the
model would score beautifully offline and collapse in production.

Both properties are tested, not asserted: `assert_no_future_leakage` walks
sampled rows back through the offline history, and
`check_training_serving_skew` compares the two retrieval paths feature by
feature.

## Three leaking features, and the rule behind them

This is the most useful thing in the project. Three times a model scored
**ROC-AUC near 0.995** and was worthless:

| column | why it leaked |
| --- | --- |
| `txn_user_subscriptions.auto_renew` | cleared at cancellation — matches the label 490/490 |
| `ms_users.tier` | downgraded to `free` at cancellation — 490/490 |
| `txn_user_subscriptions.status` | `cancelled` **is** the label |

The point-in-time join was correct every time. The leak was in the feature
**definition**: Silver holds the *current* value of a mutable column, not its
history, so reading one at a past snapshot returns a value from the future.

> **A column can only be a point-in-time feature if its history exists.**

The defence is a validation gate willing to reject a model for being too good:
`roc_auc < 0.98` and no single feature above 50% of the importance. Both now
run on every training job.

## Monitoring, with every check fired on purpose

Six checks, one per failure mode in slides 13–19, each demonstrated:

| scenario | which check fires |
| --- | --- |
| healthy | nothing — the hard part |
| distribution shift | data distribution shift |
| same mean, wider spread | data distribution shift + edge cases |
| label drift | label drift |
| concept drift | **concept drift alerts while data shift stays `ok`** |
| edge cases | edge cases |
| a feed goes quiet | missing-as-a-feature |

The concept-drift row is slide 16's distinction being *detected*: p(y|X) moved
while p(X) stood still.

Getting the healthy row quiet took five corrections, all ordinary mistakes:

1. **Features that trend by construction.** Tenure and lifetime counters grow
   every month, so comparing their level to a training window that ended months
   ago alerts for ever. Now reported as `expected_trend`; their *shape* is still
   checked.
2. **A threshold equal to the rate that occurs by definition.** The p01–p99 band
   excludes 2% of training data, so a 2% edge-case threshold warns on data drawn
   from the training distribution itself.
3. **A baseline from a different population.** Using the test-period ROC-AUC as
   the concept-drift reference against a training-period holdout reported a
   0.115 drop on untouched data — the training months run at 13–14% churn and
   the test months at 7–8%.
4. **A single moment standing in for a distribution.**
   `consistency_x_volume` warned on every healthy run with a std ratio of 0.62
   beside a PSI of 0.0000 and a KS p-value of 1.00. It is a product of two
   features, so its standard deviation is dominated by the tail, and a
   subsample that misses the tail moves it 38% while the distribution is
   identical. A spread signal is now believed only when PSI and KS do not
   contradict it — which also sharpened four other rows: `label_drift` and
   `concept_drift` now report `ok` for data shift, correctly, because neither
   touches X.
5. **A distance measured against the wrong reference, at a threshold borrowed
   from the wrong distribution.** The multivariate check took its centre and
   covariance from the *live batch itself*, so it never compared live to
   training at all, and its cutoff was a chi-squared quantile — which assumes a
   normality that zero-inflated counts and heavy-tailed products do not have.
   At chi2(0.999) the **training data itself** came out 3.7% multivariate
   outliers against a 5% threshold. It now uses the training centre and
   covariance with the training set's own 99th percentile of distance, so the
   expected rate is 1% by construction.

   Fixing that exposed what the check had been hiding: **it could not detect
   the scenario it was paired with.** Injecting 5% of rows at (volume p99,
   consistency p01) — slide 17's exact example — moved the 12-feature rate
   from 0.79% to 0.79%, while the same rows moved the *pairwise* rate for those
   two features from 1.18% to 5.99%. A contradiction between two features is a
   small displacement in twelve dimensions and a large one in the plane where
   it happens, so the check now looks at the three most correlated pairs as
   well as the whole space. A monitor that cannot see its own test case is
   worse than no monitor, because it reports `ok`.

## CI/CD

`.github/workflows/ci.yml` runs four suites, because slide 4 says ML CI
validates **data, schemas and models**, not just code — 26 checks. `cd.yml`
runs the six stages of slide 12 weekly: deploy pipeline, train, promote,
monitor, publish.

Promotion is gated on beating the incumbent. Automatic retraining without that
comparison is automatic degradation waiting for a bad week of data.

## A library version is part of the pipeline

The first run on Colab failed at the point-in-time join with

```
MergeError: incompatible merge keys [1] datetime64[ns, UTC] and
            datetime64[us, UTC], must be the same type
```

Nothing was wrong with the join. A pandas datetime column carries a
*resolution*, `merge_asof` refuses to join two keys that disagree on it, and
the two sides of this join come from different places: labels are built in
pandas, features are read back out of Delta. On pandas 2.x `date_range`
returns nanoseconds while Arrow always writes microseconds, so the two halves
disagreed by default.

Worth stating in the report, because it is the more interesting half: **the
code was developed on pandas 3, where `date_range` already returns
microseconds, so the fault could not occur — and it failed on the first cell
that ran on Colab's pandas 2.** The pipeline was never correct; it was
accidentally compatible with one environment. `requirements.txt` said
`pandas>=2.0`, which permitted both.

The fix pins the resolution rather than the library, since Colab brings its own
pandas: `feature_store.STORE_TIME_UNIT = "us"` — microseconds, because that is
what survives the Delta round trip, so normalising to nanoseconds would be
undone by the next write and the mismatch would return on the next read.
`to_store_time()` applies it on both sides of every join, `snapshot_dates()`
builds the panel at the same resolution, and a CI schema check now asserts it
across the training table and all four offline stores, so it cannot regress
quietly. Verified by running the full notebook on both pandas 2.2.3 and 3.0.5:
45 cells, 0 errors, and the same 7,931 × 44 training table from each.

## Honest limitations

- **Synthetic data.** The pipeline is proved; the relationships were placed in
  `calibration.py`.
- **Monitoring has never seen production traffic.** The detectors are proved to
  fire; the thresholds are not proved right for real behaviour.
- **ROC-AUC 0.75–0.79, about 2.2× lift at the top 20%.** A usable retention
  list and nothing more. The three excellent versions that preceded it were all
  leakage.
- **One label definition.** Churn is "cancelled within 60 days of the snapshot";
  a different horizon is a different problem.
- **Two pandas versions tested, not all of them.** 2.2.3 and 3.0.5 both pass.
  The resolution is pinned explicitly, so a third should behave, but "should"
  is the honest word.
