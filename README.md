# Lowrox ETL/ELT pipeline

OLTP (Supabase) → **Bronze** → **Silver** → **Gold**, on Delta Lake, with the three
test suites the course requires and both load modes.

Run `lowrox_etl_demo.ipynb` — it already has every output embedded, so it can be
read without running anything.

```
etl/                 the pipeline
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

1. Push this folder to a GitHub repo of your own.
2. New Colab notebook → `!git clone https://github.com/<you>/lowrox-etl.git`
3. Open `lowrox_etl_demo.ipynb` from the clone, set `PROJECT_DIR`, put your
   connection string in the setup cell, Runtime → Run all.

If you would rather not use GitHub, zip the folder, upload it through the Colab
file browser and `!unzip -oq lowrox_etl.zip -d /content`.

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
- **Parallelism buys nothing locally** (1.0×) and **1.8× at a 60 ms round trip**.
  The separate networked test demonstrates this rather than asserting it, which
  matters because Supabase is a network hop away and localhost is not.

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
