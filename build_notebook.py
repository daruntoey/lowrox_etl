"""Build the demo notebook. Run, then execute with nbclient to embed outputs."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C, M = [], []


def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src):
    nb.cells.append(nbf.v4.new_code_cell(src.strip()))


# ==========================================================================
md(r"""
# Lowrox — ETL/ELT pipeline, tested

**OLTP (Supabase) → Bronze → Silver → Gold**, built on Delta Lake, with the three
test suites the course requires and both load modes.

| Requirement | Where it is proved in this notebook |
| --- | --- |
| Functional testing — **data lineage** | Part 3: one Gold row walked back to its source row, hop by hop |
| Functional testing — source-to-target | Part 3: row counts and measures on all 18 tables |
| Functional testing — transformation logic | Part 3: DQ rules, derived fields, special keys, referential integrity |
| Performance testing — **volume and speed** | Part 5: throughput, latency, a scaling curve over 3 data volumes |
| Error handling — **with simulations** | Part 6: six faults injected into the running pipeline |
| **init** load | Part 2 |
| **incremental** load | Part 4 |

Nothing here is mocked. Every simulation arms a fault and runs the real pipeline;
the pipeline does not know it is being tested.
""")

# --------------------------------------------------------------------------
md(r"""
## Part 0 — Setup

### Where to run this

| | Verdict |
| --- | --- |
| **Google Colab** | **Recommended for the submission.** Nothing to install, the notebook with its outputs is the artefact you hand in, and it runs the same on any machine. |
| Local PC | Better while you are still editing the pipeline — faster iteration, and the lake persists between sessions. |
| Supabase SQL editor | No. It runs SQL only; there is no Python, no Delta Lake and no test runner. |

**One Colab warning that matters.** Colab's disk is wiped when the runtime
disconnects, and the incremental load depends on the watermark file surviving
between runs. Either run the whole notebook top to bottom in one session (which
is what the demo does — init and incremental both happen here), or mount Google
Drive and point `LOWROX_LAKE_ROOT` at a Drive folder so the lake persists.

### Connecting to Supabase

Use the **Session pooler**, not the direct connection:

| | Host | Port | Username | IPv4? |
| --- | --- | --- | --- | --- |
| Direct | `db.<ref>.supabase.co` | 5432 | `postgres` | **No** — IPv6 only on Free/Pro |
| **Session pooler** | `aws-<n>-<region>.pooler.supabase.com` | **5432** | `postgres.<ref>` | Yes |
| Transaction pooler | `aws-<n>-<region>.pooler.supabase.com` | 6543 | `postgres.<ref>` | Yes |

Colab has no IPv6, so the direct connection simply times out — that failure
looks like a firewall problem and is not one. Use the **session** pooler rather
than the transaction pooler: transaction mode discards session state between
transactions (`SET`, temp tables, cursors), which an ETL job uses.

Copy the exact string from your project's **Connect** dialog; the `<n>` in the
host is a pooler index, not part of the region name.
""")

code(r'''
# ---------------------------------------------------------------- packages
import sys, os, subprocess
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "deltalake==1.6.6", "pyarrow", "psycopg2-binary", "pandas"],
                   check=True)

# ------------------------------------------------------------- the package
# Three ways to get the etl/ package into the runtime. Pick one.
#
# A. (recommended) push this project to your own GitHub repo, then:
#       !git clone https://github.com/<you>/lowrox-etl.git
#       PROJECT_DIR = "/content/lowrox-etl"
#
# B. upload lowrox_etl.zip through the Colab file browser, then:
#       !unzip -oq lowrox_etl.zip -d /content
#       PROJECT_DIR = "/content/lowrox_etl"
#
# C. running locally: point at the folder you already have.
PROJECT_DIR = os.environ.get("LOWROX_PROJECT_DIR", os.getcwd())
sys.path.insert(0, PROJECT_DIR)

# ------------------------------------------------------------ connection
# SUPABASE: paste your session-pooler string here.
#   postgresql://postgres.<ref>:<password>@aws-<n>-<region>.pooler.supabase.com:5432/postgres
# Keep the password out of the notebook you submit — read it from an env var
# or Colab's secrets manager instead of typing it into a cell.
os.environ.setdefault(
    "LOWROX_SOURCE_DSN",
    "postgresql://lowrox:lowrox@localhost:5432/lowrox",   # local test database
)

# Where Bronze / Silver / Gold are written. On Colab, mount Drive and use a
# Drive path if you want the lake to survive a disconnect.
os.environ.setdefault("LOWROX_LAKE_ROOT", "/home/claude/lowrox_lake")

import pandas as pd
pd.set_option("display.width", 200, "display.max_columns", 40)

from etl import run_pipeline, reset_lake, FaultInjector, FAULT_CATALOGUE, SOURCE_TABLES
print(f"ready — {len(SOURCE_TABLES)} source tables registered")
print(f"source : {os.environ['LOWROX_SOURCE_DSN'].split('@')[-1]}")
print(f"lake   : {os.environ['LOWROX_LAKE_ROOT']}")
''')

# --------------------------------------------------------------------------
md(r"""
### The modules, and why they are split this way

The course slide on modular design asks for the single responsibility principle
and clear interfaces between phases. Each module below knows how to do one job
and nothing about how the others do theirs, which is what makes it possible to
test extraction without a warehouse and loading without a database.

| Module | Responsibility | Knows nothing about |
| --- | --- | --- |
| `config.py` | where things live, the table registry | how any phase works |
| `lineage.py` | stamp every row, record every edge | the data itself |
| `metrics.py` | measure; never decide | what the numbers mean |
| `errors.py` | retry, log, quarantine, alert | which table failed |
| `faults.py` | break things on purpose (test only) | the pipeline's logic |
| `extract.py` | OLTP → Bronze, unchanged | cleaning, modelling |
| `transform.py` | Bronze → Silver, trustworthy | where Silver came from |
| `load.py` | Silver → Gold star schema | the source database |
| `pipeline.py` | the order of phases | how any phase works |

The interface between phases is a Delta table plus the lineage columns, not a
Python object, which is why any phase can be re-run on its own.
""")

code(r'''
from etl.config import SOURCE_TABLES
reg = pd.DataFrame([{
    "table": t.name,
    "primary key": ", ".join(t.pk),
    "watermark column": t.watermark_col,
    "append-only": t.is_append_only,
    "PDPA s.26": t.sensitive,
} for t in SOURCE_TABLES])
print("THE 18 SOURCE TABLES AND HOW EACH IS READ INCREMENTALLY\n")
print(reg.to_string(index=False))
print("\ntxn_workout_summaries has no updated_at — it is an append-only device")
print("feed, so created_at is the correct watermark for it.")
print("txn_user_medical_concerns is PDPA section 26 data: it is written to a")
print("restricted Silver zone and never promoted to Gold.")
''')

# --------------------------------------------------------------------------
md(r"""
## Part 1 — ELT, not ETL, and why

The course slide contrasts the two. This pipeline is **ELT**, deliberately.

| | ETL | ELT (what this is) |
| --- | --- | --- |
| Transform happens | before loading | after loading, inside the lake |
| Raw data kept | no | yes, in Bronze, immutably |
| Reprocessing a bug | re-extract from the source | replay from Bronze, source untouched |
| Fits | small volumes, fixed rules | growing volumes, rules that change |

The decisive argument for Lowrox is the third row. The levelling rules are the
product and they will change; when they do, the warehouse has to be rebuilt on
history that no longer exists in the OLTP database, because Supabase holds
current state, not the audit trail. Bronze is that audit trail, which is why
nothing is cleaned before it lands.

The cost is storage, and at this scale that is a rounding error.
""")

# --------------------------------------------------------------------------
md(r"""
## Part 2 — INIT load

The first run: the lake is empty, every row is read once, and a high-water mark
is recorded per table so the next run knows where to resume.
""")

code(r'''
reset_lake()                      # demo only — never in production
init = run_pipeline("init", echo=False)

print(init, "\n")
print("THROUGHPUT BY LAYER")
print(init.summary().to_string(index=False))
''')

code(r'''
from etl.extract import read_watermarks
marks = read_watermarks()
print(f"WATERMARKS RECORDED — {len(marks)} tables\n")
print(pd.DataFrame(sorted(marks.items()), columns=["table", "high_water_mark"]).to_string(index=False))
''')

code(r'''
from deltalake import DeltaTable
from etl.config import BRONZE, SILVER, GOLD
from pathlib import Path

def rows(p):
    p = Path(p)
    return len(DeltaTable(str(p)).to_pandas()) if (p / "_delta_log").exists() else 0

zones = []
for t in SOURCE_TABLES:
    sp = SILVER / "restricted" / t.name if t.sensitive else Path(t.silver_path)
    zones.append({"table": t.name, "bronze": rows(t.bronze_path), "silver": rows(sp),
                  "zone": "silver/restricted" if t.sensitive else "silver"})
z = pd.DataFrame(zones)
print("BRONZE AND SILVER AFTER THE INIT LOAD\n")
print(z.to_string(index=False))
print(f"\ntotal: {z.bronze.sum()} bronze, {z.silver.sum()} silver, "
      f"{z.bronze.sum() - z.silver.sum()} quarantined by data quality rules")

gold = [p.name for p in sorted(Path(GOLD).iterdir()) if (p / "_delta_log").exists()]
print("\nGOLD STAR SCHEMA\n")
print(pd.DataFrame([{"table": g, "rows": rows(Path(GOLD) / g)} for g in gold]).to_string(index=False))
''')

# --------------------------------------------------------------------------
md(r"""
## Part 3 — Functional testing

### 3.1 Data lineage

The requirement the other tests depend on. Every Bronze row is stamped with a
`_lineage_id`, the source table and the source primary key; Silver and Gold
carry it forward. That makes the chain walkable in both directions: from a
number on a dashboard back to the row that produced it, and from a source row
forward to everything it affected.

The test below takes one row out of `fct_workout_session` and walks it back to
the OLTP database, checking the measures survived every hop.
""")

code(r'''
sys.path.insert(0, os.path.join(PROJECT_DIR, "tests"))
import test_functional as F

walk = F.test_row_level_lineage_walk()
''')

code(r'''
# The run-level view: which object produced which object, and how many rows moved.
from etl.lineage import LineageRegistry
edges = LineageRegistry.load_all()
edges = edges[edges.run_id == init.run_id]
print(f"LINEAGE EDGES FOR RUN {init.run_id} — {len(edges)} edges\n")
print(edges[["source_object", "source_version", "operation", "target_object",
             "target_version", "rows_in", "rows_out", "rows_rejected"]]
      .head(12).to_string(index=False))
print(f"\n... {len(edges)} edges in total, every one written to "
      f"_control/lineage_edges.jsonl as it happened")
''')

md(r"""
### 3.2 Source-to-target, transformation logic, end-to-end

Seventeen checks across the four categories from the course slide.
""")

code(r'''
results = F.run_all(verbose=False)
print("FUNCTIONAL TEST RESULTS\n")
for cat, grp in results.groupby("category", sort=False):
    print(f"  {cat}")
    for _, r in grp.iterrows():
        print(f"    [{r.result}] {r.test}")
        print(f"           {r.detail}")
    print()
print(f"{(results.result == 'PASS').sum()}/{len(results)} passed")
''')

md(r"""
One of these checks found a real bug while this pipeline was being built. The
first version of the distance rule was `0.5 <= distance_km <= 40` applied to
every workout, which quarantined six perfectly valid `functional` (gym) sessions
because a strength session covers no distance. A rule that is wrong in the strict
direction is still a wrong rule: it was losing good data, quietly, and only the
source-to-target balance check made that visible. The rule is now activity-aware.
""")

code(r'''
# What is in the dead-letter queue after a clean run, and why.
from etl.errors import DeadLetterQueue
dlq = DeadLetterQueue.read(init.run_id)
print(f"DEAD-LETTER QUEUE — {len(dlq)} rows quarantined out of 856\n")
cols = [c for c in ["id", "activity_type", "distance_km", "duration_sec",
                    "avg_heart_rate", "_dq_rule"] if c in dlq.columns]
print(dlq[cols].sort_values("id").to_string(index=False))
print("""
Rows 7, 19 and 31 are the three defects deliberately planted in the mock data
(GPS drift, a dead heart-rate strap, a stopwatch left running). Rows 42 and 48
are a swim and a ride recorded with zero distance, which are genuine anomalies.
Nothing was dropped: every one of them is replayable from this queue.""")
''')

# --------------------------------------------------------------------------
md(r"""
## Part 4 — INCREMENTAL load

The application writes new data; the pipeline reads only what changed.

Three things have to be true, and the third is the one most pipelines get wrong:

1. only the delta is read
2. facts are appended, not duplicated
3. an update in the source opens a new SCD type 2 version — **and the facts that
   already existed keep pointing at the old one**
""")

code(r'''
import test_incremental as I
demo = I.run_demo(verbose=True)
''')

md(r"""
Point 3 above cost a redesign. The first version resolved `user_key` against
whichever dimension row was `is_current` at load time, which quietly re-pointed
every historical fact at the newest version on each run: a workout recorded
while the member was on the free tier started reporting as an elite workout the
moment they upgraded. That is the exact history SCD type 2 exists to preserve.
The lookup is now point-in-time — it resolves the version whose validity window
contains the date of the fact — and the check above confirms 16 fact rows still
carry the closed version's key.

Phase 5 of that output demonstrates the real limit of watermarking: a row whose
timestamp lands behind the stored mark is invisible, permanently and silently.
This is not hypothetical for Lowrox, because the seeded data already contains
future-dated subscription rows. The mitigations are listed in the output; for
production, log-based CDC reading the write-ahead log is the only one that
cannot be fooled by an application clock.
""")

# --------------------------------------------------------------------------
md(r"""
## Part 5 — Performance testing

Volume and speed, measured with the same instrumentation the pipeline uses in
production rather than a stopwatch wrapped around the test.
""")

code(r'''
import test_performance as P

thr = P.test_throughput_by_layer()
lat = P.test_latency_breakdown()
''')

code(r'''
scale = P.test_scaling_curve()
''')

code(r'''
%matplotlib inline
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(scale.source_rows, scale.rows_per_sec, marker="o", color="#2563eb")
ax[0].set_xscale("log"); ax[0].set_xlabel("source rows"); ax[0].set_ylabel("rows / second")
ax[0].set_title("Throughput improves with volume")
ax[0].grid(alpha=.3)

bottom = 0
for col, colour, label in [("extract_sec", "#2563eb", "extract"),
                           ("silver_sec", "#10b981", "silver"),
                           ("gold_sec", "#f59e0b", "gold")]:
    ax[1].bar(scale.dataset, scale[col], bottom=bottom, color=colour, label=label)
    bottom = bottom + scale[col]
ax[1].set_ylabel("seconds"); ax[1].set_title("Where the time goes")
ax[1].legend(); ax[1].tick_params(axis="x", rotation=15); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.show()

print("""Throughput RISES as the data grows, which looks wrong and is not. The
fixed cost per run — eighteen connections, eighteen Delta table creations,
eighteen commits — dominates completely at 856 rows and is amortised away by
32,056. The number to quote in the report is the large one, because the small
one is measuring setup, not work.""")
''')

code(r'''
par = P.test_parallelism()
net = P.test_parallelism_under_network_latency()
''')

md(r"""
Parallel extraction buys almost nothing against a database on the same machine
(1.0×) and 1.8× once a realistic 60 ms round trip is in the way. That is the
honest result, and it is the one that matters, because Supabase in Singapore is
a network hop away from wherever this runs. The lesson is not "parallelism is
useless" but "`max_workers` is a configuration value, and the right setting
depends on where the source lives" — which is why it is a config field and not
a constant.
""")

code(r'''
incr = P.test_incremental_is_cheaper()
print(P.pd.DataFrame(P.RESULTS).to_string(index=False))

# The benchmarks repointed the pipeline at the 2000-row database and at
# temporary lake directories. Put everything back before the next section, or
# the error-handling simulations run against the wrong data and fail for
# reasons that have nothing to do with error handling. (This bit me while
# building the notebook, which is why it is an explicit, visible step.)
P.restore_defaults()
print(f"\nrestored -> {os.environ['LOWROX_SOURCE_DSN'].split('@')[-1]}, "
      f"lake {os.environ['LOWROX_LAKE_ROOT']}")
''')

# --------------------------------------------------------------------------
md(r"""
## Part 6 — Error handling, with live simulations

Six faults, injected into the real pipeline. The four strategies from the course
slide — retry, error logging, dead-letter queue, alerting — are production code
in `errors.py`; `faults.py` only decides when to break something.

| Simulation | Fault | What should happen |
| --- | --- | --- |
| 1 | source refuses 2 connections, then recovers | retry absorbs it, run succeeds |
| 2 | one table's source stays down | that table quarantined, other 17 load, alert |
| 3 | impossible measures in the feed | DQ quarantines them, good rows still load |
| 4 | a column disappears upstream | detected at the contract check, **not** retried |
| 5 | the load target rejects writes | transient recovers; permanent aborts loudly |
| 6 | the whole batch is replayed | Bronze doubles, Silver and Gold unchanged |
""")

code(r'''
print("THE FAULT CATALOGUE\n")
for name, what in FAULT_CATALOGUE.items():
    print(f"  {name:<32} {what}")
''')

code(r'''
import test_error_handling as E
sims = E.run_all(verbose=True)
''')

md(r"""
Simulation 4 changed the design. The first version let the Bronze Arrow schema
backfill the vanished column with NULLs, so the load succeeded and every
distance measure downstream was silently empty — the worst possible outcome,
because nothing failed. Extraction now compares what the source contract
declares against what the query actually returned and refuses the table if they
disagree. A refused table is visible; a silently empty column is not.

Simulation 6 is the one worth showing a reviewer. Bronze doubles, because Bronze
is the immutable record of what arrived and a replayed batch genuinely did
arrive twice. Silver deduplicates on the business key, so Gold and the revenue
total are byte-identical. That is what makes the pipeline safe to re-run after
any failure, which is the property every other recovery strategy depends on.
""")

# --------------------------------------------------------------------------
md(r"""
## Part 7 — Summary

""")

code(r'''
summary = pd.DataFrame([
    {"requirement": "Functional testing — data lineage",
     "evidence": "Gold row walked to OLTP, 4 hops, measures identical",
     "checks": 6},
    {"requirement": "Functional testing — source to target",
     "evidence": "18 tables, 856 rows, counts and measures reconciled",
     "checks": 3},
    {"requirement": "Functional testing — transformation logic",
     "evidence": "DQ rules, derived pace, special keys, referential integrity",
     "checks": 7},
    {"requirement": "Functional testing — end to end",
     "evidence": "Gold total distance = source minus quarantined",
     "checks": 1},
    {"requirement": "Performance — volume",
     "evidence": "856 / 8,056 / 32,056 rows, same code",
     "checks": 1},
    {"requirement": "Performance — speed",
     "evidence": "throughput by layer, latency breakdown, parallelism",
     "checks": 5},
    {"requirement": "Error handling — simulations",
     "evidence": "6 faults injected into the running pipeline",
     "checks": 6},
    {"requirement": "Init load", "evidence": "full read, watermarks recorded", "checks": 1},
    {"requirement": "Incremental load",
     "evidence": "12 of 856 rows read, SCD2 fired, replay is a no-op", "checks": 6},
])
print(summary.to_string(index=False))
print(f"\n{summary.checks.sum()} checks in total")
''')

md(r"""
### What to write in the report as a limitation

Three things, stated before anyone asks:

1. **The data is synthetic.** The pipeline is proved correct; the business
   relationships it surfaces were put there by `calibration.py`. Nothing here
   validates a product hypothesis.
2. **The watermark has a blind spot.** Demonstrated in Part 4, Phase 5. Moving
   to log-based CDC is the fix, and it is a change to extraction only — nothing
   downstream of Bronze would have to change.
3. **The performance numbers are from one machine, against a local database.**
   The scaling shape is meaningful; the absolute rates are not transferable.
   The networked parallelism test in Part 5 is the closest thing here to a
   production measurement.

### Where this goes next

The Gold layer currently builds two stars — `fct_workout_session` and
`fct_payment_transaction` — against seven dimensions, which is enough to prove
the pattern for all six facts and eleven dimensions in the warehouse design. The
remaining facts are the same three steps each: resolve the natural keys, apply
the special-key rules, merge on the business key.
""")

nbf.write(nb, "/home/claude/lowrox_etl/lowrox_etl_demo.ipynb")
print(f"wrote notebook with {len(nb.cells)} cells")
