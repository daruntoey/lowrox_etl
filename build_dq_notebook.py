"""
build_dq_notebook.py — writes lowrox_dq_demo.ipynb, the short live demo of
slide 9 (ETL Execution & Reliability): incremental load, idempotency, data
quality and error recovery, each proved on a running pipeline.

    python build_dq_notebook.py            # write the notebook
    python build_dq_notebook.py --execute  # write it and run it (local Postgres)

It builds its own PostgreSQL source (local_source.py, 2,000 rows per table) and
never connects to Supabase, so it can update rows and inject faults safely.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("lowrox_dq_demo.ipynb")
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["colab"] = {"provenance": []}


def md(text: str):
    nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src: str):
    nb.cells.append(nbf.v4.new_code_cell(src.strip()))


# ==========================================================================
md(r"""
# Lowrox — Data Quality & Reliability, live demo

Four claims from the slide, each proved on a running pipeline:

| # | Claim | What this notebook shows |
|---|---|---|
| 1 | **Incremental** — loads only the changes | init → no change → one user update: rows extracted per run |
| 2 | **Idempotent** — no double counting | the same batch loaded twice: Bronze doubles, Silver & Gold do not |
| 3 | **Validated** — no row goes missing | every DQ rule's result, input = accepted + rejected, Gold = Silver |
| 4 | **Recoverable** — no silent failures | a transient outage, a schema change, and bad rows, each injected live |

**Runtime:** about 2 minutes on Colab. **Run:** Runtime → Run all.

The notebook builds its own PostgreSQL source on this machine
(`local_source.py`, 2,000 rows per table, 18 tables) and never connects to
Supabase, so it is free to update rows and inject faults.
""")

code(r'''
# Setup — packages, code, local source database (about 30 s)
import os, sys, subprocess, importlib

IN_COLAB = "google.colab" in sys.modules
REPO = "https://github.com/daruntoey/lowrox_etl.git"

if IN_COLAB:
    PROJECT = "/content/lowrox_etl"
    if os.path.isdir(f"{PROJECT}/.git"):
        subprocess.run(["git", "-C", PROJECT, "pull", "-q"], check=True)
    else:
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, PROJECT], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    f"{PROJECT}/requirements.txt"], check=True)
else:
    PROJECT = os.path.abspath(".")
sys.path.insert(0, PROJECT)

# A lake of its own, so this demo never touches another notebook's lake.
os.environ["LOWROX_LAKE_ROOT"] = ("/content/dq_demo_lake" if IN_COLAB
                                  else os.path.join(PROJECT, "dq_demo_lake"))

import local_source
SRC = local_source.ensure_local_source(PROJECT, seeds=[("lowrox_s40", 40)], verbose=False)
DSN = os.environ["LOWROX_SOURCE_DSN"]
assert "localhost" in DSN, "this demo writes to its source: it must be the local database"

import pandas as pd, psycopg2
from pathlib import Path
from deltalake import DeltaTable
pd.set_option("display.width", 160, "display.max_columns", 20, "display.max_colwidth", 140)

from etl import run_pipeline, reset_lake, FaultInjector, SOURCE_TABLES
from etl.config import TABLES_BY_NAME, SILVER, GOLD
from etl.errors import DeadLetterQueue
from etl.transform import DQ_RULES, silver_path_for


def sql(query, params=None, fetch=True):
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall() if fetch else None


def delta(path) -> pd.DataFrame:
    return DeltaTable(str(path)).to_pandas()


def rows(path) -> int:
    path = Path(path)
    return len(delta(path)) if (path / "_delta_log").exists() else 0


def extracted(run) -> int:
    m = run.metrics.to_frame()
    return int(m[m.layer == "extract"].rows_out.sum())


SCORE = []
def verdict(card, claim, evidence, ok):
    SCORE.append({"card": card, "claim": claim, "evidence": evidence,
                  "result": "PASS" if ok else "FAIL"})
    print(f"\n{'PASS' if ok else 'FAIL'} — {claim}\n       {evidence}")


SOURCE_ROWS = sum(sql(f"SELECT count(*) FROM {t.name}")[0][0] for t in SOURCE_TABLES)
print(f"source  : local PostgreSQL ({DSN.split('@')[-1]})")
print(f"tables  : {len(SOURCE_TABLES)}")
print(f"rows    : {SOURCE_ROWS:,}")
print(f"lake    : {os.environ['LOWROX_LAKE_ROOT']}")
''')

# ==========================================================================
md(r"""
---
## 1 · Incremental load — loads only the changes

Each table keeps a **timestamp watermark** (`_control/watermarks.json`). An
incremental run reads only rows whose `updated_at` / `created_at` is newer than
the mark.

Three runs:
1. **Init** — the lake is empty, so everything is read.
2. **Incremental, nothing changed** — should read 0 rows.
3. **Incremental after one user is updated** — should read exactly 1 row, and
   `dim_user` should open a new SCD Type 2 version for that user.
""")

code(r'''
reset_lake()
run1 = run_pipeline("init", echo=False)
run2 = run_pipeline("incremental", echo=False)

# The application changes one member's tier.
USER_ID, OLD_TIER = sql("""SELECT u.id, u.tier FROM ms_users u
                           JOIN txn_workout_summaries w ON w.user_id = u.id
                           GROUP BY u.id, u.tier ORDER BY count(*) DESC LIMIT 1""")[0]
NEW_TIER = "elite" if OLD_TIER != "elite" else "pro"
sql("""UPDATE ms_users SET tier = %s,
         updated_at = (SELECT GREATEST(now(), max(updated_at) + interval '1 second') FROM ms_users)
       WHERE id = %s""", (NEW_TIER, USER_ID), fetch=False)
run3 = run_pipeline("incremental", echo=False)

runs = pd.DataFrame([
    {"run": "Run 1 · init load",            "rows extracted": extracted(run1), "status": "ok" if run1.ok else "FAILED"},
    {"run": "Run 2 · no changes",           "rows extracted": extracted(run2), "status": "ok" if run2.ok else "FAILED"},
    {"run": f"Run 3 · user {USER_ID} {OLD_TIER} → {NEW_TIER}", "rows extracted": extracted(run3), "status": "ok" if run3.ok else "FAILED"},
])
display(runs)

history = delta(GOLD / "dim_user")
history = history[history.user_id == USER_ID].sort_values("valid_from")
print(f"\nSCD Type 2 history of user {USER_ID} in Gold dim_user:")
display(history[["user_key", "user_id", "tier", "valid_from", "valid_to", "is_current"]])

verdict("01 Incremental", "Loads only the changes",
        f"{extracted(run1):,} → {extracted(run2)} → {extracted(run3)} rows; "
        f"user {USER_ID} has {len(history)} versions, 1 current",
        extracted(run1) == SOURCE_ROWS and extracted(run2) == 0 and extracted(run3) == 1
        and len(history) == 2 and int(history.is_current.sum()) == 1)
''')

# ==========================================================================
md(r"""
---
## 2 · Idempotency — no double counting

A job that is retried, or re-run by mistake, delivers the **same batch twice**.

- **Bronze** is append-only: it records what arrived, so it should double.
- **Silver** deduplicates on the business key, so it should not change.
- **Gold** is built from Silver: row counts and revenue must be identical.
""")

code(r'''
reset_lake()
run_pipeline("init", echo=False)

def snapshot():
    return {
        "Bronze workouts (append-only)": rows(TABLES_BY_NAME["txn_workout_summaries"].bronze_path),
        "Silver workouts (deduplicated)": rows(SILVER / "txn_workout_summaries"),
        "Gold fct_workout_session": rows(GOLD / "fct_workout_session"),
        "Gold revenue (THB)": round(float(delta(GOLD / "fct_payment_transaction").amount_thb.sum()), 2),
    }

before = snapshot()
run_pipeline("init", echo=False)          # forced replay: the same batch again
after = snapshot()

replay = pd.DataFrame({"after run 1": before, "after replay": after})
replay["expected"] = ["doubles", "unchanged", "unchanged", "unchanged"]
display(replay)

b = list(before.values()); a = list(after.values())
verdict("02 Idempotency", "No double counting",
        f"Bronze {b[0]:,} → {a[0]:,}; Silver {b[1]:,} → {a[1]:,}; Gold {b[2]:,} → {a[2]:,}; "
        f"revenue {b[3]:,.2f} → {a[3]:,.2f}",
        a[0] == 2 * b[0] and a[1] == b[1] and a[2] == b[2] and abs(a[3] - b[3]) < 0.01)
''')

# ==========================================================================
md(r"""
---
## 3 · Data quality — no row goes missing

Between Bronze and Silver every row is checked against declarative rules
(`etl/transform.py`, `DQ_RULES`). A row that fails is **not dropped**: it goes
to the dead-letter queue (`_control/dead_letter.jsonl`) with the name of the
rule it broke.

### 3.1 The rules
""")

code(r'''
rules = pd.DataFrame([{"table": t, "rule": name} for t, rs in DQ_RULES.items() for name, _ in rs])
print(f"{len(rules)} rules across {rules.table.nunique()} tables\n")
display(rules.groupby("table").rule.apply(", ".join).to_frame("rules"))
''')

md(r"""
### 3.2 A clean load, rule by rule
""")

code(r'''
reset_lake()
run = run_pipeline("init", echo=False)
dlq = DeadLetterQueue.read(run.run_id)
dq_rejects = dlq[dlq._dlq_reason == "dq_rule_failed"]

m = run.metrics.to_frame()
silver_in = m[m.layer == "silver"].set_index("step").rows_in

per_rule = []
for table, table_rules in DQ_RULES.items():
    checked = int(silver_in.get(table, 0))
    rej = dq_rejects[dq_rejects._dlq_stage == f"silver:{table}"]
    for name, _ in table_rules:
        failed = int(rej._dq_rule.astype(str).str.contains(name, regex=False).sum())
        per_rule.append({"table": table, "rule": name, "rows checked": checked,
                         "failed": failed, "pass %": round(100 * (checked - failed) / max(checked, 1), 2)})
per_rule = pd.DataFrame(per_rule)
display(per_rule[per_rule["rows checked"] > 0])   # txn_plan_checkoffs is empty in this seed

total_in = extracted(run)
passed_rows = total_in - len(dq_rejects)
print(f"\nrows extracted    : {total_in:,}")
print(f"passed every rule : {passed_rows:,}  ({100 * passed_rows / total_in:.2f}%)")
print(f"quarantined       : {len(dq_rejects):,}  "
      f"({', '.join(f'{k} {v}' for k, v in dq_rejects._dq_rule.value_counts().items())})")
''')

md(r"""
### 3.3 Reconciliation — every row is accounted for

`input = accepted + rejected`, and Gold's totals equal Silver's.
""")

code(r'''
silver_rows = sum(rows(silver_path_for(t)) for t in SOURCE_TABLES)
recon = pd.DataFrame([
    {"step": "Input rows (Bronze)",          "rows": extracted(run)},
    {"step": "Accepted (Silver)",            "rows": silver_rows},
    {"step": "Rejected (dead-letter queue)", "rows": len(dlq)},
    {"step": "Difference",                   "rows": extracted(run) - silver_rows - len(dlq)},
])
display(recon)

sw = delta(SILVER / "txn_workout_summaries")
gw = delta(GOLD / "fct_workout_session")
totals = pd.DataFrame([
    {"measure": "workouts",       "Silver": len(sw),                              "Gold": len(gw)},
    {"measure": "distance (km)",  "Silver": round(float(sw.distance_km.sum()), 2), "Gold": round(float(gw.distance_km.sum()), 2)},
])
totals["match"] = totals.Silver == totals.Gold
display(totals)

print("\nA sample of quarantined rows, with the rule that rejected each:")
cols = [c for c in ["_dlq_stage", "id", "user_id", "activity_type", "distance_km",
                    "duration_sec", "avg_heart_rate", "_dq_rule"] if c in dq_rejects.columns]
display(dq_rejects[cols].head(8))

diff = extracted(run) - silver_rows - len(dlq)
verdict("03 Data quality", "No row goes missing",
        f"{extracted(run):,} in = {silver_rows:,} accepted + {len(dlq):,} rejected "
        f"(difference {diff}); {100 * passed_rows / total_in:.2f}% pass; Gold = Silver",
        diff == 0 and bool(totals.match.all()))
''')

md(r"""
### 3.4 Live test — inject bad values and watch the rules catch them

The fault injector writes impossible measures into 3 workout rows as they are
extracted. They should land in the dead-letter queue with a rule name, and
every good row should still reach Gold.
""")

code(r'''
reset_lake()
with FaultInjector.armed("data.bad_values", fail_times=1, scope="txn_workout_summaries"):
    bad_run = run_pipeline("init", echo=False)

bad = DeadLetterQueue.read(bad_run.run_id)
bad_w = bad[bad._dlq_stage == "silver:txn_workout_summaries"]
# The injected values (etl/faults.py): distance -5 / 9999 / 0 km with duration 0 / 3 days / -1 s.
injected = bad_w[(bad_w.distance_km < 0) | (bad_w.distance_km >= 9999)
                 | (bad_w.duration_sec <= 0) | (bad_w.duration_sec >= 3 * 86_400)]
caught = list(injected.id)

print(f"injected bad rows             : 3")
print(f"caught by the DQ rules        : {len(caught)}\n")
display(injected[["id", "activity_type", "distance_km", "duration_sec", "_dq_rule"]])
print(f"Gold fct_workout_session rows: {rows(GOLD / 'fct_workout_session'):,}; run {'ok' if bad_run.ok else 'FAILED'}")
''')

# ==========================================================================
md(r"""
---
## 4 · Error recovery — no silent failures

| Failure | Policy |
|---|---|
| **Transient** (network blip, pooler busy) | retry with exponential backoff, up to 3 attempts |
| **Permanent** (authentication, schema drift) | do **not** retry, stop that table, log it, raise a critical alert |
| **Invalid data** | dead-letter queue with the row and the reason (shown in 3.4) |

### 4.1 Transient — the source refuses two connections, then recovers
""")

code(r'''
reset_lake()
with FaultInjector.armed("extract.source_unavailable", fail_times=2, scope="ms_users"):
    t_run = run_pipeline("init", echo=False, retry_attempts=3)

step = next(s for s in t_run.metrics.steps if s.layer == "extract" and s.step == "ms_users")
users_loaded = rows(TABLES_BY_NAME["ms_users"].bronze_path)
users_source = sql("SELECT count(*) FROM ms_users")[0][0]
print(f"attempts used : {step.attempts} of 3")
print(f"ms_users      : {users_loaded:,} of {users_source:,} rows loaded")
print(f"run status    : {'SUCCESS' if t_run.ok else 'FAILED'}")
print("\nlog:")
for ev in t_run.log.events:
    if "ms_users" in ev.message and ev.level in ("WARN", "ERROR", "INFO") and ("retry" in ev.message.lower() or "attempt" in ev.message.lower()):
        print(f"  {ev.level:<5} {ev.message[:110]}")
transient_ok = t_run.ok and step.attempts == 3 and users_loaded == users_source
''')

md(r"""
### 4.2 Permanent — a column disappears from the source (schema drift)

Retrying cannot bring a column back, so the table is refused on the first
attempt, nothing partial is written, the other tables still load, and a
critical alert is raised.
""")

code(r'''
reset_lake()
with FaultInjector.armed("data.schema_drift", fail_times=1, scope="txn_workout_summaries"):
    d_run = run_pipeline("init", echo=False)

d_step = next(s for s in d_run.metrics.steps if s.layer == "extract" and s.step == "txn_workout_summaries")
others = rows(TABLES_BY_NAME["txn_payments"].bronze_path)
critical = [a for a in d_run.alerts.alerts if a.severity == "critical"]
print(f"detected as        : {(d_step.error or '').split(':')[0]}")
print(f"attempts used      : {d_step.attempts} (1 = not retried)")
print(f"rows written       : {rows(TABLES_BY_NAME['txn_workout_summaries'].bronze_path)} (0 = no partial table)")
print(f"other tables       : txn_payments loaded {others:,} rows")
print(f"critical alerts    : {len(critical)}")
for a in critical[:2]:
    print(f"  · {a.title}")
drift_ok = ("SchemaDrift" in (d_step.error or "") and d_step.attempts == 1
            and rows(TABLES_BY_NAME["txn_workout_summaries"].bronze_path) == 0
            and others > 0 and len(critical) > 0)

verdict("04 Error recovery", "No silent failures",
        f"transient: success on try {step.attempts}; schema drift: stopped on try "
        f"{d_step.attempts} + {len(critical)} critical alert; bad rows: {len(caught)} sent to DLQ with a reason",
        transient_ok and drift_ok and len(caught) == 3)
''')

# ==========================================================================
md(r"""
---
## Scorecard
""")

code(r'''
score = pd.DataFrame(SCORE)
display(score)
print(f"\n{(score.result == 'PASS').sum()} of {len(score)} claims proved on this run "
      f"({SOURCE_ROWS:,} source rows, {len(SOURCE_TABLES)} tables).")
''')

code(r'''
# Clean-up: put the member's tier back, so the demo can be run again.
sql("UPDATE ms_users SET tier = %s WHERE id = %s", (OLD_TIER, USER_ID), fetch=False)
FaultInjector.clear()
print(f"user {USER_ID} restored to '{OLD_TIER}'")
''')

nbf.write(nb, OUT)
print(f"wrote {OUT.name} ({len(nb.cells)} cells)")

if "--execute" in sys.argv:
    from nbclient import NotebookClient
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(OUT.parent)}})
    client.execute()
    nbf.write(nb, OUT)
    print("executed")
