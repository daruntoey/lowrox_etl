"""
Compose the two demo notebooks into one end-to-end notebook.

    lowrox_etl_demo.ipynb    +    lowrox_mlops_demo.ipynb
                      -> lowrox_full_demo.ipynb

Composition, not a copy. The two builders are the source of truth for their own
content, so this script harvests their cells instead of restating them: fix
something in build_notebook.py and it is fixed here on the next run. What this
script owns is only what the combination needs — one setup cell instead of two,
a source database that needs no credentials, Stage A/B part numbering, and the
joins between the halves.

Run:  python3 build_full_notebook.py
Then execute it (nbclient / Colab) to embed the outputs.
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat
import nbformat as nbf

HERE = Path(__file__).parent
OUT = HERE / "lowrox_full_demo.ipynb"


# --------------------------------------------------------------------------
def harvest(builder: str) -> list:
    """Run a builder script and take its cells without letting it write."""
    real_write = nbformat.write
    nbformat.write = lambda *a, **k: None
    try:
        ns = {"__name__": "__main__", "__file__": str(HERE / builder)}
        exec(compile((HERE / builder).read_text(), builder, "exec"), ns)
        return list(ns["nb"].cells)
    finally:
        nbformat.write = real_write


def renumber(cells: list, stage: str) -> list:
    """`Part 3` -> `Part A-3`. Every reference inside one half is to that half."""
    pat = re.compile(r"(?<![\w-])Part (\d+)")
    sub = re.compile(r"^(#+\s*)(\d+)\.(\d+)", re.M)      # "### 3.2 ..." subsections
    for c in cells:
        c.source = pat.sub(lambda m: f"Part {stage}-{m.group(1)}", c.source)
        c.source = sub.sub(lambda m: f"{m.group(1)}{stage}-{m.group(2)}.{m.group(3)}",
                           c.source)
    return cells


def source_of(cell) -> str:
    return cell.source if isinstance(cell.source, str) else "".join(cell.source)


# --------------------------------------------------------------------------
etl = harvest("build_notebook.py")
ml = harvest("build_mlops_notebook.py")

nb = nbf.v4.new_notebook()


def md(text: str) -> None:
    nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src: str) -> None:
    nb.cells.append(nbf.v4.new_code_cell(src.strip()))


# ==========================================================================
# Title
# ==========================================================================
md(r"""
# Lowrox — the full pipeline, source database to dashboard

One run, one chain:

```
PostgreSQL (OLTP)  ->  Bronze  ->  Silver  ->  Gold  ->  feature store  ->  model  ->  dashboard
     18 tables         raw        cleaned     stars      offline/online    registry    predictions
                       immutable  DQ rules    SCD2       point-in-time     + gates     + monitors
```

**Stage A** is the ETL/ELT pipeline and its three test suites. **Stage B** is
MLOps: the feature store, training, CI/CD, drift monitoring and the dashboard.
Stage B learns from what Stage A produced — the model's features are computed
from the Silver zone the ETL wrote, not from a separate copy of the data.

| Requirement | Where it is proved |
| --- | --- |
| Functional testing — **data lineage** | Stage A, Part A-3: one Gold row walked back to its source row, hop by hop |
| Functional testing — source-to-target, transformation logic | Part A-3: 17 checks over 18 tables |
| Performance testing — **volume and speed** | Part A-5: throughput, latency, a scaling curve over three data volumes |
| Error handling — **with simulations** | Part A-6: six faults injected into the running pipeline |
| **init** and **incremental** load | Parts A-2 and A-4 |
| MLOps level 0 / 1 / 2 (slides 7–12) | Part B-1 for the assessment, Part B-6 for the six stages |
| Feature store — index, time index, features (slides 20–24) | Part B-3 |
| Feature engineering (slides 25–38) | Part B-4 |
| CI validates **data, schemas and models** (slide 4) | Part B-5, four suites |
| Model drift — data, label, concept (slides 13–16) | Part B-8, each one simulated |
| Edge cases, feedback loops, missing-as-a-feature (slides 17–19) | Part B-8 |
| Staging → dashboard | Part B-2 through Part B-9 |

Nothing here is mocked. Every simulation arms a fault and runs the real
pipeline; every test queries the real source and the real lake. The data is
generated, and that is the one thing to say out loud — `calibration.py` cites a
source for every parameter it places, but a relationship that was put into the
data is not evidence about real members.
""")

# ==========================================================================
# Part 0 — one setup cell for both halves
# ==========================================================================
md(r"""
## Part 0 — Setup

This cell gets the code, builds the **source database**, and wires both halves
to the same lake. It is the only cell that needs reading before you run it.

The source is the one real decision. `SOURCE = "local"` installs PostgreSQL on
the machine running this notebook and seeds it from `generate.py` at three
sizes — 50, 500 and 2,000 rows per table. Nothing to paste, no password in a
notebook that gets handed in, about 20 seconds the first time, and the
2,000-member seed is large enough that Stage B's metrics mean something. The
three sizes are also what the scaling curve in Part A-5 measures.

`SOURCE = "supabase"` extracts from your own project instead. That works, and
it is the honest demonstration that the pipeline reads a real managed database
— but the project holds 50 members, and 50 members is about nine churn events
in a test split. Stage B will say so and generate a larger set for the model
rather than report noise.
""")

setup = source_of(etl[2])

# The ML half needs scikit-learn and scipy, which the ETL-only notebook did not.
setup = setup.replace(
    '''                    "deltalake==1.6.6", "pyarrow", "psycopg2-binary", "pandas"],''',
    '''                    "deltalake==1.6.6", "pyarrow", "psycopg2-binary", "pandas",
                    "scikit-learn", "scipy"],''')
assert "scikit-learn" in setup, "pip list not patched"

SOURCE_BLOCK = '''
# ------------------------------------------------------- the source database
# The whole chain runs here, so it needs an OLTP database to extract from.
#
#   "local"     build PostgreSQL on this machine and seed it from generate.py
#               at three sizes (50 / 500 / 2,000 rows per table). No
#               credentials, nothing to paste, ~20s the first time and ~0s
#               after that. Use this for the full process.
#   "supabase"  extract from your own project. Paste the session pooler string
#               in the connection block below. 50 members proves the pipeline;
#               Stage B will fall back to a generated set for the model and
#               tell you it did.
SOURCE = "local"

LOCAL_SOURCE = None
if SOURCE == "local":
    try:
        import local_source
        LOCAL_SOURCE = local_source.ensure_local_source(PROJECT_DIR)
    except Exception as exc:
        # Not fatal: the connection block below still accepts a pasted string,
        # and its own check reports anything that is wrong with it.
        print("could not build a local source database:")
        print(f"  {type(exc).__name__}: {exc}")
        print("\\nSet SOURCE = \\"supabase\\" and paste your connection string "
              "below, or\\nset LOWROX_SOURCE_DSN to a database you can reach.")

'''

marker = "# ------------------------------------------------------------ connection"
assert setup.count(marker) == 1, "connection marker not found in the setup cell"
setup = setup.replace(marker, SOURCE_BLOCK.lstrip("\n") + marker, 1)

# One more line in the closing report, so the source mode is on the record.
setup = setup.replace(
    'print(f"colab   : {IN_COLAB}")',
    'print(f"colab   : {IN_COLAB}")\n'
    'print(f"mode    : {SOURCE}"'
    ' + (f" ({LOCAL_SOURCE[\'main_db\']}, {LOCAL_SOURCE[\'users\']:,} members)"'
    ' if LOCAL_SOURCE else ""))')
assert "mode    :" in setup

code(setup)

# The module tour and the table registry, straight from the ETL builder.
nb.cells.extend(renumber(etl[3:5], "A"))

# ==========================================================================
# Stage A
# ==========================================================================
md(r"""
---

# Stage A — ETL/ELT: source database to Gold

Extraction, the Bronze/Silver/Gold zones, both load modes, and the three test
suites the course asks for. Everything Stage B uses is produced here.
""")

nb.cells.extend(renumber(etl[5:], "A"))

# ==========================================================================
# Stage B
# ==========================================================================
md(r"""
---

# Stage B — MLOps: Gold to a dashboard

Stage A left a lakehouse behind. Stage B turns it into a model and keeps the
model honest: a feature store with point-in-time joins, training with
validation gates, CI on data and schemas and models, drift monitoring, and a
dashboard.

The lake path does not change between the stages. `LOWROX_LAKE_ROOT` was set in
Part 0 and both halves read it, which is the whole reason the model can learn
from the ETL's own output instead of from a second copy of the data.
""")

code(r'''
# ------------------------------------------------- Stage B wiring
# Same project, same lake. Only the feature store and model paths are new.
import warnings
warnings.filterwarnings("ignore")
import numpy as np

os.environ.setdefault(
    "LOWROX_MLOPS_ROOT",
    "/content/mlops_store" if IN_COLAB else os.path.join(PROJECT_DIR, "mlops_store"))
pd.set_option("display.width", 200, "display.max_columns", 50)

from mlops import (config, demo_data, feature_store as FS, features as F,
                   training as T, registry as REG, ci as CI, serving as SRV,
                   monitoring as MON, orchestrator as ORC, simulate as SIM,
                   dashboard as DASH)
from mlops.sources import load_sources

paths = config.wire()
print(f"mlops package {__import__('mlops').__version__}")
print(f"lake    : {config.LAKE_ROOT}     <- written by Stage A")
for k, v in paths.items():
    print(f"  {k:16} {v}")

# What Stage A actually left in Silver, since that is Stage B's whole input.
from deltalake import DeltaTable as _DT
rows = []
for t in ("ms_users", "txn_workout_summaries", "txn_user_subscriptions",
          "txn_payments", "txn_notifications"):
    p = config.LAKE_ROOT / "silver" / t
    rows.append({"silver table": t,
                 "rows": len(_DT(str(p)).to_pandas()) if (p / "_delta_log").exists() else 0})
print("\nSTAGE A OUTPUT THAT STAGE B READS\n")
print(pd.DataFrame(rows).to_string(index=False))
''')

# Part B-1 (maturity) comes straight over.
nb.cells.extend(renumber(ml[3:5], "B"))

# Part B-2 — staging. The markdown is rewritten, because in the combined
# notebook the data no longer arrives from a generator; it arrives from Stage A.
md(r"""
## Part B-2 — Staging: the data the model will learn from

Slide 6 puts **data extraction** and **data validation** before anything else,
and in this notebook both already happened: Stage A extracted the 18 tables,
applied the DQ rules, quarantined what failed them, and wrote the survivors to
Silver. The feature store is built on Silver rather than on the source for that
reason — Silver is the trust boundary, and a row only reaches it after the rules
have run.

The cell below asks a single question: **is there enough data in Silver for the
model's numbers to mean anything?** With the local source that is already true
(2,000 members), and it says so. With Supabase as the source there are 50
members, which is roughly nine churn events in the test split; the generator
then writes a larger set into Silver so the rest of the notebook has something
to measure, and prints that it did. Either way the pipeline is identical — only
the sample size changes, and where the data came from is reported rather than
assumed.
""")
nb.cells.extend(renumber(ml[6:7], "B"))
nb.cells.extend(renumber(ml[7:], "B"))

# ==========================================================================
md(r"""
---

## What the combination adds

Each half proves its own requirements, and the two together prove one more
thing that neither could alone: **the model is trained on the output of the
pipeline that was tested.** The features in Part B-3 are computed from the
Silver zone whose DQ rules were checked in Part A-3, whose load was measured in
Part A-5 and whose failure behaviour was simulated in Part A-6. Slide 24's
point-in-time join is applied to data whose lineage is known hop by hop.

Three things to keep in view when writing this up:

1. **The data is generated.** Stage A proves the pipeline moves it correctly;
   Stage B proves the model pipeline is sound. Neither proves anything about
   how real members behave.
2. **`SOURCE = "local"` is a PostgreSQL server on a disposable VM** with a
   superuser whose password is in the source. That is right for a demo and
   wrong for anything else. The Supabase path is the one that shows the
   pipeline reading a real managed database, with credentials kept out of the
   notebook.
3. **The seam that is now closed was real.** Before these two notebooks were
   joined, the model was trained on a set written straight into Silver, because
   the Supabase project is too small to train on. It worked, and it meant the
   model had never seen the ETL's own output. Worth saying in the report: the
   fix was not a better model, it was a bigger source.
""")

nbf.write(nb, str(OUT))
print(f"wrote {OUT.name} with {len(nb.cells)} cells "
      f"({sum(1 for c in nb.cells if c.cell_type == 'code')} code)")
