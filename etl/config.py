"""
config.py — single responsibility: WHERE things live and WHAT the contract is.

No extraction, no transformation, no loading happens here. Every other module
imports its settings from this one place, which is what makes the pipeline
configurable without editing code (slide: "Modular Design in ETL").
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# 1. Source — the OLTP database (Supabase in production, local PG in tests)
# --------------------------------------------------------------------------
# Supabase users: use the SESSION POOLER, not the direct connection.
#   host  aws-<index>-<region>.pooler.supabase.com
#   port  5432
#   user  postgres.<project-ref>
# The direct connection db.<ref>.supabase.co is IPv6-only on Free/Pro, and
# Google Colab has no IPv6, so it will simply time out.
SOURCE_DSN = os.environ.get(
    "LOWROX_SOURCE_DSN",
    "postgresql://postgres@localhost:5432/lowrox",
)

SOURCE_SYSTEM = "supabase_oltp"

# --------------------------------------------------------------------------
# 2. Lake — Bronze / Silver / Gold as Delta tables on a local or mounted path
# --------------------------------------------------------------------------
LAKE_ROOT = Path(os.environ.get("LOWROX_LAKE_ROOT", "/home/claude/lowrox_lake"))

BRONZE = LAKE_ROOT / "bronze"
SILVER = LAKE_ROOT / "silver"
GOLD = LAKE_ROOT / "gold"
CONTROL = LAKE_ROOT / "_control"      # watermarks, run log, lineage, DLQ

for _p in (BRONZE, SILVER, GOLD, CONTROL):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 3. Lineage columns — the contract between every pair of layers
# --------------------------------------------------------------------------
# Every Bronze row carries these. Silver and Gold carry them forward. This is
# what makes the lineage test possible: a Gold row can name the exact Bronze
# row it came from, which can name the exact source primary key.
LINEAGE_COLS = [
    "_lineage_id",      # uuid, unique per extracted row version
    "_run_id",          # which pipeline run produced it
    "_batch_id",        # which extract batch inside that run
    "_src_system",      # 'supabase_oltp'
    "_src_table",       # 'txn_workout_summaries'
    "_src_pk",          # primary key value in the source, as text
    "_extracted_at",    # when the extractor read it
    "_load_type",       # 'init' | 'incremental'
]

# --------------------------------------------------------------------------
# 4. Table registry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceTable:
    """One OLTP table and how to read it incrementally."""
    name: str
    pk: tuple[str, ...]
    watermark_col: str          # the column the incremental extract filters on
    is_append_only: bool        # True -> rows are never updated after insert
    sensitive: bool = False     # PDPA s.26 — never promoted past restricted Silver

    @property
    def bronze_path(self) -> str:
        return str(BRONZE / self.name)

    @property
    def silver_path(self) -> str:
        return str(SILVER / self.name)


# The 18 OLTP tables. watermark_col is 'updated_at' wherever the source
# maintains one; txn_workout_summaries is append-only from the device feed and
# has only created_at, which is the correct watermark for it.
SOURCE_TABLES: list[SourceTable] = [
    SourceTable("ms_users",                ("id",),      "updated_at", False),
    SourceTable("ms_user_demographics",    ("user_id",), "updated_at", False),
    SourceTable("ms_user_verifications",   ("id",),      "updated_at", False),
    SourceTable("ms_area",                 ("id",),      "updated_at", False),
    SourceTable("ms_subscription_plans",   ("id",),      "updated_at", False),
    SourceTable("ms_device_connections",   ("id",),      "updated_at", False),
    SourceTable("txn_events",              ("id",),      "updated_at", False),
    SourceTable("txn_workout_summaries",   ("id",),      "created_at", True),
    SourceTable("txn_user_history_level",  ("id",),      "updated_at", False),
    SourceTable("txn_user_goals",          ("id",),      "updated_at", False),
    SourceTable("txn_user_medical_concerns", ("id",),    "updated_at", False, sensitive=True),
    SourceTable("txn_user_constraints",    ("id",),      "updated_at", False),
    SourceTable("txn_event_participants",  ("id",),      "updated_at", False),
    SourceTable("txn_group_comments",      ("id",),      "updated_at", False),
    SourceTable("txn_training_plans",      ("id",),      "updated_at", False),
    SourceTable("txn_user_subscriptions",  ("id",),      "updated_at", False),
    SourceTable("txn_payments",            ("id",),      "updated_at", False),
    SourceTable("txn_notifications",       ("id",),      "updated_at", False),
]

TABLES_BY_NAME = {t.name: t for t in SOURCE_TABLES}

# --------------------------------------------------------------------------
# 5. Runtime behaviour
# --------------------------------------------------------------------------
@dataclass
class RunConfig:
    load_type: str = "init"            # 'init' | 'incremental'
    max_workers: int = 4               # parallel extraction (slide: Parallel Extraction)
    batch_rows: int = 5_000            # chunked load (slide: Parallel Loading)
    retry_attempts: int = 3            # slide: Retry Mechanism
    retry_base_delay: float = 0.25     # seconds, exponential backoff
    retry_max_delay: float = 4.0
    fail_fast: bool = False            # False -> bad rows go to the DLQ, run continues
    tables: tuple[str, ...] = field(default_factory=tuple)   # empty = all

    def selected(self) -> list[SourceTable]:
        if not self.tables:
            return SOURCE_TABLES
        return [TABLES_BY_NAME[n] for n in self.tables]


# Special dimension keys — the convention fixed in the warehouse design.
KEY_NOT_APPLICABLE = 0
KEY_UNKNOWN = -1
KEY_INVALID = -2
