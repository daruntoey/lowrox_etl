"""Lowrox ETL/ELT pipeline — OLTP -> Bronze -> Silver -> Gold."""

# Bumped whenever the package changes in a way the notebook depends on. The
# setup cell compares this against what it expects, because the commonest
# confusing failure is a NEW notebook running against an OLD checkout: the
# notebook is downloaded directly, the package arrives by git clone, and a
# repo that has not been re-pushed silently supplies last week's code.
__version__ = "2026.11.10"
from .pipeline import run_pipeline, reset_lake, RunResult
from .config import RunConfig, SOURCE_TABLES, LAKE_ROOT
from .faults import FaultInjector, FAULT_CATALOGUE
from .preflight import diagnose, lake_status, require_gold
__all__ = ["run_pipeline", "reset_lake", "RunResult", "RunConfig",
           "SOURCE_TABLES", "LAKE_ROOT", "FaultInjector", "FAULT_CATALOGUE",
           "diagnose", "lake_status", "require_gold"]
