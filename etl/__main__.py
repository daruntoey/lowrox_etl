"""
python -m etl [--publish] [--init]

The scheduled entry point (.github/workflows/etl-daily.yml):
  * incremental when the lake already has Gold, init on an empty lake (first run)
  * --publish copies Gold into Supabase schema dw afterwards ($LOWROX_DW_DSN)
Exit code 1 when the run or the publish fails, so the workflow goes red.
"""
from __future__ import annotations

import argparse
import sys

from .config import GOLD
from .pipeline import reset_lake, run_pipeline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m etl")
    ap.add_argument("--init", action="store_true", help="rebuild the lake from nothing")
    ap.add_argument("--publish", action="store_true", help="publish Gold to schema dw afterwards")
    args = ap.parse_args(argv)

    first_run = not (GOLD / "dim_user" / "_delta_log").exists()
    load_type = "init" if (args.init or first_run) else "incremental"
    if load_type == "init":
        reset_lake()
    result = run_pipeline(load_type, echo=True)
    print(result)
    if not result.ok:
        return 1
    if args.publish:
        from .publish import main as publish_main
        return publish_main([])
    return 0


if __name__ == "__main__":
    sys.exit(main())
