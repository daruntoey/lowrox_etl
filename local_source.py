"""
local_source.py — build the OLTP source database on the machine that is running
the notebook, so the full pipeline can run with no credentials at all.

Why this exists
---------------
The ETL half of this project extracts from a PostgreSQL source. That was
Supabase, which means a connection string, a password in a notebook that gets
handed in, and a project that holds 50 members — enough to prove a pipeline,
far too small to train a model on. The MLOps half worked around the size by
generating a larger set and writing it straight into Silver, which left a seam
in the middle of the story: the model was not learning from the ETL's own
output.

This module removes both problems. It installs PostgreSQL where the notebook is
running, seeds it from `generate.py` at three sizes, and hands back a DSN. The
pipeline then extracts from a real database, Silver and Gold are built by the
real ETL, and the feature store is computed from that Gold-grade Silver — one
chain, no seam, nothing to paste.

    source PostgreSQL  ->  Bronze  ->  Silver  ->  Gold  ->  feature store  ->  model

The three seeds are the same 18-table schema at 50, 500 and 2,000 rows per
table, which is also exactly what the performance suite's scaling curve looks
for (`lowrox`, `lowrox_s10`, `lowrox_s40`).

What this is not
----------------
A production deployment. The role is a superuser with a well-known password on
a loopback-only server holding generated data, which is appropriate for a
throwaway Colab VM and would be indefensible anywhere else.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROLE = "lowrox"
PASSWORD = "lowrox"          # loopback only, generated data, disposable VM
HOST = "localhost"
PORT = 5432

# (database, generator scale) — 50 rows per table per unit of scale.
SEEDS: list[tuple[str, int]] = [
    ("lowrox", 1),           # 50 rows/table
    ("lowrox_s10", 10),      # 500
    ("lowrox_s40", 40),      # 2000
]
MAIN_DB = "lowrox_s40"       # what the pipeline extracts from
ROWS_PER_UNIT = 50


def dsn_for(db: str) -> str:
    return f"postgresql://{ROLE}:{PASSWORD}@{HOST}:{PORT}/{db}"


# --------------------------------------------------------------------------
def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kw)


def _as_postgres(args: list[str]) -> list[str]:
    """
    Run a command as the postgres system user.

    `sudo -n` never prompts. A notebook that stops for a password prompt looks
    like a hang with no output, which is the worst failure mode here — better
    to fail with a message.
    """
    if os.geteuid() == 0 and shutil.which("sudo"):
        return ["sudo", "-n", "-u", "postgres", *args]
    return args


def _psql(sql: str, db: str = "postgres") -> subprocess.CompletedProcess:
    return _run(_as_postgres(["psql", "-qtAc", sql, db]))


def _psql_file(path: str | Path, db: str) -> subprocess.CompletedProcess:
    return _run(_as_postgres(["psql", "-q", "-v", "ON_ERROR_STOP=1",
                              "-f", str(path), db]))


def _server_up() -> bool:
    if not shutil.which("pg_isready"):
        return False
    return _run(["pg_isready", "-q", "-h", HOST, "-p", str(PORT)]).returncode == 0


def _count(db: str) -> int | None:
    """ms_users row count over TCP as the application role, or None."""
    try:
        import psycopg2
    except ImportError:
        return None
    try:
        with psycopg2.connect(dsn_for(db), connect_timeout=5) as c, c.cursor() as cur:
            cur.execute("SELECT count(*) FROM ms_users")
            return int(cur.fetchone()[0])
    except Exception:
        return None


# --------------------------------------------------------------------------
def _install() -> None:
    if shutil.which("psql") and Path("/etc/postgresql").exists():
        return
    if not shutil.which("apt-get"):
        raise RuntimeError(
            "PostgreSQL is not installed and this is not a Debian/Ubuntu machine,\n"
            "so it cannot be installed from here. Either:\n"
            "  - run this notebook on Colab, where it can be, or\n"
            "  - install PostgreSQL yourself and set LOWROX_SOURCE_DSN, or\n"
            "  - set SOURCE = \"supabase\" and use your own project.")
    if os.geteuid() != 0:
        raise RuntimeError(
            "PostgreSQL is not installed and this process is not root, so it\n"
            "cannot install it. Set SOURCE = \"supabase\", or install it yourself\n"
            "and set LOWROX_SOURCE_DSN.")
    print("installing PostgreSQL (one-off, about a minute)...", flush=True)
    env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
    r = _run(["apt-get", "install", "-y", "-qq", "postgresql"], env=env)
    if r.returncode != 0:
        _run(["apt-get", "update", "-qq"], env=env)
        r = _run(["apt-get", "install", "-y", "-qq", "postgresql"], env=env)
    if r.returncode != 0 or not shutil.which("psql"):
        raise RuntimeError("could not install PostgreSQL:\n"
                           + (r.stderr or r.stdout)[-1500:])
    print("  installed", flush=True)


def _start(timeout: int = 60) -> None:
    if _server_up():
        return
    # No systemd in a container or on Colab, so `service` (the SysV script) and
    # pg_ctlcluster are the two that work. Try both before giving up.
    attempts: list[list[str]] = []
    if shutil.which("service"):
        attempts.append(["service", "postgresql", "start"])
    if shutil.which("pg_ctlcluster"):
        versions = sorted(p.name for p in Path("/etc/postgresql").glob("*")) \
            if Path("/etc/postgresql").exists() else []
        for v in reversed(versions):
            attempts.append(["pg_ctlcluster", v, "main", "start"])
    for cmd in attempts:
        _run(cmd)
        deadline = time.time() + timeout / max(1, len(attempts))
        while time.time() < deadline:
            if _server_up():
                print(f"  server started ({' '.join(cmd[:2])})", flush=True)
                return
            time.sleep(1)
    raise RuntimeError(
        "PostgreSQL is installed but will not start. Tried: "
        + "; ".join(" ".join(c) for c in attempts))


def _ensure_role() -> None:
    # CREATE ROLE is not idempotent and an existing role cannot be dropped
    # while it owns anything, so the check goes inside the statement.
    r = _psql(
        f"DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='{ROLE}') "
        f"THEN CREATE ROLE {ROLE} LOGIN PASSWORD '{PASSWORD}' SUPERUSER; "
        f"ELSE ALTER ROLE {ROLE} LOGIN PASSWORD '{PASSWORD}' SUPERUSER; END IF; END $$;")
    if r.returncode != 0:
        raise RuntimeError(f"could not create the {ROLE} role:\n{r.stderr[-800:]}")


def _seed(db: str, scale: int, project_dir: Path, verbose: bool) -> int:
    expected = scale * ROWS_PER_UNIT
    have = _count(db)
    if have == expected:
        if verbose:
            print(f"  {db:11s} already holds {have:,} users")
        return have

    _psql(f'DROP DATABASE IF EXISTS "{db}"')
    r = _psql(f'CREATE DATABASE "{db}" OWNER {ROLE}')
    if r.returncode != 0:
        raise RuntimeError(f"could not create database {db}:\n{r.stderr[-800:]}")

    # generate.py writes out/02_seed.sql and out/csv/ for ONE scale at a time,
    # so each database is generated immediately before it is loaded.
    g = _run([sys.executable, str(project_dir / "generate.py"), "--scale", str(scale)],
             cwd=str(project_dir))
    if g.returncode != 0:
        raise RuntimeError(f"generate.py --scale {scale} failed:\n{g.stderr[-1500:]}")

    schema = project_dir / "sql" / "01_schema.sql"
    seed = project_dir / "out" / "02_seed.sql"
    if not seed.exists():
        raise RuntimeError(f"the generator did not write {seed}")
    for f in (schema, seed):
        # The files are read by the postgres system user, which cannot see a
        # project folder that is only readable by its owner.
        try:
            os.chmod(f, 0o644)
        except OSError:
            pass
        r = _psql_file(f, db)
        if r.returncode != 0:
            raise RuntimeError(f"loading {f.name} into {db} failed:\n{r.stderr[-1500:]}")

    n = _count(db)
    if n != expected:
        raise RuntimeError(
            f"{db} loaded but holds {n} users, expected {expected}. "
            f"The role may not be able to read the tables over TCP.")
    if verbose:
        print(f"  {db:11s} seeded with {n:,} users", flush=True)
    return n


# --------------------------------------------------------------------------
def ensure_local_source(project_dir: str | Path = ".", *, main_db: str = MAIN_DB,
                        seeds: list[tuple[str, int]] | None = None,
                        verbose: bool = True) -> dict:
    """
    Install, start and seed a local PostgreSQL source, and return its DSN.

    Idempotent: a database that already holds the right number of rows is left
    alone, so re-running the setup cell costs a few seconds rather than a
    rebuild.
    """
    project_dir = Path(project_dir).resolve()
    if not (project_dir / "generate.py").is_file():
        raise FileNotFoundError(f"generate.py not found in {project_dir}")
    seeds = seeds or SEEDS

    _install()
    _start()
    _ensure_role()

    counts = {}
    for db, scale in seeds:
        counts[db] = _seed(db, scale, project_dir, verbose)

    dsn = dsn_for(main_db)
    os.environ["LOWROX_SOURCE_DSN"] = dsn
    if verbose:
        print(f"\nsource ready: {main_db} with {counts[main_db]:,} members")
        print(f"  the other seeds ({', '.join(d for d in counts if d != main_db)}) "
              f"are what the\n  performance suite's scaling curve measures.")
    return {"dsn": dsn, "main_db": main_db, "counts": counts,
            "users": counts[main_db]}


if __name__ == "__main__":
    import json
    out = ensure_local_source(Path(__file__).parent)
    print(json.dumps({k: v for k, v in out.items() if k != "dsn"}, indent=2))
