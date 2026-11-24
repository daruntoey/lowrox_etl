-- ============================================================
-- 13 — database roles for the scheduled pipeline (GitHub Actions)
--
--   lowrox_etl_reader   reads the app tables (extract). Read-only, sees every row
--                       through a SELECT policy, cannot write anything.
--   lowrox_dw_writer    owns schema dw and rewrites it (etl/publish.py). Cannot
--                       read or write the app tables.
--
-- Neither role is the service-role key or the postgres password, so a leaked
-- GitHub secret cannot change the app's data.
-- Both are created WITHOUT a password and NOLOGIN; you set the passwords yourself
-- (step at the end) — never paste them into chat or commit them.
--
-- Run after 12 (or after loading 12_dw_mock_data.sql): existing dw tables are
-- handed to lowrox_dw_writer. Safe to re-run.
-- ============================================================

BEGIN;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'lowrox_etl_reader') THEN
    CREATE ROLE lowrox_etl_reader NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'lowrox_dw_writer') THEN
    CREATE ROLE lowrox_dw_writer NOLOGIN;
  END IF;
END $$;

-- reader: SELECT on every ms_ / txn_ table, plus a policy so RLS shows all rows
GRANT USAGE ON SCHEMA public TO lowrox_etl_reader;
DO $$
DECLARE t record;
BEGIN
  FOR t IN SELECT tablename FROM pg_tables
           WHERE schemaname = 'public' AND (tablename LIKE 'ms\_%' OR tablename LIKE 'txn\_%') LOOP
    EXECUTE format('GRANT SELECT ON public.%I TO lowrox_etl_reader', t.tablename);
    EXECUTE format('DROP POLICY IF EXISTS lowrox_etl_read ON public.%I', t.tablename);
    EXECUTE format('CREATE POLICY lowrox_etl_read ON public.%I FOR SELECT TO lowrox_etl_reader USING (true)',
                   t.tablename);
  END LOOP;
END $$;
ALTER ROLE lowrox_etl_reader SET default_transaction_read_only = on;
ALTER ROLE lowrox_etl_reader SET statement_timeout = '5min';

-- writer: owns dw, nothing else. Handing dw to it needs membership on PG16+.
DO $$ BEGIN EXECUTE format('GRANT lowrox_dw_writer TO %I', current_user); END $$;
CREATE SCHEMA IF NOT EXISTS dw;
ALTER SCHEMA dw OWNER TO lowrox_dw_writer;
DO $$
DECLARE t record;
BEGIN
  FOR t IN SELECT tablename FROM pg_tables WHERE schemaname = 'dw' LOOP
    EXECUTE format('ALTER TABLE dw.%I OWNER TO lowrox_dw_writer', t.tablename);
  END LOOP;
END $$;
REVOKE ALL ON SCHEMA dw FROM PUBLIC;
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
    EXECUTE 'REVOKE ALL ON SCHEMA dw FROM anon, authenticated';
  END IF;
END $$;
ALTER ROLE lowrox_dw_writer SET statement_timeout = '5min';

COMMIT;

-- ------------------------------------------------------------
-- Then, in the SQL Editor, give each role a password you make up and turn LOGIN on
-- (run these two lines yourself with your own passwords; do not save them in a file):
--
--   ALTER ROLE lowrox_etl_reader LOGIN PASSWORD '<a long random password>';
--   ALTER ROLE lowrox_dw_writer  LOGIN PASSWORD '<another long random password>';
--
-- GitHub secrets (Session pooler, IPv4 — Dashboard → Connect → Session pooler):
--   LOWROX_SOURCE_DSN = postgresql://lowrox_etl_reader.avvxdvvcqexmfpsgtzqm:<password>@<pooler host>:5432/postgres
--   LOWROX_DW_DSN     = postgresql://lowrox_dw_writer.avvxdvvcqexmfpsgtzqm:<password>@<pooler host>:5432/postgres
--
-- Check:
--   SELECT rolname, rolcanlogin FROM pg_roles WHERE rolname LIKE 'lowrox_%';
--   SELECT tableowner, count(*) FROM pg_tables WHERE schemaname = 'dw' GROUP BY 1;
-- ------------------------------------------------------------
