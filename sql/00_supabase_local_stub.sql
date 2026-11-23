-- ============================================================
-- LOCAL / CI ONLY — never run this on Supabase.
--
-- Supabase ships the auth and storage schemas and the anon /
-- authenticated roles. A plain PostgreSQL (the local source, CI) has
-- none of them, so 03, 04, 06 and 07 would fail there. This file adds the
-- smallest stand-ins those scripts reference, so the local source gets
-- exactly the same public schema as Supabase.
--
-- Order on a plain PostgreSQL:
--   01_schema → 02_seed → 00_supabase_local_stub → 03 → 04 → 06 → 07 → 08 → 09
-- ใช้เฉพาะเครื่อง local / CI เท่านั้น ห้ามรันบน Supabase
-- ============================================================
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN CREATE ROLE anon NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN CREATE ROLE authenticated NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN CREATE ROLE service_role NOLOGIN BYPASSRLS; END IF;
END $$;

CREATE SCHEMA IF NOT EXISTS auth;
CREATE SCHEMA IF NOT EXISTS storage;

CREATE TABLE IF NOT EXISTS auth.users (
  id uuid PRIMARY KEY,
  email text,
  raw_user_meta_data jsonb DEFAULT '{}'
);
CREATE OR REPLACE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS $$
  SELECT NULLIF(COALESCE(NULLIF(current_setting('request.jwt.sub', true), ''),
                         NULLIF(current_setting('request.jwt.claims', true), '')::jsonb ->> 'sub'), '')::uuid
$$;

CREATE TABLE IF NOT EXISTS storage.buckets (
  id text PRIMARY KEY, name text, public boolean, file_size_limit bigint, allowed_mime_types text[]
);
CREATE TABLE IF NOT EXISTS storage.objects (id serial PRIMARY KEY, bucket_id text, name text);
ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;
CREATE OR REPLACE FUNCTION storage.foldername(name text) RETURNS text[] LANGUAGE sql AS $$
  SELECT string_to_array(name, '/')
$$;

GRANT USAGE ON SCHEMA public, auth, storage TO anon, authenticated;
GRANT EXECUTE ON FUNCTION auth.uid() TO anon, authenticated;
