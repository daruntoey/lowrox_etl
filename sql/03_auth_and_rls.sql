-- ============================================================
-- Lowrox — Supabase Auth integration + Row Level Security
-- Run AFTER 01_schema.sql and 02_seed.sql
--
-- WHY THIS IS URGENT  |  ทำไมต้องรีบ
-- Supabase exposes every table in the public schema through its auto
-- generated REST API. The anon key sits in the browser bundle and is
-- meant to be public. Until RLS is on, anyone who opens the site can
-- read and write every row, including txn_user_medical_concerns.
-- ตราบใดที่ยังไม่เปิด RLS ใครก็ตามที่เปิดเว็บอ่านและเขียนได้ทุกแถว
-- รวมถึงข้อมูลสุขภาพ ซึ่งเป็นข้อมูลอ่อนไหวตาม PDPA มาตรา 26
-- ============================================================

BEGIN;

-- ------------------------------------------------------------
-- 1. Link ms_users to Supabase Auth
-- ------------------------------------------------------------
-- auth.users owns passwords now, so the column goes.
ALTER TABLE ms_users DROP COLUMN IF EXISTS password_hash;

-- Nullable on purpose: the 50 seeded users have no auth account.
-- They stay for the warehouse; only real sign-ups get an auth_user_id.
-- ตั้งเป็น nullable เพราะ user ตัวอย่าง 50 คนไม่มี auth account
ALTER TABLE ms_users
  ADD COLUMN IF NOT EXISTS auth_user_id uuid UNIQUE
  REFERENCES auth.users(id) ON DELETE CASCADE;

-- Every new auth sign-up gets its ms_users profile row automatically.
-- Without this trigger you end up with auth accounts that have no profile.
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
  INSERT INTO public.ms_users (auth_user_id, username, email, tier)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data->>'username', split_part(NEW.email, '@', 1)),
    NEW.email,
    'free'
  )
  ON CONFLICT (auth_user_id) DO NOTHING;
  RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ------------------------------------------------------------
-- 2. Helper: the signed-in person's ms_users.id
-- ------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.current_user_id()
RETURNS int LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
  SELECT id FROM public.ms_users WHERE auth_user_id = auth.uid()
$$;

-- ------------------------------------------------------------
-- 3. Private data — a row is visible only to the person it belongs to
-- ------------------------------------------------------------
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'ms_user_demographics', 'ms_user_verifications', 'ms_device_connections',
    'txn_user_goals', 'txn_user_medical_concerns', 'txn_user_constraints',
    'txn_user_history_level', 'txn_workout_summaries',
    'txn_user_subscriptions', 'txn_payments', 'txn_notifications']
  LOOP
    EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS own_rows ON public.%I', t);
    EXECUTE format(
      'CREATE POLICY own_rows ON public.%I FOR ALL TO authenticated '
      'USING (user_id = public.current_user_id()) '
      'WITH CHECK (user_id = public.current_user_id())', t);
  END LOOP;
END $$;

-- ------------------------------------------------------------
-- 4. ms_users — read and update your own row only
-- ------------------------------------------------------------
ALTER TABLE ms_users ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS own_profile ON ms_users;
CREATE POLICY own_profile ON ms_users FOR SELECT TO authenticated
  USING (auth_user_id = auth.uid());

DROP POLICY IF EXISTS update_own_profile ON ms_users;
CREATE POLICY update_own_profile ON ms_users FOR UPDATE TO authenticated
  USING (auth_user_id = auth.uid()) WITH CHECK (auth_user_id = auth.uid());

-- Social features need usernames without exposing email. A view owned by
-- postgres reads past RLS and publishes only the safe columns.
-- view นี้เปิดเฉพาะคอลัมน์ที่ปลอดภัย ไม่มี email
CREATE OR REPLACE VIEW public.v_user_public AS
  SELECT id, username, tier FROM public.ms_users;
GRANT SELECT ON public.v_user_public TO authenticated;

-- ------------------------------------------------------------
-- 5. Reference data — everyone signed in may read, nobody may write
-- ------------------------------------------------------------
ALTER TABLE ms_area ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS read_areas ON ms_area;
CREATE POLICY read_areas ON ms_area FOR SELECT TO authenticated USING (true);

-- The pricing page is public, so anon may read plans and nothing else.
ALTER TABLE ms_subscription_plans ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS read_plans ON ms_subscription_plans;
CREATE POLICY read_plans ON ms_subscription_plans FOR SELECT TO anon, authenticated
  USING (is_active = true);

-- ------------------------------------------------------------
-- 6. Social data — discoverable by all, writable by the owner
-- ------------------------------------------------------------
ALTER TABLE txn_events ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS read_events ON txn_events;
CREATE POLICY read_events ON txn_events FOR SELECT TO authenticated USING (true);
DROP POLICY IF EXISTS write_own_events ON txn_events;
CREATE POLICY write_own_events ON txn_events FOR ALL TO authenticated
  USING (event_owner = public.current_user_id())
  WITH CHECK (event_owner = public.current_user_id());

ALTER TABLE txn_event_participants ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS read_participants ON txn_event_participants;
CREATE POLICY read_participants ON txn_event_participants FOR SELECT TO authenticated
  USING (true);
DROP POLICY IF EXISTS write_own_participation ON txn_event_participants;
CREATE POLICY write_own_participation ON txn_event_participants FOR ALL TO authenticated
  USING (user_id = public.current_user_id())
  WITH CHECK (user_id = public.current_user_id());

ALTER TABLE txn_group_comments ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS read_comments ON txn_group_comments;
CREATE POLICY read_comments ON txn_group_comments FOR SELECT TO authenticated USING (true);
DROP POLICY IF EXISTS write_own_comments ON txn_group_comments;
CREATE POLICY write_own_comments ON txn_group_comments FOR ALL TO authenticated
  USING (user_id = public.current_user_id())
  WITH CHECK (user_id = public.current_user_id());

ALTER TABLE txn_training_plans ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS read_plans_txn ON txn_training_plans;
CREATE POLICY read_plans_txn ON txn_training_plans FOR SELECT TO authenticated
  USING (plan_status = 'published');
DROP POLICY IF EXISTS write_own_plans ON txn_training_plans;
CREATE POLICY write_own_plans ON txn_training_plans FOR ALL TO authenticated
  USING (created_by = public.current_user_id())
  WITH CHECK (created_by = public.current_user_id());

COMMIT;

-- ============================================================
-- Verify: every table should report rowsecurity = true
-- ตรวจสอบ: ทุกตารางต้องขึ้น true
-- ============================================================
-- SELECT tablename, rowsecurity FROM pg_tables
-- WHERE schemaname = 'public' ORDER BY rowsecurity, tablename;
