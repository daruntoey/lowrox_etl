-- ============================================================
-- Lowrox — wire the web app (aom-thanya/lowrox) to the OLTP schema
-- Run AFTER lowrox_etl/sql/01_schema.sql, 02_seed.sql, 03_auth_and_rls.sql
-- Supabase Dashboard → SQL Editor → paste → Run.  Safe to re-run.
--
-- ms_/txn_ tables stay the single source of truth (the ETL reads them).
-- This script only ADDS what the web app needs: profile columns, event
-- display columns, a few RPCs that must be atomic, storage buckets and RLS.
-- ไฟล์นี้เพิ่มเฉพาะสิ่งที่หน้าเว็บต้องใช้ ไม่ลบข้อมูลเดิม
-- ============================================================

BEGIN;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_schema = 'public' AND table_name = 'ms_users' AND column_name = 'auth_user_id') THEN
    RAISE EXCEPTION 'Run lowrox_etl/sql/03_auth_and_rls.sql first';
  END IF;
END $$;

-- ------------------------------------------------------------
-- 0. Lenient casts: bad form input becomes NULL instead of failing a save
-- ------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.try_date(v text) RETURNS date LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN RETURN NULLIF(v, '')::date; EXCEPTION WHEN others THEN RETURN NULL; END $$;
CREATE OR REPLACE FUNCTION public.try_time(v text) RETURNS time LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN RETURN NULLIF(v, '')::time; EXCEPTION WHEN others THEN RETURN NULL; END $$;
CREATE OR REPLACE FUNCTION public.try_numeric(v text) RETURNS numeric LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN RETURN NULLIF(v, '')::numeric; EXCEPTION WHEN others THEN RETURN NULL; END $$;

-- ------------------------------------------------------------
-- 1. ms_users — profile fields the app edits
-- ------------------------------------------------------------
ALTER TABLE ms_users
  ADD COLUMN IF NOT EXISTS display_name           varchar(50),
  ADD COLUMN IF NOT EXISTS bio                    varchar(300),
  ADD COLUMN IF NOT EXISTS province_id            varchar(10),
  ADD COLUMN IF NOT EXISTS district_id            varchar(10),
  ADD COLUMN IF NOT EXISTS avatar_url             text,
  ADD COLUMN IF NOT EXISTS onboarding_status      varchar(20) NOT NULL DEFAULT 'not_started',
  ADD COLUMN IF NOT EXISTS onboarding_data        jsonb,
  ADD COLUMN IF NOT EXISTS show_profile           boolean NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS public_profile_visible boolean NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS public_area_visible    boolean NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS interests              text[] NOT NULL DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS deleted_at             timestamptz;

COMMENT ON COLUMN ms_users.onboarding_data IS
  'The onboarding form exactly as the app shows it (UI selections included), for
   round-trip editing. The analysable facts are ALSO written to
   ms_user_demographics / txn_user_goals / txn_user_constraints /
   txn_user_medical_concerns / txn_user_history_level by save_onboarding().';

-- A signed-in user may change only these columns of their own row.
-- tier, email and auth_user_id stay server-controlled.
REVOKE UPDATE ON ms_users FROM authenticated;
GRANT UPDATE (display_name, bio, province_id, district_id, avatar_url,
              onboarding_status, onboarding_data, show_profile,
              public_profile_visible, public_area_visible, interests, updated_at)
  ON ms_users TO authenticated;

-- Sign-up → profile row. Replaces the 03 version:
--  * a verified email that matches a seeded member claims that row
--  * usernames get a suffix so two people called "nok@..." don't collide
--  * never blocks the sign-up itself
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE base text;
BEGIN
  UPDATE public.ms_users SET auth_user_id = NEW.id, updated_at = now()
   WHERE lower(email) = lower(NEW.email) AND auth_user_id IS NULL;
  IF NOT FOUND THEN
    base := left(regexp_replace(COALESCE(NEW.raw_user_meta_data->>'username', split_part(NEW.email, '@', 1)),
                                '[^A-Za-z0-9_.]', '', 'g'), 80);
    INSERT INTO public.ms_users (auth_user_id, username, email, tier, display_name)
    VALUES (NEW.id,
            COALESCE(NULLIF(base, ''), 'member') || '_' || left(replace(NEW.id::text, '-', ''), 8),
            NEW.email, 'free',
            left(COALESCE(NEW.raw_user_meta_data->>'display_name', split_part(NEW.email, '@', 1)), 50))
    ON CONFLICT DO NOTHING;
  END IF;
  RETURN NEW;
EXCEPTION WHEN others THEN
  RAISE WARNING 'handle_new_user failed for %: %', NEW.id, SQLERRM;
  RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Called by the app after every sign-in: returns the profile, creating it if the
-- trigger was missed (accounts made before this script ran).
CREATE OR REPLACE FUNCTION public.ensure_my_profile()
RETURNS SETOF public.ms_users LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE u auth.users%ROWTYPE;
BEGIN
  IF auth.uid() IS NULL THEN RAISE EXCEPTION 'not authenticated' USING ERRCODE = '28000'; END IF;
  IF NOT EXISTS (SELECT 1 FROM ms_users WHERE auth_user_id = auth.uid()) THEN
    SELECT * INTO u FROM auth.users WHERE id = auth.uid();
    PERFORM public.handle_new_user_row(u);
  END IF;
  RETURN QUERY SELECT * FROM ms_users WHERE auth_user_id = auth.uid();
END $$;

CREATE OR REPLACE FUNCTION public.handle_new_user_row(u auth.users)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE base text;
BEGIN
  UPDATE ms_users SET auth_user_id = u.id, updated_at = now()
   WHERE lower(email) = lower(u.email) AND auth_user_id IS NULL;
  IF NOT FOUND THEN
    base := left(regexp_replace(split_part(u.email, '@', 1), '[^A-Za-z0-9_.]', '', 'g'), 80);
    INSERT INTO ms_users (auth_user_id, username, email, tier, display_name)
    VALUES (u.id, COALESCE(NULLIF(base, ''), 'member') || '_' || left(replace(u.id::text, '-', ''), 8),
            u.email, 'free', left(split_part(u.email, '@', 1), 50))
    ON CONFLICT DO NOTHING;
  END IF;
END $$;
REVOKE ALL ON FUNCTION public.handle_new_user_row(auth.users) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.ensure_my_profile() TO authenticated;

-- Public face of a member. Owned by postgres, so it reads past RLS and
-- publishes only safe columns, honouring the member's visibility switches.
-- (column order of 03's view is kept so CREATE OR REPLACE works)
CREATE OR REPLACE VIEW public.v_user_public AS
  SELECT u.id, u.username, u.tier,
         COALESCE(NULLIF(u.display_name, ''), u.username)                         AS display_name,
         u.avatar_url,
         CASE WHEN u.public_profile_visible THEN u.bio END                         AS bio,
         CASE WHEN u.public_profile_visible AND u.public_area_visible THEN u.province_id END AS province_id,
         CASE WHEN u.public_profile_visible AND u.public_area_visible THEN u.district_id END AS district_id,
         CASE WHEN u.public_profile_visible THEN u.interests ELSE '{}' END          AS interests,
         (u.public_profile_visible AND u.deleted_at IS NULL)                        AS profile_visible,
         (SELECT h.fitness_level FROM txn_user_history_level h
           WHERE h.user_id = u.id ORDER BY h.recorded_at DESC NULLS LAST, h.id DESC LIMIT 1) AS fitness_level
    FROM public.ms_users u;
GRANT SELECT ON public.v_user_public TO anon, authenticated;

-- ------------------------------------------------------------
-- 2. txn_events — what the event and party pages display
-- ------------------------------------------------------------
ALTER TABLE txn_events
  ADD COLUMN IF NOT EXISTS title               varchar(150),
  ADD COLUMN IF NOT EXISTS description         text,
  ADD COLUMN IF NOT EXISTS location_label      varchar(200),
  ADD COLUMN IF NOT EXISTS ends_at             timestamptz,
  ADD COLUMN IF NOT EXISTS join_url            text,
  ADD COLUMN IF NOT EXISTS cover_url           text,
  ADD COLUMN IF NOT EXISTS visibility          varchar(20) NOT NULL DEFAULT 'public',
  ADD COLUMN IF NOT EXISTS is_party            boolean NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS party_style         varchar(20),
  ADD COLUMN IF NOT EXISTS registration_closed boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS deleted_at          timestamptz;

-- Member-hosted parties have a free-text meeting point, not a master area.
ALTER TABLE txn_events ALTER COLUMN area_id DROP NOT NULL;
ALTER TABLE txn_events DROP CONSTRAINT IF EXISTS txn_events_status_check;
ALTER TABLE txn_events ADD CONSTRAINT txn_events_status_check
  CHECK (status IN ('draft', 'scheduled', 'full', 'completed', 'cancelled'));
ALTER TABLE txn_events DROP CONSTRAINT IF EXISTS txn_events_visibility_check;
ALTER TABLE txn_events ADD CONSTRAINT txn_events_visibility_check
  CHECK (visibility IN ('public', 'private'));

ALTER TABLE txn_group_comments
  ADD COLUMN IF NOT EXISTS images jsonb NOT NULL DEFAULT '[]';

-- Deleting a workout must not erase the level history it produced (keep the row, drop the link).
ALTER TABLE txn_user_history_level DROP CONSTRAINT IF EXISTS txn_user_history_level_source_workout_id_fkey;
ALTER TABLE txn_user_history_level ADD CONSTRAINT txn_user_history_level_source_workout_id_fkey
  FOREIGN KEY (source_workout_id) REFERENCES txn_workout_summaries(id) ON DELETE SET NULL;

-- Device settings the Devices page toggles.
ALTER TABLE ms_device_connections
  ADD COLUMN IF NOT EXISTS import_runs   boolean NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS post_to_party boolean NOT NULL DEFAULT false;

-- Weekly plan check-offs (My Level page).
CREATE TABLE IF NOT EXISTS txn_plan_checkoffs (
  user_id    int  NOT NULL REFERENCES ms_users(id) ON DELETE CASCADE,
  week_start date NOT NULL,
  session_id varchar(50) NOT NULL,
  created_at timestamptz DEFAULT now(),
  PRIMARY KEY (user_id, week_start, session_id)
);

-- Seeded tables use explicit ids; move sequences past them so app inserts don't collide.
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['ms_users','txn_events','txn_event_participants','txn_group_comments',
                           'txn_workout_summaries','txn_user_history_level','txn_user_goals',
                           'txn_user_constraints','txn_user_medical_concerns','ms_device_connections']
  LOOP
    EXECUTE format('SELECT setval(pg_get_serial_sequence(%L, ''id''), GREATEST((SELECT COALESCE(MAX(id), 0) FROM %I), 1))', t, t);
  END LOOP;
END $$;

-- ------------------------------------------------------------
-- 3. RLS — visitors may browse public events; members own their rows
-- ------------------------------------------------------------
ALTER TABLE txn_plan_checkoffs ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS own_rows ON txn_plan_checkoffs;
CREATE POLICY own_rows ON txn_plan_checkoffs FOR ALL TO authenticated
  USING (user_id = public.current_user_id()) WITH CHECK (user_id = public.current_user_id());

CREATE OR REPLACE FUNCTION public.event_is_public(p_event_id int)
RETURNS boolean LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
  SELECT EXISTS (SELECT 1 FROM txn_events e WHERE e.id = p_event_id
                 AND e.visibility = 'public' AND e.status <> 'draft' AND e.deleted_at IS NULL)
$$;

DROP POLICY IF EXISTS read_areas ON ms_area;
CREATE POLICY read_areas ON ms_area FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS read_events ON txn_events;
CREATE POLICY read_events ON txn_events FOR SELECT TO anon, authenticated
  USING ((visibility = 'public' AND status <> 'draft' AND deleted_at IS NULL)
         OR event_owner = public.current_user_id());

DROP POLICY IF EXISTS read_participants ON txn_event_participants;
CREATE POLICY read_participants ON txn_event_participants FOR SELECT TO anon, authenticated
  USING (public.event_is_public(event_id) OR user_id = public.current_user_id());

DROP POLICY IF EXISTS read_comments ON txn_group_comments;
CREATE POLICY read_comments ON txn_group_comments FOR SELECT TO anon, authenticated
  USING (public.event_is_public(event_id));

-- ------------------------------------------------------------
-- 4. RPCs — the writes that must be checked or atomic
-- ------------------------------------------------------------

-- Join with a capacity check that two people can't race past.
CREATE OR REPLACE FUNCTION public.join_event(p_event_id int)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE uid int := public.current_user_id(); e txn_events%ROWTYPE; taken int;
BEGIN
  IF uid IS NULL THEN RAISE EXCEPTION 'not authenticated' USING ERRCODE = '28000'; END IF;
  SELECT * INTO e FROM txn_events WHERE id = p_event_id FOR UPDATE;
  IF NOT FOUND OR e.visibility <> 'public' OR e.deleted_at IS NOT NULL OR e.status NOT IN ('scheduled', 'full') THEN
    RAISE EXCEPTION 'Event is no longer active';
  END IF;
  IF COALESCE(e.ends_at, e.scheduled_at) <= now() OR e.registration_closed THEN
    RAISE EXCEPTION 'Event is no longer active';
  END IF;
  IF EXISTS (SELECT 1 FROM txn_event_participants WHERE event_id = p_event_id AND user_id = uid AND status = 'joined') THEN
    RETURN;
  END IF;
  SELECT count(*) INTO taken FROM txn_event_participants WHERE event_id = p_event_id AND status = 'joined';
  IF e.max_participants IS NOT NULL AND taken >= e.max_participants THEN
    RAISE EXCEPTION 'Party is full';
  END IF;
  INSERT INTO txn_event_participants (user_id, event_id, joined_at, role, status)
  VALUES (uid, p_event_id, now(), CASE WHEN e.event_owner = uid THEN 'host' ELSE 'participant' END, 'joined')
  ON CONFLICT (user_id, event_id) DO UPDATE SET status = 'joined', joined_at = now(), updated_at = now();
  UPDATE txn_events SET status = CASE WHEN e.max_participants IS NOT NULL AND taken + 1 >= e.max_participants
                                      THEN 'full' ELSE 'scheduled' END, updated_at = now()
   WHERE id = p_event_id;
END $$;

CREATE OR REPLACE FUNCTION public.leave_event(p_event_id int)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE uid int := public.current_user_id();
BEGIN
  IF uid IS NULL THEN RAISE EXCEPTION 'not authenticated' USING ERRCODE = '28000'; END IF;
  DELETE FROM txn_event_participants WHERE event_id = p_event_id AND user_id = uid;
  UPDATE txn_events SET status = 'scheduled', updated_at = now() WHERE id = p_event_id AND status = 'full';
END $$;

-- Create a party and seat the host in one transaction.
CREATE OR REPLACE FUNCTION public.create_party(
  p_title text, p_workout_type text, p_scheduled_at timestamptz, p_location text,
  p_description text, p_target_level text, p_party_style text, p_max_participants int)
RETURNS int LANGUAGE plpgsql SECURITY INVOKER SET search_path = public AS $$
DECLARE uid int := public.current_user_id(); new_id int;
BEGIN
  IF uid IS NULL THEN RAISE EXCEPTION 'not authenticated' USING ERRCODE = '28000'; END IF;
  INSERT INTO txn_events (event_owner, title, workout_type, scheduled_at, location_label, description,
                          target_level, party_style, max_participants, status, visibility, is_party)
  VALUES (uid, left(p_title, 150), p_workout_type, p_scheduled_at, left(p_location, 200), p_description,
          p_target_level, p_party_style, p_max_participants, 'scheduled', 'public', true)
  RETURNING id INTO new_id;
  INSERT INTO txn_event_participants (user_id, event_id, joined_at, role, status)
  VALUES (uid, new_id, now(), 'host', 'joined');
  RETURN new_id;
END $$;

-- Save the onboarding form: keep the form JSON for the editor, and write the
-- analysable facts into the normalised tables the ETL reads.
CREATE OR REPLACE FUNCTION public.save_onboarding(p_data jsonb, p_complete boolean, p_fitness_level text)
RETURNS void LANGUAGE plpgsql SECURITY INVOKER SET search_path = public AS $$
DECLARE
  uid int := public.current_user_id();
  item jsonb;
  v_dist numeric := public.try_numeric(p_data #>> '{fitnessLevel,runningDistance}');
  v_dur  numeric := public.try_numeric(p_data #>> '{fitnessLevel,runningDuration}');
  last_level txn_user_history_level%ROWTYPE;
BEGIN
  IF uid IS NULL THEN RAISE EXCEPTION 'not authenticated' USING ERRCODE = '28000'; END IF;

  UPDATE ms_users SET onboarding_data = p_data,
         onboarding_status = CASE WHEN p_complete THEN 'completed' ELSE onboarding_status END,
         updated_at = now()
   WHERE id = uid;

  INSERT INTO ms_user_demographics (user_id, birth_date, gender)
  VALUES (uid, public.try_date(p_data #>> '{demographics,birthDate}'), NULLIF(p_data #>> '{demographics,gender}', ''))
  ON CONFLICT (user_id) DO UPDATE SET birth_date = EXCLUDED.birth_date, gender = EXCLUDED.gender, updated_at = now();

  -- Goals, availability and health describe the member now: replace them.
  DELETE FROM txn_user_goals WHERE user_id = uid;
  FOR item IN SELECT * FROM jsonb_array_elements(COALESCE(p_data -> 'goals', '[]')) LOOP
    CONTINUE WHEN COALESCE(item ->> 'goalType', '') = '';
    INSERT INTO txn_user_goals (user_id, goal_type, target_value, target_date, status)
    VALUES (uid, left(item ->> 'goalType', 50), left(NULLIF(item ->> 'targetValue', ''), 100),
            public.try_date(item ->> 'targetDate'), 'active');
  END LOOP;

  DELETE FROM txn_user_constraints WHERE user_id = uid;
  FOR item IN SELECT * FROM jsonb_array_elements(COALESCE(p_data -> 'constraints', '[]')) LOOP
    CONTINUE WHEN COALESCE(item ->> 'constraintType', '') = '';
    INSERT INTO txn_user_constraints (user_id, constraint_type, area_type, area_label, day_of_week,
                                      available_from, available_to, weekly_frequency, note, is_active)
    VALUES (uid, left(item ->> 'constraintType', 30), left(NULLIF(item ->> 'areaType', ''), 30),
            left(NULLIF(item ->> 'areaLabel', ''), 100), left(NULLIF(item ->> 'dayOfWeek', ''), 20),
            public.try_time(item ->> 'availableFrom'), public.try_time(item ->> 'availableTo'),
            LEAST(GREATEST(public.try_numeric(item ->> 'weeklyFrequency')::int, 1), 14),
            NULLIF(item ->> 'note', ''), COALESCE((item ->> 'isActive')::boolean, true));
  END LOOP;

  DELETE FROM txn_user_medical_concerns WHERE user_id = uid;
  FOR item IN SELECT * FROM jsonb_array_elements(COALESCE(p_data #> '{safetyCheck,concerns}', '[]')) LOOP
    CONTINUE WHEN COALESCE(item ->> 'concern_type', '') = '';
    INSERT INTO txn_user_medical_concerns (user_id, concern_type, concern_name, medication_name,
                                           restriction_level, restriction_note, is_active)
    VALUES (uid, left(item ->> 'concern_type', 30),
            left(NULLIF(concat_ws(', ',
                   CASE jsonb_typeof(item -> 'concern_name')
                     WHEN 'array' THEN (SELECT string_agg(x, ', ') FROM jsonb_array_elements_text(item -> 'concern_name') x)
                     ELSE item ->> 'concern_name' END,
                   NULLIF(item ->> 'custom_concern_name', '')), ''), 255),
            left(NULLIF(item ->> 'medication_name', ''), 255),
            CASE item ->> 'restriction_level' WHEN 'mild' THEN 'low' WHEN 'moderate' THEN 'medium'
                                              WHEN 'high' THEN 'high' END,
            NULLIF(item ->> 'restriction_note', ''),
            COALESCE((item ->> 'is_active')::boolean, true));
  END LOOP;

  -- Level history is append-only: add a row only when the self-reported run changed.
  IF p_complete AND v_dist IS NOT NULL THEN
    SELECT * INTO last_level FROM txn_user_history_level
     WHERE user_id = uid AND calculation_method = 'assessment_test'
     ORDER BY recorded_at DESC NULLS LAST, id DESC LIMIT 1;
    IF NOT FOUND OR last_level.running_distance IS DISTINCT FROM v_dist
       OR last_level.running_duration_min IS DISTINCT FROM v_dur THEN
      INSERT INTO txn_user_history_level (user_id, fitness_level, calculation_method,
                                          running_distance, running_duration_min, recorded_at)
      VALUES (uid, NULLIF(p_fitness_level, ''), 'assessment_test', v_dist, v_dur, now());
    END IF;
  END IF;
END $$;

-- Log a run and the level it puts the member at, together.
CREATE OR REPLACE FUNCTION public.log_workout(
  p_started_at timestamptz, p_distance_km numeric, p_duration_min numeric,
  p_source text, p_fitness_level text)
RETURNS int LANGUAGE plpgsql SECURITY INVOKER SET search_path = public AS $$
DECLARE uid int := public.current_user_id(); new_id int;
BEGIN
  IF uid IS NULL THEN RAISE EXCEPTION 'not authenticated' USING ERRCODE = '28000'; END IF;
  INSERT INTO txn_workout_summaries (user_id, activity_type, source, started_at, duration_sec, distance_km)
  VALUES (uid, 'running', p_source, p_started_at, GREATEST(round(p_duration_min * 60)::int, 1), p_distance_km)
  RETURNING id INTO new_id;
  INSERT INTO txn_user_history_level (user_id, source_workout_id, fitness_level, calculation_method,
                                      running_distance, running_duration_min, recorded_at)
  VALUES (uid, new_id, NULLIF(p_fitness_level, ''), 'manual', p_distance_km, p_duration_min, now());
  RETURN new_id;
END $$;

GRANT EXECUTE ON FUNCTION public.join_event(int), public.leave_event(int),
  public.create_party(text, text, timestamptz, text, text, text, text, int),
  public.save_onboarding(jsonb, boolean, text),
  public.log_workout(timestamptz, numeric, numeric, text, text) TO authenticated;
REVOKE EXECUTE ON FUNCTION public.join_event(int), public.leave_event(int),
  public.create_party(text, text, timestamptz, text, text, text, text, int),
  public.save_onboarding(jsonb, boolean, text),
  public.log_workout(timestamptz, numeric, numeric, text, text), public.ensure_my_profile() FROM anon;

-- ------------------------------------------------------------
-- 5. Storage — avatars and comment photos, public read, own-folder write
-- ------------------------------------------------------------
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types) VALUES
  ('avatars',        'avatars',        true, 5242880, ARRAY['image/jpeg', 'image/png', 'image/webp']),
  ('comment-images', 'comment-images', true, 8388608, ARRAY['image/jpeg'])
ON CONFLICT (id) DO NOTHING;

DROP POLICY IF EXISTS lowrox_own_uploads_insert ON storage.objects;
CREATE POLICY lowrox_own_uploads_insert ON storage.objects FOR INSERT TO authenticated
  WITH CHECK (bucket_id IN ('avatars', 'comment-images') AND (storage.foldername(name))[1] = auth.uid()::text);
DROP POLICY IF EXISTS lowrox_own_uploads_update ON storage.objects;
CREATE POLICY lowrox_own_uploads_update ON storage.objects FOR UPDATE TO authenticated
  USING (bucket_id IN ('avatars', 'comment-images') AND (storage.foldername(name))[1] = auth.uid()::text);
DROP POLICY IF EXISTS lowrox_own_uploads_delete ON storage.objects;
CREATE POLICY lowrox_own_uploads_delete ON storage.objects FOR DELETE TO authenticated
  USING (bucket_id IN ('avatars', 'comment-images') AND (storage.foldername(name))[1] = auth.uid()::text);

COMMIT;

-- ============================================================
-- Verify / ตรวจสอบ
-- ============================================================
-- SELECT tablename, rowsecurity FROM pg_tables WHERE schemaname = 'public' ORDER BY 1;
-- SELECT * FROM v_user_public LIMIT 5;
