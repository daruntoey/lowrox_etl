-- ============================================================
-- 06 — Real LINE Login
-- Run after 04. Safe to re-run.
--
-- The line-auth Edge Function creates the Supabase account with the LINE profile in
-- user_metadata (display_name, avatar_url, line_user_id) and links ms_users.line_user_id.
-- ============================================================
BEGIN;

ALTER TABLE ms_users ADD COLUMN IF NOT EXISTS line_user_id varchar(64);
CREATE UNIQUE INDEX IF NOT EXISTS ms_users_line_user_id_key ON ms_users (line_user_id) WHERE line_user_id IS NOT NULL;

-- Members can't change which LINE account they are (04 grants UPDATE per column, so
-- line_user_id is already excluded; this is a reminder, not a change).

-- Same as 04, plus the LINE picture and user ID for accounts the Edge Function creates.
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE base text;
BEGIN
  UPDATE public.ms_users SET auth_user_id = NEW.id, updated_at = now()
   WHERE lower(email) = lower(NEW.email) AND auth_user_id IS NULL;
  IF NOT FOUND THEN
    base := left(regexp_replace(COALESCE(NEW.raw_user_meta_data->>'username', split_part(NEW.email, '@', 1)),
                                '[^A-Za-z0-9_.]', '', 'g'), 80);
    INSERT INTO public.ms_users (auth_user_id, username, email, tier, display_name, avatar_url, line_user_id)
    VALUES (NEW.id,
            COALESCE(NULLIF(base, ''), 'member') || '_' || left(replace(NEW.id::text, '-', ''), 8),
            NEW.email, 'free',
            left(COALESCE(NULLIF(NEW.raw_user_meta_data->>'display_name', ''), split_part(NEW.email, '@', 1)), 50),
            NULLIF(NEW.raw_user_meta_data->>'avatar_url', ''),
            NULLIF(NEW.raw_user_meta_data->>'line_user_id', ''))
    ON CONFLICT DO NOTHING;
  END IF;
  RETURN NEW;
EXCEPTION WHEN others THEN
  RAISE WARNING 'handle_new_user failed for %: %', NEW.id, SQLERRM;
  RETURN NEW;
END $$;

COMMIT;

-- Verify / ตรวจสอบ
-- SELECT id, username, display_name, line_user_id FROM ms_users WHERE line_user_id IS NOT NULL;
