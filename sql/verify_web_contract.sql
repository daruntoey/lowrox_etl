-- ============================================================
-- READ-ONLY check: does this database have everything the web app
-- (aom-thanya/lowrox) and the ETL (this repo) expect?
-- Safe to run any time on Supabase (SQL Editor → Run). Changes nothing.
--
-- Every row should say ok = true. A false row names the script to run.
-- ตรวจอย่างเดียว ไม่แก้ข้อมูล ทุกแถวควรได้ ok = true
-- ============================================================
WITH checks(item, ok, fix) AS (
  VALUES
  ('03 auth: ms_users.auth_user_id',
     EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema = 'public' AND table_name = 'ms_users' AND column_name = 'auth_user_id'),
     'sql/03_auth_and_rls.sql'),
  ('03 RLS on every ms_/txn_ table',
     NOT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public'
                 AND (tablename LIKE 'ms\_%' OR tablename LIKE 'txn\_%') AND NOT rowsecurity),
     'sql/03_auth_and_rls.sql + 04'),
  ('04 profile columns on ms_users',
     (SELECT count(*) = 12 FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = 'ms_users'
        AND column_name IN ('display_name', 'bio', 'province_id', 'district_id', 'avatar_url',
                            'onboarding_status', 'onboarding_data', 'show_profile',
                            'public_profile_visible', 'public_area_visible', 'interests', 'deleted_at')),
     'sql/04_frontend_integration.sql'),
  ('04 event columns on txn_events',
     (SELECT count(*) = 11 FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = 'txn_events'
        AND column_name IN ('title', 'description', 'location_label', 'ends_at', 'join_url', 'cover_url',
                            'visibility', 'is_party', 'party_style', 'registration_closed', 'deleted_at')),
     'sql/04_frontend_integration.sql'),
  ('04 txn_plan_checkoffs table',
     to_regclass('public.txn_plan_checkoffs') IS NOT NULL,
     'sql/04_frontend_integration.sql'),
  ('04 RPCs (7)',
     (SELECT count(DISTINCT proname) = 7 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
      WHERE n.nspname = 'public'
        AND proname IN ('ensure_my_profile', 'save_onboarding', 'create_party', 'join_event',
                        'leave_event', 'log_workout', 'event_is_public')),
     'sql/04_frontend_integration.sql'),
  ('04 storage buckets avatars + comment-images',
     (SELECT count(*) = 2 FROM storage.buckets WHERE id IN ('avatars', 'comment-images')),
     'sql/04_frontend_integration.sql'),
  ('06 LINE: ms_users.line_user_id',
     EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema = 'public' AND table_name = 'ms_users' AND column_name = 'line_user_id'),
     'sql/06_line_login.sql'),
  ('06 LINE: sign-up trigger stores line_user_id',
     EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'handle_new_user'
             AND prosrc LIKE '%line_user_id%'),
     'sql/06_line_login.sql')
)
SELECT item, ok, CASE WHEN ok THEN '' ELSE 'run ' || fix END AS action
FROM checks;

-- What the web app has written so far (counts only, no personal data):
SELECT 'members signed up through the app' AS what, count(*) AS n FROM ms_users WHERE auth_user_id IS NOT NULL
UNION ALL SELECT 'members linked to LINE', count(*) FROM ms_users
  WHERE to_jsonb(ms_users) ? 'line_user_id' AND to_jsonb(ms_users) ->> 'line_user_id' IS NOT NULL
UNION ALL SELECT 'runs logged in the app', count(*) FROM txn_workout_summaries WHERE source IN ('log', 'onboarding')
UNION ALL SELECT 'parties hosted in the app', count(*) FROM txn_events WHERE title IS NOT NULL
UNION ALL SELECT 'weekly plan ticks', count(*) FROM txn_plan_checkoffs;
