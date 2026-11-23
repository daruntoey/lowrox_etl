-- ============================================================
-- 10 — automatic LINE reminder the day before a party
--
-- Every day at 18:00 Bangkok time (Supabase Cron = pg_cron) every member who has
-- joined an event that starts TOMORROW gets one LINE message:
--
--   พรุ่งนี้มีนัดนะ! <title>
--
--   <HH:MM> มาเจอกันที่ <location> นะ!
--
--   ไม่ต้องรีบ ไม่ต้องแข่งกับใคร วิ่งชิล ๆ คุยกันไป เหนื่อยก็พัก
--
--   อย่าลืมพกน้ำด้วยนะ แล้วเจอกัน 💚
--
--   https://lowrox.vercel.app/events/<id>          (line-push adds the link)
--
-- How it fits with 09: queue_event_reminders() INSERTs txn_notifications rows
-- (channel 'line', notification_type 'event_reminder_1d'); the 09 trigger calls
-- line-push, which sends them. A unique index means one reminder per member per
-- event, however often the job runs. Cancelled events and members who left are skipped.
--
-- A second job runs every 10 minutes and calls line-push only if LINE rows are still
-- queued (more than one batch of 20, or LINE was busy / over quota earlier).
--
-- ส่งเตือนอัตโนมัติทุกวัน 18:00 (เวลาไทย) ให้ทุกคนที่ join ตี้ที่นัดวันพรุ่งนี้ คนละ 1 ข้อความต่อตี้
--
-- Run after 09. Safe to re-run. Needs pg_cron (Supabase: Integrations → Cron); without
-- it the functions are still created and you can call them by hand.
-- ============================================================

BEGIN;

-- one reminder per member per event, even if the job runs twice
CREATE UNIQUE INDEX IF NOT EXISTS txn_notifications_reminder_1d_once
  ON txn_notifications (event_id, user_id) WHERE notification_type = 'event_reminder_1d';

-- p_day: the event date (Bangkok) to remind about; default tomorrow. Returns rows queued.
CREATE OR REPLACE FUNCTION public.queue_event_reminders(
  p_day date DEFAULT ((now() AT TIME ZONE 'Asia/Bangkok')::date + 1))
RETURNS int LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE queued int;
BEGIN
  INSERT INTO txn_notifications (user_id, event_id, notification_type, channel, title, body, deep_link, send_status)
  SELECT p.user_id, e.id, 'event_reminder_1d', 'line',
         left('พรุ่งนี้มีนัดนะ! ' || coalesce(e.title, 'ตี้วิ่ง'), 255),
         to_char(e.scheduled_at AT TIME ZONE 'Asia/Bangkok', 'HH24:MI')
           || ' มาเจอกันที่ ' || coalesce(nullif(trim(e.location_label), ''), 'จุดนัด') || ' นะ!'
           || E'\n\nไม่ต้องรีบ ไม่ต้องแข่งกับใคร วิ่งชิล ๆ คุยกันไป เหนื่อยก็พัก'
           || E'\n\nอย่าลืมพกน้ำด้วยนะ แล้วเจอกัน 💚',
         '/events/' || e.id,
         'queued'
  FROM txn_events e
  JOIN txn_event_participants p ON p.event_id = e.id AND p.status = 'joined'
  JOIN ms_users u ON u.id = p.user_id AND u.line_user_id IS NOT NULL
  WHERE (e.scheduled_at AT TIME ZONE 'Asia/Bangkok')::date = p_day
    AND e.status IN ('scheduled', 'full')
  ON CONFLICT (event_id, user_id) WHERE notification_type = 'event_reminder_1d' DO NOTHING;
  GET DIAGNOSTICS queued = ROW_COUNT;
  RETURN queued;
END $$;

REVOKE ALL ON FUNCTION public.queue_event_reminders(date) FROM PUBLIC, anon, authenticated;

-- Calls line-push (same request as the 09 trigger) when LINE rows are waiting.
CREATE OR REPLACE FUNCTION public.kick_line_push()
RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE fn_url text; hook_secret text;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM txn_notifications
                 WHERE channel = 'line' AND send_status IN ('queued', 'sending')
                   AND created_at > now() - interval '24 hours' AND attempts < 3) THEN
    RETURN false;
  END IF;
  IF to_regnamespace('vault') IS NULL OR to_regnamespace('net') IS NULL THEN
    RETURN false;
  END IF;
  EXECUTE $q$SELECT decrypted_secret FROM vault.decrypted_secrets WHERE name = 'line_push_url'$q$ INTO fn_url;
  EXECUTE $q$SELECT decrypted_secret FROM vault.decrypted_secrets WHERE name = 'line_push_hook_secret'$q$ INTO hook_secret;
  IF fn_url IS NULL OR hook_secret IS NULL THEN
    RETURN false;
  END IF;
  EXECUTE $q$SELECT net.http_post(url := $1, body := '{"reason":"cron"}'::jsonb,
            headers := jsonb_build_object('Content-Type', 'application/json', 'x-lowrox-hook', $2),
            timeout_milliseconds := 10000)$q$ USING fn_url, hook_secret;
  RETURN true;
END $$;

REVOKE ALL ON FUNCTION public.kick_line_push() FROM PUBLIC, anon, authenticated;

COMMIT;

-- ------------------------------------------------------------
-- Schedules (pg_cron runs in UTC: 11:00 UTC = 18:00 Bangkok).
-- cron.schedule with an existing job name replaces that job, so re-running is safe.
-- ------------------------------------------------------------
DO $outer$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_cron') THEN
    RAISE NOTICE '10: pg_cron not available — schedules not created (fine for local/CI)';
    RETURN;
  END IF;
  BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_cron;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '10: could not enable pg_cron here (%). Enable it in Dashboard → Integrations → Cron, then run this file again.', SQLERRM;
    RETURN;
  END;
  PERFORM cron.schedule('lowrox-event-reminders-1d', '0 11 * * *', 'SELECT public.queue_event_reminders()');
  PERFORM cron.schedule('lowrox-line-push-retry', '*/10 * * * *', 'SELECT public.kick_line_push()');
END $outer$;

-- ------------------------------------------------------------
-- Check / ตรวจสอบ
--   SELECT jobname, schedule, command, active FROM cron.job WHERE jobname LIKE 'lowrox-%';
--   SELECT j.jobname, d.status, d.return_message, d.start_time
--     FROM cron.job_run_details d JOIN cron.job j USING (jobid)
--    WHERE j.jobname LIKE 'lowrox-%' ORDER BY d.start_time DESC LIMIT 10;
-- Preview who would be reminded tomorrow (writes nothing):
--   SELECT e.id, e.title, e.scheduled_at AT TIME ZONE 'Asia/Bangkok' AS bkk_time, count(*) AS members
--     FROM txn_events e JOIN txn_event_participants p ON p.event_id = e.id AND p.status = 'joined'
--     JOIN ms_users u ON u.id = p.user_id AND u.line_user_id IS NOT NULL
--    WHERE (e.scheduled_at AT TIME ZONE 'Asia/Bangkok')::date = (now() AT TIME ZONE 'Asia/Bangkok')::date + 1
--      AND e.status IN ('scheduled', 'full') GROUP BY 1, 2, 3;
-- Run now instead of waiting for 18:00:   SELECT public.queue_event_reminders();
-- Stop:   SELECT cron.unschedule('lowrox-event-reminders-1d');
-- ------------------------------------------------------------
