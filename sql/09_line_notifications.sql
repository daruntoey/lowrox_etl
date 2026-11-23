-- ============================================================
-- 09 — LINE notifications from txn_notifications
--
-- A row with channel = 'line' and send_status = 'queued' is pushed to the member
-- through the Lowrox LINE Official Account (@590mecpu):
--
--   INSERT txn_notifications  ──trigger (pg_net)──▶  Edge Function line-push
--        ▲                                               │ claim_line_notifications()
--        └──────── send_status sent / failed ◀───────────┘ LINE /v2/bot/message/push
--
-- What this file changes
--   * channel may be 'line'; send_status may be 'sending' (claimed by line-push)
--   * retry_key (X-Line-Retry-Key, so a retry is never delivered twice), attempts,
--     error_message, provider_message_id
--   * members keep READ access to their own notifications but can no longer
--     INSERT / UPDATE / DELETE them (the web never wrote this table; without this a
--     member could queue LINE pushes to themselves and use up the monthly quota)
--   * claim_line_notifications(): service_role only
--   * a statement trigger that calls line-push once per INSERT that queued LINE rows.
--     It reads the function URL and a shared secret from Supabase Vault; until both
--     are set (see the end of this file) rows simply wait in the queue.
--
-- Only rows created in the last 24 h are ever sent, so old data can't go out by mistake.
-- แถว channel='line' + queued จะถูกส่งเข้า LINE ของสมาชิก ส่งเฉพาะแถวที่สร้างภายใน 24 ชม.
--
-- Run after 08. Safe to re-run. Does not touch any table the web app writes.
-- ============================================================

BEGIN;

ALTER TABLE txn_notifications DROP CONSTRAINT IF EXISTS txn_notifications_channel_check;
ALTER TABLE txn_notifications ADD CONSTRAINT txn_notifications_channel_check
  CHECK (channel IN ('push', 'email', 'in_app', 'line'));

ALTER TABLE txn_notifications DROP CONSTRAINT IF EXISTS txn_notifications_send_status_check;
ALTER TABLE txn_notifications ADD CONSTRAINT txn_notifications_send_status_check
  CHECK (send_status IN ('queued', 'sending', 'sent', 'failed'));

ALTER TABLE txn_notifications
  ADD COLUMN IF NOT EXISTS retry_key           uuid        NOT NULL DEFAULT gen_random_uuid(),
  ADD COLUMN IF NOT EXISTS attempts            smallint    NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS error_message       text,
  ADD COLUMN IF NOT EXISTS provider_message_id varchar(64);

COMMENT ON COLUMN txn_notifications.retry_key IS 'X-Line-Retry-Key: the same key on every attempt, so LINE delivers once';
COMMENT ON COLUMN txn_notifications.error_message IS 'Why the last send failed (LINE API message), NULL when sent';

CREATE INDEX IF NOT EXISTS txn_notifications_line_queue_idx
  ON txn_notifications (id) WHERE channel = 'line' AND send_status IN ('queued', 'sending');

-- Members read their own notifications; only the server writes them.
REVOKE INSERT, UPDATE, DELETE ON txn_notifications FROM anon, authenticated;

-- line-push takes up to p_limit rows at a time. A row stuck in 'sending' for 5 minutes
-- (the function crashed) is taken again; after 3 attempts it is left for a person.
CREATE OR REPLACE FUNCTION public.claim_line_notifications(p_limit int DEFAULT 20)
RETURNS TABLE (id int, retry_key uuid, title text, body text, deep_link text,
               line_user_id text, attempts smallint)
LANGUAGE sql SECURITY DEFINER SET search_path = public AS $$
  WITH picked AS (
    SELECT n.id
    FROM txn_notifications n
    WHERE n.channel = 'line'
      AND (n.send_status = 'queued'
           OR (n.send_status = 'sending' AND n.updated_at < now() - interval '5 minutes'))
      AND n.created_at > now() - interval '24 hours'
      AND n.attempts < 3
    ORDER BY n.id
    LIMIT greatest(1, least(p_limit, 100))
    FOR UPDATE SKIP LOCKED
  )
  UPDATE txn_notifications n
     SET send_status = 'sending', attempts = n.attempts + 1, updated_at = now()
    FROM picked, ms_users u
   WHERE n.id = picked.id AND u.id = n.user_id
  RETURNING n.id, n.retry_key, n.title::text, n.body, n.deep_link::text,
            u.line_user_id::text, n.attempts;
$$;

REVOKE ALL ON FUNCTION public.claim_line_notifications(int) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.claim_line_notifications(int) TO service_role;
GRANT SELECT, UPDATE ON txn_notifications TO service_role;

COMMIT;

-- ------------------------------------------------------------
-- The trigger needs pg_net and Vault (both on Supabase). A plain PostgreSQL
-- (local / CI) skips this part; nothing else depends on it.
-- ------------------------------------------------------------
DO $outer$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_net')
     OR to_regnamespace('vault') IS NULL THEN
    RAISE NOTICE '09: pg_net or Vault not available — LINE push trigger not installed (fine for local/CI)';
    RETURN;
  END IF;

  CREATE EXTENSION IF NOT EXISTS pg_net;

  CREATE OR REPLACE FUNCTION public.notify_line_push()
  RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $fn$
  DECLARE fn_url text; hook_secret text;
  BEGIN
    IF NOT EXISTS (SELECT 1 FROM new_rows WHERE channel = 'line' AND send_status = 'queued') THEN
      RETURN NULL;
    END IF;
    SELECT decrypted_secret INTO fn_url FROM vault.decrypted_secrets WHERE name = 'line_push_url';
    SELECT decrypted_secret INTO hook_secret FROM vault.decrypted_secrets WHERE name = 'line_push_hook_secret';
    IF fn_url IS NULL OR hook_secret IS NULL THEN
      RETURN NULL;  -- not configured yet: the rows wait in the queue
    END IF;
    -- pg_net sends after this transaction commits, so line-push sees the new rows.
    PERFORM net.http_post(
      url := fn_url,
      body := jsonb_build_object('reason', 'insert'),
      headers := jsonb_build_object('Content-Type', 'application/json', 'x-lowrox-hook', hook_secret),
      timeout_milliseconds := 10000);
    RETURN NULL;
  END $fn$;

  REVOKE ALL ON FUNCTION public.notify_line_push() FROM PUBLIC, anon, authenticated;

  DROP TRIGGER IF EXISTS txn_notifications_line_push ON txn_notifications;
  CREATE TRIGGER txn_notifications_line_push
    AFTER INSERT ON txn_notifications
    REFERENCING NEW TABLE AS new_rows
    FOR EACH STATEMENT EXECUTE FUNCTION public.notify_line_push();
END $outer$;

-- ------------------------------------------------------------
-- One-time setup on Supabase (run separately, NOT committed with real values):
--
--   SELECT vault.create_secret('https://avvxdvvcqexmfpsgtzqm.supabase.co/functions/v1/line-push', 'line_push_url');
--   SELECT vault.create_secret(encode(extensions.gen_random_bytes(32), 'hex'), 'line_push_hook_secret');
--   -- copy this value into Edge Functions → Secrets as LINE_PUSH_HOOK_SECRET:
--   SELECT decrypted_secret FROM vault.decrypted_secrets WHERE name = 'line_push_hook_secret';
--
-- Test (your own member id; you must have added @590mecpu as a friend):
--   INSERT INTO txn_notifications (user_id, notification_type, channel, title, body, deep_link, send_status)
--   VALUES (<your id>, 'test', 'line', 'ทดสอบจาก Lowrox', 'แจ้งเตือนผ่าน LINE ใช้ได้แล้ว',
--           'https://lowrox.vercel.app/events', 'queued');
--   SELECT id, send_status, attempts, sent_at, error_message FROM txn_notifications ORDER BY id DESC LIMIT 3;
-- ------------------------------------------------------------
