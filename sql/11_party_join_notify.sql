-- ============================================================
-- 11 — tell the host on LINE when someone joins their party
--
-- When a member joins (join_event → txn_event_participants status 'joined'), the
-- event owner gets one LINE message through the 09 pipeline:
--
--   🎉 ตี้เริ่มคึกคักแล้ว!
--
--   คุณ <display name> มาเป็นส่วนหนึ่งของตี้ “<title>” แล้ว
--   ตอนนี้มากัน <joined>/<max> คน            (ตี้เต็มแล้ว 🙌 when full)
--
--   มีเพื่อนเพิ่มอีกหนึ่งคน ภารกิจพิชิตเป้าหมายสนุกขึ้นอีกแล้ว!
--
--   https://lowrox.vercel.app/events/<id>      (line-push adds the link)
--
-- Rules
--   * only the event owner, only if they log in with LINE; never for the host's own join
--   * one message per joiner per event: leaving and joining again does not send it again
--     (txn_notifications.actor_user_id + a unique index)
--   * cancelled events: nothing
--   * a problem here never stops the join: errors become a WARNING in the logs
--
-- เมื่อมีคนเข้าร่วมตี้ หัวตี้ได้ LINE 1 ข้อความต่อคน (ออกแล้วเข้าใหม่ไม่ส่งซ้ำ) และไม่ทำให้การ join ล้มเหลว
--
-- Run after 09. Safe to re-run. join_event / leave_event are unchanged.
-- ============================================================

BEGIN;

-- who caused the notification (the member who joined)
ALTER TABLE txn_notifications
  ADD COLUMN IF NOT EXISTS actor_user_id int REFERENCES ms_users(id) ON DELETE SET NULL;
COMMENT ON COLUMN txn_notifications.actor_user_id IS 'Member whose action caused the notification (e.g. who joined the party)';

CREATE UNIQUE INDEX IF NOT EXISTS txn_notifications_party_joined_once
  ON txn_notifications (event_id, user_id, actor_user_id) WHERE notification_type = 'party_joined';

CREATE OR REPLACE FUNCTION public.notify_host_party_joined()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
  e         txn_events%ROWTYPE;
  joiner    text;
  joined    int;
  seats     text;
BEGIN
  BEGIN
    SELECT * INTO e FROM txn_events WHERE id = NEW.event_id;
    IF NOT FOUND OR e.event_owner IS NULL OR e.event_owner = NEW.user_id
       OR e.status NOT IN ('scheduled', 'full') THEN
      RETURN NULL;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM ms_users WHERE id = e.event_owner AND line_user_id IS NOT NULL) THEN
      RETURN NULL;  -- the host can't get LINE messages
    END IF;

    SELECT COALESCE(NULLIF(trim(display_name), ''), username, 'เพื่อนใหม่') INTO joiner
      FROM ms_users WHERE id = NEW.user_id;
    SELECT count(*) INTO joined FROM txn_event_participants WHERE event_id = e.id AND status = 'joined';
    seats := CASE
      WHEN e.max_participants IS NULL THEN 'ตอนนี้มากัน ' || joined || ' คน'
      WHEN joined >= e.max_participants THEN 'ตอนนี้มากัน ' || joined || '/' || e.max_participants || ' คน ตี้เต็มแล้ว 🙌'
      ELSE 'ตอนนี้มากัน ' || joined || '/' || e.max_participants || ' คน'
    END;

    INSERT INTO txn_notifications (user_id, event_id, actor_user_id, notification_type, channel,
                                   title, body, deep_link, send_status)
    VALUES (e.event_owner, e.id, NEW.user_id, 'party_joined', 'line',
            '🎉 ตี้เริ่มคึกคักแล้ว!',
            'คุณ ' || joiner || ' มาเป็นส่วนหนึ่งของตี้ “' || coalesce(e.title, 'ตี้วิ่ง') || '” แล้ว'
              || E'\n' || seats
              || E'\n\nมีเพื่อนเพิ่มอีกหนึ่งคน ภารกิจพิชิตเป้าหมายสนุกขึ้นอีกแล้ว!',
            '/events/' || e.id, 'queued')
    ON CONFLICT (event_id, user_id, actor_user_id) WHERE notification_type = 'party_joined' DO NOTHING;
  EXCEPTION WHEN OTHERS THEN
    -- never block a join because of a notification
    RAISE WARNING 'notify_host_party_joined: % (event %, member %)', SQLERRM, NEW.event_id, NEW.user_id;
  END;
  RETURN NULL;
END $$;

REVOKE ALL ON FUNCTION public.notify_host_party_joined() FROM PUBLIC, anon, authenticated;

DROP TRIGGER IF EXISTS txn_event_participants_notify_host ON txn_event_participants;
CREATE TRIGGER txn_event_participants_notify_host
  AFTER INSERT OR UPDATE OF status ON txn_event_participants
  FOR EACH ROW
  WHEN (NEW.status = 'joined' AND NEW.role IS DISTINCT FROM 'host')
  EXECUTE FUNCTION public.notify_host_party_joined();

COMMIT;

-- ------------------------------------------------------------
-- Check / ตรวจสอบ
--   SELECT id, user_id AS host, actor_user_id AS joiner, event_id, send_status, error_message
--     FROM txn_notifications WHERE notification_type = 'party_joined' ORDER BY id DESC LIMIT 10;
-- Turn off:   DROP TRIGGER txn_event_participants_notify_host ON txn_event_participants;
-- ------------------------------------------------------------
