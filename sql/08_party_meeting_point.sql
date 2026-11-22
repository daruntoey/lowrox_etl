-- ============================================================
-- 08 — a pin on the map for each party's meeting point
--
-- The party form (/parties/new) now lets the host drop a pin; the party page
-- (/events/:id) shows it on a map with a "directions" link. The place name
-- stays in txn_events.location_label; this adds the coordinates.
--
--   txn_events.meeting_lat / meeting_lng   WGS84 degrees, both set or both NULL
--   create_party(..., p_meeting_lat, p_meeting_lng)   two optional arguments
--
-- The old 8-argument create_party is dropped so PostgREST has exactly one
-- function to call. The new arguments default to NULL, so a web build that
-- does not send them yet keeps working: run this BEFORE merging the web change.
--
-- ตี้มีพิกัดจุดนัด (lat/lng) เพิ่มจากชื่อสถานที่เดิม
-- รันไฟล์นี้ก่อน merge เว็บ เพราะ argument ใหม่มีค่า default เว็บเวอร์ชันเก่ายังใช้ได้
--
-- Run after 04, 06 and 07. Safe to re-run.
-- If you ever re-run 04, run 06, 07 and 08 again after it (04 re-creates the
-- 8-argument create_party).
-- ============================================================

BEGIN;

ALTER TABLE txn_events
  ADD COLUMN IF NOT EXISTS meeting_lat numeric(9,6),
  ADD COLUMN IF NOT EXISTS meeting_lng numeric(9,6);

ALTER TABLE txn_events DROP CONSTRAINT IF EXISTS txn_events_meeting_point_check;
ALTER TABLE txn_events ADD CONSTRAINT txn_events_meeting_point_check CHECK (
  (meeting_lat IS NULL AND meeting_lng IS NULL)
  OR (meeting_lat BETWEEN -90 AND 90 AND meeting_lng BETWEEN -180 AND 180));

COMMENT ON COLUMN txn_events.meeting_lat IS 'Meeting point latitude (WGS84), set by the host on /parties/new';
COMMENT ON COLUMN txn_events.meeting_lng IS 'Meeting point longitude (WGS84), set by the host on /parties/new';

DROP FUNCTION IF EXISTS public.create_party(text, text, timestamptz, text, text, text, text, int);

CREATE OR REPLACE FUNCTION public.create_party(
  p_title text, p_workout_type text, p_scheduled_at timestamptz, p_location text,
  p_description text, p_target_level text, p_party_style text, p_max_participants int,
  p_meeting_lat numeric DEFAULT NULL, p_meeting_lng numeric DEFAULT NULL)
RETURNS int LANGUAGE plpgsql SECURITY INVOKER SET search_path = public AS $$
DECLARE uid int := public.current_user_id(); new_id int;
BEGIN
  IF uid IS NULL THEN RAISE EXCEPTION 'not authenticated' USING ERRCODE = '28000'; END IF;
  -- half a pin is no pin
  IF (p_meeting_lat IS NULL) <> (p_meeting_lng IS NULL) THEN
    p_meeting_lat := NULL; p_meeting_lng := NULL;
  END IF;
  INSERT INTO txn_events (event_owner, title, workout_type, scheduled_at, location_label, description,
                          target_level, party_style, max_participants, status, visibility, is_party,
                          meeting_lat, meeting_lng)
  VALUES (uid, left(p_title, 150), p_workout_type, p_scheduled_at, left(p_location, 200), p_description,
          p_target_level, p_party_style, p_max_participants, 'scheduled', 'public', true,
          round(p_meeting_lat, 6), round(p_meeting_lng, 6))
  RETURNING id INTO new_id;
  INSERT INTO txn_event_participants (user_id, event_id, joined_at, role, status)
  VALUES (uid, new_id, now(), 'host', 'joined');
  RETURN new_id;
END $$;

REVOKE EXECUTE ON FUNCTION public.create_party(text, text, timestamptz, text, text, text, text, int, numeric, numeric)
  FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.create_party(text, text, timestamptz, text, text, text, text, int, numeric, numeric)
  TO authenticated;

COMMIT;

-- PostgREST: pick up the new function signature now
NOTIFY pgrst, 'reload schema';

-- Verify / ตรวจสอบ: expect 2 columns and 1 function with 10 arguments
-- SELECT column_name FROM information_schema.columns
--  WHERE table_name = 'txn_events' AND column_name LIKE 'meeting_%';
-- SELECT pg_get_function_identity_arguments(oid) FROM pg_proc WHERE proname = 'create_party';
