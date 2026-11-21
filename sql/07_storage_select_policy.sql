-- ============================================================
-- 07 — let members see (and so delete) their own uploads
--
-- 04 gave storage.objects INSERT, UPDATE and DELETE policies for each member's
-- own folder, but no SELECT. Storage runs remove() as
--   DELETE ... WHERE name = $1 RETURNING *
-- and without SELECT the row is invisible, so the delete matches nothing and
-- returns an empty list — no error, file still there. Found by
-- `npm run test:api` in the web app (REMOVE avatars / comment-images).
--
-- The buckets stay public for reading through public URLs; this only lets a
-- member's own session see the objects in their own folder.
--
-- 04 ให้สิทธิ์ INSERT/UPDATE/DELETE แต่ไม่ให้ SELECT จึงลบไฟล์ของตัวเองไม่ได้
-- (ลบแล้วได้ลิสต์ว่าง ไฟล์ยังอยู่) ไฟล์นี้เพิ่มสิทธิ์ SELECT เฉพาะโฟลเดอร์ตัวเอง
--
-- Run after 04 (and 06). Safe to re-run.
-- ============================================================

BEGIN;

DROP POLICY IF EXISTS lowrox_own_uploads_select ON storage.objects;
CREATE POLICY lowrox_own_uploads_select ON storage.objects FOR SELECT TO authenticated
  USING (bucket_id IN ('avatars', 'comment-images') AND (storage.foldername(name))[1] = auth.uid()::text);

COMMIT;

-- Verify / ตรวจสอบ: expect one row
-- SELECT policyname, cmd FROM pg_policies
--  WHERE schemaname = 'storage' AND tablename = 'objects' AND policyname = 'lowrox_own_uploads_select';
