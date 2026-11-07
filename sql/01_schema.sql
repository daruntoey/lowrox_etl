-- ============================================================
-- Lowrox OLTP schema — 18 tables, PostgreSQL
-- Matches social_fitness_platform_schema_v3.dbml
-- Run this BEFORE 02_seed.sql
--
-- SAFE TO RE-RUN: the drops below clear any previous attempt first.
-- รันซ้ำได้: คำสั่ง drop ข้างล่างล้างของเดิมให้ก่อน
--
-- WARNING: this deletes all data in these 18 tables.
-- คำเตือน: คำสั่งนี้ลบข้อมูลใน 18 ตารางนี้ทิ้งทั้งหมด
-- ============================================================

BEGIN;

-- drop in reverse dependency order so no foreign key blocks the drop
-- ลบย้อนลำดับ dependency เพื่อไม่ให้ foreign key ขวาง
DROP TABLE IF EXISTS txn_notifications        CASCADE;
DROP TABLE IF EXISTS txn_payments             CASCADE;
DROP TABLE IF EXISTS txn_user_subscriptions   CASCADE;
DROP TABLE IF EXISTS txn_training_plans       CASCADE;
DROP TABLE IF EXISTS txn_group_comments       CASCADE;
DROP TABLE IF EXISTS txn_event_participants   CASCADE;
DROP TABLE IF EXISTS txn_user_history_level   CASCADE;
DROP TABLE IF EXISTS txn_user_constraints     CASCADE;
DROP TABLE IF EXISTS txn_user_medical_concerns CASCADE;
DROP TABLE IF EXISTS txn_user_goals           CASCADE;
DROP TABLE IF EXISTS txn_workout_summaries    CASCADE;
DROP TABLE IF EXISTS txn_events               CASCADE;
DROP TABLE IF EXISTS ms_device_connections    CASCADE;
DROP TABLE IF EXISTS ms_subscription_plans    CASCADE;
DROP TABLE IF EXISTS ms_area                  CASCADE;
DROP TABLE IF EXISTS ms_user_verifications    CASCADE;
DROP TABLE IF EXISTS ms_user_demographics     CASCADE;
DROP TABLE IF EXISTS ms_users                 CASCADE;

-- ---------- 1. CORE USER ----------

CREATE TABLE ms_users (
  id              serial PRIMARY KEY,
  username        varchar(100) NOT NULL UNIQUE,
  email           varchar(255) NOT NULL UNIQUE,
  password_hash   varchar(255) NOT NULL,
  tier            varchar(20) DEFAULT 'free',
  created_at      timestamptz DEFAULT now(),
  updated_at      timestamptz DEFAULT now()
);
COMMENT ON COLUMN ms_users.tier IS
  'denormalized cache; source of truth = txn_user_subscriptions';

CREATE TABLE ms_user_demographics (
  user_id     int PRIMARY KEY REFERENCES ms_users(id) ON DELETE CASCADE,
  birth_date  date,
  gender      varchar(20),
  created_at  timestamptz DEFAULT now(),
  updated_at  timestamptz DEFAULT now()
);

CREATE TABLE ms_user_verifications (
  id                serial PRIMARY KEY,
  user_id           int NOT NULL REFERENCES ms_users(id) ON DELETE CASCADE,
  verification_type varchar(30),
  status            varchar(20) CHECK (status IN ('pending','verified','rejected','expired')),
  verified_at       timestamptz,
  expires_at        timestamptz,
  rejected_at       timestamptz,
  rejection_reason  varchar(255),
  created_at        timestamptz DEFAULT now(),
  updated_at        timestamptz DEFAULT now()
);
CREATE INDEX idx_verif_user ON ms_user_verifications (user_id);

-- ---------- 2. AREA & PLANS (master) ----------

CREATE TABLE ms_area (
  id          serial PRIMARY KEY,
  area_type   varchar(30),
  area_label  varchar(100),
  created_at  timestamptz DEFAULT now(),
  updated_at  timestamptz DEFAULT now()
);

CREATE TABLE ms_subscription_plans (
  id             serial PRIMARY KEY,
  plan_code      varchar(30) NOT NULL UNIQUE,
  plan_name      varchar(100),
  tier           varchar(20) CHECK (tier IN ('free','pro','elite')),
  price          numeric(10,2) CHECK (price >= 0),
  currency       varchar(3) DEFAULT 'THB',
  billing_period varchar(20) CHECK (billing_period IN ('monthly','annual')),
  is_active      boolean DEFAULT true,
  created_at     timestamptz DEFAULT now(),
  updated_at     timestamptz DEFAULT now()
);
COMMENT ON TABLE ms_subscription_plans IS
  'Master table. Merging into txn_user_subscriptions would create the transitive
   dependency plan_code -> price and break 3NF.';

CREATE TABLE ms_device_connections (
  id                serial PRIMARY KEY,
  user_id           int NOT NULL REFERENCES ms_users(id) ON DELETE CASCADE,
  provider          varchar(30),
  external_user_id  varchar(100),
  connection_status varchar(20) CHECK (connection_status IN ('active','revoked','expired')),
  last_synced_at    timestamptz,
  created_at        timestamptz DEFAULT now(),
  updated_at        timestamptz DEFAULT now(),
  CONSTRAINT uq_device_user_provider UNIQUE (user_id, provider)
);
COMMENT ON TABLE ms_device_connections IS
  'OAuth tokens are NOT stored here - keep them in a secrets vault';

-- ---------- 3. EVENTS & COMMUNITY ----------

CREATE TABLE txn_events (
  id               serial PRIMARY KEY,
  event_owner      int NOT NULL REFERENCES ms_users(id),
  workout_type     varchar(50),
  area_id          int NOT NULL REFERENCES ms_area(id),
  scheduled_at     timestamptz,
  target_level     varchar(50),
  max_participants int CHECK (max_participants > 0),
  status           varchar(20) CHECK (status IN ('scheduled','full','completed','cancelled')),
  created_at       timestamptz DEFAULT now(),
  updated_at       timestamptz DEFAULT now()
);
CREATE INDEX idx_events_sched ON txn_events (scheduled_at);
CREATE INDEX idx_events_area  ON txn_events (area_id);

CREATE TABLE txn_event_participants (
  id         serial PRIMARY KEY,
  user_id    int NOT NULL REFERENCES ms_users(id) ON DELETE CASCADE,
  event_id   int NOT NULL REFERENCES txn_events(id) ON DELETE CASCADE,
  joined_at  timestamptz,
  role       varchar(20) CHECK (role IN ('host','participant','co_host')),
  status     varchar(20) CHECK (status IN ('joined','attended','no_show','cancelled')),
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  CONSTRAINT uq_participant UNIQUE (user_id, event_id)
);

CREATE TABLE txn_group_comments (
  id           serial PRIMARY KEY,
  user_id      int NOT NULL REFERENCES ms_users(id) ON DELETE CASCADE,
  event_id     int NOT NULL REFERENCES txn_events(id) ON DELETE CASCADE,
  comment_text text,
  created_at   timestamptz DEFAULT now(),
  updated_at   timestamptz DEFAULT now()
);
CREATE INDEX idx_comments_event ON txn_group_comments (event_id);

CREATE TABLE txn_training_plans (
  id              serial PRIMARY KEY,
  created_by      int NOT NULL REFERENCES ms_users(id),
  event_id        int NOT NULL REFERENCES txn_events(id) ON DELETE CASCADE,
  plan_details    text,
  plan_status     varchar(20) CHECK (plan_status IN ('draft','published','archived')),
  is_ai_suggested boolean DEFAULT false,
  scaled_level    varchar(20),
  created_at      timestamptz DEFAULT now(),
  updated_at      timestamptz DEFAULT now()
);

-- ---------- 4. WEARABLE & WORKOUT ----------

CREATE TABLE txn_workout_summaries (
  id                   serial PRIMARY KEY,
  user_id              int NOT NULL REFERENCES ms_users(id) ON DELETE CASCADE,
  device_id            int REFERENCES ms_device_connections(id),
  event_id             int REFERENCES txn_events(id),
  external_activity_id varchar(100),
  activity_type        varchar(50),
  source               varchar(30),
  started_at           timestamptz,
  duration_sec         int CHECK (duration_sec > 0),
  distance_km          numeric(6,2) CHECK (distance_km >= 0),
  calories_kcal        int,
  avg_heart_rate       int,
  created_at           timestamptz DEFAULT now(),
  CONSTRAINT uq_workout_external UNIQUE (device_id, external_activity_id)
);
CREATE INDEX idx_workout_user_time ON txn_workout_summaries (user_id, started_at);
COMMENT ON TABLE txn_workout_summaries IS
  'avg_pace is derived from distance_km and duration_sec - compute on read';
COMMENT ON COLUMN txn_workout_summaries.device_id IS 'NULL means manual entry';
COMMENT ON COLUMN txn_workout_summaries.event_id  IS 'NULL means solo workout';

-- ---------- 5. PROFILE: GOALS, HEALTH, CONSTRAINTS, LEVEL ----------

CREATE TABLE txn_user_goals (
  id           serial PRIMARY KEY,
  user_id      int NOT NULL REFERENCES ms_users(id) ON DELETE CASCADE,
  goal_type    varchar(50),
  target_value varchar(100),
  target_date  date,
  status       varchar(20) CHECK (status IN ('active','achieved','expired','abandoned')),
  created_at   timestamptz DEFAULT now(),
  updated_at   timestamptz DEFAULT now()
);

CREATE TABLE txn_user_medical_concerns (
  id                serial PRIMARY KEY,
  user_id           int NOT NULL REFERENCES ms_users(id) ON DELETE CASCADE,
  concern_type      varchar(30),
  concern_name      varchar(255),
  medication_name   varchar(255),
  restriction_level varchar(30) CHECK (restriction_level IN ('low','medium','high')),
  restriction_note  text,
  is_active         boolean DEFAULT true,
  created_at        timestamptz DEFAULT now(),
  updated_at        timestamptz DEFAULT now()
);
COMMENT ON TABLE txn_user_medical_concerns IS
  'PDPA sec.26 sensitive data - encrypt at rest, restricted access role,
   must support right to erasure';

CREATE TABLE txn_user_constraints (
  id               serial PRIMARY KEY,
  user_id          int NOT NULL REFERENCES ms_users(id) ON DELETE CASCADE,
  constraint_type  varchar(30),
  area_type        varchar(30),
  area_label       varchar(100),
  day_of_week      varchar(20),
  available_from   time,
  available_to     time,
  weekly_frequency int CHECK (weekly_frequency BETWEEN 1 AND 14),
  note             text,
  is_active        boolean DEFAULT true,
  created_at       timestamptz DEFAULT now(),
  updated_at       timestamptz DEFAULT now()
);

CREATE TABLE txn_user_history_level (
  id                   serial PRIMARY KEY,
  user_id              int NOT NULL REFERENCES ms_users(id) ON DELETE CASCADE,
  source_workout_id    int REFERENCES txn_workout_summaries(id),
  fitness_level        varchar(20),
  calculation_method   varchar(20) CHECK (calculation_method IN ('device','manual','assessment_test')),
  running_distance     numeric(6,2),
  running_duration_min numeric(8,2),
  recorded_at          timestamptz,
  created_at           timestamptz DEFAULT now(),
  updated_at           timestamptz DEFAULT now()
);
CREATE INDEX idx_level_user_time ON txn_user_history_level (user_id, recorded_at);

-- ---------- 6. SUBSCRIPTION & PAYMENT ----------

CREATE TABLE txn_user_subscriptions (
  id                  serial PRIMARY KEY,
  user_id             int NOT NULL REFERENCES ms_users(id) ON DELETE CASCADE,
  plan_id             int NOT NULL REFERENCES ms_subscription_plans(id),
  referred_by_user_id int REFERENCES ms_users(id),
  status              varchar(20) CHECK (status IN ('trialing','active','past_due','cancelled','expired')),
  started_at          timestamptz,
  current_period_end  timestamptz,
  cancelled_at        timestamptz,
  auto_renew          boolean DEFAULT true,
  created_at          timestamptz DEFAULT now(),
  updated_at          timestamptz DEFAULT now()
);
CREATE INDEX idx_sub_user ON txn_user_subscriptions (user_id);
COMMENT ON COLUMN txn_user_subscriptions.referred_by_user_id IS
  'trainer attribution for the reward mechanism - nullable';

CREATE TABLE txn_payments (
  id              serial PRIMARY KEY,
  subscription_id int NOT NULL REFERENCES txn_user_subscriptions(id) ON DELETE CASCADE,
  user_id         int NOT NULL REFERENCES ms_users(id) ON DELETE CASCADE,
  amount          numeric(10,2) CHECK (amount >= 0),
  currency        varchar(3) DEFAULT 'THB',
  payment_method  varchar(30),
  gateway_txn_ref varchar(100) UNIQUE,
  status          varchar(20) CHECK (status IN ('pending','succeeded','failed','refunded')),
  paid_at         timestamptz,
  created_at      timestamptz DEFAULT now(),
  updated_at      timestamptz DEFAULT now()
);
COMMENT ON COLUMN txn_payments.gateway_txn_ref IS
  'idempotency key - prevents duplicate webhook charges';

-- ---------- 7. NOTIFICATION ----------

CREATE TABLE txn_notifications (
  id                serial PRIMARY KEY,
  user_id           int NOT NULL REFERENCES ms_users(id) ON DELETE CASCADE,
  event_id          int REFERENCES txn_events(id) ON DELETE SET NULL,
  notification_type varchar(40),
  channel           varchar(20) CHECK (channel IN ('push','email','in_app')),
  title             varchar(255),
  body              text,
  deep_link         varchar(255),
  send_status       varchar(20) CHECK (send_status IN ('queued','sent','failed')),
  sent_at           timestamptz,
  delivered_at      timestamptz,
  opened_at         timestamptz,
  clicked_at        timestamptz,
  dismissed_at      timestamptz,
  created_at        timestamptz DEFAULT now(),
  updated_at        timestamptz DEFAULT now()
);
CREATE INDEX idx_notif_user_sent ON txn_notifications (user_id, sent_at);
COMMENT ON TABLE txn_notifications IS
  'Funnel stored as columns, not a child table. The engagement sequence is
   linear and each step happens at most once.';

COMMIT;
