"""
Lowrox mock data generator — 18 OLTP tables.

Run:  python3 generate.py            (default 50 rows per table)
      python3 generate.py --scale 20 (20x bigger, same shape)

Outputs into ./out/
  csv/<table>.csv        one file per table, for bulk load or for the lake
  02_seed.sql            INSERT statements in dependency order
  source_mock/*.json     raw provider payloads for the Bronze zone, with the
                         duplicates and unit differences the OLTP tables cannot
                         hold because of their unique indexes

Design rules this generator follows:
  1. every parameter comes from calibration.py, which cites its source
  2. the data is deliberately dirty where Silver is supposed to clean it
  3. pace improves over time so dim_user SCD2 has something to track
  4. workouts follow a power law across users, not a uniform spread
  5. some users churn, so the retention mart is not a flat 100%
  6. the relationships listed in calibration.py are baked in on purpose
"""
import argparse, csv, json, random, math
from pathlib import Path
from datetime import datetime, timedelta, timezone

import calibration as C

BKK = timezone(timedelta(hours=7))
OUT = Path(__file__).parent / "out"
TODAY = datetime(2026, 10, 21, tzinfo=BKK)
WINDOW_DAYS = 270                      # 9 months of history

rnd = random.Random(20261021)          # deterministic: same output every run


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ts(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S%z")


def pick(weighted):
    r, acc = rnd.random(), 0.0
    for value, w in weighted:
        acc += w
        if r <= acc:
            return value
    return weighted[-1][0]


def band_for(pace):
    if pace is None:
        return "not rated"
    for name, lo, hi in C.LEVEL_BANDS:
        if lo <= pace < hi:
            return name
    return "starter"


def lognormal_pace():
    mu = math.log(C.POPULATION_MEDIAN_PACE_SEC_PER_KM)
    return rnd.lognormvariate(mu, C.POPULATION_PACE_LOG_SIGMA)


TABLES = {}


def emit(name, rows):
    TABLES[name] = rows
    return rows


# ---------------------------------------------------------------------------
# 1-2. ms_users + ms_user_demographics
# ---------------------------------------------------------------------------
FIRST = ["Nattapong", "Siriporn", "Thanakrit", "Pimchanok", "Worawut", "Kanyarat",
         "Peerapat", "Chayanit", "Sarawut", "Napatsorn", "Ittipat", "Benjawan",
         "Kittisak", "Warunee", "Teerapong", "Supansa", "Anuchit", "Nichapat",
         "Phuwadon", "Jiraporn", "Chatchai", "Ornanong", "Narongrit", "Patcharee",
         "Somchai", "Malinee", "Ekkachai", "Rattana", "Wichian", "Suphattra"]
LAST = ["Srisuk", "Wongchai", "Thongdee", "Phongsak", "Kaewmanee", "Boonmee",
        "Chaiyaporn", "Rattanakosin", "Intharat", "Saetang", "Limthong",
        "Prasertsak", "Niyomrat", "Chanthavong", "Mekwattana"]


def gen_users(n_users):
    users, demos = [], []
    taken = set()                       # usernames must stay unique at any scale
    for i in range(1, n_users + 1):
        sex = pick([(s, w) for s, w in C.SEX_DISTRIBUTION])
        lo, hi, _ = pick([((a, b, w), w) for a, b, w in C.AGE_DISTRIBUTION])
        age = rnd.randint(lo, hi)
        first, last = rnd.choice(FIRST), rnd.choice(LAST)
        for _ in range(40):
            uname = f"{first.lower()}{rnd.randint(10, 99)}"
            if uname not in taken:
                break
        else:                           # name space exhausted: fall back to the id
            uname = f"{first.lower()}{last[:3].lower()}{i}"
        taken.add(uname)
        joined = TODAY - timedelta(days=rnd.randint(5, WINDOW_DAYS))
        # tier is the denormalised cache the ER diagram documents; it is
        # recomputed from txn_user_subscriptions below, not set independently
        users.append({
            "id": i, "username": uname, "email": f"{uname}@example.com",
            "password_hash": f"$2b$12${'x' * 22}{i:06d}",
            "tier": "free",
            "created_at": ts(joined), "updated_at": ts(joined),
        })
        demos.append({
            "user_id": i,
            "birth_date": (TODAY - timedelta(days=age * 365 + rnd.randint(0, 364))).strftime("%Y-%m-%d"),
            "gender": {"M": "male", "F": "female", "O": "prefer_not_to_say"}[sex],
            "created_at": ts(joined), "updated_at": ts(joined),
        })
        users[-1]["_sex"] = sex
        users[-1]["_age"] = age
        users[-1]["_joined"] = joined
        # base pace: population lognormal, adjusted for sex and age per SOURCE 1
        base = lognormal_pace() * C.SEX_PACE_FACTOR[sex]
        if age > 39:
            base *= 1 + C.AGE_PENALTY_PER_YEAR_PAST_39 * (age - 39)
        users[-1]["_base_pace"] = base
        # a few users are trainers, which drives ms_user_verifications and referrals
        users[-1]["_is_trainer"] = i <= max(3, n_users // 10)
        users[-1]["_group_oriented"] = rnd.random() < 0.45
    return emit("ms_users", users), emit("ms_user_demographics", demos)


# ---------------------------------------------------------------------------
# 3. ms_user_verifications
# ---------------------------------------------------------------------------
def gen_verifications(users, n):
    rows, i = [], 1
    pool = users[:]
    rnd.shuffle(pool)
    while len(rows) < n:
        u = pool[(i - 1) % len(pool)]
        vtype = "trainer_certificate" if u["_is_trainer"] and rnd.random() < 0.7 else \
            rnd.choice(["email", "phone", "identity_document"])
        status = pick([("verified", 0.74), ("pending", 0.14), ("rejected", 0.08), ("expired", 0.04)])
        at = u["_joined"] + timedelta(days=rnd.randint(0, 20), hours=rnd.randint(0, 23))
        rows.append({
            "id": i, "user_id": u["id"], "verification_type": vtype, "status": status,
            "verified_at": ts(at) if status == "verified" else "",
            "expires_at": ts(at + timedelta(days=365)) if status in ("verified", "expired") else "",
            "rejected_at": ts(at) if status == "rejected" else "",
            "rejection_reason": "document unreadable" if status == "rejected" else "",
            "created_at": ts(at), "updated_at": ts(at),
        })
        i += 1
    return emit("ms_user_verifications", rows)


# ---------------------------------------------------------------------------
# 4. ms_area — real Bangkok venues, see calibration SOURCE 6
# ---------------------------------------------------------------------------
def gen_areas(n):
    rows, i = [], 1
    for label in C.BANGKOK_PARKS:
        rows.append({"id": i, "area_type": "park", "area_label": label}); i += 1
    for label in C.BANGKOK_VENUES:
        rows.append({"id": i, "area_type": "stadium", "area_label": label}); i += 1
    for label in C.BANGKOK_DISTRICTS:
        rows.append({"id": i, "area_type": "district", "area_label": label}); i += 1
    rows = rows[:n]
    created = ts(TODAY - timedelta(days=WINDOW_DAYS + 30))
    for r in rows:
        r["created_at"] = created
        r["updated_at"] = created
    return emit("ms_area", rows)


# ---------------------------------------------------------------------------
# 5. ms_subscription_plans — the one table that is NOT 50 rows, on purpose
# ---------------------------------------------------------------------------
def gen_plans():
    created = ts(TODAY - timedelta(days=WINDOW_DAYS + 60))
    rows = []
    for i, (code, name, tier, price, period) in enumerate(C.PLAN_PRICING, start=1):
        rows.append({
            "id": i, "plan_code": code, "plan_name": name, "tier": tier,
            "price": price, "currency": "THB", "billing_period": period,
            "is_active": "false" if code.endswith("_OLD") else "true",
            "created_at": created, "updated_at": created,
        })
    return emit("ms_subscription_plans", rows)


# ---------------------------------------------------------------------------
# 6. ms_device_connections
# ---------------------------------------------------------------------------
PROVIDERS = [("apple_health", 0.42), ("strava", 0.28), ("garmin", 0.16),
             ("google_fit", 0.10), ("polar", 0.04)]


def gen_devices(users, n):
    rows, i = [], 1
    pool = [u for u in users]
    rnd.shuffle(pool)
    used = set()
    while len(rows) < n:
        u = pool[(i - 1) % len(pool)]
        prov = pick(PROVIDERS)
        if (u["id"], prov) in used:
            i += 1
            continue
        used.add((u["id"], prov))
        status = pick([("active", 0.82), ("revoked", 0.12), ("expired", 0.06)])
        linked = u["_joined"] + timedelta(days=rnd.randint(0, 30))
        rows.append({
            "id": len(rows) + 1, "user_id": u["id"], "provider": prov,
            "external_user_id": f"{prov[:2].upper()}-{rnd.randint(100000, 999999)}",
            "connection_status": status,
            "last_synced_at": ts(TODAY - timedelta(days=rnd.randint(0, 9), hours=rnd.randint(0, 23)))
            if status == "active" else ts(linked + timedelta(days=rnd.randint(5, 40))),
            "created_at": ts(linked), "updated_at": ts(linked),
        })
        i += 1
    return emit("ms_device_connections", rows)


# ---------------------------------------------------------------------------
# 7. txn_events
# ---------------------------------------------------------------------------
def gen_events(users, areas, n):
    rows = []
    owners = [u for u in users if u["_is_trainer"] or u["_group_oriented"]] or users
    for i in range(1, n + 1):
        owner = rnd.choice(owners)
        area = rnd.choice(areas)
        when = TODAY - timedelta(days=rnd.randint(-21, WINDOW_DAYS - 10))
        when = when.replace(hour=rnd.choice([6, 6, 7, 17, 18, 19]), minute=rnd.choice([0, 30]))
        status = "scheduled" if when > TODAY else pick([("completed", 0.82), ("cancelled", 0.18)])
        rows.append({
            "id": i, "event_owner": owner["id"],
            "workout_type": rnd.choice(C.WORKOUT_TYPES),
            "area_id": area["id"], "scheduled_at": ts(when),
            "target_level": rnd.choice([b[0] for b in C.LEVEL_BANDS]),
            "max_participants": rnd.choice([4, 6, 8, 10, 12, 20]),
            "status": status,
            "created_at": ts(when - timedelta(days=rnd.randint(3, 21))),
            "updated_at": ts(when),
        })
        rows[-1]["_when"] = when
    return emit("txn_events", rows)


# ---------------------------------------------------------------------------
# 8. txn_workout_summaries  +  the raw provider payloads for Bronze
# ---------------------------------------------------------------------------
def gen_workouts(users, devices, events, n):
    # power law: a few users log a lot, most log once or twice
    weights = [1 / (idx + 1) ** 0.85 for idx in range(len(users))]
    total = sum(weights)
    weights = [w / total for w in weights]
    shuffled = users[:]
    rnd.shuffle(shuffled)

    dev_by_user = {}
    for d in devices:
        if d["connection_status"] == "active":
            dev_by_user.setdefault(d["user_id"], []).append(d)

    completed_events = [e for e in events if e["status"] == "completed"]
    rows, raw = [], []
    for i in range(1, n + 1):
        u = shuffled[pick(list(zip(range(len(shuffled)), weights)))]
        days_ago = rnd.randint(0, min(WINDOW_DAYS, max(1, (TODAY - u["_joined"]).days)))
        started = TODAY - timedelta(days=days_ago)
        started = started.replace(hour=rnd.choice([5, 6, 6, 7, 18, 19, 20]),
                                  minute=rnd.choice([0, 5, 10, 15, 30, 45]))

        # improvement over time, faster for group-oriented users
        months_active = max(0.0, (started - u["_joined"]).days / 30.0)
        rate = C.IMPROVEMENT_PER_MONTH * (C.GROUP_TRAINING_BONUS if u["_group_oriented"] else 1.0)
        gain = min(C.IMPROVEMENT_CEILING, rate * months_active)
        pace = u["_base_pace"] * (1 - gain) * rnd.uniform(0.95, 1.06)

        atype = pick([("running", 0.62), ("functional", 0.18), ("hyrox_prep", 0.12),
                      ("cycling", 0.05), ("swimming", 0.03)])
        if atype in ("running", "hyrox_prep"):
            distance = round(rnd.choice([3.0, 5.0, 5.0, 6.0, 8.0, 10.0, 12.0, 21.1]) *
                             rnd.uniform(0.94, 1.08), 2)
            duration = int(distance * pace)
        else:
            distance = 0.0
            duration = rnd.choice([1800, 2400, 2700, 3000, 3600])

        weight_kg = rnd.uniform(52, 88)
        kcal = int(C.KCAL_PER_KG_PER_KM * weight_kg * distance) if distance else int(duration / 60 * 8.5)
        hrm = C.hr_max(u["_age"])
        avg_hr = int(hrm * rnd.uniform(*C.HR_INTENSITY_RANGE))

        devs = dev_by_user.get(u["id"], [])
        use_device = bool(devs) and rnd.random() < 0.68
        dev = rnd.choice(devs) if use_device else None
        ev = rnd.choice(completed_events) if (completed_events and rnd.random() < 0.3) else None

        # --- deliberate dirt, so Silver has something to catch -------------
        defect = None
        if i == 7:                       # GPS drift
            distance, defect = 84.3, "gps_drift"
        elif i == 19:                    # chest strap never connected
            avg_hr, defect = 12, "bad_hr"
        elif i == 31:                    # forgot to stop the watch
            duration, defect = duration + 19800, "forgot_stop"

        row = {
            "id": i, "user_id": u["id"],
            "device_id": dev["id"] if dev else "",
            "event_id": ev["id"] if ev else "",
            "external_activity_id": f"{(dev['provider'][:2].upper() if dev else 'MAN')}-{900000 + i}"
            if dev else "",
            "activity_type": atype,
            "source": dev["provider"] if dev else "manual",
            "started_at": ts(started),
            "duration_sec": duration,
            "distance_km": distance,
            "calories_kcal": kcal,
            "avg_heart_rate": avg_hr,
            "created_at": ts(started + timedelta(minutes=rnd.randint(1, 90))),
        }
        rows.append(row)
        raw.append({**row, "_provider": dev["provider"] if dev else "manual",
                    "_defect": defect, "_started_dt": started})
    return emit("txn_workout_summaries", rows), raw


# ---------------------------------------------------------------------------
# 9. txn_user_history_level
# ---------------------------------------------------------------------------
def gen_history_level(workouts, n):
    runs = [w for w in workouts if w["distance_km"] and w["distance_km"] < 40 and w["duration_sec"] < 20000]
    runs.sort(key=lambda w: w["started_at"])
    rows = []
    for i, w in enumerate(runs[:n], start=1):
        pace = w["duration_sec"] / w["distance_km"]
        rows.append({
            "id": i, "user_id": w["user_id"], "source_workout_id": w["id"],
            "fitness_level": band_for(pace),
            "calculation_method": "manual" if w["source"] == "manual" else "device",
            "running_distance": w["distance_km"],
            "running_duration_min": round(w["duration_sec"] / 60, 2),
            "recorded_at": w["created_at"],
            "created_at": w["created_at"], "updated_at": w["created_at"],
        })
    # top up from assessment tests if the run pool was short
    i = len(rows)
    while len(rows) < n:
        i += 1
        w = rnd.choice(workouts)
        rows.append({
            "id": i, "user_id": w["user_id"], "source_workout_id": "",
            "fitness_level": rnd.choice([b[0] for b in C.LEVEL_BANDS]),
            "calculation_method": "assessment_test",
            "running_distance": "", "running_duration_min": "",
            "recorded_at": w["created_at"], "created_at": w["created_at"],
            "updated_at": w["created_at"],
        })
    return emit("txn_user_history_level", rows)


# ---------------------------------------------------------------------------
# 10-12. goals, medical concerns, constraints
# ---------------------------------------------------------------------------
def gen_goals(users, n):
    rows = []
    for i in range(1, n + 1):
        u = users[(i - 1) % len(users)]
        gtype = rnd.choice(C.GOAL_TYPES)
        if gtype.startswith("hyrox"):
            secs = C.HYROX_MEDIAN_SECONDS[u["_sex"]] * rnd.uniform(0.92, 1.18)
            target = f"{int(secs // 3600)}:{int(secs % 3600 // 60):02d}:{int(secs % 60):02d}"
        elif gtype == "marathon":
            target = f"{rnd.choice([4, 4, 5, 5, 6])}:{rnd.choice(['00', '15', '30', '45'])}:00"
        elif gtype == "half_marathon":
            target = f"{rnd.choice([1, 2, 2, 2])}:{rnd.choice(['45', '00', '15', '30'])}:00"
        else:
            target = f"00:{rnd.choice([50, 55, 60, 65, 70])}:00"
        tdate = TODAY + timedelta(days=rnd.randint(-60, 210))
        rows.append({
            "id": i, "user_id": u["id"], "goal_type": gtype, "target_value": target,
            "target_date": tdate.strftime("%Y-%m-%d"),
            "status": "achieved" if tdate < TODAY and rnd.random() < 0.35
            else ("expired" if tdate < TODAY else "active"),
            "created_at": ts(u["_joined"]), "updated_at": ts(u["_joined"]),
        })
    return emit("txn_user_goals", rows)


CONCERNS = [("chronic_condition", "asthma", "salbutamol inhaler", "medium"),
            ("injury", "runner's knee", "", "high"),
            ("chronic_condition", "hypertension", "amlodipine", "high"),
            ("injury", "plantar fasciitis", "", "medium"),
            ("allergy", "pollen allergy", "cetirizine", "low"),
            ("injury", "ankle sprain history", "", "medium"),
            ("chronic_condition", "mild anaemia", "ferrous sulfate", "low"),
            ("injury", "lower back strain", "", "high")]


def gen_medical(users, n):
    rows = []
    subset = users[:max(1, int(len(users) * 0.7))]
    for i in range(1, n + 1):
        u = subset[(i - 1) % len(subset)]
        ctype, cname, med, level = rnd.choice(CONCERNS)
        rows.append({
            "id": i, "user_id": u["id"], "concern_type": ctype, "concern_name": cname,
            "medication_name": med, "restriction_level": level,
            "restriction_note": "avoid maximal effort in high heat" if level == "high" else "",
            "is_active": "true" if rnd.random() < 0.8 else "false",
            "created_at": ts(u["_joined"]), "updated_at": ts(u["_joined"]),
        })
    return emit("txn_user_medical_concerns", rows)


def gen_constraints(users, areas, n):
    rows = []
    for i in range(1, n + 1):
        u = users[(i - 1) % len(users)]
        a = rnd.choice(areas)
        ctype = pick([("time_window", 0.5), ("location", 0.3), ("frequency", 0.2)])
        after_work = rnd.random() < 0.62
        rows.append({
            "id": i, "user_id": u["id"], "constraint_type": ctype,
            "area_type": a["area_type"], "area_label": a["area_label"],
            "day_of_week": rnd.choice(["mon", "tue", "wed", "thu", "fri", "sat", "sun", "weekday", "weekend"]),
            "available_from": "18:00:00" if after_work else "05:30:00",
            "available_to": "21:00:00" if after_work else "08:00:00",
            "weekly_frequency": pick([(2, 0.28), (3, 0.34), (4, 0.22), (5, 0.16)]),
            "note": "after office hours" if after_work else "before work",
            "is_active": "true",
            "created_at": ts(u["_joined"]), "updated_at": ts(u["_joined"]),
        })
    return emit("txn_user_constraints", rows)


# ---------------------------------------------------------------------------
# 13-15. participants, comments, training plans
# ---------------------------------------------------------------------------
def gen_participants(users, events, n):
    rows, seen, i = [], set(), 0
    while len(rows) < n:
        i += 1
        e = events[rnd.randrange(len(events))]
        u = users[rnd.randrange(len(users))]
        if (u["id"], e["id"]) in seen:
            continue
        seen.add((u["id"], e["id"]))
        joined = e["_when"] - timedelta(days=rnd.randint(0, 14), hours=rnd.randint(0, 23))
        is_owner = u["id"] == e["event_owner"]
        if e["status"] == "cancelled":
            status = "cancelled"
        elif e["_when"] > TODAY:
            status = "joined"
        else:
            status = pick([("attended", 0.78), ("no_show", 0.14), ("cancelled", 0.08)])
        rows.append({
            "id": len(rows) + 1, "user_id": u["id"], "event_id": e["id"],
            "joined_at": ts(joined), "role": "host" if is_owner else "participant",
            "status": status, "created_at": ts(joined), "updated_at": ts(joined),
        })
    return emit("txn_event_participants", rows)


COMMENTS = ["เจอกันหน้าสวนนะครับ", "ขอจอยด้วยคนครับ เลเวลใกล้เคียงกันพอดี",
            "วันนี้ฝนตก เลื่อนไหมครับ", "pace เท่าไหร่ครับรอบนี้",
            "มาสายนิดนึงนะคะ รอด้วย", "ขอบคุณที่ตั้งตี้ครับ สนุกมาก",
            "สัปดาห์หน้าจัดอีกไหมครับ", "ใครมีคู่ buddy สำหรับ Hyrox บ้างคะ",
            "วอร์มอัพก่อน 10 นาทีนะครับ", "ที่จอดรถเต็มแล้ว แนะนำมารถไฟฟ้า"]


def gen_comments(users, events, n):
    rows = []
    for i in range(1, n + 1):
        e = rnd.choice(events)
        u = rnd.choice(users)
        at = e["_when"] - timedelta(days=rnd.randint(0, 10), hours=rnd.randint(0, 23))
        rows.append({
            "id": i, "user_id": u["id"], "event_id": e["id"],
            "comment_text": rnd.choice(COMMENTS),
            "created_at": ts(at), "updated_at": ts(at),
        })
    return emit("txn_group_comments", rows)


PLAN_BLOCKS = ["Warm-up 10 min easy jog | 6 x 400m @ 5K pace | cooldown 10 min",
               "Zone 2 base run 45 min, conversational pace",
               "Hyrox sim: 1km run + 50 wall balls x 4 rounds",
               "Tempo 20 min @ threshold + 2 x 1km strides",
               "Long run 90 min, negative split second half",
               "Strength: deadlift 5x5, sled push 6x20m, farmers carry 4x40m"]


def gen_plans_txn(users, events, n):
    rows = []
    for i in range(1, n + 1):
        e = rnd.choice(events)
        ai = rnd.random() < 0.55
        at = e["_when"] - timedelta(days=rnd.randint(1, 14))
        rows.append({
            "id": i, "created_by": e["event_owner"], "event_id": e["id"],
            "plan_details": rnd.choice(PLAN_BLOCKS),
            "plan_status": pick([("published", 0.7), ("draft", 0.2), ("archived", 0.1)]),
            "is_ai_suggested": "true" if ai else "false",
            "scaled_level": rnd.choice([b[0] for b in C.LEVEL_BANDS]) if ai else e["target_level"],
            "created_at": ts(at), "updated_at": ts(at),
        })
    return emit("txn_training_plans", rows)


# ---------------------------------------------------------------------------
# 16-17. subscriptions + payments
# ---------------------------------------------------------------------------
def gen_subscriptions(users, plans, n):
    trainers = [u["id"] for u in users if u["_is_trainer"]]
    paid = [p for p in plans if p["price"] > 0 and p["is_active"] == "true"]
    rows = []
    for i in range(1, n + 1):
        u = users[(i - 1) % len(users)]
        plan = rnd.choice(paid)
        referred = rnd.random() < 0.38
        ref_id = rnd.choice(trainers) if (referred and trainers) else ""
        start = u["_joined"] + timedelta(days=rnd.randint(1, 45))
        if start > TODAY:
            start = TODAY - timedelta(days=rnd.randint(1, 20))
        # trainer-referred users churn less (baked-in relationship)
        churn_p = 0.30 * (C.REFERRED_CHURN_FACTOR if ref_id else 1.0)
        r = rnd.random()
        if r < churn_p:
            status = "cancelled"
        elif r < churn_p + 0.07:
            status = "past_due"
        elif r < churn_p + 0.13:
            status = "trialing"
        else:
            status = "active"
        period = 365 if plan["billing_period"] == "annual" else 30
        rows.append({
            "id": 500 + i, "user_id": u["id"], "plan_id": plan["id"],
            "referred_by_user_id": ref_id,
            "status": status,
            "started_at": ts(start),
            "current_period_end": ts(start + timedelta(days=period)),
            "cancelled_at": ts(start + timedelta(days=rnd.randint(20, 120))) if status == "cancelled" else "",
            "auto_renew": "false" if status == "cancelled" else "true",
            "created_at": ts(start), "updated_at": ts(start + timedelta(days=rnd.randint(0, 60))),
        })
        rows[-1]["_plan_price"] = plan["price"]
        rows[-1]["_start"] = start
        # keep the denormalised cache consistent with the source of truth
        if status in ("active", "trialing", "past_due"):
            u["tier"] = plan["tier"]
    return emit("txn_user_subscriptions", rows)


def gen_payments(subs, n):
    rows = []
    i = 0
    while len(rows) < n:
        s = subs[i % len(subs)]
        i += 1
        attempt = s["_start"] + timedelta(days=rnd.randint(0, 60))
        if s["status"] == "past_due" and rnd.random() < 0.6:
            status, paid = "failed", ""
        elif s["status"] == "cancelled" and rnd.random() < 0.2:
            status, paid = "refunded", ts(attempt)
        else:
            status, paid = "succeeded", ts(attempt)
        rows.append({
            "id": 9000 + len(rows) + 1, "subscription_id": s["id"], "user_id": s["user_id"],
            "amount": s["_plan_price"], "currency": "THB",
            "payment_method": pick([("credit_card", 0.52), ("promptpay", 0.33),
                                    ("truemoney", 0.09), ("bank_transfer", 0.06)]),
            "gateway_txn_ref": f"omise_ch_{700000 + len(rows) + 1}",
            "status": status, "paid_at": paid,
            "created_at": ts(attempt), "updated_at": ts(attempt),
        })
    return emit("txn_payments", rows)


# ---------------------------------------------------------------------------
# 18. txn_notifications — funnel stored as columns, per the ER diagram
# ---------------------------------------------------------------------------
NOTIF = [("event_reminder", "ซ้อมพรุ่งนี้ 06:00", "อย่าลืมนัดซ้อมที่"),
         ("buddy_matched", "เจอคู่ซ้อมแล้ว", "มีคนเลเวลเดียวกับคุณ"),
         ("level_up", "ขึ้นเลเวลแล้ว", "pace ของคุณดีขึ้น"),
         ("plan_due", "ถึงรอบแผนซ้อม", "แผนสัปดาห์นี้รออยู่"),
         ("subscription_expiring", "สมาชิกใกล้หมดอายุ", "ต่ออายุเพื่อดูแผนซ้อมต่อ")]


def gen_notifications(users, events, n):
    rows = []
    for i in range(1, n + 1):
        u = users[rnd.randrange(len(users))]
        ntype, title, body = rnd.choice(NOTIF)
        ev = rnd.choice(events) if ntype in ("event_reminder", "buddy_matched") else None
        sent = TODAY - timedelta(days=rnd.randint(0, 120), hours=rnd.randint(0, 23))
        send_status = pick([("sent", 0.93), ("failed", 0.05), ("queued", 0.02)])
        delivered = opened = clicked = dismissed = ""
        if send_status == "sent":
            if rnd.random() < 0.95:
                d = sent + timedelta(seconds=rnd.randint(2, 90)); delivered = ts(d)
                if rnd.random() < 0.42:
                    o = d + timedelta(minutes=rnd.randint(1, 600)); opened = ts(o)
                    if rnd.random() < 0.37:
                        clicked = ts(o + timedelta(seconds=rnd.randint(3, 180)))
                    elif rnd.random() < 0.3:
                        dismissed = ts(o + timedelta(seconds=rnd.randint(1, 30)))
        rows.append({
            "id": i, "user_id": u["id"], "event_id": ev["id"] if ev else "",
            "notification_type": ntype,
            "channel": pick([("push", 0.68), ("in_app", 0.22), ("email", 0.10)]),
            "title": title, "body": body,
            "deep_link": f"lowrox://event/{ev['id']}" if ev else "lowrox://profile/level",
            "send_status": send_status, "sent_at": ts(sent) if send_status != "queued" else "",
            "delivered_at": delivered, "opened_at": opened,
            "clicked_at": clicked, "dismissed_at": dismissed,
            "created_at": ts(sent), "updated_at": ts(sent),
        })
    return emit("txn_notifications", rows)


# ---------------------------------------------------------------------------
# raw provider payloads for the Bronze zone
# ---------------------------------------------------------------------------
def write_source_mock(raw, devices, payments, users):
    d = OUT / "source_mock"
    dev_by_id = {x["id"]: x for x in devices}

    apple, strava = [], []
    for w in raw:
        if not w["device_id"]:
            continue
        prov = dev_by_id[w["device_id"]]["provider"]
        if prov == "apple_health":
            apple.append({
                "uuid": w["external_activity_id"], "user": f"U{w['user_id']:03d}",
                "type": w["activity_type"],
                "start": w["_started_dt"].isoformat(),          # +07:00
                "duration_min": round(w["duration_sec"] / 60, 1),
                "distance_km": w["distance_km"],                 # kilometres
                "kcal": w["calories_kcal"], "avg_hr": w["avg_heart_rate"],
            })
        elif prov == "strava":
            strava.append({
                "id": w["external_activity_id"], "athlete": f"U{w['user_id']:03d}",
                "sport_type": "Run" if w["activity_type"] == "running" else "Workout",
                "start_date": w["_started_dt"].astimezone(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),                 # UTC, different format
                "elapsed_time": w["duration_sec"],
                "distance": w["distance_km"] * 1000,             # METRES, different unit
                "calories": w["calories_kcal"],
                "average_heartrate": w["avg_heart_rate"],
            })

    # the export ran twice: the second file is a duplicate of the first
    (d / "apple_health_0735.json").write_text(json.dumps(
        {"provider": "apple_health", "exported_at": "2026-10-21T07:35:00+07:00",
         "workouts": apple}, ensure_ascii=False, indent=1))
    (d / "apple_health_0752.json").write_text(json.dumps(
        {"provider": "apple_health", "exported_at": "2026-10-21T07:52:00+07:00",
         "workouts": apple}, ensure_ascii=False, indent=1))
    (d / "strava_0810.json").write_text(json.dumps(
        {"provider": "strava", "exported_at": "2026-10-21T08:10:00Z",
         "activities": strava}, ensure_ascii=False, indent=1))

    # payment gateway fires the first webhook twice
    hooks = [{"op": "c", "ts_ms": int(datetime.now().timestamp() * 1000),
              "source": {"table": "txn_payments"}, "after": p} for p in payments[:12]]
    hooks.insert(1, hooks[0])
    (d / "payment_webhooks.json").write_text(json.dumps(hooks, ensure_ascii=False, indent=1))

    # clickstream: week 1 everyone, week 2 a churned subset — the retention signal
    # that the 18 OLTP tables cannot provide
    lines = []
    active = [u["id"] for u in users]
    for wk, cohort in [(1, active), (2, active[:int(len(active) * 0.62)])]:
        day = TODAY - timedelta(days=14 - 7 * (wk - 1))
        for uid in cohort:
            for k, ev in enumerate(["app_open", "view_event", "view_plan"]):
                lines.append(json.dumps({
                    "user": f"U{uid:03d}",
                    "ts": (day + timedelta(hours=7, minutes=k * 3)).isoformat(),
                    "event": ev}, ensure_ascii=False))
            if rnd.random() < 0.3:
                lines.append(json.dumps({
                    "user": f"U{uid:03d}",
                    "ts": (day + timedelta(hours=9)).isoformat(),
                    "event": "paywall_view"}, ensure_ascii=False))
                if rnd.random() < C.PAYWALL_TO_SUBSCRIBE_RATE:
                    lines.append(json.dumps({
                        "user": f"U{uid:03d}",
                        "ts": (day + timedelta(hours=9, minutes=2)).isoformat(),
                        "event": "subscribe_click"}, ensure_ascii=False))
    (d / "clickstream_2026w42_w43.ndjson").write_text("\n".join(lines), encoding="utf-8")
    return len(apple), len(strava), len(lines)


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
ORDER = ["ms_users", "ms_user_demographics", "ms_user_verifications", "ms_area",
         "ms_subscription_plans", "ms_device_connections", "txn_events",
         "txn_workout_summaries", "txn_user_history_level", "txn_user_goals",
         "txn_user_medical_concerns", "txn_user_constraints",
         "txn_event_participants", "txn_group_comments", "txn_training_plans",
         "txn_user_subscriptions", "txn_payments", "txn_notifications"]

NUMERIC = {"id", "user_id", "event_id", "device_id", "area_id", "plan_id",
           "subscription_id", "source_workout_id", "referred_by_user_id",
           "event_owner", "created_by", "max_participants", "duration_sec",
           "distance_km", "calories_kcal", "avg_heart_rate", "weekly_frequency",
           "running_distance", "running_duration_min", "amount", "price"}
BOOLEAN = {"is_active", "auto_renew", "is_ai_suggested"}


def clean(rows):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def write_csv():
    for t in ORDER:
        rows = clean(TABLES[t])
        with open(OUT / "csv" / f"{t}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def sql_value(col, val):
    if val == "" or val is None:
        return "NULL"
    if col in BOOLEAN:
        return str(val).lower()
    if col in NUMERIC:
        return str(val)
    return "'" + str(val).replace("'", "''") + "'"


def write_sql():
    parts = ["-- Lowrox mock seed data, generated by generate.py",
             "-- Insert order respects every foreign key in the ER diagram.",
             "-- ลำดับ insert เคารพ foreign key ทุกตัวใน ER diagram", "BEGIN;", ""]
    for t in ORDER:
        rows = clean(TABLES[t])
        cols = list(rows[0].keys())
        parts.append(f"-- {t}: {len(rows)} rows")
        parts.append(f"INSERT INTO {t} ({', '.join(cols)}) VALUES")
        vals = [f"  ({', '.join(sql_value(c, r[c]) for c in cols)})" for r in rows]
        parts.append(",\n".join(vals) + ";")
        parts.append("")
    # keep the sequences in step with the explicit ids we inserted
    parts.append("-- resync sequences so the app can insert without colliding")
    for t in ORDER:
        if t == "ms_user_demographics":
            continue
        parts.append(f"SELECT setval(pg_get_serial_sequence('{t}', 'id'), "
                     f"(SELECT MAX(id) FROM {t}));")
    parts += ["", "COMMIT;"]
    (OUT / "02_seed.sql").write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50, help="rows per table")
    ap.add_argument("--scale", type=int, default=1, help="multiply rows by this")
    args = ap.parse_args()
    n = args.rows * args.scale

    users, demos = gen_users(n)
    gen_verifications(users, n)
    areas = gen_areas(n)
    plans = gen_plans()
    devices = gen_devices(users, n)
    events = gen_events(users, areas, n)
    workouts, raw = gen_workouts(users, devices, events, n)
    gen_history_level(workouts, n)
    gen_goals(users, n)
    gen_medical(users, n)
    gen_constraints(users, areas, n)
    gen_participants(users, events, n)
    gen_comments(users, events, n)
    gen_plans_txn(users, events, n)
    subs = gen_subscriptions(users, plans, n)
    payments = gen_payments(subs, n)
    gen_notifications(users, events, n)

    write_csv()
    write_sql()
    a, s, c = write_source_mock(raw, devices, clean(payments), users)

    print(f"{'table':<30}{'rows':>6}")
    print("-" * 36)
    for t in ORDER:
        print(f"{t:<30}{len(TABLES[t]):>6}")
    print("-" * 36)
    print(f"{'TOTAL':<30}{sum(len(TABLES[t]) for t in ORDER):>6}")
    print(f"\nsource mock: apple {a} activities, strava {s}, clickstream {c} events")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
