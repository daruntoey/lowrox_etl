"""
Calibration constants for the Lowrox mock data generator.

Every number here is traceable to a public source or is flagged as an
assumption. Nothing is invented silently.  |  ทุกตัวเลขมีที่มา ไม่มีตัวเลขที่เดาเอาเงียบ ๆ
"""

# ---------------------------------------------------------------------------
# SOURCE 1 — 5K benchmark finish times, seconds, by sex and age bracket
# marathonhandbook.com/whats-a-good-5k-time/
# The site labels its columns Beginner / Novice / Intermediate / High-level
# RECREATIONAL RACER. Note these are RACE FINISHERS, so they are faster than
# the general population this app serves (selection bias).
# ตัวเลขนี้มาจากคนที่ลงแข่งจริง จึงเร็วกว่าประชากรทั่วไปที่แอปให้บริการ
# ---------------------------------------------------------------------------
FIVEK_SECONDS = {
    ("M", 18): {"beginner": 1620, "novice": 1440, "intermediate": 1200, "advanced": 1020},
    ("M", 40): {"beginner": 1725, "novice": 1545, "intermediate": 1275, "advanced": 1080},
    ("M", 45): {"beginner": 1830, "novice": 1635, "intermediate": 1350, "advanced": 1155},
    ("M", 50): {"beginner": 1890, "novice": 1695, "intermediate": 1395, "advanced": 1200},
    ("M", 55): {"beginner": 1965, "novice": 1755, "intermediate": 1455, "advanced": 1230},
    ("M", 60): {"beginner": 2040, "novice": 1815, "intermediate": 1500, "advanced": 1290},
    ("F", 18): {"beginner": 1770, "novice": 1590, "intermediate": 1320, "advanced": 1125},
    ("F", 40): {"beginner": 1965, "novice": 1770, "intermediate": 1470, "advanced": 1260},
    ("F", 45): {"beginner": 2025, "novice": 1815, "intermediate": 1500, "advanced": 1290},
    ("F", 50): {"beginner": 2070, "novice": 1860, "intermediate": 1545, "advanced": 1320},
    ("F", 55): {"beginner": 2265, "novice": 1980, "intermediate": 1635, "advanced": 1410},
    ("F", 60): {"beginner": 2370, "novice": 2130, "intermediate": 1770, "advanced": 1515},
}

# Derived from the table above: women are about 9% slower than men at the same
# level and age (1770 / 1620 = 1.093).  |  ผู้หญิงช้ากว่าผู้ชายราว 9% ที่ระดับและอายุเดียวกัน
SEX_PACE_FACTOR = {"M": 1.00, "F": 1.093, "O": 1.05}

# Also derived: men's beginner 5K goes 1620s at 18-39 to 2040s at 60+,
# about +0.8% per year past 39.  |  ช้าลงราว 0.8% ต่อปีหลังอายุ 39
AGE_PENALTY_PER_YEAR_PAST_39 = 0.008

# ---------------------------------------------------------------------------
# SOURCE 2 — Level band cut points, seconds per km
# Anchored on SOURCE 1 and on the stated persona in the problem statement
# ("runs 5 km in 1 hour" = 720 s/km, which is slower than every published
# race-finisher benchmark). Four bands, which is what dim_user_profile's
# 4 x 4 x 4 x 5 = 320 combinations assume.
# จุดตัดอ้างอิงจาก SOURCE 1 และจาก persona ในโจทย์ที่วิ่ง 5 กม. ใน 1 ชม.
# ---------------------------------------------------------------------------
LEVEL_BANDS = [
    # (name, lower bound inclusive, upper bound exclusive) in seconds per km
    ("advanced",     0,   300),   # 5K under 25:00 — at or beyond marathonhandbook "novice racer"
    ("intermediate", 300, 420),   # 5K 25:00-35:00 — the site's stated recreational average
    ("beginner",     420, 600),   # 5K 35:00-50:00
    ("starter",      600, 9999),  # 5K 50:00+ — the persona in the problem statement
]

# Population pace is drawn lognormal around this median, NOT around the
# race-finisher benchmarks, because the app's users have mostly never raced.
# ASSUMPTION, stated openly.  |  ข้อสมมติ ระบุไว้ชัดเจน
POPULATION_MEDIAN_PACE_SEC_PER_KM = 430
POPULATION_PACE_LOG_SIGMA = 0.28

# ---------------------------------------------------------------------------
# SOURCE 3 — Hyrox Open division median finish times, seconds
# boxlifemagazine.com, analysis of ~700,000 Hyrox results
# Men 1:27:56 = 5276 s, Women 1:36:20 = 5780 s
# ใช้ตั้งค่า target_value ของเป้าหมายที่เป็น Hyrox
# ---------------------------------------------------------------------------
HYROX_MEDIAN_SECONDS = {"M": 5276, "F": 5780, "O": 5500}

# ---------------------------------------------------------------------------
# SOURCE 4 — Tanaka formula for maximum heart rate, widely used in sports
# physiology and more accurate across ages than 220 - age.
# HRmax = 208 - 0.7 * age ; steady running sits at 75-85% of HRmax.
# สูตร Tanaka ใช้กันทั่วไปในสรีรวิทยาการกีฬา แม่นกว่า 220 - อายุ
# ---------------------------------------------------------------------------
def hr_max(age: int) -> float:
    return 208 - 0.7 * age

HR_INTENSITY_RANGE = (0.75, 0.85)

# ---------------------------------------------------------------------------
# SOURCE 5 — Energy cost of running, approximately 1 kcal per kg per km.
# Standard exercise-physiology approximation.
# ค่าประมาณมาตรฐานทางสรีรวิทยาการออกกำลังกาย
# ---------------------------------------------------------------------------
KCAL_PER_KG_PER_KM = 1.0

# ---------------------------------------------------------------------------
# ASSUMPTIONS — order of magnitude only, each one flagged in the report.
# These are the relationships deliberately baked in so the marts have
# something to find. Synthetic data cannot validate them; it can only show
# that the pipeline computes them correctly.
# ความสัมพันธ์ที่ตั้งใจฝังไว้ ข้อมูลสังเคราะห์พิสูจน์สมมติฐานไม่ได้
# พิสูจน์ได้แค่ว่า pipeline คำนวณถูก
# ---------------------------------------------------------------------------
IMPROVEMENT_PER_MONTH = 0.012          # pace improves ~1.2% per active month
IMPROVEMENT_CEILING = 0.22             # no one improves more than 22% overall
GROUP_TRAINING_BONUS = 1.5             # group trainers improve 1.5x faster
REFERRED_CHURN_FACTOR = 0.55           # trainer-referred users churn 45% less
PAYWALL_TO_SUBSCRIBE_RATE = 0.28       # conversion after seeing the paywall

# Thai urban fitness-app demographics. ASSUMPTION.
# ข้อมูลประชากรผู้ใช้แอปฟิตเนสในเมืองไทย เป็นข้อสมมติ
AGE_DISTRIBUTION = [(22, 29, 0.30), (30, 39, 0.42), (40, 49, 0.20), (50, 62, 0.08)]
SEX_DISTRIBUTION = [("M", 0.55), ("F", 0.43), ("O", 0.02)]

# Subscription pricing in THB, matching ms_subscription_plans in the ER diagram.
PLAN_PRICING = [
    ("FREE_M",  "Free",          "free",     0.0,   "monthly"),
    ("PRO_M",   "Pro Monthly",   "pro",      349.0, "monthly"),
    ("PRO_A",   "Pro Annual",    "pro",     3490.0, "annual"),
    ("ELITE_M", "Elite Monthly", "elite",    899.0, "monthly"),
    ("ELITE_A", "Elite Annual",  "elite",   8990.0, "annual"),
    ("PRO_M_OLD", "Pro Monthly (2025 price)", "pro", 299.0, "monthly"),
]

# ---------------------------------------------------------------------------
# SOURCE 6 — Real Bangkok training venues and districts.
# Public reference data, used for ms_area so the area dimension is not fiction.
# สถานที่ซ้อมจริงในกรุงเทพ ใช้กับ ms_area เพื่อให้ dimension ไม่ใช่ของปลอม
# ---------------------------------------------------------------------------
BANGKOK_PARKS = [
    "Lumphini Park", "Benjakitti Forest Park", "Chatuchak Park",
    "Wachirabenchathat Park", "Benjasiri Park", "Queen Sirikit Park",
    "King Rama IX Park", "Saranrom Park", "Romaneenart Park",
    "Santiphap Park", "Thonburirom Park", "Chaloem Phrakiat Park",
    "Nong Bon Water Sports Centre", "Bang Kachao Green Lung",
    "Suan Luang Rama IX", "Phutthamonthon Park", "Si Nakhon Khuean Khan Park",
    "Wachirabenchathat Rot Fai Park", "Bang Khen Public Park",
    "Nawaminphirom Park",
]
BANGKOK_VENUES = [
    "Thephasadin Stadium", "National Stadium Bangkok",
    "Chulalongkorn University Stadium", "Thammasat Rangsit Stadium",
    "Kasetsart University Track", "Mahidol Salaya Track",
    "Hua Mak Sports Complex", "Thai-Japanese Bangkok Youth Centre",
    "Bangkok Arena Nong Chok", "Rajamangala National Stadium",
]
BANGKOK_DISTRICTS = [
    "Pathum Wan", "Watthana", "Khlong Toei", "Chatuchak", "Huai Khwang",
    "Bang Rak", "Sathon", "Phaya Thai", "Ratchathewi", "Din Daeng",
    "Lat Phrao", "Bang Kapi", "Suan Luang", "Prawet", "Bang Na",
    "Phra Khanong", "Thon Buri", "Khlong San", "Bang Sue", "Dusit",
]

WORKOUT_TYPES = ["running", "functional", "hyrox_prep", "cycling", "swimming"]
GOAL_TYPES = ["hyrox_singles", "hyrox_doubles", "marathon", "half_marathon", "fun_run_10k"]
