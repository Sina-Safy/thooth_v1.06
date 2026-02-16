import cv2
import numpy as np

# ============================================================
# Reference LABs (same as desktop APP.py)
# ============================================================
REAL_LIGHT_LAB = np.array([86.2, 0.0, 5.8], dtype=np.float64)
REAL_DARK_LAB  = np.array([62.8, 3.9, 18.8], dtype=np.float64)

# ============================================================
# Shade guide (same as desktop APP.py)
# Note: website uses ΔE2000 for "nearest shade" (per your request)
# ============================================================
SHADE_GUIDE = [
    {"name": "A1",   "lab": np.array([85.0,  0.0, 10.0], dtype=np.float64)},
    {"name": "A2",   "lab": np.array([80.0,  1.5, 15.0], dtype=np.float64)},
    {"name": "A3",   "lab": np.array([75.0,  3.0, 20.0], dtype=np.float64)},
    {"name": "A3.5", "lab": np.array([72.0,  4.0, 22.0], dtype=np.float64)},
    {"name": "B1",   "lab": np.array([86.0, -1.0, 12.0], dtype=np.float64)},
    {"name": "B2",   "lab": np.array([82.0,  0.0, 18.0], dtype=np.float64)},
    {"name": "C1",   "lab": np.array([80.0,  0.0,  8.0], dtype=np.float64)},
    {"name": "C2",   "lab": np.array([76.0,  1.5, 12.0], dtype=np.float64)},
    {"name": "D2",   "lab": np.array([78.0,  2.0, 10.0], dtype=np.float64)},
]

# ============================================================
# Params for each mode (same as desktop APP.py)
# SR preprocess is intentionally NOT applied here (per your request)
# ============================================================
FA_PARAMS = {
    "SECOND_STAGE_OFFSET": np.array([1.22543944, -0.54819939, 5.66859776], dtype=np.float64),
    "STAGE3_L_GAMMA": 0.49310,
    "FINAL_STAGE_OFFSET": np.array([-0.50868, -0.16668, -4.49899], dtype=np.float64),

    "STAGE1_S_GAMMA_L": 1.0,
    "STAGE1_S_GAMMA_A": 1.0,
    "STAGE1_S_GAMMA_B": 1.0,

    "SATURATION_L_DROP": 3.50,
    "SAT_THRESHOLD": 0.70,
    "SOFT_BRIGHT_L": 84.5,

    "GAMMA_LOW_L": 76.0,
    "GAMMA_HIGH_L": 86.5,
    "MAX_STAGE3_BOOST_L": 3.0,
}

FR_PARAMS = {
    "SECOND_STAGE_OFFSET": np.array([1.22543944, -0.54819939, 5.66859776], dtype=np.float64),
    "STAGE3_L_GAMMA": 0.49310,
    "FINAL_STAGE_OFFSET": np.array([-0.50868, -0.16668, -4.49899], dtype=np.float64),

    "STAGE1_S_GAMMA_L": 1.0,
    "STAGE1_S_GAMMA_A": 1.0,
    "STAGE1_S_GAMMA_B": 1.0,

    "SATURATION_L_DROP": 3.50,
    "SAT_THRESHOLD": 0.70,
    "SOFT_BRIGHT_L": 84.5,

    "GAMMA_LOW_L": 76.0,
    "GAMMA_HIGH_L": 86.5,
    "MAX_STAGE3_BOOST_L": 3.0,
}

SA_PARAMS = {
    "SECOND_STAGE_OFFSET": np.array([1.22543944, -0.54819939, 5.66859776], dtype=np.float64),
    "STAGE3_L_GAMMA": 0.49310,

    "STAGE1_S_GAMMA_L": 1.0,
    "STAGE1_S_GAMMA_A": 1.0,
    "STAGE1_S_GAMMA_B": 1.0,

    "SATURATION_L_DROP": 3.50,
    "SAT_THRESHOLD": 0.70,
    "SOFT_BRIGHT_L": 84.5,

    "GAMMA_LOW_L": 70.0,
    "GAMMA_HIGH_L": 86.5,
    "MAX_STAGE3_BOOST_L": 3.0,

    "FINAL_STAGE_OFFSET": np.array([-6.02771, -1.94433, 0.60999], dtype=np.float64),
}

SR_PARAMS = {
    "SECOND_STAGE_OFFSET": np.array([1.22543944, -0.54819939, 5.66859776], dtype=np.float64),
    "STAGE3_L_GAMMA": 0.49310,

    "STAGE1_S_GAMMA_L": 1.0,
    "STAGE1_S_GAMMA_A": 1.0,
    "STAGE1_S_GAMMA_B": 1.0,

    "SATURATION_L_DROP": 3.50,
    "SAT_THRESHOLD": 0.70,
    "SOFT_BRIGHT_L": 84.5,

    "GAMMA_LOW_L": 70.0,
    "GAMMA_HIGH_L": 86.5,
    "MAX_STAGE3_BOOST_L": 3.0,

    "FINAL_STAGE_OFFSET": np.array([-2.72114, -2.33791, 0.53112], dtype=np.float64),
}

MODE_CONFIG = {
    "fa": {"label": "FA", "params": FA_PARAMS, "safe_ab": False},
    "fr": {"label": "FR", "params": FR_PARAMS, "safe_ab": False},
    "sa": {"label": "SA", "params": SA_PARAMS, "safe_ab": True},
    "sr": {"label": "SR", "params": SR_PARAMS, "safe_ab": True},
}

MASK_SHRINK_PX = 2

# ============================================================
# Color conversions
# ============================================================
def bgr_to_lab_opencv(img_bgr: np.ndarray) -> np.ndarray:
    lab8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab = lab8.astype(np.float64)
    lab[..., 0] = (lab[..., 0] / 255.0) * 100.0
    lab[..., 1] = lab[..., 1] - 128.0
    lab[..., 2] = lab[..., 2] - 128.0
    return lab

def lab_to_bgr_opencv(lab: np.ndarray) -> np.ndarray:
    lab = np.array(lab, dtype=np.float64)
    lab8 = np.empty_like(lab, dtype=np.uint8)
    L8 = np.clip((lab[..., 0] / 100.0) * 255.0, 0, 255)
    a8 = np.clip(lab[..., 1] + 128.0, 0, 255)
    b8 = np.clip(lab[..., 2] + 128.0, 0, 255)
    lab8[..., 0] = L8.astype(np.uint8)
    lab8[..., 1] = a8.astype(np.uint8)
    lab8[..., 2] = b8.astype(np.uint8)
    return cv2.cvtColor(lab8, cv2.COLOR_LAB2BGR)

# ============================================================
# ROI measurement (same behavior as desktop)
# ============================================================
def _hex_pts(points) -> np.ndarray:
    pts = np.array(points, dtype=np.float64).reshape((-1, 2))
    pts = np.round(pts).astype(np.int32)
    return pts.reshape((-1, 1, 2))

def trimmed_mean_lab_in_hex(img_bgr: np.ndarray, points, low_trim_pct=0.0, high_trim_pct=0.0):
    if img_bgr is None:
        return None
    pts = _hex_pts(points) if points is not None else None
    if pts is None or len(pts) < 3:
        return None

    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    if MASK_SHRINK_PX > 0:
        k = 2 * MASK_SHRINK_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.erode(mask, kernel, iterations=1)

    lab_img = bgr_to_lab_opencv(img_bgr)
    pixels = lab_img[mask == 255].reshape(-1, 3)
    if pixels.size == 0:
        return None

    L_vals = pixels[:, 0]
    if len(L_vals) < 5:
        return pixels.mean(axis=0)

    low_trim = float(np.clip(low_trim_pct, 0, 100))
    high_trim = float(np.clip(high_trim_pct, 0, 100))
    if low_trim + high_trim >= 100:
        low_trim, high_trim = 0.0, 0.0

    low_q = np.percentile(L_vals, low_trim)
    high_q = np.percentile(L_vals, 100 - high_trim)

    filtered = pixels[(L_vals >= low_q) & (L_vals <= high_q)]
    if len(filtered) == 0:
        filtered = pixels

    return filtered.mean(axis=0)

# ============================================================
# Stage 0: global WB using delta_illum (same as desktop)
# ============================================================
def apply_global_delta_illum(img_bgr: np.ndarray, delta_illum: np.ndarray) -> np.ndarray:
    lab = bgr_to_lab_opencv(img_bgr)
    lab2 = lab + delta_illum.reshape((1, 1, 3))
    lab2[..., 0] = np.clip(lab2[..., 0], 0, 100)
    lab2[..., 1:] = np.clip(lab2[..., 1:], -128, 127)
    return lab_to_bgr_opencv(lab2)

# ============================================================
# Stage1..4 (same as desktop)
# ============================================================
def scurve_escape_channel(meas_x, meas_light_x, meas_dark_x, real_light_x, real_dark_x, gamma_s):
    denom = (meas_light_x - meas_dark_x)
    if abs(denom) < 1e-5:
        return real_dark_x

    t = (meas_x - meas_dark_x) / denom

    if 0.0 <= t <= 1.0:
        g = max(gamma_s, 1e-6)
        num = t ** g
        den = num + (1.0 - t) ** g
        t_prime = num / (den + 1e-9)
        return real_dark_x + t_prime * (real_light_x - real_dark_x)

    a_lin = (real_light_x - real_dark_x) / (meas_light_x - meas_dark_x + 1e-6)
    b_lin = real_dark_x - a_lin * meas_dark_x
    return a_lin * meas_x + b_lin

def scurve_escape_channel_safe(meas_x, meas_light_x, meas_dark_x, real_light_x, real_dark_x, gamma_s, denom_floor=5.0):
    denom = (meas_light_x - meas_dark_x)
    if abs(denom) < denom_floor:
        denom = np.sign(denom) * denom_floor if denom != 0 else denom_floor

    t = (meas_x - meas_dark_x) / denom

    if 0.0 <= t <= 1.0:
        g = max(gamma_s, 1e-6)
        num = t ** g
        den = num + (1.0 - t) ** g
        t_prime = num / (den + 1e-9)
        return real_dark_x + t_prime * (real_light_x - real_dark_x)

    a_lin = (real_light_x - real_dark_x) / denom
    b_lin = real_dark_x - a_lin * meas_dark_x
    return a_lin * meas_x + b_lin

def apply_stage1_generic(measured_lab, measured_light_lab_wb, measured_dark_lab_wb, params, safe_ab):
    measured_lab = np.array(measured_lab, dtype=np.float64)

    L_real = scurve_escape_channel(
        measured_lab[0], measured_light_lab_wb[0], measured_dark_lab_wb[0],
        REAL_LIGHT_LAB[0], REAL_DARK_LAB[0], params["STAGE1_S_GAMMA_L"]
    )

    if safe_ab:
        a_real = scurve_escape_channel_safe(
            measured_lab[1], measured_light_lab_wb[1], measured_dark_lab_wb[1],
            REAL_LIGHT_LAB[1], REAL_DARK_LAB[1], params["STAGE1_S_GAMMA_A"]
        )
        b_real = scurve_escape_channel_safe(
            measured_lab[2], measured_light_lab_wb[2], measured_dark_lab_wb[2],
            REAL_LIGHT_LAB[2], REAL_DARK_LAB[2], params["STAGE1_S_GAMMA_B"]
        )
    else:
        a_real = scurve_escape_channel(
            measured_lab[1], measured_light_lab_wb[1], measured_dark_lab_wb[1],
            REAL_LIGHT_LAB[1], REAL_DARK_LAB[1], params["STAGE1_S_GAMMA_A"]
        )
        b_real = scurve_escape_channel(
            measured_lab[2], measured_light_lab_wb[2], measured_dark_lab_wb[2],
            REAL_LIGHT_LAB[2], REAL_DARK_LAB[2], params["STAGE1_S_GAMMA_B"]
        )

    lab_stage1 = np.array([L_real, a_real, b_real], dtype=np.float64)
    lab_stage1[0] = np.clip(lab_stage1[0], 0, 100)
    lab_stage1[1] = np.clip(lab_stage1[1], -128, 127)
    lab_stage1[2] = np.clip(lab_stage1[2], -128, 127)
    return lab_stage1

def apply_stage2_offset(lab, params):
    lab = np.array(lab, dtype=np.float64) + params["SECOND_STAGE_OFFSET"]
    lab[0] = np.clip(lab[0], 0, 100)
    lab[1] = np.clip(lab[1], -128, 127)
    lab[2] = np.clip(lab[2], -128, 127)
    return lab

def apply_stage3_L_gamma_conditional(lab_stage2, params, stage1_L):
    lab = np.array(lab_stage2, dtype=np.float64)

    sat_thresh_stage1 = REAL_LIGHT_LAB[0] - params["SAT_THRESHOLD"]
    stage2_L = lab[0]
    soft_bright_L = params["SOFT_BRIGHT_L"]

    saturated = (
        (stage1_L >= sat_thresh_stage1) or
        (stage2_L >= (REAL_LIGHT_LAB[0] + 0.5)) or
        (stage2_L >= soft_bright_L)
    )

    if saturated:
        gamma_eff = 1.0
        lab[0] = lab[0] - params["SATURATION_L_DROP"]
    else:
        base_gamma = params["STAGE3_L_GAMMA"]
        L_low = params["GAMMA_LOW_L"]
        L_high = params["GAMMA_HIGH_L"]

        t = (stage2_L - L_low) / (L_high - L_low + 1e-9)
        t = np.clip(t, 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        gamma_eff = 1.0 - (1.0 - base_gamma) * t

    L = np.clip(lab[0], 0, 100) / 100.0
    L_new = np.clip(100.0 * (L ** gamma_eff), 0, 100)

    if not saturated:
        L_new = min(L_new, stage2_L + params["MAX_STAGE3_BOOST_L"])

    lab[0] = L_new
    return lab

def apply_stage4_final_offset(lab, params):
    lab = np.array(lab, dtype=np.float64) + params["FINAL_STAGE_OFFSET"]
    lab[0] = np.clip(lab[0], 0, 100)
    lab[1] = np.clip(lab[1], -128, 127)
    lab[2] = np.clip(lab[2], -128, 127)
    return lab

# ============================================================
# ΔE2000 (CIEDE2000)
# ============================================================
def delta_e_2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0) -> float:
    L1, a1, b1 = [float(x) for x in np.array(lab1, dtype=np.float64).reshape(3,)]
    L2, a2, b2 = [float(x) for x in np.array(lab2, dtype=np.float64).reshape(3,)]

    C1 = (a1*a1 + b1*b1) ** 0.5
    C2 = (a2*a2 + b2*b2) ** 0.5
    Cbar = (C1 + C2) / 2.0

    Cbar7 = Cbar ** 7
    G = 0.5 * (1.0 - (Cbar7 / (Cbar7 + 25.0**7)) ** 0.5)

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = (a1p*a1p + b1*b1) ** 0.5
    C2p = (a2p*a2p + b2*b2) ** 0.5

    def hp(ap, b):
        if ap == 0 and b == 0:
            return 0.0
        h = np.degrees(np.arctan2(b, ap))
        return h + 360.0 if h < 0 else h

    h1p = hp(a1p, b1)
    h2p = hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    if C1p*C2p == 0:
        dHp = 0.0
    else:
        if dhp > 180:
            dhp -= 360
        elif dhp < -180:
            dhp += 360
        dHp = 2.0 * (C1p*C2p) ** 0.5 * np.sin(np.radians(dhp / 2.0))

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0

    if C1p*C2p == 0:
        hbarp = h1p + h2p
    else:
        hsum = h1p + h2p
        if abs(h1p - h2p) > 180:
            hbarp = (hsum + 360.0) / 2.0 if hsum < 360.0 else (hsum - 360.0) / 2.0
        else:
            hbarp = hsum / 2.0

    T = (
        1
        - 0.17 * np.cos(np.radians(hbarp - 30))
        + 0.24 * np.cos(np.radians(2 * hbarp))
        + 0.32 * np.cos(np.radians(3 * hbarp + 6))
        - 0.20 * np.cos(np.radians(4 * hbarp - 63))
    )

    dtheta = 30.0 * np.exp(-((hbarp - 275.0) / 25.0) ** 2)
    Rc = 2.0 * (Cbarp ** 7 / (Cbarp ** 7 + 25.0**7)) ** 0.5
    Sl = 1.0 + (0.015 * (Lbarp - 50.0) ** 2) / (20.0 + (Lbarp - 50.0) ** 2) ** 0.5
    Sc = 1.0 + 0.045 * Cbarp
    Sh = 1.0 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2.0 * dtheta)) * Rc

    dE = ((dLp / (kL * Sl)) ** 2 + (dCp / (kC * Sc)) ** 2 + (dHp / (kH * Sh)) ** 2 +
          Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))) ** 0.5
    return float(dE)

def find_best_shade(lab):
    best_name = "N/A"
    best_de = float("inf")
    for shade in SHADE_GUIDE:
        de = delta_e_2000(lab, shade["lab"])
        if de < best_de:
            best_de = de
            best_name = shade["name"]
    return best_name, best_de

# ============================================================
# REQUIRED BY app.py
# ============================================================
def analyze_image_with_rois(img_bgr: np.ndarray, payload: dict):
    mode_key = str(payload.get("mode", "fa")).strip().lower()
    cfg = MODE_CONFIG.get(mode_key, MODE_CONFIG["fa"])
    params = cfg["params"]
    safe_ab = bool(cfg["safe_ab"])

    light_obj = payload.get("light", {}) or {}
    dark_obj  = payload.get("dark",  {}) or {}
    teeth_arr = payload.get("teeth", []) or []

    measured_light_lab = trimmed_mean_lab_in_hex(
        img_bgr,
        light_obj.get("points"),
        low_trim_pct=light_obj.get("low_trim", 0),
        high_trim_pct=light_obj.get("high_trim", 0),
    )
    measured_dark_lab = trimmed_mean_lab_in_hex(
        img_bgr,
        dark_obj.get("points"),
        low_trim_pct=dark_obj.get("low_trim", 0),
        high_trim_pct=dark_obj.get("high_trim", 0),
    )

    if measured_light_lab is None or measured_dark_lab is None:
        return {"error": "Light/Dark ROI نامعتبر است یا پیکسل کافی ندارد."}

    M_meas = (measured_light_lab + measured_dark_lab) / 2.0
    M_real = (REAL_LIGHT_LAB + REAL_DARK_LAB) / 2.0
    delta_illum = M_real - M_meas

    img_wb_bgr = apply_global_delta_illum(img_bgr, delta_illum)

    measured_light_lab_wb = measured_light_lab + delta_illum
    measured_dark_lab_wb  = measured_dark_lab  + delta_illum

    measured_light_lab_wb[0] = np.clip(measured_light_lab_wb[0], 0, 100)
    measured_dark_lab_wb[0]  = np.clip(measured_dark_lab_wb[0], 0, 100)
    measured_light_lab_wb[1:] = np.clip(measured_light_lab_wb[1:], -128, 127)
    measured_dark_lab_wb[1:]  = np.clip(measured_dark_lab_wb[1:], -128, 127)

    teeth_results = []
    for i, t in enumerate(teeth_arr, start=1):
        raw_lab = trimmed_mean_lab_in_hex(
            img_wb_bgr,
            t.get("points"),
            low_trim_pct=t.get("low_trim", 0),
            high_trim_pct=t.get("high_trim", 0),
        )

        if raw_lab is None:
            teeth_results.append({
                "tooth": i,
                "lab_raw": None,
                "stage4_lab": None,
                "lab_calibrated": None,
                "stage1": None,
                "stage2": None,
                "stage3": None,
                "stage4": None,
                "shade": "N/A",
                "deltaE00": None,
                "error": "Tooth ROI نامعتبر است یا پیکسل کافی ندارد."
            })
            continue

        stage1 = apply_stage1_generic(raw_lab, measured_light_lab_wb, measured_dark_lab_wb, params, safe_ab)
        stage2 = apply_stage2_offset(stage1, params)
        stage3 = apply_stage3_L_gamma_conditional(stage2, params, stage1[0])
        stage4 = apply_stage4_final_offset(stage3, params)

        shade_name, de00 = find_best_shade(stage4)

        teeth_results.append({
            "tooth": i,
            "lab_raw": [round(float(x), 2) for x in raw_lab.tolist()],
            "stage4_lab": [round(float(x), 2) for x in stage4.tolist()],
            "lab_calibrated": [round(float(x), 2) for x in stage4.tolist()],
            "stage1": [round(float(x), 2) for x in stage1.tolist()],
            "stage2": [round(float(x), 2) for x in stage2.tolist()],
            "stage3": [round(float(x), 2) for x in stage3.tolist()],
            "stage4": [round(float(x), 2) for x in stage4.tolist()],
            "shade": shade_name,
            "deltaE00": round(float(de00), 2),
        })

    return {
        "mode": cfg["label"],
        "light_label": "0m1",
        "dark_label": "5m1",

        "delta_illum": [round(float(x), 2) for x in delta_illum.tolist()],

        "light_lab_raw": [round(float(x), 2) for x in measured_light_lab.tolist()],
        "dark_lab_raw":  [round(float(x), 2) for x in measured_dark_lab.tolist()],

        "light_lab_wb": [round(float(x), 2) for x in measured_light_lab_wb.tolist()],
        "dark_lab_wb":  [round(float(x), 2) for x in measured_dark_lab_wb.tolist()],

        "teeth": teeth_results,
    }

# ============================================================
# PATCH: Analyze using ROI patches only (no full image upload)
# Each ROI has its own small img_bgr, and points are in that patch coords.
# ============================================================
def analyze_patches_with_rois(payload: dict):
    mode_key = str(payload.get("mode", "fa")).strip().lower()
    cfg = MODE_CONFIG.get(mode_key, MODE_CONFIG["fa"])
    params = cfg["params"]
    safe_ab = bool(cfg["safe_ab"])

    light_obj = payload.get("light", {}) or {}
    dark_obj  = payload.get("dark",  {}) or {}
    teeth_arr = payload.get("teeth", []) or []

    light_img = light_obj.get("img_bgr", None)
    dark_img  = dark_obj.get("img_bgr", None)

    measured_light_lab = trimmed_mean_lab_in_hex(
        light_img,
        light_obj.get("points"),
        low_trim_pct=light_obj.get("low_trim", 0),
        high_trim_pct=light_obj.get("high_trim", 0),
    )
    measured_dark_lab = trimmed_mean_lab_in_hex(
        dark_img,
        dark_obj.get("points"),
        low_trim_pct=dark_obj.get("low_trim", 0),
        high_trim_pct=dark_obj.get("high_trim", 0),
    )

    if measured_light_lab is None or measured_dark_lab is None:
        return {"error": "Light/Dark ROI نامعتبر است یا پیکسل کافی ندارد."}

    M_meas = (measured_light_lab + measured_dark_lab) / 2.0
    M_real = (REAL_LIGHT_LAB + REAL_DARK_LAB) / 2.0
    delta_illum = M_real - M_meas

    measured_light_lab_wb = measured_light_lab + delta_illum
    measured_dark_lab_wb  = measured_dark_lab  + delta_illum

    measured_light_lab_wb[0] = np.clip(measured_light_lab_wb[0], 0, 100)
    measured_dark_lab_wb[0]  = np.clip(measured_dark_lab_wb[0], 0, 100)
    measured_light_lab_wb[1:] = np.clip(measured_light_lab_wb[1:], -128, 127)
    measured_dark_lab_wb[1:]  = np.clip(measured_dark_lab_wb[1:], -128, 127)

    teeth_results = []
    for i, t in enumerate(teeth_arr, start=1):
        tooth_img = t.get("img_bgr", None)

        # Apply WB to the tooth patch (same delta_illum)
        tooth_wb_bgr = apply_global_delta_illum(tooth_img, delta_illum)

        raw_lab = trimmed_mean_lab_in_hex(
            tooth_wb_bgr,
            t.get("points"),
            low_trim_pct=t.get("low_trim", 0),
            high_trim_pct=t.get("high_trim", 0),
        )

        if raw_lab is None:
            teeth_results.append({
                "tooth": i,
                "lab_raw": None,
                "stage4_lab": None,
                "lab_calibrated": None,
                "stage1": None,
                "stage2": None,
                "stage3": None,
                "stage4": None,
                "shade": "N/A",
                "deltaE00": None,
                "error": "Tooth ROI نامعتبر است یا پیکسل کافی ندارد."
            })
            continue

        stage1 = apply_stage1_generic(raw_lab, measured_light_lab_wb, measured_dark_lab_wb, params, safe_ab)
        stage2 = apply_stage2_offset(stage1, params)
        stage3 = apply_stage3_L_gamma_conditional(stage2, params, stage1[0])
        stage4 = apply_stage4_final_offset(stage3, params)

        shade_name, de00 = find_best_shade(stage4)

        teeth_results.append({
            "tooth": i,
            "lab_raw": [round(float(x), 2) for x in raw_lab.tolist()],
            "stage4_lab": [round(float(x), 2) for x in stage4.tolist()],
            "lab_calibrated": [round(float(x), 2) for x in stage4.tolist()],
            "stage1": [round(float(x), 2) for x in stage1.tolist()],
            "stage2": [round(float(x), 2) for x in stage2.tolist()],
            "stage3": [round(float(x), 2) for x in stage3.tolist()],
            "stage4": [round(float(x), 2) for x in stage4.tolist()],
            "shade": shade_name,
            "deltaE00": round(float(de00), 2),
        })

    return {
        "mode": cfg["label"],
        "light_label": "0m1",
        "dark_label": "5m1",
        "delta_illum": [round(float(x), 2) for x in delta_illum.tolist()],
        "light_lab_raw": [round(float(x), 2) for x in measured_light_lab.tolist()],
        "dark_lab_raw":  [round(float(x), 2) for x in measured_dark_lab.tolist()],
        "light_lab_wb": [round(float(x), 2) for x in measured_light_lab_wb.tolist()],
        "dark_lab_wb":  [round(float(x), 2) for x in measured_dark_lab_wb.tolist()],
        "teeth": teeth_results,
    }

