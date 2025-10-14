#CGT
"""
Full-flow wafer dataset generator targeted at wafer-sort for discrete power MOSFETs,
with industry-standard local defect density defined in defects/cm^2 (default 1.0 cm^2).

- RNG: numpy.default_rng (SEED for reproducibility)
- Multi-lot / multi-wafer: configure NUM_LOTS and WAFERS_PER_LOT (max 25)
- Simulates full process sequence (POLISH -> ... -> PASSIVATION) cumulatively,
  then produces wafer-sort electrical test data for each die.
- Dies touching the excluded edge are removed using corner-based rule.
- Local defect density uses a survey area (SURVEY_AREA_CM2) -> neighbor radius derived.
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from datetime import datetime, timedelta, time
import uuid
from math import atan2, degrees, pi

# ----------------------
# User-configurable settings
# ----------------------
SEED = 42
rng = np.random.default_rng(SEED)

NUM_LOTS = 1                # number of lots to generate
WAFERS_PER_LOT = 1          # wafers per lot (must be <= MAX_WAFERS_PER_LOT)
MAX_WAFERS_PER_LOT = 25     # enforced maximum per your request

OUT_CSV = "synth_power_mosfet.csv"
PRESET = "power_mainstream"  # or "power_low_leak"

# Industry-standard survey area for local defect density (defects per cm^2).
# Common choice: 1.0 cm^2. Change if you prefer a different survey window.
SURVEY_AREA_CM2 = 1.0

# sanity clamp
if WAFERS_PER_LOT > MAX_WAFERS_PER_LOT:
    print(f"WAFERS_PER_LOT ({WAFERS_PER_LOT}) > MAX ({MAX_WAFERS_PER_LOT}); clamping.")
    WAFERS_PER_LOT = MAX_WAFERS_PER_LOT

TOTAL_WAFERS = int(NUM_LOTS * WAFERS_PER_LOT)

# ----------------------
# Wafer / die geometry (mm)
# ----------------------
DIE_W_MM = 10.0
DIE_H_MM = 10.0
SCRIBE_MM = 1.0
WAFER_DIA_MM = 300.0
WAFER_EDGE_EXCLUSION_MM = 2.0
USABLE_RADIUS_MM = WAFER_DIA_MM / 2.0 - WAFER_EDGE_EXCLUSION_MM

half_w = DIE_W_MM / 2.0
half_h = DIE_H_MM / 2.0

# Create die center grid (centered)
pitch_x = DIE_W_MM + SCRIBE_MM
pitch_y = DIE_H_MM + SCRIBE_MM
n_cols = int(np.floor(WAFER_DIA_MM / pitch_x))
n_rows = int(np.floor(WAFER_DIA_MM / pitch_y))
xs_grid = (np.arange(n_cols) - (n_cols - 1) / 2) * pitch_x
ys_grid = (np.arange(n_rows) - (n_rows - 1) / 2) * pitch_y
xx_grid, yy_grid = np.meshgrid(xs_grid, ys_grid)
xx_flat_all = xx_grid.ravel()
yy_flat_all = yy_grid.ravel()

# Corner-based inclusion: every die rectangle corner must be inside usable circle.
corner_dist = np.hypot(np.abs(xx_flat_all) + half_w, np.abs(yy_flat_all) + half_h)
mask_corners = corner_dist <= USABLE_RADIUS_MM

usable_centers_x = xx_flat_all[mask_corners]
usable_centers_y = yy_flat_all[mask_corners]
radial_all = np.hypot(usable_centers_x, usable_centers_y)
BASE_NUM_DIES = int(usable_centers_x.size)

# ----------------------
# Process sequence (realistic)
# ----------------------
PROCESS_SEQUENCE = ["POLISH", "LITH", "IMPL", "DIFF", "ETCH", "CMP", "METAL", "PASSIVATION"]

# Pools and nominals
chemical_lots = np.array([f"CL{n}" for n in range(200, 240)])
tester_ids = np.array([f"T-{c}{i}" for c in ["A","B","C"] for i in range(1,6)])
probe_cards = np.array([f"PC-{1000 + i}" for i in range(12)])
cleanroom_classes = np.array(["ISO5","ISO6"])
tool_pool = {
    "ETCH": np.array(["E1","E2","E3"]),
    "CMP": np.array(["C1","C2"]),
    "IMPL": np.array(["I1","I2"]),
    "LITH": np.array(["L1","L2"]),
    "DIFF": np.array(["D1","D2"]),
    "METAL": np.array(["M1","M2"]),
    "PASSIVATION": np.array(["P1","P2"]),
    "POLISH": np.array(["PZ1","PZ2"])
}

# CMP slurry subset + bad slurry selection
slurry_lots = rng.choice(chemical_lots, size=6, replace=False)
bad_slurry_lots = rng.choice(slurry_lots, size=2, replace=False)

# Bad chemical lots (systematic bias candidates)
bad_chem_lots = rng.choice(chemical_lots, size=2, replace=False)

# Chamber nominal conditions
process_chamber_temps = {
    "POLISH": 25.0, "LITH": 23.0, "IMPL": 25.0, "DIFF": 1000.0,
    "ETCH": 65.0, "CMP": 30.0, "METAL": 200.0, "PASSIVATION": 120.0
}
process_chamber_pressures = {
    "POLISH": 760.0, "LITH": 760.0, "IMPL": 1e-5, "DIFF": 760.0,
    "ETCH": 0.05, "CMP": 760.0, "METAL": 1e-3, "PASSIVATION": 0.1
}

# Test limits: tweak these to represent product acceptance criteria
if PRESET == "power_low_leak":
    limits = {
        "Vth_V": (1.8, 3.5), "RON_mOhm": (None, 80.0),
        "Idss_A": (None, 1e-7), "Ioff_nA": (None, 200.0), "Cgd_pF": (None, 80.0)
    }
else:
    limits = {
        "Vth_V": (1.5, 4.0), "RON_mOhm": (None, 200.0),
        "Idss_A": (None, 1e-6), "Ioff_nA": (None, 2000.0), "Cgd_pF": (None, 200.0)
    }

# per-tool biases (small wafer-to-wafer drift)
tool_biases = {}
for tlist in tool_pool.values():
    for tid in tlist:
        tool_biases[tid] = {
            "vth_shift_v": float(rng.normal(0, 0.02)),
            "ron_scale": float(1.0 + rng.normal(0, 0.01)),
            "short_bump": float(max(0.0, rng.normal(0, 0.003)))
        }

# helper: robust numpy choice
def choice_one(arr):
    return arr[int(rng.integers(0, len(arr)))] if (arr is not None and len(arr) > 0) else None

# ----------------------
# Per-step cumulative effect functions (conservative, wafer-sort oriented)
# ----------------------
def apply_polish_effects(state):
    state["Cgd_pF"] += rng.normal(-0.5, 0.8, state["N"])
    state["physical_prob"] *= 0.995
    return state

def apply_lith_effects(state):
    hum = float(state["Ambient Humidity (%)"].mean()) if "Ambient Humidity (%)" in state else 40.0
    lith_contact_prob_bump = max(0.0, (hum - 40.0) * 0.006)
    if state["Cleanroom Class"] == "ISO6":
        lith_contact_prob_bump += 0.01
    state["lith_contact_prob_bump"] += lith_contact_prob_bump
    state["Threshold Voltage (V)"] += rng.normal(0.0, 0.008, state["N"])
    state["On-Resistance (mΩ)"] *= (1.0 + rng.normal(0.0, 0.008, state["N"]))
    return state

def apply_implant_effects(state):
    nominal_dose = 1e13
    wafer_impl_dose = nominal_dose * (1.0 + float(rng.normal(0.0, 0.05)))
    dose_rel = (wafer_impl_dose - nominal_dose) / nominal_dose
    vth_shift_from_impl = 0.75 * dose_rel
    state["Threshold Voltage (V)"] += vth_shift_from_impl + rng.normal(0.0, 0.008, state["N"])
    if abs(dose_rel) > 0.12:
        factor = 1.0 + abs(dose_rel) * rng.uniform(1.5, 4.0)
        state["Idss_A"] *= factor
        state["Ioff_nA"] *= factor
    state["physical_prob"] += abs(dose_rel) * 0.005
    return state

def apply_diffusion_effects(state):
    state["Threshold Voltage (V)"] += rng.normal(-0.008, 0.006, state["N"])
    state["Idss_A"] *= (1.0 - np.clip(rng.normal(0.01, 0.005, state["N"]), -0.02, 0.02))
    return state

def apply_etch_effects(state):
    wafer_mean_pressure = float(state["Chamber Pressure (Torr)"].mean())
    nominal = process_chamber_pressures["ETCH"]
    pressure_rel = (wafer_mean_pressure - nominal) / (nominal if nominal>0 else 1.0)
    state["Cgd_pF"] += rng.normal(0.0, max(0.8, abs(pressure_rel) * 4.0), state["N"])
    state["physical_prob"] += min(0.03, abs(pressure_rel) * 0.06)
    state["On-Resistance (mΩ)"] *= (1.0 + abs(pressure_rel) * rng.uniform(0.003, 0.01))
    return state

def apply_cmp_effects(state):
    slurry = state.get("Slurry Lot ID", None)
    if slurry is None:
        slurry = choice_one(slurry_lots)
        state["Slurry Lot ID"] = slurry
    if slurry in bad_slurry_lots:
        state["Cgd_pF"] += rng.normal(2.0, 1.5, state["N"])
        state["physical_prob"] += 0.01 + abs(rng.normal(0.0, 0.005))
    else:
        state["physical_prob"] += abs(rng.normal(0.0, 0.001))
    return state

def apply_metal_effects(state):
    wafer_mean_temp = float(state["Chamber Temperature (°C)"].mean())
    state["On-Resistance (mΩ)"] *= (1.0 + 0.0015 * (wafer_mean_temp - process_chamber_temps["METAL"]))
    tid = state.get("Tool ID", "")
    short_bump = tool_biases.get(tid, {}).get("short_bump", 0.0)
    state["metal_short_bump"] += short_bump
    return state

def apply_passivation_effects(state):
    pass_humidity = float(state["Ambient Humidity (%)"].mean())
    pass_temp = float(state["Chamber Temperature (°C)"].mean())
    pass_pinhole_bump = max(0.0, (pass_humidity - 45.0) * 0.004)
    if pass_temp < 100:
        pass_pinhole_bump += 0.005
    state["physical_prob"] += pass_pinhole_bump
    return state

STEP_APPLY = {
    "POLISH": apply_polish_effects,
    "LITH": apply_lith_effects,
    "IMPL": apply_implant_effects,
    "DIFF": apply_diffusion_effects,
    "ETCH": apply_etch_effects,
    "CMP": apply_cmp_effects,
    "METAL": apply_metal_effects,
    "PASSIVATION": apply_passivation_effects
}

# ----------------------
# Helper: recompute param-derived columns after any wafer-wide update
# Also computes neighbor counts and local defect density using SURVEY_AREA_CM2.
# ----------------------
def recompute_parametrics_and_spatial(wafer_df):
    # param pass/fail
    def param_pass_row(vth, ron, idss_row, ioff_row, cgd_row):
        vth_ok = True
        vth_limits = limits.get("Vth_V", (None, None))
        if vth_limits[0] is not None and vth < vth_limits[0]: vth_ok = False
        if vth_limits[1] is not None and vth > vth_limits[1]: vth_ok = False
        ron_ok = True
        ron_limits = limits.get("RON_mOhm", (None, None))
        if ron_limits[1] is not None and ron > ron_limits[1]: ron_ok = False
        idss_ok = True
        idss_limits = limits.get("Idss_A", (None, None))
        if idss_limits[1] is not None and idss_row > idss_limits[1]: idss_ok = False
        ioff_ok = True
        ioff_limits = limits.get("Ioff_nA", (None, None))
        if ioff_limits[1] is not None and ioff_row > ioff_limits[1]: ioff_ok = False
        cgd_ok = True
        cgd_limits = limits.get("Cgd_pF", (None, None))
        if cgd_limits[1] is not None and cgd_row > cgd_limits[1]: cgd_ok = False
        return vth_ok and ron_ok and idss_ok and ioff_ok and cgd_ok

    wafer_df["Param Pass/Fail"] = wafer_df.apply(
        lambda r: "Pass" if param_pass_row(r["Threshold Voltage (V)"], r["On-Resistance (mΩ)"],
                                          r["Idss_A"], r["Ioff_nA"], r["Cgd_pF"]) else "Fail", axis=1)

    wafer_df["Test Result"] = wafer_df.apply(lambda r: "Fail" if (r["PhysicalDefect"] is not None and pd.notna(r["PhysicalDefect"])) else r["Param Pass/Fail"], axis=1)

    def defect_type_from_row(r):
        if r["PhysicalDefect"] is not None and pd.notna(r["PhysicalDefect"]):
            return "Physical"
        if r["Test Result"] == "Fail" and (r["PhysicalDefect"] is None or pd.isna(r["PhysicalDefect"])):
            return "Parametric"
        return "None"
    wafer_df["Defect Type"] = wafer_df.apply(lambda r: defect_type_from_row(r), axis=1)

    # FailureCause_All, First, Severity
    def violations_list_row(r):
        parts = []
        if pd.notna(r["PhysicalDefect"]):
            parts.append(f"Physical:{r['PhysicalDefect']}")
        if pd.notna(r["Idss_A"]) and limits.get("Idss_A",(None,))[1] is not None and r["Idss_A"] > limits["Idss_A"][1]:
            parts.append(f"Idss>{limits['Idss_A'][1]:.1e}A")
        if pd.notna(r["Ioff_nA"]) and limits.get("Ioff_nA",(None,))[1] is not None and r["Ioff_nA"] > limits["Ioff_nA"][1]:
            parts.append(f"Ioff>{limits['Ioff_nA'][1]}nA")
        vth_limits = limits.get("Vth_V", (None, None))
        if pd.notna(r["Threshold Voltage (V)"]) and ((vth_limits[0] is not None and r["Threshold Voltage (V)"] < vth_limits[0]) or (vth_limits[1] is not None and r["Threshold Voltage (V)"] > vth_limits[1])):
            parts.append("Vth OOS")
        if pd.notna(r["On-Resistance (mΩ)"]) and limits.get("RON_mOhm",(None,))[1] is not None and r["On-Resistance (mΩ)"] > limits["RON_mOhm"][1]:
            parts.append(f"RON>{limits['RON_mOhm'][1]}")
        if pd.notna(r["Cgd_pF"]) and limits.get("Cgd_pF",(None,))[1] is not None and r["Cgd_pF"] > limits["Cgd_pF"][1]:
            parts.append(f"Cgd>{limits['Cgd_pF'][1]}pF")
        return "; ".join(parts) if parts else None

    wafer_df["FailureCause_All"] = wafer_df.apply(lambda r: violations_list_row(r), axis=1)

    def first_fail_attr_row(r):
        if r["Defect Type"] == "Physical" and pd.notna(r["PhysicalDefect"]):
            return r["PhysicalDefect"]
        if pd.notna(r["Idss_A"]) and limits.get("Idss_A",(None,))[1] is not None and r["Idss_A"] > limits["Idss_A"][1]:
            return "Idss"
        if pd.notna(r["Ioff_nA"]) and limits.get("Ioff_nA",(None,))[1] is not None and r["Ioff_nA"] > limits["Ioff_nA"][1]:
            return "Ioff"
        vth_limits = limits.get("Vth_V", (None, None))
        if pd.notna(r["Threshold Voltage (V)"]) and ((vth_limits[0] is not None and r["Threshold Voltage (V)"] < vth_limits[0]) or (vth_limits[1] is not None and r["Threshold Voltage (V)"] > vth_limits[1])):
            return "Vth"
        if pd.notna(r["On-Resistance (mΩ)"]) and limits.get("RON_mOhm",(None,))[1] is not None and r["On-Resistance (mΩ)"] > limits["RON_mOhm"][1]:
            return "RON"
        if pd.notna(r["Cgd_pF"]) and limits.get("Cgd_pF",(None,))[1] is not None and r["Cgd_pF"] > limits["Cgd_pF"][1]:
            return "Cgd"
        return None

    wafer_df["FailureCause_First"] = wafer_df.apply(lambda r: first_fail_attr_row(r), axis=1)

    def severity_row(r):
        s = 0.0
        if pd.notna(r["Idss_A"]) and limits.get("Idss_A",(None,))[1] is not None and r["Idss_A"] > limits["Idss_A"][1]:
            denom = abs(limits["Idss_A"][1]) if abs(limits["Idss_A"][1])>1e-12 else 1.0
            s += (r["Idss_A"] - limits["Idss_A"][1]) / denom
        if pd.notna(r["Ioff_nA"]) and limits.get("Ioff_nA",(None,))[1] is not None and r["Ioff_nA"] > limits["Ioff_nA"][1]:
            denom = (limits["Ioff_nA"][1] if limits["Ioff_nA"][1] else 1.0)
            s += (r["Ioff_nA"] - limits["Ioff_nA"][1]) / denom
        vth_limits = limits.get("Vth_V", (None, None))
        if pd.notna(r["Threshold Voltage (V)"]):
            if vth_limits[0] is not None and r["Threshold Voltage (V)"] < vth_limits[0]:
                s += (vth_limits[0] - r["Threshold Voltage (V)"]) / abs(vth_limits[0])
            if vth_limits[1] is not None and r["Threshold Voltage (V)"] > vth_limits[1]:
                s += (r["Threshold Voltage (V)"] - vth_limits[1]) / abs(vth_limits[1])
        if pd.notna(r["On-Resistance (mΩ)"]) and limits.get("RON_mOhm",(None,))[1] is not None and r["On-Resistance (mΩ)"] > limits["RON_mOhm"][1]:
            denom = (limits["RON_mOhm"][1] if limits["RON_mOhm"][1] else 1.0)
            s += (r["On-Resistance (mΩ)"] - limits["RON_mOhm"][1]) / denom
        if pd.notna(r["Cgd_pF"]) and limits.get("Cgd_pF",(None,))[1] is not None and r["Cgd_pF"] > limits["Cgd_pF"][1]:
            denom = (limits["Cgd_pF"][1] if limits["Cgd_pF"][1] else 1.0)
            s += (r["Cgd_pF"] - limits["Cgd_pF"][1]) / denom
        return s

    wafer_df["FailureSeverity"] = wafer_df.apply(lambda r: severity_row(r), axis=1)

    # Spatial neighbor analysis (KDTree): count neighbor fails excluding self
    coords_mm = wafer_df[["Die Center X (mm)", "Die Center Y (mm)"]].values
    tree = KDTree(coords_mm)

    # derive neighbor radius from SURVEY_AREA_CM2 (area in cm^2 -> radius in mm)
    # radius_cm = sqrt(area/pi)  => radius_mm = radius_cm * 10
    neighbor_radius_mm = float(np.sqrt(SURVEY_AREA_CM2 / pi) * 10.0)

    nbrs = tree.query_ball_point(coords_mm, r=neighbor_radius_mm)
    neighbor_fail_count = []
    neighbor_total_count = []
    for idx, nlist in enumerate(nbrs):
        nb = [j for j in nlist if j != idx]
        neighbor_total_count.append(len(nb))
        count_fail = sum(1 for j in nb if wafer_df.iloc[j]["Test Result"] == "Fail")
        neighbor_fail_count.append(int(count_fail))
    wafer_df["Neighbor Fail Count"] = neighbor_fail_count
    wafer_df["Neighbor Total Count"] = neighbor_total_count

    # Local defect density in defects per cm^2 using SURVEY_AREA_CM2
    wafer_df["Local Defect Density (defects/cm^2)"] = wafer_df["Neighbor Fail Count"] / SURVEY_AREA_CM2

    return wafer_df

# ----------------------
# Wafer generation function (full flow up to PASSIVATION)
# Accepts `lot_bad_chem` to apply lot-level chemical bias consistently.
# ----------------------
def generate_wafer(lot_idx, wafer_idx_in_lot, base_date, lot_bad_chem=None):
    N = BASE_NUM_DIES
    die_ids = [f"L{lot_idx:02d}-W{wafer_idx_in_lot:02d}-D{str(i+1).zfill(4)}" for i in range(N)]
    cx = usable_centers_x.copy()
    cy = usable_centers_y.copy()
    radial = np.hypot(cx, cy)

    wafer_id = uuid.uuid4().hex[:8]
    lot_number = f"L{int(lot_idx):03d}"
    fab_id = f"F{int(rng.integers(1,10))}"
    process_code = choice_one(np.array(["PowerLP","PowerHP"]))
    wafer_polish_date = (base_date + timedelta(days=int(rng.integers(0,30)))).date()
    wafer_edge_excl = WAFER_EDGE_EXCLUSION_MM

    cleanroom = choice_one(cleanroom_classes)
    chem = choice_one(chemical_lots)
    tester = choice_one(tester_ids)
    probe = choice_one(probe_cards)

    # If this lot is flagged as bad chemical lot, override chem
    if lot_bad_chem is not None:
        chem = lot_bad_chem

    # Always simulate through PASSIVATION so dataset reflects wafer-sort stage
    current_step = "PASSIVATION"
    tool = choice_one(tool_pool.get(current_step, np.array(["GEN"])))

    # base correlated params (Vth vs R_on)
    mean_vth = 3.0
    mean_ron = 50.0
    sigma_vth = 0.20
    sigma_ron = 18.0
    rho_vth_ron = 0.35
    cov_vr = rho_vth_ron * sigma_vth * sigma_ron
    cov_vr_matrix = [[sigma_vth**2, cov_vr], [cov_vr, sigma_ron**2]]
    vth_ron = rng.multivariate_normal([mean_vth, mean_ron], cov_vr_matrix, N)
    vth = np.clip(vth_ron[:,0], 0.5, None)
    ron = np.clip(vth_ron[:,1], 0.5, None)

    # Idss / Ioff log-normal (conservative sigma, adjusted for mature process)
    if PRESET == "power_low_leak":
        log_mean_idss = np.log(1e-7)
        log_mean_ioff = np.log(2e-7)
    else:
        log_mean_idss = np.log(2e-7)
        log_mean_ioff = np.log(5e-7)

    sigma_log_idss = 0.6
    sigma_log_ioff = 0.8
    rho_log = 0.60
    cov_log = rho_log * sigma_log_idss * sigma_log_ioff
    cov_log_matrix = [[sigma_log_idss**2, cov_log], [cov_log, sigma_log_ioff**2]]
    log_idss, log_ioff = rng.multivariate_normal([log_mean_idss, log_mean_ioff], cov_log_matrix, N).T
    idss = np.exp(log_idss)         # A
    ioff = np.exp(log_ioff) * 1e9   # convert to nA for consistency with limits

    base_cgd_mean = 60.0
    cgd = rng.normal(base_cgd_mean, 6.0, N)

    # initialize state dictionary
    state = {
        "N": N,
        "Threshold Voltage (V)": vth.copy(),
        "On-Resistance (mΩ)": ron.copy(),
        "Idss_A": idss.copy(),
        "Ioff_nA": ioff.copy(),
        "Cgd_pF": cgd.copy(),
        "Die Center X (mm)": cx.copy(),
        "Die Center Y (mm)": cy.copy(),
        "Defect Radius (mm)": radial.copy(),
        "physical_prob": np.full(N, 0.0002),   # lower baseline for mature process
        "lith_contact_prob_bump": 0.0,
        "metal_short_bump": 0.0,
        "Wafer ID": wafer_id,
        "Lot Number": lot_number,
        "Fab ID": fab_id,
        "Process Code": process_code,
        "Wafer Polish Date": wafer_polish_date,
        "Wafer Edge Exclusion (mm)": wafer_edge_excl,
        "ProcessHistory": [],
        "Process Step ID": current_step,
        "Chemical Lot ID": chem,
        "Tester ID": tester,
        "Probe Card ID": probe,
        "Cleanroom Class": cleanroom,
        "Tool ID": tool,
        "Test Program ID": f"TPR-{int(rng.integers(100,999))}"
    }

    # create a realistic process timeline (one wafer per day-ish series)
    step_time = datetime.combine(base_date.date(), time(6,0,0)) + timedelta(days=int(wafer_idx_in_lot))

    # simulate each step in sequence and apply cumulative effects
    for step in PROCESS_SEQUENCE:
        step_tool = choice_one(tool_pool.get(step, np.array(["GEN"])))
        step_chem = choice_one(chemical_lots) if step in ("CMP", "POLISH") else state["Chemical Lot ID"]
        ch_temp = float(rng.normal(process_chamber_temps.get(step, 25.0), 1.5))
        ch_press = float(rng.normal(process_chamber_pressures.get(step, 760.0), max(1e-6, process_chamber_pressures.get(step,760.0)*0.02)))
        ambient_temp = float(rng.normal(22.0 + rng.normal(0,0.2), 0.3))
        ambient_humidity = float(rng.normal(40.0 + rng.normal(0,1.0), 0.8))

        state["ProcessHistory"].append(f"{step}@{step_time.isoformat()}|Tool={step_tool}|Chem={step_chem}")

        state["Chamber Temperature (°C)"] = np.full(N, ch_temp)
        state["Chamber Pressure (Torr)"] = np.full(N, ch_press)
        state["Ambient Temperature (°C)"] = np.full(N, ambient_temp)
        state["Ambient Humidity (%)"] = np.full(N, ambient_humidity)

        bias = tool_biases.get(step_tool, {"vth_shift_v":0.0, "ron_scale":1.0, "short_bump":0.0})
        state["Threshold Voltage (V)"] += bias["vth_shift_v"]
        state["On-Resistance (mΩ)"] *= bias["ron_scale"]
        state["metal_short_bump"] += bias["short_bump"]

        if step == "CMP":
            state["Slurry Lot ID"] = step_chem

        func = STEP_APPLY.get(step, None)
        if func:
            state = func(state)

        step_time = step_time + timedelta(hours=int(rng.integers(4, 36)))

    # spatial clusters (particle clusters, local yield excursions)
    coords = np.column_stack((state["Die Center X (mm)"], state["Die Center Y (mm)"]))
    n_clusters = int(rng.integers(0, 3))
    for _ in range(n_clusters):
        r = rng.uniform(0, USABLE_RADIUS_MM * 0.6)
        theta = rng.uniform(0, 2*np.pi)
        cx_c = r * np.cos(theta)
        cy_c = r * np.sin(theta)
        sigma = rng.uniform(6.0, 15.0)
        amp = rng.uniform(0.002, 0.02)
        d2 = ((coords[:,0] - cx_c)**2 + (coords[:,1] - cy_c)**2)
        state["physical_prob"] += amp * np.exp(-d2 / (2* sigma * sigma))

    # edge clusters
    n_edge_clusters = int(rng.integers(0, 2))
    for _ in range(n_edge_clusters):
        r = rng.uniform(USABLE_RADIUS_MM * 0.82, USABLE_RADIUS_MM * 0.96)
        theta = rng.uniform(0, 2*np.pi)
        cx_c = r * np.cos(theta)
        cy_c = r * np.sin(theta)
        sigma = rng.uniform(3.0, 7.0)
        amp = rng.uniform(0.008, 0.05)
        d2 = ((coords[:,0] - cx_c)**2 + (coords[:,1] - cy_c)**2)
        state["physical_prob"] += amp * np.exp(-d2 / (2* sigma * sigma))

    # edge region bump
    edge_region_idx = state["Defect Radius (mm)"] > (USABLE_RADIUS_MM * 0.85)
    state["physical_prob"][edge_region_idx] += 0.005

    # include lith / metal / probe bumps
    state["physical_prob"] += state.get("lith_contact_prob_bump", 0.0)
    state["physical_prob"] += state.get("metal_short_bump", 0.0)
    # ensure probabilities are bounded
    state["physical_prob"] = np.minimum(state["physical_prob"], 0.5)

    # sample physical defects
    is_physical = rng.random(N) < state["physical_prob"]
    defect_types = np.array([None]*N, dtype=object)

    for i in np.where(is_physical)[0]:
        radial_pos = state["Defect Radius (mm)"][i]
        if radial_pos > USABLE_RADIUS_MM * 0.85:
            defect_type = rng.choice(["EdgeCrack", "EdgeDelamination", "EdgeContamination"], p=[0.55, 0.30, 0.15])
        else:
            base_p = {"GateOxidePinhole": 0.12, "SourceContactDefect": 0.18, "DrainContactDefect": 0.18,
                      "ParticleContamination": 0.42, "MetalShort": 0.10}
            if state.get("Slurry Lot ID", None) in bad_slurry_lots:
                base_p["ParticleContamination"] += 0.12
                base_p["MetalShort"] += 0.02
            if state.get("lith_contact_prob_bump", 0.0) > 0:
                base_p["SourceContactDefect"] += 0.08 * state["lith_contact_prob_bump"]
                base_p["DrainContactDefect"] += 0.08 * state["lith_contact_prob_bump"]
            names = list(base_p.keys())
            vals = np.array(list(base_p.values()), dtype=float)
            vals = np.clip(vals, 0.0, None)
            probs = vals / vals.sum() if vals.sum() > 0 else np.ones(len(vals))/len(vals)
            defect_type = rng.choice(names, p=probs)

        defect_types[i] = defect_type

        # apply conservative electrical effects for wafer-sort
        if defect_type == "GateOxidePinhole":
            state["Idss_A"][i] *= rng.uniform(6, 30)
            state["Ioff_nA"][i] *= rng.uniform(6, 30)
            state["Threshold Voltage (V)"][i] *= rng.uniform(0.7, 0.95)
        elif defect_type in ["EdgeCrack", "EdgeDelamination"]:
            state["Idss_A"][i] *= rng.uniform(3, 15)
            state["Ioff_nA"][i] *= rng.uniform(3, 15)
            state["On-Resistance (mΩ)"][i] *= rng.uniform(1.1, 1.8)
        elif defect_type in ["SourceContactDefect", "DrainContactDefect"]:
            state["On-Resistance (mΩ)"][i] *= rng.uniform(1.4, 2.8)
            state["Idss_A"][i] *= rng.uniform(1.3, 4.0)
        elif defect_type == "ParticleContamination":
            state["Idss_A"][i] *= rng.uniform(2.5, 10)
            state["Ioff_nA"][i] *= rng.uniform(2.5, 10)
            state["On-Resistance (mΩ)"][i] *= rng.uniform(1.02, 1.5)
            state["Cgd_pF"][i] *= rng.uniform(0.9, 1.2)
        elif defect_type == "MetalShort":
            if rng.random() < 0.6:
                state["Threshold Voltage (V)"][i] *= rng.uniform(0.6, 0.9)
                state["On-Resistance (mΩ)"][i] *= rng.uniform(0.5, 0.85)
            else:
                state["Idss_A"][i] *= 500
                state["Ioff_nA"][i] *= 500
        elif defect_type == "EdgeContamination":
            state["Idss_A"][i] *= rng.uniform(4, 12)
            state["Ioff_nA"][i] *= rng.uniform(4, 12)

    # temp dependence (TCR for metal, Vth temp coefficient)
    TCR = 0.0039
    VTH_TEMP_COEFF = -0.002
    state["On-Resistance (mΩ)"] = state["On-Resistance (mΩ)"] * (1.0 + TCR * (state["Chamber Temperature (°C)"] - 25.0))
    state["Threshold Voltage (V)"] = state["Threshold Voltage (V)"] + VTH_TEMP_COEFF * (state["Chamber Temperature (°C)"] - 25.0)

    # clip physically reasonable bounds
    state["Idss_A"] = np.clip(state["Idss_A"], 1e-12, 1e-3)
    state["Ioff_nA"] = np.clip(state["Ioff_nA"], 0.0, 1e7)
    state["Threshold Voltage (V)"] = np.clip(state["Threshold Voltage (V)"], 0.05, 10.0)
    state["On-Resistance (mΩ)"] = np.clip(state["On-Resistance (mΩ)"], 0.1, 1000.0)
    state["Cgd_pF"] = np.clip(state["Cgd_pF"], 0.01, 500.0)

    # build wafer-level dataframe
    wafer_df = pd.DataFrame({
        "Wafer ID": [state["Wafer ID"]] * N,
        "Lot Number": [state["Lot Number"]] * N,
        "Fab ID": [state["Fab ID"]] * N,
        "Process Code": [state["Process Code"]] * N,
        "Wafer Polish Date": [state["Wafer Polish Date"]] * N,
        "Wafer Edge Exclusion (mm)": [state["Wafer Edge Exclusion (mm)"]] * N,
        "Die ID": die_ids,
        "Die Center X (mm)": state["Die Center X (mm)"],
        "Die Center Y (mm)": state["Die Center Y (mm)"],
        "Defect Radius (mm)": state["Defect Radius (mm)"]
    })

    wafer_df["Process Step ID"] = state["Process Step ID"]
    wafer_df["Chemical Lot ID"] = state["Chemical Lot ID"]
    wafer_df["Tester ID"] = state["Tester ID"]
    wafer_df["Probe Card ID"] = state["Probe Card ID"]
    wafer_df["Cleanroom Class"] = state["Cleanroom Class"]
    wafer_df["Tool ID"] = state["Tool ID"]
    wafer_df["Test Program ID"] = state["Test Program ID"]

    wafer_df["Chamber Temperature (°C)"] = state["Chamber Temperature (°C)"]
    wafer_df["Chamber Pressure (Torr)"] = state["Chamber Pressure (Torr)"]
    wafer_df["Ambient Temperature (°C)"] = state["Ambient Temperature (°C)"]
    wafer_df["Ambient Humidity (%)"] = state["Ambient Humidity (%)"]

    wafer_df["Threshold Voltage (V)"] = state["Threshold Voltage (V)"]
    wafer_df["On-Resistance (mΩ)"] = state["On-Resistance (mΩ)"]
    wafer_df["Idss_A"] = state["Idss_A"]
    wafer_df["Ioff_nA"] = state["Ioff_nA"]
    wafer_df["Cgd_pF"] = state["Cgd_pF"]

    wafer_df["PhysicalDefect"] = [dt if dt is not None else None for dt in defect_types]
    wafer_df["ProcessHistory"] = ";".join(state["ProcessHistory"])

    # wafer-sort test timestamps (orderly per-die, within a single day)
    base_test_date = base_date + timedelta(days=int(rng.integers(0,30)))
    test_duration_hours = float(rng.uniform(1.5, 4.0))   # wafer-sort is relatively quick
    seconds_per_die = 3600.0 * test_duration_hours / N
    test_start = datetime.combine(base_test_date.date(), time(8,0,0))
    test_ts = []
    for i in range(N):
        time_variance = float(rng.normal(0, seconds_per_die * 0.08))
        die_test_time = test_start + timedelta(seconds = i * seconds_per_die + time_variance)
        test_ts.append(die_test_time)
    wafer_df["Test Timestamp"] = test_ts
    wafer_df["Test Date"] = [ts.date() for ts in test_ts]
    wafer_df["Test Time"] = [ts.time() for ts in test_ts]

    # compute parametrics + spatial (keeps all fields consistent)
    wafer_df = recompute_parametrics_and_spatial(wafer_df)

    # add some geometry columns
    wafer_df["Defect Angle (deg)"] = wafer_df.apply(lambda r: (degrees(atan2(r["Die Center Y (mm)"], r["Die Center X (mm)"])) + 360) % 360, axis=1)
    wafer_df["Sector ID"] = (wafer_df["Defect Angle (deg)"] // (360.0/12.0) + 1).astype(int)
    wafer_df["Die Width (mm)"] = DIE_W_MM
    wafer_df["Die Height (mm)"] = DIE_H_MM
    wafer_df["Wafer Diameter (mm)"] = WAFER_DIA_MM

    return wafer_df

# ----------------------
# Generate lots & wafers (select lot-level bad chem ahead of generation)
# ----------------------
all_wafers = []
start_date = datetime(2025, 7, 1)

for lot_idx in range(1, NUM_LOTS+1):
    # decide per-lot if it is affected by bad chemical lot
    lot_bad_flag = rng.random() < 0.10
    chosen_bad = None
    if lot_bad_flag:
        chosen_bad = choice_one(bad_chem_lots)

    for widx in range(1, WAFERS_PER_LOT+1):
        df_w = generate_wafer(lot_idx, widx, start_date, lot_bad_chem=chosen_bad)
        # if a lot is flagged bad, apply a small consistent bump and recompute
        if chosen_bad is not None:
            df_w["Chemical Lot ID"] = chosen_bad
            df_w["Ioff_nA"] = df_w["Ioff_nA"] * (1.0 + abs(rng.normal(0.08, 0.03)))
            df_w = recompute_parametrics_and_spatial(df_w)
        all_wafers.append(df_w)

df_all = pd.concat(all_wafers, ignore_index=True)
df_all.to_csv(OUT_CSV, index=False)

# ----------------------
# Summary & basic checks
# ----------------------
total = len(df_all)
fails = (df_all["Test Result"] == "Fail").sum()
passes = total - fails
yield_pct = 100.0 * passes / total if total > 0 else 0.0

print(f"Generated {total} dies across {TOTAL_WAFERS} wafers ({NUM_LOTS} lots, {WAFERS_PER_LOT} wafers/lot).")
print(f"Saved CSV: {OUT_CSV}")
print(f"Yield: {passes}/{total} = {yield_pct:.2f}%")
print("\nTop failure causes (by FailureCause_First):")
print(df_all[df_all["Test Result"]=="Fail"]["FailureCause_First"].value_counts().head(10))

