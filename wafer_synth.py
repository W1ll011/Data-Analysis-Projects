#!/usr/bin/env python3
"""
wafer_synth.py

Generate synthetic wafer/die test data for yield engineering analysis.

Produces per-die records with:
 - wafer/lots/process metadata
 - die spatial coordinates on a circular wafer
 - clustered defects (to simulate hotspots)
 - parametric measurements (Vth, RON, Ioff, Idss, Cgd) with correlations
 - pass/fail logic (per parameter and overall)
 - spatial aggregates (SectorID, NeighborFailCount, LocalDefectDensity)

Outputs:
 - CSV file per wafer (default: ./synthetic_wafer_<WaferID>.csv)
 - Prints a yield summary and causes-of-failure breakdown

Author: ChatGPT (synthetic)
"""
import numpy as np
import pandas as pd
import random
import uuid
from datetime import datetime, timedelta
from math import atan2, degrees
from scipy.spatial import KDTree

# -----------------------------
# Config / tunables
# -----------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# number of wafers to simulate
NUMBER_OF_WAFERS = 1

# die & wafer geometry (mm)
DIE_W_mm = 10.0
DIE_H_mm = 10.0
SCRIBE_mm = 1.0
WAFER_DIA_mm = 300.0
WAFER_EDGE_EXCLUSION_mm = 2.0

# defect clustering: choose between 'uniform' or 'clustered'
DEFECT_MODE = 'clustered'  # 'uniform' or 'clustered'
MEAN_DEFECTS_PER_WAFER = 50   # mean number of physical defects to seed (clusters or singletons)

# parametric generation settings
# We use correlated Vth (V) and RON (mOhm) as an example
VTH_MEAN = 0.65
VTH_VAR = 0.0004
RON_MEAN = 120.0
RON_VAR = 100.0
VTH_RON_RHO = -0.5  # correlation coefficient

# Limits used to determine pass/fail (tunable)
PARAM_LIMITS = {
    'Vth_V':    (0.60, 0.70),    # [low, high] in volts
    'Ioff_nA':  (None, 10),      # max off-state leakage (nA)
    'Idss_A':   (None, 1e-9),    # max source-drain leakage (A)
    'RON_mOhm': (None, 150),     # max on-resistance (mΩ)
    'Cgd_pF':   (None, 0.15)     # max capacitance (pF)
}

# environmental baselines
AMBIENT_TEMP_MEAN = 23.5  # deg C
AMBIENT_TEMP_SIGMA = 0.5
AMBIENT_HUM_LO = 40.0
AMBIENT_HUM_HI = 60.0

# test logistics
TEST_START_DATE = datetime(2025, 7, 1)

# output folder & naming
OUTPUT_PREFIX = "./synthetic_wafer"

# -----------------------------
# Helper functions
# -----------------------------
def make_die_grid(die_w_mm, die_h_mm, scribe_mm, wafer_dia_mm, wafer_edge_excl_mm):
    """Return DataFrame of die centers within usable wafer circle."""
    pitch_x = die_w_mm + scribe_mm
    pitch_y = die_h_mm + scribe_mm
    n_cols = int(np.floor(wafer_dia_mm / pitch_x))
    n_rows = int(np.floor(wafer_dia_mm / pitch_y))
    xs = (np.arange(n_cols) - (n_cols - 1) / 2) * pitch_x
    ys = (np.arange(n_rows) - (n_rows - 1) / 2) * pitch_y
    xx, yy = np.meshgrid(xs, ys)
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()
    radial = np.hypot(xx_flat, yy_flat)
    usable_radius = (wafer_dia_mm / 2) - wafer_edge_excl_mm
    mask = radial <= usable_radius
    df = pd.DataFrame({
        'DieCenterX_mm': xx_flat[mask],
        'DieCenterY_mm': yy_flat[mask],
        'dist_from_center_mm': radial[mask]
    })
    df = df.reset_index(drop=True)
    df['DieID'] = [f"D{str(i+1).zfill(4)}" for i in range(len(df))]
    return df

def seed_physical_defects(df_dies, mean_defects, mode='clustered'):
    """
    Create a binary defect indicator per die by seeding a number of physical defects
    and marking dies within a small radius as impacted. If 'uniform', sprinkle
    defects individually (Poisson).
    Returns a boolean array same length as df_dies indicating physical defect presence.
    """
    num_dies = len(df_dies)
    coords = df_dies[['DieCenterX_mm', 'DieCenterY_mm']].values
    tree = KDTree(coords)
    n_defects = max(0, int(np.random.poisson(mean_defects)))
    impacted = np.zeros(num_dies, dtype=bool)

    if mode == 'uniform' or n_defects == 0:
        # Pick defected dies uniformly
        idxs = np.random.choice(num_dies, size=min(n_defects, num_dies), replace=False)
        impacted[idxs] = True
    else:
        # clustered: create a few cluster centers; each cluster affects multiple nearby dies
        n_clusters = max(1, int(np.round(n_defects / 6)))  # heuristic: ~6 defects per cluster center
        for _ in range(n_clusters):
            # pick a random seed center on wafer (uniformly)
            seed_idx = np.random.randint(0, num_dies)
            seed_pt = coords[seed_idx]
            # choose cluster radius (mm) and number of defect points in cluster
            cluster_radius = np.random.uniform(1.5 * min(DIE_W_mm, DIE_H_mm), 10.0)
            cluster_points = max(1, int(np.random.poisson(6)))
            # for each cluster point, mark neighboring dies within a small radius
            for _ in range(cluster_points):
                # jitter around seed
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0, cluster_radius)
                cx, cy = seed_pt + np.array([r * np.cos(angle), r * np.sin(angle)])
                # mark dies within influence radius (e.g., 1 die diagonal)
                influence_radius = max(DIE_W_mm, DIE_H_mm) * 0.75 + np.random.normal(0, 0.5)
                idxs = tree.query_ball_point([cx, cy], r=influence_radius)
                impacted[idxs] = True

    return impacted

def simulate_parametrics(num_dies):
    """Return a dict of arrays for parametric measurements (Vth, RON, Ioff, Idss, Cgd)."""
    # Correlated Vth & RON (units: V and mOhm). Build covariance from variances and rho.
    cov = VTH_RON_RHO * np.sqrt(VTH_VAR * RON_VAR)
    cov_matrix = np.array([[VTH_VAR, cov], [cov, RON_VAR]])
    mean_vec = np.array([VTH_MEAN, RON_MEAN])
    vth_ron = np.random.multivariate_normal(mean_vec, cov_matrix, size=num_dies)
    vth = vth_ron[:, 0]
    ron = vth_ron[:, 1]

    # Ambient conditions
    amb_temp = np.random.normal(AMBIENT_TEMP_MEAN, AMBIENT_TEMP_SIGMA, size=num_dies)
    amb_hum = np.random.uniform(AMBIENT_HUM_LO, AMBIENT_HUM_HI, size=num_dies)

    # Ioff: lognormal around nA range, sensitive to temp
    base_Ioff_nA = np.random.lognormal(mean=1.0, sigma=0.6, size=num_dies)  # ~e^1 ~ 2.7 nA typical
    alpha = 0.03
    ioff = base_Ioff_nA * np.exp(alpha * (amb_temp - AMBIENT_TEMP_MEAN))

    # Idss: small currents in A, lognormal; typical ~1e-9 to 1e-8
    base_Idss_A = np.random.lognormal(mean=-20.5, sigma=1.0, size=num_dies)  # tweak mean for scale
    # temperature dependence doubling per 10°C
    idss = base_Idss_A * (2 ** ((amb_temp - 25.0) / 10.0))

    # Cgd: pF range around 0.12 with small humidity dependency
    base_Cgd = np.random.normal(0.12, 0.01, size=num_dies)
    beta = 0.001
    cgd = base_Cgd + beta * (amb_hum - 50.0)

    return {
        'Vth_V': vth,
        'RON_mOhm': ron,
        'AmbientTemp_C': amb_temp,
        'AmbientHumidity_%': amb_hum,
        'Ioff_nA': ioff,
        'Idss_A': idss,
        'Cgd_pF': cgd
    }

def evaluate_params(df, param_limits):
    """Compute ParamPassFail and TestResult columns based on limits."""
    def check_row(r):
        for p, (lo, hi) in param_limits.items():
            v = r[p]
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                return False
        return True
    df['Param Pass/Fail'] = df.apply(check_row, axis=1)
    df['Test Result'] = df['Param Pass/Fail'].map({True: 'Pass', False: 'Fail'})
    return df

def compute_spatial_metrics(df):
    """Compute SectorID, DefectRadius/Angle, NeighborFailCount, LocalDefectDensity."""
    # angular sector (12 sectors)
    df['DefectAngle_deg'] = df.apply(
        lambda r: (degrees(atan2(r['DieCenterY_mm'], r['DieCenterX_mm'])) + 360) % 360,
        axis=1
    )
    df['SectorID'] = (df['DefectAngle_deg'] // (360 / 12) + 1).astype(int)
    coords = df[['DieCenterX_mm', 'DieCenterY_mm']].values
    tree = KDTree(coords)
    # neighbor radius: 1.5 * die width (mm)
    nbrs = tree.query_ball_point(coords, r=max(DIE_W_mm, DIE_H_mm) * 1.5)
    df['NeighborFailCount'] = [sum(df.iloc[n]['Test Result'] == 'Fail') for n in nbrs]
    # correct die area in cm^2 (mm->cm)
    area_cm2 = (DIE_W_mm / 10.0) * (DIE_H_mm / 10.0)
    df['LocalDefectDensity (defects/cm^2)'] = df['NeighborFailCount'] / area_cm2
    df['Defect Radius (mm)'] = df['dist_from_center_mm']
    return df

def failure_breakdown(df, param_limits):
    """Return a dict counting fails caused by each parameter (first-fail attribution)."""
    reasons = {p: 0 for p in param_limits}
    other = 0
    for _, r in df.iterrows():
        if r['Test Result'] == 'Pass':
            continue
        # attribute to first failing param in order
        attributed = False
        for p, (lo, hi) in param_limits.items():
            v = r[p]
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                reasons[p] += 1
                attributed = True
                break
        if not attributed:
            other += 1
    if other > 0:
        reasons['Other'] = other
    return reasons

# -----------------------------
# Main wafer generator
# -----------------------------
def generate_wafer(wafer_index=0):
    wafer_meta = {
        'WaferID': str(uuid.uuid4())[:8],
        'LotNumber': f"L{random.randint(10000,99999)}",
        'FabID': f"F{random.randint(1,9)}",
        'ProcessCode': random.choice(['N7','N7P','N5','N3','28nm']),
        'WaferOrientation': random.choice(['Flat','Notch']),
        'WaferPolishDate': (TEST_START_DATE + timedelta(days=random.randint(0,30))).date(),
        'WaferEdgeExclusion_mm': WAFER_EDGE_EXCLUSION_mm
    }

    # die grid
    df = make_die_grid(DIE_W_mm, DIE_H_mm, SCRIBE_mm, WAFER_DIA_mm, WAFER_EDGE_EXCLUSION_mm)
    num_dies = len(df)

    # parametrics
    params = simulate_parametrics(num_dies)
    for k, v in params.items():
        df[k] = v

    # add other test logistics
    df['Test Date'] = [TEST_START_DATE + timedelta(days=random.randint(0, 30)) for _ in range(num_dies)]
    df['Test Program ID'] = [f"TPR-{random.randint(100,999)}" for _ in range(num_dies)]
    df['Tester ID'] = [f"T-{random.choice(['A','B','C'])}{random.randint(1,5)}" for _ in range(num_dies)]
    df['Probe Card ID'] = [f"PC-{random.randint(1000,9999)}" for _ in range(num_dies)]
    df['Process Step ID'] = [random.choice(['ETCH','CMP','IMPL','LITH','CVD']) for _ in range(num_dies)]
    df['ChamberTemperature_C'] = np.random.normal(200, 5, size=num_dies)
    df['ChamberPressure_Torr'] = np.random.normal(5, 0.2, size=num_dies)
    df['Cleanroom Class'] = random.choice(['ISO5', 'ISO6'])
    df['Chemical Lot ID'] = [f"CL{random.randint(100,999)}" for _ in range(num_dies)]

    # physical defects seeded (spatially correlated)
    impacted = seed_physical_defects(df, MEAN_DEFECTS_PER_WAFER, mode=DEFECT_MODE)
    # Mark a defect type for those impacted dies (for realism, not necessarily failing)
    defect_types = ['Crack', 'Contamination', 'Open', 'Short', 'OxideDefect']
    df['PhysicalDefect'] = np.where(impacted, np.random.choice(defect_types, size=num_dies), 'None')

    # param pass/fail
    df = evaluate_params(df, PARAM_LIMITS)

    # for failed dies, optionally overwrite defect reason if physical defect exists (makes traceability)
    df['Defect Type'] = df.apply(lambda r: 'Physical' if r['PhysicalDefect'] != 'None' and r['Test Result'] == 'Fail' else ('Parametric' if r['Test Result']=='Fail' else 'None'), axis=1)

    # spatial metrics & neighbor/failure density
    df = compute_spatial_metrics(df)

    # add wafer metadata columns
    for k, v in wafer_meta.items():
        df[k] = v

    # reorder columns in a sensible way
    cols_front = [
        'WaferID', 'LotNumber', 'FabID', 'ProcessCode', 'WaferPolishDate', 'WaferEdgeExclusion_mm',
        'DieID', 'DieCenterX_mm', 'DieCenterY_mm', 'dist_from_center_mm',
        'Test Date', 'Test Program ID', 'Tester ID', 'Probe Card ID'
    ]
    # ensure these exist (map matching column names)
    final_cols = []
    for c in cols_front:
        if c in df.columns:
            final_cols.append(c)
    # append rest
    rest_cols = [c for c in df.columns if c not in final_cols]
    df = df[final_cols + rest_cols]

    # rename wafer meta columns to nicer names for CSV
    rename_map = {
        'WaferID': 'Wafer ID',
        'LotNumber': 'Lot Number',
        'FabID': 'Fab ID',
        'ProcessCode': 'Process Code',
        'WaferPolishDate': 'Wafer Polish Date',
        'WaferEdgeExclusion_mm': 'Wafer Edge Exclusion (mm)'
    }
    df.rename(columns=rename_map, inplace=True)

    return df

# -----------------------------
# Run simulation(s)
# -----------------------------
def main():
    all_wafers = []
    for i in range(NUMBER_OF_WAFERS):
        df_w = generate_wafer(wafer_index=i)
        wafer_id = df_w.iloc[0]['Wafer ID']
        out_name = f"{OUTPUT_PREFIX}_{wafer_id}.csv"
        df_w.to_csv(out_name, index=False)
        print(f"Saved wafer data to: {out_name}  (dies: {len(df_w)})")

        # yield summary
        total = len(df_w)
        passes = (df_w['Test Result'] == 'Pass').sum()
        fails = total - passes
        yield_pct = 100.0 * passes / total
        print(f"Wafer {wafer_id} summary: Total dies={total}, Pass={passes}, Fail={fails}, Yield={yield_pct:.2f}%")

        # failure breakdown
        breakdown = failure_breakdown(df_w, PARAM_LIMITS)
        print("Failure attribution (counts):")
        for k, v in breakdown.items():
            print(f"  {k}: {v}")
        print("-" * 60)

        all_wafers.append(df_w)

    # combine if multiple wafers requested
    if len(all_wafers) > 1:
        combined = pd.concat(all_wafers, ignore_index=True)
        combined_name = f"{OUTPUT_PREFIX}_combined.csv"
        combined.to_csv(combined_name, index=False)
        print(f"Saved combined wafers to {combined_name}")

if __name__ == "__main__":
    main()