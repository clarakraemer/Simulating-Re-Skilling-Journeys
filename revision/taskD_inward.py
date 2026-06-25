"""Task D — inward (shortage) intensity sweep over the optional weight.

Outward (at_risk) is saturated (~0 skills to first transition), so it cannot show
intensity sensitivity; inward (shortage, ~23 skills) is the only flow that can. Sample
country only (DE), journey-30, three weights, feasibility-only isolation (programs fixed
at 0.5). Gives direction + rough magnitude of intensity movement, not EU precision.

The w=0 leg is the heaviest config in the revision (q_viable ~1.42 -> many feasible
targets -> heavy per-worker loop), so its wall-time is the benchmark for what the w=0 leg
of an eventual full run will cost. Each leg is timed; w=0 timing is reported explicitly.
"""
import sys, os, time, numpy as np, pandas as pd, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from src.modelling.reskilling import ReskillingPathways
from src.modelling import occupation_distance
from revision.run_sample import assemble_lfs_data

COUNTRIES, FLOW, J = ["DE"], "shortage", 30
lfs = assemble_lfs_data()
rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc", lfs_data=lfs, year=2023)
osm0 = rp.occ_skills_mat.copy()
ess = (osm0.values == 1.0).astype(float); opt = (osm0.values == 0.5).astype(float)

def set_weight(w):
    Mw = osm0.copy(); Mw.iloc[:, :] = ess + w * opt
    rp.occ_skills_mat = Mw
    rp.occ_skills_mat_3d = Mw.groupby(level=3).mean()
    rp.df_occ_sim = occupation_distance.occ_sim_matrix_by_levels(
        occ_skills_mat=Mw, osm_version="weighted", sim_metric="cooc", diagonal_zeros=rp.osim_diag_zeros)

def threshold():
    cont = rp.calc_sim_means_by_level()
    msd = lambda l: (lambda s: float(s.mean() - s.std()))(cont[cont.level == l]["sim_mean"].astype(float))
    return msd("isco_4_digit"), msd("esco_5_digit")

def feas_intensity(df):
    tv = sorted([c for c in df.columns if re.fullmatch(r"transition_viable_step_\d+", c)],
                key=lambda c: int(c.split("_")[-1]))
    steps = [int(c.split("_")[-1]) for c in tv]
    M = df[tv].to_numpy(); first = np.full(len(df), np.nan)
    for j, s in enumerate(steps):
        first[M[:, j].astype(bool) & np.isnan(first)] = s
    w = df["COEFFY"].to_numpy(float); r = ~np.isnan(first)
    pct = float(np.nansum(w[r]) / np.nansum(w)) * 100
    mst = float(np.average(first[r], weights=w[r])) if r.any() else np.nan
    return pct, mst

print(f"=== Task D inward intensity sweep — {COUNTRIES} / {FLOW} / journey {J} ===", flush=True)
print(f"{'weight':>6} {'q_viable':>9} | {'program':12} {'%reached':>9} {'mean steps→1st':>14} {'leg time(s)':>11}", flush=True)
for w in [0.0, 0.5, 1.0]:
    set_weight(w); qv, qh = threshold()
    for prog, lab in [("coreness_ranked", "transferable"), ("optimal", "tailored")]:
        t0 = time.time()
        df = rp.simulate_regional(level="isco_3_digit", countries=COUNTRIES, scenarios=[FLOW],
            transition_optimisation="wage", reskilling=prog, reskilling_journey_length=J,
            region_constraints=False, mask_diagonal=True, transition_thresholds=(qv, qh),
            optional_weight=w, out_dir="/tmp/taskD_inward_out")[FLOW][COUNTRIES[0]]
        dt = time.time() - t0
        pct, mst = feas_intensity(df)
        tag = "  <-- w=0 BENCHMARK" if w == 0.0 else ""
        print(f"{w:6.2f} {qv:9.3f} | {lab:12} {pct:8.1f}% {mst:14.2f} {dt:11.1f}{tag}", flush=True)
    print(flush=True)
print("note: programs held at 0.5 (feasibility-only); 0.5 row reproduces Phase-1.", flush=True)
