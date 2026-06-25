"""Task D — optional-skill-weight robustness sweep (feasibility-only isolation).

Varies the optional weight in the FEASIBILITY measure only (M_os -> M_oo -> threshold),
holding program definitions fixed (transferable coreness order + tailored upskilling
table stay at 0.5). Isolates how feasibility/intensity respond to the treatment of
optional skills (SI sensitivity row 14). Disable switch: weight stays 0.5 by default.

set_weight(w) rebuilds all three weight-dependent feasibility objects consistently:
  occ_skills_mat(w)  (full M_os: essential=1, optional=w)
  occ_skills_mat_3d(w) (ISCO-3 mean aggregate — drives the journey reskill())
  df_occ_sim(w)      (full M_oo — drives the step-0 baseline via sim_matrix_at_level)
Threshold(w) = ISCO-4 mean-SD (q_viable), ESCO-5 mean-SD (q_highly_viable), same rule.
"""
import sys, os, numpy as np, pandas as pd, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from src.modelling.reskilling import ReskillingPathways
from src.modelling import occupation_distance
from revision.run_sample import assemble_lfs_data

lfs = assemble_lfs_data()
rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc", lfs_data=lfs, year=2023)
osm0 = rp.occ_skills_mat.copy()
osm3d0 = rp.occ_skills_mat_3d.copy()
dfsim0 = rp.df_occ_sim.copy()
ess = (osm0.values == 1.0).astype(float)
opt = (osm0.values == 0.5).astype(float)

def set_weight(w):
    Mw = osm0.copy(); Mw.iloc[:, :] = ess + w * opt
    rp.occ_skills_mat = Mw
    rp.occ_skills_mat_3d = Mw.groupby(level=3).mean()
    rp.df_occ_sim = occupation_distance.occ_sim_matrix_by_levels(
        occ_skills_mat=Mw, osm_version="weighted", sim_metric="cooc",
        diagonal_zeros=rp.osim_diag_zeros)

def threshold():
    cont = rp.calc_sim_means_by_level()
    def msd(lvl):
        sm = cont[cont.level == lvl]["sim_mean"].astype(float)
        return float(sm.mean() - sm.std())
    return (msd("isco_4_digit"), msd("esco_5_digit"))   # (q_viable, q_highly_viable)

def feas_intensity(res_df, J):
    tv = sorted([c for c in res_df.columns if re.fullmatch(r"transition_viable_step_\d+", c)],
                key=lambda c: int(c.split("_")[-1]))
    steps = [int(c.split("_")[-1]) for c in tv]
    M = res_df[tv].to_numpy(); first = np.full(len(res_df), np.nan)
    for j, s in enumerate(steps):
        first[M[:, j].astype(bool) & np.isnan(first)] = s
    w = res_df["COEFFY"].to_numpy(float); reached = ~np.isnan(first)
    pct = float(np.nansum(w[reached]) / np.nansum(w)) * 100
    mst = (float(np.average(first[reached], weights=w[reached])) if reached.any() else np.nan)
    return pct, mst

FLOW, J = "at_risk", 12
print("=== Task D 0.5 matrix regression (set_weight(0.5) must restore exactly) ===")
set_weight(0.5)
print("  occ_skills_mat_3d max|Δ| vs original:", float((rp.occ_skills_mat_3d - osm3d0).abs().values.max()))
print("  df_occ_sim        max|Δ| vs original:", float((rp.df_occ_sim - dfsim0).abs().values.max()))

print(f"\n=== Task D sweep — DE / {FLOW} / no-regC / journey {J} (feasibility-only) ===")
print(f"{'weight':>6} {'q_viable':>9} {'q_high':>8} | {'program':12} {'%reached':>9} {'mean steps→1st':>14}")
print("-" * 70)
rows = {}
for w in [0.0, 0.5, 1.0]:
    set_weight(w); qv, qh = threshold()
    for prog, lab in [("coreness_ranked", "transferable"), ("optimal", "tailored")]:
        res = rp.simulate_regional(level="isco_3_digit", countries=["DE"], scenarios=[FLOW],
            transition_optimisation="wage", reskilling=prog, reskilling_journey_length=J,
            region_constraints=False, mask_diagonal=True, transition_thresholds=(qv, qh),
            optional_weight=w, out_dir="/tmp/taskD_out")[FLOW]["DE"]
        pct, mst = feas_intensity(res, J)
        rows[(w, lab)] = (pct, mst)
        print(f"{w:6.2f} {qv:9.3f} {qh:8.3f} | {lab:12} {pct:8.1f}% {mst:14.2f}")
    print()
print("note: thresholds rise with weight; feasibility-only (programs fixed at 0.5).")
