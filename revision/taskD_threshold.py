"""Task D — re-derive the feasibility threshold per optional-skill weight.

The rule is held FIXED (across-group mean minus one SD of the within-group occupation-
overlap distribution, computed by the model's own calc_sim_means_by_level); only the
optional-skill weight varies. M_os(w) = essential(=1) + w*optional, rebuilt from the
loaded 0.5-weighted matrix; M_oo(w) = M_os(w) @ M_os(w).T via the production
occ_sim_matrix_by_levels. {0, 0.5, 1.0} are all dyadic/integer so M_oo(w) is exact.

Self-check: at w=0.5 the derived (q_viable, q_highly_viable) must come back to the
~(3.68, 10.80) used in Phase-1.
"""
import sys, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from src.modelling.reskilling import ReskillingPathways
from src.modelling import occupation_distance
from revision.run_sample import assemble_lfs_data

lfs = assemble_lfs_data()
rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc", lfs_data=lfs, year=2023)
osm = rp.occ_skills_mat
vals = osm.values
ess = (vals == 1.0).astype(float)
opt = (vals == 0.5).astype(float)
print(f"M_os structure: essential cells={int(ess.sum())}, optional cells={int(opt.sum())}")

LEVELS = ["esco_5_digit", "isco_4_digit", "isco_3_digit"]

def thresholds(w):
    Mw = osm.copy(); Mw.iloc[:, :] = ess + w * opt
    rp.df_occ_sim = occupation_distance.occ_sim_matrix_by_levels(
        occ_skills_mat=Mw, osm_version="weighted", sim_metric="cooc",
        diagonal_zeros=rp.osim_diag_zeros)
    cont = rp.calc_sim_means_by_level()
    out = {}
    for lvl in LEVELS:
        sm = cont[cont.level == lvl]["sim_mean"].astype(float)
        m, s = float(sm.mean()), float(sm.std())
        out[lvl] = (m, s, m - s, m + s)
    return out

print("\n=== TASK D: feasibility threshold re-derivation (same rule, weight varies) ===")
print(f"{'weight':>6} | {'level':13} {'mean':>8} {'sd':>8} {'q_viable':>9} {'q_high':>9}  note")
print("-" * 72)
labels = {0.0: "essential-only feasibility", 0.5: "midpoint (used)", 1.0: "optional = essential"}
for w in [0.0, 0.5, 1.0]:
    t = thresholds(w)
    for lvl in LEVELS:
        m, s, lo, hi = t[lvl]
        note = ""
        if lvl == "isco_4_digit":
            note = labels[w] + ("  <- ISCO-4 rule" if w == 0.5 else "  <- ISCO-4 rule")
        print(f"{w:6.2f} | {lvl:13} {m:8.3f} {s:8.3f} {lo:9.3f} {hi:9.3f}  {note}")
    print()
print("self-check: w=0.5 ISCO-4 (q_viable, q_high) should ~= (3.68, 10.80).")
