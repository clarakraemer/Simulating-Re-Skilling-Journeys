"""Focused before/after for the Task A correctness fix (journey-aware default).

transferable runs to journey-20 (the response letter quotes the step-20 intensity);
tailored/green/digital run to journey-12 to confirm Δ=0 (identical by construction).
Each program prints as soon as it finishes (flush) so a kill can't lose earlier results.

before = baseline-only (journey_aware=False, frozen-reference behaviour)
after  = journey-aware (journey_aware=True,  corrected default)
"""
import sys, os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from src.modelling.reskilling import ReskillingPathways
from revision.run_sample import assemble_lfs_data

COUNTRIES = ["DE", "HR", "CY", "LV"]
SCENARIO = "at_risk"
THRESH = (3.68, 10.80)
# (label, reskilling-mode, journey, steps-to-report)
PROGRAMS = [
    ("transferable", "coreness_ranked", 20, (4, 12, 20)),
    ("tailored",     "optimal",         12, (4, 12)),
    ("green",        "green",           12, (4, 12)),
    ("digital",      "digital",         12, (4, 12)),
]

def agg(frames, steps):
    df = pd.concat(frames, ignore_index=True)
    w = df["COEFFY"].values
    def wm(col):
        if col not in df: return float("nan")
        v = df[col].values; m = ~np.isnan(v) & ~np.isnan(w)
        return float(np.average(v[m], weights=w[m])) if m.any() and w[m].sum() > 0 else float("nan")
    inten = {s: wm(f"n_viable_transitions_step_{s}") for s in steps}
    cols = sorted([c for c in df.columns if c.startswith("transition_viable_step_")],
                  key=lambda c: int(c.split("_")[-1]))
    sl = [int(c.split("_")[-1]) for c in cols]
    first = np.array([next((s for s, c in zip(sl, cols) if bool(r[c])), np.nan)
                      for _, r in df.iterrows()], float)
    ok = ~np.isnan(first)
    wfirst = float(np.average(first[ok], weights=w[ok])) if ok.any() else float("nan")
    return inten, wfirst

def run(rp, mode, journey, ja, steps):
    rp.journey_aware = ja
    res = rp.simulate_regional(
        level="isco_3_digit", countries=COUNTRIES, scenarios=[SCENARIO],
        transition_optimisation="wage", reskilling=mode,
        reskilling_journey_length=journey, region_constraints=False,
        target_job_availability_coeffy="COEFFY_mean+sd", mask_diagonal=True,
        transition_thresholds=THRESH, out_dir=f"/tmp/baf_{mode}_{ja}")[SCENARIO]
    return agg([res[c] for c in COUNTRIES if c in res], steps)

def main():
    lfs = assemble_lfs_data()
    rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc", lfs_data=lfs, year=2023)
    print(f"BEFORE/AFTER (focused) — sample={COUNTRIES} {SCENARIO} no-regC", flush=True)
    print(f"{'program':12} {'metric':16} {'before':>9} {'after':>9} {'Δ':>9} {'Δ%':>8}", flush=True)
    print("-" * 68, flush=True)
    for label, mode, journey, steps in PROGRAMS:
        ib, fb = run(rp, mode, journey, False, steps)
        ia, fa = run(rp, mode, journey, True, steps)
        for s in steps:
            b, a = ib[s], ia[s]
            d = f"{a-b:+.3f}" if not (np.isnan(a) or np.isnan(b)) else "n/a"
            dp = f"{(a-b)/b*100:+.1f}%" if b else "n/a"
            print(f"{label:12} {'intensity@'+str(s):16} {b:9.3f} {a:9.3f} {d:>9} {dp:>8}", flush=True)
        db = f"{fa-fb:+.3f}" if not (np.isnan(fa) or np.isnan(fb)) else "n/a"
        print(f"{label:12} {'1st-trans step':16} {fb:9.3f} {fa:9.3f} {db:>9} {'':>8}", flush=True)
        print("-" * 68, flush=True)

if __name__ == "__main__":
    main()
