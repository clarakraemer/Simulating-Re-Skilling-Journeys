"""Task A correctness fix — before/after table of the manuscript numbers it moves.

before = baseline-only (rp.journey_aware=False, = the frozen-reference behaviour)
after  = journey-aware (rp.journey_aware=True,  = the corrected default)

Reports, per program, COEFFY-weighted-mean intensity (n_viable_transitions) at steps
4 / 12 / 20 and the mean first-transition step, aggregated across the awkward-path
sample. Expectation: transferable moves; tailored and green/digital show no change.
The EXACT EU figures come from the single end-of-phase full-country run; this sample
table gives direction + magnitude for drafting the response letter.
"""
import sys, os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from src.modelling.reskilling import ReskillingPathways
from revision.run_sample import assemble_lfs_data

COUNTRIES = ["DE", "HR", "CY", "LV"]
SCENARIO = "at_risk"
JOURNEY = 20
THRESH = (3.68, 10.80)
PROGRAMS = [("transferable", "coreness_ranked"), ("tailored", "optimal"),
            ("green", "green"), ("digital", "digital")]

def agg(frames):
    """Concatenate country frames; return weighted-mean intensity by step + mean 1st-trans step."""
    df = pd.concat(frames, ignore_index=True)
    w = df["COEFFY"].values
    def wm(col):
        v = df[col].values; m = ~np.isnan(v) & ~np.isnan(w)
        return float(np.average(v[m], weights=w[m])) if m.any() and w[m].sum() > 0 else float("nan")
    inten = {s: wm(f"n_viable_transitions_step_{s}") for s in (4, 12, 20)}
    # first-transition step (NaN if none within journey)
    cols = [c for c in df.columns if c.startswith("transition_viable_step_")]
    cols.sort(key=lambda c: int(c.split("_")[-1]))
    steps = [int(c.split("_")[-1]) for c in cols]
    first = []
    for _, r in df.iterrows():
        first.append(next((s for s, c in zip(steps, cols) if bool(r[c])), np.nan))
    first = np.array(first, float)
    wfirst = float(np.average(first[~np.isnan(first)], weights=w[~np.isnan(first)])) \
        if (~np.isnan(first)).any() else float("nan")
    return inten, wfirst

def run(rp, prog, ja):
    rp.journey_aware = ja
    res = rp.simulate_regional(
        level="isco_3_digit", countries=COUNTRIES, scenarios=[SCENARIO],
        transition_optimisation="wage", reskilling=prog,
        reskilling_journey_length=JOURNEY, region_constraints=False,
        target_job_availability_coeffy="COEFFY_mean+sd", mask_diagonal=True,
        transition_thresholds=THRESH, out_dir=f"/tmp/ba_{prog}_{ja}",
    )[SCENARIO]
    return agg([res[c] for c in COUNTRIES if c in res])

def main():
    lfs = assemble_lfs_data()
    rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc", lfs_data=lfs, year=2023)
    print(f"BEFORE/AFTER — sample={COUNTRIES} {SCENARIO} no-regC journey={JOURNEY}")
    print(f"{'program':12} {'metric':16} {'before':>9} {'after':>9} {'Δ':>9}")
    print("-" * 60)
    for label, prog in PROGRAMS:
        ib, fb = run(rp, prog, False)
        ia, fa = run(rp, prog, True)
        for s in (4, 12, 20):
            b, a = ib[s], ia[s]
            d = f"{(a-b):+.3f}" if (not np.isnan(a) and not np.isnan(b)) else "n/a"
            print(f"{label:12} {'intensity@'+str(s):16} {b:9.3f} {a:9.3f} {d:>9}")
        db = f"{(fa-fb):+.3f}" if (not np.isnan(fa) and not np.isnan(fb)) else "n/a"
        print(f"{label:12} {'1st-trans step':16} {fb:9.3f} {fa:9.3f} {db:>9}")
        print("-" * 60)

if __name__ == "__main__":
    main()
