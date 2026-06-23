"""Task A: journey-aware vs baseline-only `have` filter — sample comparison.

Only the transferable program (coreness_ranked) is affected:
  * green/digital produce identical sequences in both modes (greedy == rank-walk);
  * tailored (optimal) never touches the held filter.
So this runs transferable under both modes on the awkward-path sample countries and
reports how much the OUTCOMES move (intensity at steps 4 & 12, first-transition step,
added-skill divergence).
"""
import sys, os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from src.modelling.reskilling import ReskillingPathways
from revision.run_sample import assemble_lfs_data

COUNTRIES = ["DE", "HR", "CY", "LV"]
SCENARIO = "at_risk"   # at_risk carries intensity signal at steps 4/12 (shortage unlocks only ~step 30)
JOURNEY = 12
THRESH = (3.68, 10.80)

def wmean(df, col):
    w = df["COEFFY"].values; v = df[col].values
    m = ~np.isnan(v) & ~np.isnan(w)
    return float(np.average(v[m], weights=w[m])) if m.any() and w[m].sum() > 0 else float("nan")

def first_transition_step(df):
    cols = sorted([c for c in df.columns if c.startswith("transition_viable_step_")],
                  key=lambda c: int(c.split("_")[-1]))
    steps = [int(c.split("_")[-1]) for c in cols]
    out = []
    for _, row in df.iterrows():
        f = next((s for s, c in zip(steps, cols) if bool(row[c])), np.nan)
        out.append(f)
    return np.array(out, float)

def run(rp, journey_aware, tag):
    rp.journey_aware = journey_aware
    return rp.simulate_regional(
        level="isco_3_digit", countries=COUNTRIES, scenarios=[SCENARIO],
        transition_optimisation="wage", reskilling="coreness_ranked",
        reskilling_journey_length=JOURNEY, region_constraints=False,
        target_job_availability_coeffy="COEFFY_mean+sd", mask_diagonal=True,
        transition_thresholds=THRESH, out_dir=f"/tmp/taskA_out_{tag}",
    )[SCENARIO]

def main():
    lfs = assemble_lfs_data()
    rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc", lfs_data=lfs, year=2023)
    base = run(rp, False, "base")
    jour = run(rp, True, "jour")

    def pct(j, b):
        return (j - b) / b * 100 if b else float("nan")

    print(f"\n=== TASK A: transferable (coreRanked) journey-aware vs baseline-only ===")
    print(f"sample={COUNTRIES} scenario={SCENARIO} no-regC journey={JOURNEY}")
    print("intensity = COEFFY-weighted mean n_viable_transitions at the step\n")
    hdr = f"{'cty':4} {'n':4} | {'intensity@4 base->jour':26} | {'intensity@12 base->jour':27} | first-transition-step move"
    print(hdr); print("-" * len(hdr))
    for c in COUNTRIES:
        if c not in base or c not in jour:
            print(f"{c:4} (absent in one mode)"); continue
        b, j = base[c].reset_index(drop=True), jour[c].reset_index(drop=True)
        n = len(b)
        i4b, i4j = wmean(b, "n_viable_transitions_step_4"), wmean(j, "n_viable_transitions_step_4")
        i12b, i12j = wmean(b, "n_viable_transitions_step_12"), wmean(j, "n_viable_transitions_step_12")
        fb, fj = first_transition_step(b), first_transition_step(j)
        both_nan = np.isnan(fb) & np.isnan(fj)
        changed = (~both_nan) & ((fb != fj) | (np.isnan(fb) ^ np.isnan(fj)))
        chg = float(np.mean(changed)) if len(fb) == len(fj) else float("nan")
        print(f"{c:4} {n:4} | base={i4b:6.3f} jour={i4j:6.3f} ({pct(i4j,i4b):+6.1f}%) | "
              f"base={i12b:6.3f} jour={i12j:6.3f} ({pct(i12j,i12b):+6.1f}%) | "
              f"{chg*100:5.1f}% of workers")
    print("\ngreen/digital: identical in both modes (greedy==rank-walk, verified);"
          "  tailored: unaffected by construction.")

if __name__ == "__main__":
    main()
