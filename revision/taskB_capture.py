"""Task B sample — capture phase.

Runs the regional model with capture on, recording at the report horizon the feasible
target set (mode-invariant) and, per worker, the deterministic draw seed + the choice
the run actually made. Countries DE + HR (granular-NUTS2; CY/LV drop out of regC).

Produces:
  /tmp/taskB_cap_at_risk.pkl   off-mode, at_risk @ step 20   (feeds the sweep + validation A)
  /tmp/taskB_val_at_risk.pkl   band=0.25, at_risk @ step 20  (validation B: live choices)
  /tmp/taskB_cap_shortage.pkl  off-mode, shortage @ step 30  (feeds the sweep, inward)
"""
import sys, os, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from src.modelling.reskilling import ReskillingPathways
from revision.run_sample import assemble_lfs_data

# Income-eligible (notebook-01/06 KEEP list) and regC-surviving (granular NUTS-2), so the
# income-cost column is computed on observed-earnings countries, consistent with the
# notebook-06 reporting filter. (DE+HR were income-excluded -> their income column was
# invalid; the employment-share frontier from DE+HR is unaffected and stands.)
COUNTRIES = ["SE", "SK", "DK"]
THRESH = (3.68, 10.80)
PROGRAM = "optimal"

def run(rp, scen, J, dw, band=None):
    rp.destination_weighting = dw
    if band is not None:
        rp.income_band = band; rp.acceptability = "band"
    rp._capture_at_step = J
    print(f">>> {scen} j={J} dw={dw} band={band} regC DE+HR", flush=True)
    rp.simulate_regional(
        level="isco_3_digit", countries=COUNTRIES, scenarios=[scen],
        transition_optimisation="wage", reskilling=PROGRAM,
        reskilling_journey_length=J, region_constraints=True,
        target_job_availability_coeffy="COEFFY_mean+sd", mask_diagonal=True,
        transition_thresholds=THRESH, out_dir=f"/tmp/taskB_out_{scen}_{dw}_{band}")
    return list(rp._capture)

def main():
    lfs = assemble_lfs_data()
    rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc", lfs_data=lfs, year=2023)

    cap = run(rp, "at_risk", 20, "off")
    pd.to_pickle({"capture": cap, "step": 20, "scenario": "at_risk"}, "/tmp/taskB_cap_at_risk.pkl")
    print(f">>> saved at_risk off: {len(cap)} regions", flush=True)

    val = run(rp, "at_risk", 20, "share_income_acceptable", band=0.25)
    pd.to_pickle({"capture": val, "step": 20, "scenario": "at_risk"}, "/tmp/taskB_val_at_risk.pkl")
    print(f">>> saved at_risk band=0.25 (validation): {len(val)} regions", flush=True)

    cap2 = run(rp, "shortage", 30, "off")
    pd.to_pickle({"capture": cap2, "step": 30, "scenario": "shortage"}, "/tmp/taskB_cap_shortage.pkl")
    print(f">>> saved shortage off: {len(cap2)} regions", flush=True)

if __name__ == "__main__":
    main()
