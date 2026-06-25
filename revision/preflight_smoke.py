"""Pre-flight smoke pass: all countries, journey-1, every (program, flow, regC) combo.

Catches country-specific crashes (regC drop logic, region fallback, NaN handling) in
minutes instead of mid-run. Uses the committed defaults (journey_aware=True,
destination_weighting=above_current, weight 0.5, symmetric off) — i.e. exactly the full
run's configuration, just at journey-1. Each combo runs all 27 countries at once; a crash
in any country fails that combo and is reported with the traceback head.
"""
import sys, os, time, traceback
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from src.modelling.reskilling import ReskillingPathways
from revision.run_sample import assemble_lfs_data

COUNTRIES = ["AT","BE","CH","CY","CZ","DE","DK","EL","EE","ES","FI","FR","HR","HU","IE",
             "IS","IT","LT","LU","LV","NL","NO","PL","PT","RO","SE","SK"]
PROGRAMS = ["coreness_ranked", "optimal", "green", "digital"]
FLOWS = ["at_risk", "shortage"]
REGC = [True, False]

lfs = assemble_lfs_data()
rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc", lfs_data=lfs, year=2023)
print(f"defaults: journey_aware={rp.journey_aware} dest={rp.destination_weighting} "
      f"acceptability={getattr(rp,'acceptability',None)}", flush=True)

npass = nfail = 0
for flow in FLOWS:
    for regc in REGC:
        for prog in PROGRAMS:
            t0 = time.time()
            try:
                res = rp.simulate_regional(level="isco_3_digit", countries=COUNTRIES,
                    scenarios=[flow], transition_optimisation="wage", reskilling=prog,
                    reskilling_journey_length=1, region_constraints=regc, mask_diagonal=True,
                    transition_thresholds=(3.68, 10.80),
                    out_dir="/tmp/preflight_smoke_out")[flow]
                ncty = len(res)
                print(f"PASS {flow:8} {'regC' if regc else 'no-regC':7} {prog:16} "
                      f"countries={ncty:2}  {time.time()-t0:5.1f}s", flush=True)
                npass += 1
            except Exception as e:
                print(f"FAIL {flow:8} {'regC' if regc else 'no-regC':7} {prog:16} :: {e}", flush=True)
                print("   " + " | ".join(traceback.format_exc().splitlines()[-3:]), flush=True)
                nfail += 1
print(f"\nSMOKE PASS SUMMARY: {npass} passed, {nfail} failed", flush=True)
print("VERDICT:", "ALL GREEN — no country-specific crash at journey-1" if nfail == 0
      else f"{nfail} FAILING COMBO(S) — fix before full run", flush=True)
