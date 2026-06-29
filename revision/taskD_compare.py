"""Task-D (reviewer's question): are optional skills needed AT ALL?

Compares w=0 (optional skills EXCLUDED) vs w=0.5 (production) **holding the production
viability rule fixed** (threshold 3.68/10.80). This isolates the weight's effect: the
re-derived per-weight threshold is NOT used (the green decomposition showed it collapses the
bar — green reaches in 1.9 vs 19 skills — so a per-weight sweep confounds bar-movement with
skill-accounting). Fixed-threshold => the only thing that changes is whether optional-skill
overlap contributes to M_oo.

Framing for the rebuttal/SI:
  reach(w=0.5) - reach(w=0, fixed thr) = the share of feasible transitions ATTRIBUTABLE TO
  OPTIONAL-SKILL OVERLAP, holding the production viability rule fixed. (Pre-empts the
  "of course removing skills lowers overlaps" objection: the bar is unchanged; what falls is
  exactly the feasibility that optional skills were providing.)

Inputs (local pickles; run after the fixed-threshold w=0 leg lands):
  production : <flow>/reskill-*_wage-opt_regC_2023/                 (w=0.5)
  w=0 fixed  : <flow>/reskill-*_wage-opt_regC_2023_optw0_fixthr2/    (w=0, threshold 3.68/10.80)

(The per-weight sweep _optw0/_optw1 is demoted to an SI footnote; this script is the headline.)

    python revision/taskD_compare.py
"""
import os, pickle, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "results", "figures", "reskilling_simulation")
SIM = {"optimal": "reskill-optimal", "coreness_ranked": "reskill-coreRanked",
       "green": "reskill-green", "digital": "reskill-digital"}
LAB = {"optimal": "tailored", "coreness_ranked": "transferable", "green": "green", "digital": "digital"}
ORDER = ["optimal", "coreness_ranked", "green", "digital"]
JL = {"at_risk": 20, "shortage": 30}


def load(flow, prog, suffix=""):
    tag = f"{SIM[prog]}_wage-opt_regC_2023{suffix}"
    p = os.path.join(B, flow, tag, f"{tag}.pkl")
    return pickle.load(open(p, "rb"))[flow] if os.path.exists(p) else None


def wmean(v, w):
    v = np.asarray(v, float); w = np.asarray(w, float); m = ~np.isnan(v) & ~np.isnan(w)
    return np.average(v[m], weights=w[m]) if m.any() and w[m].sum() > 0 else np.nan


def reach_first(per, last):
    df = pd.concat(list(per.values()))
    w = df["COEFFY"].to_numpy(float)
    cols = [f"transition_viable_step_{s}" for s in range(last + 1) if f"transition_viable_step_{s}" in df.columns]
    V = np.nan_to_num(df[cols].astype(float).to_numpy(), nan=0) > 0.5
    reached = V.any(1); fs = V.argmax(1).astype(float); fs[~reached] = np.nan
    return (w[reached].sum() / w.sum() * 100), wmean(fs[reached], w[reached])


def main():
    print("=" * 104)
    print("TASK-D: feasibility attributable to OPTIONAL-SKILL OVERLAP (production viability rule FIXED 3.68/10.80)")
    print("        w=0.5 = production (optional incl) ; w=0 = _optw0_fixthr2 (optional excl, SAME bar)")
    print("=" * 104)
    any_missing = False
    for flow in ["at_risk", "shortage"]:
        last = JL[flow]
        print(f"\n--- {flow} (regC, EU-pooled, COEFFY-weighted) ---")
        print(f"{'program':12} | {'reach w0.5':>10} {'reach w0':>9} {'Δreach pp':>10} {'%attrib':>8} | "
              f"{'1st w0.5':>9} {'1st w0':>8} {'Δ1st':>7}")
        for prog in ORDER:
            prod = load(flow, prog, "")
            w0 = load(flow, prog, "_optw0_fixthr2")
            if prod is None or w0 is None:
                any_missing = True
                miss = "production" if prod is None else "_optw0_fixthr2"
                print(f"{LAB[prog]:12} | MISSING ({miss}) — run the fixed-threshold w=0 leg first")
                continue
            r_p, f_p = reach_first(prod, last)
            r_0, f_0 = reach_first(w0, last)
            d_reach = r_p - r_0
            attrib = (d_reach / r_p * 100) if r_p else np.nan
            print(f"{LAB[prog]:12} | {r_p:9.1f}% {r_0:8.1f}% {d_reach:+9.1f} {attrib:7.1f}% | "
                  f"{f_p:9.1f} {f_0:8.1f} {(f_0 - f_p):+7.1f}")
    print("\nReading: Δreach (pp) and %attrib = feasible-transition reach that optional-skill overlap")
    print("provides, holding the production viability rule fixed. Δreach>0 => optional skills ARE needed")
    print("(excluding them at the SAME bar makes fewer transitions feasible). Δ1st>0 => excluding optional")
    print("skills also slows time-to-first-transition. This answers the reviewer directly.")
    if any_missing:
        print("\n[!] Missing legs — launch the fixed-threshold w=0 run:")
        print("    python revision/run_parallel.py --workers 20 --optional-weight 0 \\")
        print("      --threshold 3.68,10.80 --scenarios at_risk,shortage --regc true --tag-suffix _fixthr2")


if __name__ == "__main__":
    main()
