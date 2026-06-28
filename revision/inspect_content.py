"""Content sanity check of the 16 parallel-run pickles — headline numbers, no plotting.

Reads the saved result frames directly and reports, per program x flow (regC, EU-pooled,
COEFFY-weighted):
  * reach rate  = pop-weighted share of worker-groups that EVER reach a viable transition
  * skills-to-first = pop-weighted mean first step with a viable transition (over reachers)
  * income %    = pop-weighted mean earnings_delta_closest_switch_pct, INCOME_KEEP only
And a Task-A check: transferable (coreRanked) at_risk intensity at steps 12/20 vs SEP25.

All metrics use COEFFY (population weight). Income is reported ONLY for the income-eligible
KEEP list; DROP-list countries are excluded and flagged.
"""
import os, pickle, re
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results", "figures", "reskilling_simulation")
SEP25 = os.path.join(BASE, "SEP25")
PROG = {"optimal": ("reskill-optimal", "tailored"),
        "coreness_ranked": ("reskill-coreRanked", "transferable"),
        "green": ("reskill-green", "green"),
        "digital": ("reskill-digital", "digital")}
ORDER = ["optimal", "coreness_ranked", "green", "digital"]
JOURNEY = {"at_risk": 20, "shortage": 30}
INCOME_KEEP = {"SE", "CH", "SK", "DK", "FR", "EE", "EL", "IE", "NO", "FI", "PT", "LU"}


def load(scenario, sim_name, regc, root=BASE):
    tag = f"{sim_name}_wage-opt_{'regC' if regc else 'no-regC'}_2023"
    p = os.path.join(root, scenario, tag, f"{tag}.pkl")
    if not os.path.exists(p):
        return None
    return pickle.load(open(p, "rb"))[scenario]


def pooled(per_country, add_country=True):
    """Concat all country frames into one, tagging the source country."""
    frames = []
    for c, df in per_country.items():
        d = df.copy()
        d["__country"] = c
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def wmean(v, w):
    v = np.asarray(v, float); w = np.asarray(w, float)
    m = ~np.isnan(v) & ~np.isnan(w)
    return np.average(v[m], weights=w[m]) if m.any() and w[m].sum() > 0 else np.nan


def reach_and_first(df, last):
    w = df["COEFFY"].to_numpy(float)
    cols = [f"transition_viable_step_{s}" for s in range(0, last + 1) if f"transition_viable_step_{s}" in df.columns]
    V = df[cols].to_numpy(float)
    V = np.nan_to_num(V, nan=0.0) > 0.5
    reached = V.any(axis=1)
    first = np.full(len(df), np.nan)
    idx = V.argmax(axis=1).astype(float)        # first True position (0 if none, masked next)
    first[reached] = idx[reached]
    reach_rate = (w[reached].sum() / w.sum()) if w.sum() > 0 else np.nan
    mean_first = wmean(first[reached], w[reached])
    return reach_rate, mean_first


def main():
    print("=" * 92)
    print("CONTENT SANITY CHECK  (regC, EU-pooled, COEFFY-weighted)")
    print("=" * 92)
    hdr = f"{'flow':9} {'program':12} | {'reach%':>7} {'skills→1st':>10} {'never%':>7} | {'income% (KEEP)':>14}"
    for scenario in ["at_risk", "shortage"]:
        print(f"\n--- {scenario}  (journey 0..{JOURNEY[scenario]}) ---")
        print(hdr); print("-" * 92)
        last = JOURNEY[scenario]
        for prog in ORDER:
            sim_name, label = PROG[prog]
            per = load(scenario, sim_name, True)
            if per is None:
                print(f"{scenario:9} {label:12} | MISSING"); continue
            df = pooled(per)
            rr, mf = reach_and_first(df, last)
            # income, KEEP only
            keep = df[df["__country"].isin(INCOME_KEEP)]
            inc = wmean(keep["earnings_delta_closest_switch_pct"], keep["COEFFY"]) if "earnings_delta_closest_switch_pct" in df.columns else np.nan
            print(f"{scenario:9} {label:12} | {rr*100:6.1f}% {mf:10.2f} {(1-rr)*100:6.1f}% | {inc:13.2f}%")
    # income DROP-list audit (must never be reported)
    drop = sorted(set().union(*[set(load("at_risk", PROG[p][0], True).keys()) for p in ORDER]) - INCOME_KEEP)
    print(f"\n[income audit] KEEP={sorted(INCOME_KEEP)}")
    print(f"[income audit] EXCLUDED from income (DROP-list, regC): {drop}")

    # ---- Task A: transferable at_risk intensity vs SEP25 published ----
    print("\n" + "=" * 92)
    print("TASK A CHECK — transferable (coreRanked) at_risk intensity, current vs SEP25 published")
    print("(intensity = COEFFY-weighted mean n_viable_transitions at the step; vs-published")
    print(" approximates the journey-aware effect since Task B does not change viable-counts)")
    print("=" * 92)
    cur = load("at_risk", "reskill-coreRanked", True)
    ref = load("at_risk", "reskill-coreRanked", True, root=SEP25)
    if ref is None:
        print("  SEP25 baseline not found — skipping Task-A delta (only absolute levels shown).")
    print(f"{'program':12} {'step':>4} {'current':>9} {'SEP25':>9} {'Δ%':>8}")
    for prog in ORDER:
        c = load("at_risk", PROG[prog][0], True)
        r = load("at_risk", PROG[prog][0], True, root=SEP25)
        cd = pooled(c)
        rd = pooled(r) if r is not None else None
        for step in (12, 20):
            col = f"n_viable_transitions_step_{step}"
            cv = wmean(cd[col], cd["COEFFY"]) if col in cd.columns else np.nan
            rv = wmean(rd[col], rd["COEFFY"]) if (rd is not None and col in rd.columns) else np.nan
            d = (cv - rv) / rv * 100 if rv and not np.isnan(rv) else np.nan
            print(f"{PROG[prog][1]:12} {step:>4} {cv:9.3f} {rv:9.3f} {d:+7.1f}%")


if __name__ == "__main__":
    main()
