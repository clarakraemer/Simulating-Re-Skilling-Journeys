"""Income from the weight-0.5 production pickles. No rerun. KEEP-list only.

FIX: weight by the AT-RISK pool, not full employment. Per-cell income (pct/EUR) is correct
in the pickle; only the cross-cell rollup weight was wrong. The model's own
earnings_delta_closest_switch_sum_step_N already encodes the at-risk weighting, so the
per-cell at-risk population is Wc = sum_step / delta_step (step-invariant), recovered here.

Reports (KEEP-only, at-risk-weighted):
  (1) COMMON-STEP % (switchers-at-step) at steps 4/8/12/16/20, per program x flow  [LEAD]
  (2) FIRST-TRANSITION %, abs EUR, base EUR, per program x flow
  (3) CONVERGENCE tailored vs transferable (crossover step)
  (4) PER-COUNTRY aggregate EUR at first transition (fiscal scale), via the _sum column
Asserts COUNTRYW subset of KEEP (no DROP leak).
"""
import os, pickle, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASEDIR = os.path.join(ROOT, "results", "figures", "reskilling_simulation")
KEEP = {"SE", "CH", "SK", "DK", "FR", "EE", "EL", "IE", "NO", "FI", "PT", "LU"}
PROG = [("reskill-optimal", "tailored"), ("reskill-coreRanked", "transferable"),
        ("reskill-green", "green"), ("reskill-digital", "digital")]
JL = {"at_risk": 20, "shortage": 30}


def load(sc, sim):
    tag = f"{sim}_wage-opt_regC_2023"
    return pickle.load(open(os.path.join(BASEDIR, sc, tag, f"{tag}.pkl"), "rb"))[sc]


def keep_concat(per):
    df = pd.concat([d[d.index.get_level_values("COUNTRYW").isin(KEEP)] for d in per.values()])
    assert set(df.index.get_level_values("COUNTRYW")) <= KEEP, "DROP LEAK"
    return df


def atrisk_weight(df, last):
    """Wc = at-risk population per cell = median_step(sum_step/delta_step) (step-invariant).
    Fallback for all-zero-delta cells: COEFFY x global median at-risk share."""
    coeffy = df["COEFFY"].to_numpy(float)
    ratios = []
    for s in range(last + 1):
        de = df.get(f"earnings_delta_closest_switch_step_{s}")
        su = df.get(f"earnings_delta_closest_switch_sum_step_{s}")
        if de is None or su is None:
            continue
        de = de.to_numpy(float); su = su.to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(np.abs(de) > 1e-9, su / de, np.nan)
        ratios.append(r)
    with np.errstate(invalid="ignore"):
        Wc = np.nanmedian(np.vstack(ratios), axis=0) if ratios else np.full(len(df), np.nan)
        share = np.nanmedian(Wc / coeffy) if np.isfinite(Wc).any() else np.nan
    Wc = np.where(np.isnan(Wc), coeffy * share, Wc)
    return Wc


def wmean(v, w):
    v = np.asarray(v, float); w = np.asarray(w, float); m = ~np.isnan(v) & ~np.isnan(w)
    return np.average(v[m], weights=w[m]) if m.any() and w[m].sum() > 0 else np.nan


def viable_at(df, s):
    c = f"transition_viable_step_{s}"
    return np.nan_to_num(df[c].astype(float).to_numpy(), nan=0) > 0.5 if c in df else np.zeros(len(df), bool)


def first_step(df, last):
    V = np.nan_to_num(df[[f"transition_viable_step_{s}" for s in range(last + 1)
                          if f"transition_viable_step_{s}" in df.columns]].astype(float).to_numpy(), nan=0) > 0.5
    return V.any(axis=1), V.argmax(axis=1)


def part1_common_step():
    # flow-specific common steps: outward saturates early; inward unlocks ~step 24-26.
    STEPS = {"at_risk": [4, 8, 12, 16, 20], "shortage": [16, 20, 24, 28, 30]}
    print("=" * 100)
    print("(1) COMMON-STEP income % — switchers-at-step, at-risk-weighted, KEEP  [LEAD: fair constant-effort]")
    print("=" * 100)
    for sc in ["at_risk", "shortage"]:
        last = JL[sc]
        steps = [s for s in STEPS[sc] if s <= last]
        print(f"\n  {sc}:   " + "  ".join(f"step{ s:>2}" for s in steps))
        for sim, lab in PROG:
            df = keep_concat(load(sc, sim)); Wc = atrisk_weight(df, last)
            cells = []
            for s in steps:
                vi = viable_at(df, s)
                pc = df.get(f"earnings_delta_closest_switch_pct_step_{s}")
                v = pc.to_numpy(float) if pc is not None else np.full(len(df), np.nan)
                cells.append(wmean(v[vi], Wc[vi]) * 100)
            print(f"  {lab:12} " + "  ".join(f"{x:6.1f}%" for x in cells))
    print("\n  outward: tailored cushions best per unit of effort through ~step 12 (see convergence).")


def part2_first_transition():
    print("\n" + "=" * 100)
    print("(2) FIRST-TRANSITION income — at-risk-weighted, KEEP  (each program at its OWN first-unlock step)")
    print("=" * 100)
    print(f"{'flow':9} {'program':12} | {'% change':>9} {'abs EUR':>10} {'base EUR':>9}")
    for sc in ["at_risk", "shortage"]:
        last = JL[sc]
        for sim, lab in PROG:
            df = keep_concat(load(sc, sim)); Wc = atrisk_weight(df, last)
            reached, fs = first_step(df, last)
            pct = np.array([df.iloc[i].get(f"earnings_delta_closest_switch_pct_step_{fs[i]}", np.nan)
                            if reached[i] else np.nan for i in range(len(df))], float)
            eur = np.array([df.iloc[i].get(f"earnings_delta_closest_switch_step_{fs[i]}", np.nan)
                            if reached[i] else np.nan for i in range(len(df))], float)
            base = df["annual_earnings"].to_numpy(float)
            m = reached
            print(f"{sc:9} {lab:12} | {wmean(pct[m], Wc[m])*100:8.1f}% {wmean(eur[m], Wc[m]):10.0f} {wmean(base[m], Wc[m]):9.0f}")
    print("  speed/income tradeoff: tailored switches soonest to a nearer lower-paid job (smaller")
    print("  gain here); transferable reskills longer (~13 vs ~4 skills) and lands better-paid.")


def part3_convergence():
    print("\n" + "=" * 100)
    print("(3) CONVERGENCE — at_risk switcher income %, tailored vs transferable, at-risk-weighted")
    print("=" * 100)
    last = JL["at_risk"]
    dt = keep_concat(load("at_risk", "reskill-optimal")); Wt = atrisk_weight(dt, last)
    dx = keep_concat(load("at_risk", "reskill-coreRanked")); Wx = atrisk_weight(dx, last)
    print(f"{'step':>4} {'tailored%':>10} {'transf%':>9} {'gap pp':>7}")
    prev = None; cross = None
    for s in range(1, last + 1):
        t = wmean(dt.get(f"earnings_delta_closest_switch_pct_step_{s}").to_numpy(float)[viable_at(dt, s)], Wt[viable_at(dt, s)]) * 100
        x = wmean(dx.get(f"earnings_delta_closest_switch_pct_step_{s}").to_numpy(float)[viable_at(dx, s)], Wx[viable_at(dx, s)]) * 100
        gap = t - x
        if prev is not None and prev > 0 >= gap and cross is None:
            cross = s
        prev = gap
        if s in (4, 8, 12, 13, 14, 16, 20):
            print(f"{s:>4} {t:10.1f} {x:9.1f} {gap:+7.1f}")
    print(f"  -> tailored advantage (gap>0) flips sign at step {cross} (old draft: 13). Front-loaded.")


def part4_country_aggregate():
    print("\n" + "=" * 100)
    print("(4) AGGREGATE EUR at first transition, per country (at_risk, KEEP) — via model _sum column")
    print("=" * 100)
    for sim, lab in [("reskill-optimal", "tailored"), ("reskill-coreRanked", "transferable")]:
        per = load("at_risk", sim); rows = []
        for c, d in per.items():
            if c not in KEEP:
                continue
            reached, fs = first_step(d, JL["at_risk"])
            tot = sum(d.iloc[i].get(f"earnings_delta_closest_switch_sum_step_{fs[i]}", 0.0)
                      for i in range(len(d)) if reached[i]
                      and pd.notna(d.iloc[i].get(f"earnings_delta_closest_switch_sum_step_{fs[i]}", np.nan)))
            rows.append((c, tot))
        rows.sort(key=lambda r: r[1])
        print(f"\n  {lab} (most negative first):  " +
              "   ".join(f"{c} {tot/1e9:+.2f}bn" for c, tot in rows[:6]))


if __name__ == "__main__":
    part1_common_step()
    part2_first_transition()
    part3_convergence()
    part4_country_aggregate()
    print("\n[OK] at-risk weighted via model _sum column; COUNTRYW subset of KEEP (no DROP leak).")
    print("[note] above_current is a SOFT preference (not a hard floor): some switchers still lose.")
