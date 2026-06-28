"""Income at the FIRST unlocked transition (switchers), from the weight-0.5 production
pickles. No rerun. KEEP-list only, COEFFY-weighted.

Reports, per program x flow:
  (a) % change vs pre-reskilling income, (b) absolute EUR annual change, (c) mean
      pre-reskilling annual income (the base)  -> "-X% (-EURY on EURZ prior)"
Plus: tailored-vs-transferable income convergence step, and per-country aggregate EUR loss
at first transition (fiscal-scale). Asserts COUNTRYW subset of KEEP (no DROP leak).
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
    fr = [d[d.index.get_level_values("COUNTRYW").isin(KEEP)] for d in per.values()]
    df = pd.concat(fr)
    assert set(df.index.get_level_values("COUNTRYW")) <= KEEP, "DROP LEAK"
    return df


def first_step(df, last):
    vcols = [f"transition_viable_step_{s}" for s in range(last + 1) if f"transition_viable_step_{s}" in df.columns]
    V = np.nan_to_num(df[vcols].astype(float).to_numpy(), nan=0) > 0.5
    reached = V.any(axis=1)
    fs = V.argmax(axis=1)
    return reached, fs


def at_first(df, last, col_tmpl):
    """Pull col_tmpl.format(step) at each row's first viable step; NaN for non-reachers."""
    reached, fs = first_step(df, last)
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        if reached[i]:
            col = col_tmpl.format(fs[i])
            if col in df.columns:
                out[i] = df.iloc[i][col]
    return out, reached


def wmean(v, w):
    v = np.asarray(v, float); w = np.asarray(w, float); m = ~np.isnan(v) & ~np.isnan(w)
    return np.average(v[m], weights=w[m]) if m.any() and w[m].sum() > 0 else np.nan


def part1_table():
    print("=" * 96)
    print("(1) INCOME AT FIRST UNLOCKED TRANSITION — switchers, regC, KEEP-only, COEFFY-weighted")
    print("=" * 96)
    print(f"{'flow':9} {'program':12} | {'% change':>9} {'abs EUR':>11} {'base EUR (prior)':>17}")
    for sc in ["at_risk", "shortage"]:
        last = JL[sc]
        for sim, lab in PROG:
            df = keep_concat(load(sc, sim))
            w = df["COEFFY"].to_numpy(float)
            pct, reached = at_first(df, last, "earnings_delta_closest_switch_pct_step_{}")
            eur, _ = at_first(df, last, "earnings_delta_closest_switch_step_{}")
            base = df["annual_earnings"].to_numpy(float)
            m = reached
            p = wmean(pct[m], w[m]) * 100
            e = wmean(eur[m], w[m])
            z = wmean(base[m], w[m])
            print(f"{sc:9} {lab:12} | {p:8.1f}% {e:11.0f} {z:17.0f}")
    print("  (outward = stepping out of at-risk jobs; inward = into low-carbon/keep-job)")


def part2_convergence():
    print("\n" + "=" * 96)
    print("(2) CONVERGENCE — at_risk income %% (switchers-at-step), tailored vs transferable, KEEP")
    print("=" * 96)
    last = JL["at_risk"]
    dt = keep_concat(load("at_risk", "reskill-optimal"))
    dx = keep_concat(load("at_risk", "reskill-coreRanked"))
    print(f"{'step':>4} {'tailored%':>10} {'transf%':>9} {'gap pp':>7}")
    conv = None
    for s in range(last + 1):
        pc = f"earnings_delta_closest_switch_pct_step_{s}"; vc = f"transition_viable_step_{s}"
        def stat(d):
            w = d["COEFFY"].to_numpy(float)
            vi = np.nan_to_num(d[vc].astype(float).to_numpy(), nan=0) > 0.5 if vc in d else np.zeros(len(d), bool)
            v = d[pc].to_numpy(float) if pc in d else np.full(len(d), np.nan)
            return wmean(v[vi], w[vi]) * 100
        t, x = stat(dt), stat(dx)
        gap = t - x
        if s in (0, 4, 8, 10, 12, 13, 14, 16, 20):
            print(f"{s:>4} {t:10.1f} {x:9.1f} {gap:7.1f}")
        if conv is None and s >= 1 and abs(gap) <= 1.0:
            conv = s
    print(f"  -> income gap closes to <=1pp at step {conv} (old draft: 13). Tailored advantage is front-loaded.")


def part3_country_aggregate():
    print("\n" + "=" * 96)
    print("(3) AGGREGATE EUR CHANGE AT FIRST TRANSITION, per country (at_risk, KEEP) — fiscal scale")
    print("=" * 96)
    last = JL["at_risk"]
    for sim, lab in [("reskill-optimal", "tailored"), ("reskill-coreRanked", "transferable")]:
        per = load("at_risk", sim)
        rows = []
        for c, d in per.items():
            if c not in KEEP:
                continue
            reached, fs = first_step(d, last)
            w = d["COEFFY"].to_numpy(float)
            tot = 0.0
            for i in range(len(d)):
                if reached[i]:
                    col = f"earnings_delta_closest_switch_step_{fs[i]}"
                    if col in d.columns and pd.notna(d.iloc[i][col]):
                        tot += d.iloc[i][col] * w[i]
            rows.append((c, tot))
        rows.sort(key=lambda r: r[1])
        print(f"\n  {lab} — aggregate EUR change at first transition (most negative first):")
        for c, tot in rows[:6]:
            print(f"     {c}: {tot/1e9:+.2f} bn  ({tot/1e6:+.0f} m)")


if __name__ == "__main__":
    part1_table()
    part2_convergence()
    part3_country_aggregate()
    print("\n[OK] COUNTRYW subset of KEEP enforced in every aggregation (no DROP leakage).")
