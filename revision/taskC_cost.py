"""Task C — program cost post-processing (scaffold; euro values are PLACEHOLDERS).

Reads per-program simulation outputs, derives steps-to-first-transition per participant
(a JOURNEY-COUNT quantity: first step where transition_viable_step_*==True), and combines
it with per-participant cost bands to produce a small cost table for the Results/SI.

FRAMING (maintainer-decided):
  * Flows: report BOTH, lead with inward (shortage). The cheap-exit (outward ~0 skills) vs
    expensive-entry (inward ~23 skills) contrast is the inward/outward asymmetry, in euros.
  * Cost: two separate columns — €/participant (spend) and %-reached (yield) — NOT a blended
    "€/reached transition" headline (that metric explodes for low-reach programs: it is a
    0.2%-denominator artifact for green and undefined for digital). A secondary
    €/reached-transition is shown ONLY for high-reach programs (>= REACH_THRESHOLD), where the
    denominator is meaningful. Cleaner answer to the R2 ROI ask: spend vs yield, not
    denominator-driven.
  * Step-0: cost is conditioned on participants who NEED >=1 skill; the share who reach a
    transition with ZERO skills is reported separately as context (it conflates "cheap
    reskilling" with "no reskilling needed" if folded in — and the zero-skill share is itself
    a result, especially for outward).

INHERITANCE NOTE (income filter): uses ONLY journey-count columns (transition_viable_step_*)
and the population weight COEFFY — never an income/earnings column (enforced by
assert_no_income_columns_used). So it is independent of the income-eligibility (KEEP-list)
filter and needs no country exclusion. If a future cost band were earnings-derived, that band
WOULD require the KEEP list (which until Phase 3 lives only in notebook-06) — apply explicitly.
"""
import os, re, argparse, numpy as np, pandas as pd

# ======================================================================================
#  COST BANDS — PLACEHOLDERS. DO NOT CITE. €/skill acquired per participant. Maintainer
#  supplies real values (German training voucher, Spain, OECD, WEF). The repeating-digit
#  numbers are deliberately, unmistakably fake so no scaffold run is read as real.
# ======================================================================================
COST_BANDS_EUR_PER_SKILL = {
    "German_voucher": 1111.0,   # PLACEHOLDER
    "Spain":          2222.0,   # PLACEHOLDER
    "OECD":           3333.0,   # PLACEHOLDER
    "WEF":            4444.0,   # PLACEHOLDER
}
PLACEHOLDER = True                 # flip to False only when real bands are filled in
COUNT_NONREACHER_COST = True       # non-reachers invest the full journey (counts in spend)
REACH_THRESHOLD = 0.50             # show secondary €/reached only above this reach rate

PROGRAMS = [("transferable", "coreRanked"), ("green", "green"),
            ("digital", "digital"), ("tailored", "optimal")]
FLOWS_IN_ORDER = [("shortage", "inward"), ("at_risk", "outward")]   # lead with inward
INCOME_TOKENS = ("earn", "income", "wage")

def assert_no_income_columns_used(cols):
    bad = [c for c in cols if any(t in c.lower() for t in INCOME_TOKENS)]
    assert not bad, f"Task C must not read income columns, but used: {bad}"

def steps_to_first(df):
    tv = sorted([c for c in df.columns if re.fullmatch(r"transition_viable_step_\d+", c)],
                key=lambda c: int(c.split("_")[-1]))
    assert_no_income_columns_used(tv + ["COEFFY"])
    steps_idx = [int(c.split("_")[-1]) for c in tv]
    M = df[tv].to_numpy()
    first = np.full(len(df), np.nan)
    for j, s in enumerate(steps_idx):
        hit = M[:, j].astype(bool) & np.isnan(first)
        first[hit] = s
    return first, df["COEFFY"].to_numpy(float), max(steps_idx)

def wsum(w, mask=None):
    return float(np.nansum(w if mask is None else w[mask]))

def wmean(v, w):
    m = ~np.isnan(v) & ~np.isnan(w)
    return float(np.average(v[m], weights=w[m])) if m.any() and w[m].sum() > 0 else np.nan

def analyse_program(path, flow):
    if not os.path.exists(path):
        return None
    res = pd.read_pickle(path)[flow]
    fa, wa, J = [], [], 0
    for _, df in res.items():
        f, w, j = steps_to_first(df); J = max(J, j)
        fa.append(f); wa.append(w)
    first = np.concatenate(fa); w = np.concatenate(wa)
    N = wsum(w)
    step0 = wsum(w, first == 0)                       # reach with zero skills (context)
    subset = first != 0                               # need >=1 skill (NaN counts as needing)
    sub_w = wsum(w, subset)
    reached1 = first > 0                              # reach with >=1 skill
    # skills invested within the >=1-skill subset: reachers -> steps; non-reachers -> full J
    invested = np.where(reached1, first, J if COUNT_NONREACHER_COST else np.nan)
    invested_sub_w = float(np.nansum((invested * w)[subset]))
    return {
        "J": J, "N": N,
        "step0_share": step0 / N if N else np.nan,
        "pct_reached_ge1": wsum(w, reached1) / sub_w if sub_w else np.nan,
        "mean_steps_ge1": wmean(np.where(reached1, first, np.nan), w),
        "mean_invested_per_participant": invested_sub_w / sub_w if sub_w else np.nan,
        "invested_sub_w": invested_sub_w,
        "reached1_w": wsum(w, reached1),
    }

def print_flow(base, flow, f_label, reg):
    print(f"\n================  {f_label.upper()} ({flow}) / {reg}  ================")
    data = {lab: analyse_program(os.path.join(base, flow, f"reskill-{tg}_wage-opt_{reg}_2023",
                                              f"reskill-{tg}_wage-opt_{reg}_2023.pkl"), flow)
            for lab, tg in PROGRAMS}
    bands = COST_BANDS_EUR_PER_SKILL
    # Table A: spend + yield
    h = f"{'program':12} {'0-skill%':>8} {'%reach(≥1)':>10} {'mean sk(≥1)':>11} |" + \
        "".join(f" {b[:9]:>9}" for b in bands) + "   €/participant"
    print(h); print("-" * len(h))
    for lab, _ in PROGRAMS:
        d = data[lab]
        if d is None: print(f"{lab:12}  (artifact missing)"); continue
        row = f"{lab:12} {d['step0_share']*100:7.1f}% {d['pct_reached_ge1']*100:9.1f}% {d['mean_steps_ge1']:11.2f} |"
        for b, band in bands.items():
            row += f" {d['mean_invested_per_participant']*band:9,.0f}"
        print(row)
    # Table B: secondary €/reached-transition, high-reach programs only
    print(f"\n  secondary — €/reached-transition (only programs with reach ≥ {REACH_THRESHOLD*100:.0f}%):")
    for lab, _ in PROGRAMS:
        d = data[lab]
        if d is None: continue
        if (d["pct_reached_ge1"] or 0) >= REACH_THRESHOLD and d["reached1_w"] > 0:
            cells = "  ".join(f"{b}: {d['invested_sub_w']*band/d['reached1_w']:,.0f}"
                              for b, band in bands.items())
            print(f"    {lab:12} {cells}")
        else:
            print(f"    {lab:12} — low reach ({(d['pct_reached_ge1'] or 0)*100:.1f}%), denominator not meaningful")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="../review-copy/results/figures/reskilling_simulation/SEP25")
    ap.add_argument("--reg", default="no-regC", choices=["regC", "no-regC"])
    a = ap.parse_args()
    print("################  TASK C COST TABLE  ################")
    if PLACEHOLDER:
        print("***** EURO VALUES ARE PLACEHOLDERS (1111/2222/3333/4444) — DO NOT CITE *****")
    for flow, fl in FLOWS_IN_ORDER:
        print_flow(a.base, flow, fl, a.reg)
    print("\nnotes: €/participant = band × mean skills invested over the ≥1-skill subset "
          "(reachers: steps-to-first; non-reachers: full journey).")
    print("       0-skill% = share reaching a transition with no reskilling (context, not in cost).")
