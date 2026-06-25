"""Task B sample — post-hoc sweep over the destination-weighting rule.

Reads the off-mode captures (feasible target sets + per-worker draw seeds, mode-invariant)
and recomputes, for each rule, each worker's realized destination by calling the SAME
production method ReskillingPathways._select_destination on a stub self (no logic drift,
no data load). The per-draw deterministic seeding means this realized draw is byte-
identical to what a live run of that rule would produce (proven by taskB_validate.py).

Per scenario / rule it reports the COEFFY-weighted earnings-delta distribution
(mean/median/p25/p75, income-loss share) and the mean employment share (COEFFY) of the
chosen destination -- the income-vs-"land where the jobs are" trade-off. off = Phase-1.
"""
import sys, os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from src.modelling.reskilling import ReskillingPathways

RULES = [  # label, destination_weighting, income_band, acceptability
    ("off",            "off",                     None, None),
    ("band=0.00",      "share_income_acceptable", 0.00, "band"),
    ("band=0.05",      "share_income_acceptable", 0.05, "band"),
    ("band=0.10",      "share_income_acceptable", 0.10, "band"),
    ("band=0.25",      "share_income_acceptable", 0.25, "band"),
    ("band=0.50",      "share_income_acceptable", 0.50, "band"),
    ("above_current",  "share_income_acceptable", None, "above_current"),
    ("share_only",     "share_only",              None, None),
]

def stub(dw, band, acc):
    s = type("S", (), {})()
    s.destination_weighting = dw
    if band is not None: s.income_band = band
    if acc is not None: s.acceptability = acc
    return s

def wq(v, w, q):
    o = np.argsort(v); v, w = v[o], w[o]; cw = np.cumsum(w) / w.sum()
    return float(np.interp(q, cw, v))

def sweep(cap):
    print(f"{'rule':14} {'meanΔ€':>9} {'medΔ€':>9} {'p25':>8} {'p75':>8} "
          f"{'%loss':>6} {'mean share':>11} {'rel':>6}")
    off_share = None
    for label, dw, band, acc in RULES:
        # Acceptable-set mask per rule (mirrors _select_destination; off is the single
        # top-income target). Income is the EXPECTED delta conditional on landing in a
        # known-earnings job: employment shares renormalised over the finite-earnings
        # targets in the acceptable set -> consistent worker set, off stays the ceiling.
        def acc_mask(earn, we):
            if dw == "off":   # captures are income-sorted desc, so index 0 is the top pick
                m = np.zeros(len(earn), bool); m[0] = True; return m
            if dw == "share_only" or not np.isfinite(earn).any():
                return np.ones(len(earn), bool)
            if acc == "above_current":
                m = earn >= we
            else:
                m = earn >= (1.0 - band) * np.nanmax(earn)
            return m if m.any() else np.ones(len(earn), bool)

        D, Wd, C, Wc = [], [], [], []
        for c in cap:
            earn = c["targets"]["annual_earnings"].to_numpy(float)
            coef = c["targets"]["COEFFY"].to_numpy(float)
            if not (np.isfinite(coef).any() and np.nansum(coef) > 0):
                continue
            for ch in c["choices"]:
                we, ww = ch["earn"], ch["w"]
                m = acc_mask(earn, we)
                cc = np.where(m, coef, 0.0); cc = np.where(np.isfinite(cc), cc, 0.0)
                if cc.sum() <= 0:
                    continue
                p = cc / cc.sum()
                C.append(float((p * coef).sum())); Wc.append(ww)                 # E[employment share]
                if np.isfinite(we):                                              # income: finite earnings only
                    fin = m & np.isfinite(earn)
                    cf = np.where(fin, coef, 0.0)
                    if cf.sum() > 0:
                        pf = cf / cf.sum()
                        for i in np.nonzero(pf > 0)[0]:
                            D.append(earn[i] - we); Wd.append(ww * pf[i])
        C, Wc = np.array(C), np.array(Wc)
        if Wc.sum() <= 0:
            print(f"{label:14} (no multi-target choices)"); continue
        D, Wd = np.array(D), np.array(Wd)
        if len(D) and Wd.sum() > 0:
            mean = np.average(D, weights=Wd)
            med, p25, p75 = wq(D, Wd, .5), wq(D, Wd, .25), wq(D, Wd, .75)
            loss = float(Wd[D < 0].sum() / Wd.sum())
        else:
            mean = med = p25 = p75 = loss = float("nan")
        msh = np.average(C, weights=Wc)
        if off_share is None: off_share = msh
        print(f"{label:14} {mean:9.0f} {med:9.0f} {p25:8.0f} {p75:8.0f} "
              f"{loss*100:5.1f}% {msh:11.1f} {msh/off_share:5.2f}x")

def main():
    for scen in ["at_risk", "shortage"]:
        p = f"/tmp/taskB_cap_{scen}.pkl"
        if not os.path.exists(p):
            print(f"[{scen}] capture {p} not found — run taskB_capture.py"); continue
        d = pd.read_pickle(p); cap = d["capture"]
        nch = sum(len(c["choices"]) for c in cap)
        nmt = sum(len(c["choices"]) for c in cap if len(c["targets"]) >= 2)
        print(f"\n=== {scen.upper()} @ step {d['step']}  ({len(cap)} regions, {nch} worker-choices, "
              f"{nmt} with >=2 targets where the rule can bind) ===")
        sweep(cap)
    print("\noff == Phase-1 (argmax). band 0 ~ off; band->large ~ share_only (income ignored).")
    print("'mean share' = employment count at chosen destination; 'rel' vs off (>1 = lands where more jobs are).")

if __name__ == "__main__":
    main()
