"""Task B headline — employment-share of REALIZED destinations, post-processed from the
production (weight-0.5) pickles. NO model rerun.

For every worker-group that reaches a transition, take the destination it actually moved to
(target occupation at its FIRST viable step), look up that occupation's share of NUTS-2
employment (region from the row's index), and report the COEFFY-weighted mean destination
employment share, per flow and per program.

IMPORTANT — the baseline caveat (read before quoting a ratio):
  The pickles store ONLY the Task-B-chosen destination, not the feasible set and not the
  income-max ("no-weighting") alternative. So the literal "vs no-weighting" ratio (the ~1.8x
  from the sample capture, off vs share) CANNOT be reconstructed from these pickles. What we
  CAN compute without a rerun is the realized share vs a REGION-AVERAGE baseline:
     * uniform     : 1 / n_occupations_in_region  (a destination picked ignoring employment)
     * empweighted : sum(share^2)                 (where employment actually concentrates)
  These answer "how much does employment-weighting concentrate destinations in larger
  occupations", which is RELATED to but NOT the income-max counterfactual. To get the exact
  income-max ratio at production scale, run a destination_weighting='off' leg (a rerun).
"""
import os, pickle, numpy as np, pandas as pd, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from src.modelling.reskilling import ReskillingPathways  # noqa
from revision.run_sample import assemble_lfs_data  # noqa

BASEDIR = os.path.join(ROOT, "results", "figures", "reskilling_simulation")
PROG = [("reskill-optimal", "tailored"), ("reskill-coreRanked", "transferable"),
        ("reskill-green", "green"), ("reskill-digital", "digital")]
JL = {"at_risk": 20, "shortage": 30}


def load(sc, sim, regc=True):
    tag = f"{sim}_wage-opt_{'regC' if regc else 'no-regC'}_2023"
    p = os.path.join(BASEDIR, sc, tag, f"{tag}.pkl")
    return pickle.load(open(p, "rb"))[sc] if os.path.exists(p) else None


def region_share_tables(rp):
    """Return per-region: dict region -> {isco3: share}, plus uniform & empweighted baselines."""
    jobs = rp.jobs_by_country_and_region()
    jobs = jobs.copy()
    jobs["ISCO08_3D"] = jobs["ISCO08_3D"].astype(str)
    share, uniform, empw = {}, {}, {}
    for reg, g in jobs.groupby("NUTS_ID"):
        tot = g["COEFFY"].sum()
        s = (g.set_index("ISCO08_3D")["COEFFY"] / tot) if tot > 0 else g.set_index("ISCO08_3D")["COEFFY"] * 0
        share[reg] = s.to_dict()
        uniform[reg] = 1.0 / len(s) if len(s) else np.nan
        empw[reg] = float((s.values ** 2).sum())
    return share, uniform, empw


def first_target(df, last):
    """Per row: (reached?, destination isco3 code at first viable step)."""
    vcols = [f"transition_viable_step_{s}" for s in range(last + 1) if f"transition_viable_step_{s}" in df.columns]
    tcols = [f"transition_target_code_step_{s}" for s in range(last + 1)]
    V = np.nan_to_num(df[vcols].astype(float).to_numpy(), nan=0) > 0.5
    reached = V.any(axis=1)
    firststep = V.argmax(axis=1)
    codes = []
    for i, (r, fs) in enumerate(zip(reached, firststep)):
        if not r:
            codes.append(None); continue
        col = f"transition_target_code_step_{fs}"
        codes.append(str(df.iloc[i][col]) if col in df.columns and pd.notna(df.iloc[i][col]) else None)
    return reached, codes


def wmean(v, w):
    v = np.asarray(v, float); w = np.asarray(w, float); m = ~np.isnan(v) & ~np.isnan(w)
    return np.average(v[m], weights=w[m]) if m.any() and w[m].sum() > 0 else np.nan


def main():
    rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc",
                            lfs_data=assemble_lfs_data(), year=2023)
    share, uniform, empw = region_share_tables(rp)
    print("=" * 92)
    print("TASK B — realized destination employment share (regC, COEFFY-weighted over reachers)")
    print("baseline = region-average occupation share (NOT the income-max counterfactual; see header)")
    print("=" * 92)
    print(f"{'flow':9} {'program':12} | {'chosen share':>12} {'uniform base':>12} {'ratio×':>7} | {'empwt base':>10} {'ratio×':>7}")
    for sc in ["at_risk", "shortage"]:
        last = JL[sc]
        for sim, lab in PROG:
            per = load(sc, sim)
            if per is None:
                continue
            cs, ub, eb, ws = [], [], [], []
            for c, d in per.items():
                regs = d.index.get_level_values("NUTS_ID")
                reached, codes = first_target(d, last)
                w = d["COEFFY"].to_numpy(float)
                for i in range(len(d)):
                    if not reached[i] or codes[i] is None:
                        continue
                    reg = regs[i]
                    cs.append(share.get(reg, {}).get(codes[i], 0.0))
                    ub.append(uniform.get(reg, np.nan))
                    eb.append(empw.get(reg, np.nan))
                    ws.append(w[i])
            if not cs:
                print(f"{sc:9} {lab:12} | (no reachers)"); continue
            mc, mu, me = wmean(cs, ws), wmean(ub, ws), wmean(eb, ws)
            print(f"{sc:9} {lab:12} | {mc:12.4f} {mu:12.4f} {mc/mu:7.2f} | {me:10.4f} {mc/me:7.2f}")
    print("\nNOTE: 'ratio×' here is realized-vs-average-occupation, a no-rerun proxy. The literal")
    print("income-max 'no-weighting' ratio (sample ~1.8x) needs a destination_weighting='off' run.")


if __name__ == "__main__":
    main()
