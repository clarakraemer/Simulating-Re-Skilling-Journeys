"""Three analyses on the local extra-leg pickles (no rerun):
  (1) Task-D weight sensitivity: shortage reach + skills-to-first, {tailored,transferable}
      x {0(_optw0), 0.5(production), 1.0(_optw1)}, gap = tailored - transferable.
  (2) Employment-share on(production) vs off(_dwoff): chosen-destination employment share,
      per flow, ratio (the ~1.8x claim).
  (3) Clean correction: production(on) vs _jaoff(journey_aware off), at_risk EU-regC,
      transferable n_viable intensity at steps 12/20 (expect ~ -2.4% / +2.2%); others ~0.
"""
import os, pickle, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "results", "figures", "reskilling_simulation")
SIM = {"optimal": "reskill-optimal", "coreness_ranked": "reskill-coreRanked",
       "green": "reskill-green", "digital": "reskill-digital"}
JL = {"at_risk": 20, "shortage": 30}


def load(sc, prog, suffix=""):
    tag = f"{SIM[prog]}_wage-opt_regC_2023{suffix}"
    p = os.path.join(B, sc, tag, f"{tag}.pkl")
    return pickle.load(open(p, "rb"))[sc] if os.path.exists(p) else None


def concat(per):
    return pd.concat(list(per.values()))


def wmean(v, w):
    v = np.asarray(v, float); w = np.asarray(w, float); m = ~np.isnan(v) & ~np.isnan(w)
    return np.average(v[m], weights=w[m]) if m.any() and w[m].sum() > 0 else np.nan


def reach_first(df, last):
    w = df["COEFFY"].to_numpy(float)
    cols = [f"transition_viable_step_{s}" for s in range(last + 1) if f"transition_viable_step_{s}" in df.columns]
    V = np.nan_to_num(df[cols].astype(float).to_numpy(), nan=0) > 0.5
    reached = V.any(1); fs = V.argmax(1).astype(float); fs[~reached] = np.nan
    return (w[reached].sum() / w.sum() * 100), wmean(fs[reached], w[reached]), reached, V


# ---------------------------------------------------------------- (1) Task D
def task_d():
    print("=" * 90)
    print("(1) TASK-D weight sensitivity — shortage (inward), regC, COEFFY-weighted")
    print("=" * 90)
    legs = [("0.0", "_optw0"), ("0.5", ""), ("1.0", "_optw1")]
    print(f"{'weight':>6} | {'tailored reach/1st':>20} {'transf reach/1st':>18} | {'gap 1st (T-X)':>13}")
    for wlab, suf in legs:
        t = concat(load("shortage", "optimal", suf)); x = concat(load("shortage", "coreness_ranked", suf))
        tr, t1, *_ = reach_first(t, 30); xr, x1, *_ = reach_first(x, 30)
        print(f"{wlab:>6} | {tr:6.1f}% {t1:11.1f}   {xr:6.1f}% {x1:9.1f}   | {t1 - x1:+13.1f}")
    print("\n  green/digital inward reach (must stay << tailored/transferable at every weight):")
    for wlab, suf in legs:
        g = reach_first(concat(load("shortage", "green", suf)), 30)[0]
        d = reach_first(concat(load("shortage", "digital", suf)), 30)[0]
        print(f"   weight {wlab}: green {g:5.1f}%   digital {d:5.1f}%")


# ---------------------------------------------------------------- (2) employment-share
def emp_share(rp):
    print("\n" + "=" * 90)
    print("(2) EMPLOYMENT-SHARE of chosen destinations — on(production) vs off(_dwoff), regC")
    print("=" * 90)
    jobs = rp.jobs_by_country_and_region().copy()
    jobs["ISCO"] = jobs["ISCO08_3D"].astype(str).str.zfill(3)
    emp = jobs.groupby(["NUTS_ID", "ISCO"])["COEFFY"].sum()
    regtot = jobs.groupby("NUTS_ID")["COEFFY"].sum()

    def mean_share(df, last):
        _, _, reached, V = reach_first(df, last)
        fs = V.argmax(1)
        nuts = df.index.get_level_values("NUTS_ID")
        w = df["COEFFY"].to_numpy(float)
        sh = np.full(len(df), np.nan)
        for i in range(len(df)):
            if reached[i]:
                code = df.iloc[i].get(f"transition_target_code_step_{fs[i]}")
                if pd.notna(code):
                    iso = str(code).split(".")[0].zfill(3); r = nuts[i]
                    e = emp.get((r, iso), np.nan); tt = regtot.get(r, np.nan)
                    if tt and not np.isnan(e):
                        sh[i] = e / tt
        return wmean(sh, w)

    print(f"{'flow':9} {'program':12} {'on share':>9} {'off share':>10} {'ratio on/off':>13}")
    for sc in ["at_risk", "shortage"]:
        ons, offs = [], []
        for prog in ["optimal", "coreness_ranked", "green", "digital"]:
            pon = load(sc, prog, ""); poff = load(sc, prog, "_dwoff")
            if pon is None or poff is None:
                continue
            son = mean_share(concat(pon), JL[sc]); soff = mean_share(concat(poff), JL[sc])
            ons.append(son); offs.append(soff)
            r = son / soff if soff else np.nan
            print(f"{sc:9} {prog:12} {son:9.4f} {soff:10.4f} {r:12.2f}x")
        ratio = np.nanmean(ons) / np.nanmean(offs) if np.nanmean(offs) else np.nan
        print(f"{sc:9} {'POOLED':12} {np.nanmean(ons):9.4f} {np.nanmean(offs):10.4f} {ratio:12.2f}x  <==")


# ---------------------------------------------------------------- (3) clean correction
def correction():
    print("\n" + "=" * 90)
    print("(3) CLEAN CORRECTION — production(journey_aware ON) vs _jaoff(OFF), at_risk EU-regC")
    print("=" * 90)
    print(f"{'program':12} {'step':>4} {'ON':>8} {'OFF':>8} {'Δ%':>8}")
    for prog in ["coreness_ranked", "optimal", "green", "digital"]:
        on = concat(load("at_risk", prog, "")); off = concat(load("at_risk", prog, "_jaoff"))
        lab = {"coreness_ranked": "transferable", "optimal": "tailored"}.get(prog, prog)
        for step in (12, 20):
            col = f"n_viable_transitions_step_{step}"
            o = wmean(on[col], on["COEFFY"]); f = wmean(off[col], off["COEFFY"])
            d = (o - f) / f * 100 if f else np.nan
            print(f"{lab:12} {step:>4} {o:8.3f} {f:8.3f} {d:+7.1f}%")


if __name__ == "__main__":
    task_d()
    import sys
    sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
    from src.modelling.reskilling import ReskillingPathways
    from revision.run_sample import assemble_lfs_data
    rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc", lfs_data=assemble_lfs_data(), year=2023)
    emp_share(rp)
    correction()
