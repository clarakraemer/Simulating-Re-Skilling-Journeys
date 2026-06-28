"""Task D SI table — does the inward tailored-vs-transferable gap close/hold/reverse with the
optional-skill weight? Run AFTER the sweep lands (weights 0 and 1.0, shortage, regC).

Reads the weight-0.5 production pickles and the _optw0/_optw1 variant pickles, and tabulates,
for shortage (inward), regC, EU-pooled COEFFY-weighted:
    reach %  and  skills-to-first-transition
for {tailored, transferable} x {0.0, 0.5, 1.0}. Prints the tailored-minus-transferable gap
per weight so the SI can state the trend in one line.

    python revision/taskD_compare.py
"""
import os, pickle, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASEDIR = os.path.join(ROOT, "results", "figures", "reskilling_simulation")
SCEN, LAST, REGC = "shortage", 30, True
PROG = [("reskill-optimal", "tailored"), ("reskill-coreRanked", "transferable")]
WEIGHTS = [0.0, 0.5, 1.0]


def variant(weight):
    if abs(weight - 0.5) < 1e-12:
        return ""
    return "_optw{:g}".format(weight)


def load(sim, weight):
    tag = f"{sim}_wage-opt_{'regC' if REGC else 'no-regC'}_2023" + variant(weight)
    p = os.path.join(BASEDIR, SCEN, tag, f"{tag}.pkl")
    return pickle.load(open(p, "rb"))[SCEN] if os.path.exists(p) else None


def metrics(per):
    """EU-pooled COEFFY-weighted reach% and mean skills-to-first over reachers."""
    df = pd.concat(per.values(), ignore_index=True)
    w = df["COEFFY"].to_numpy(float)
    cols = [f"transition_viable_step_{s}" for s in range(LAST + 1) if f"transition_viable_step_{s}" in df.columns]
    V = np.nan_to_num(df[cols].astype(float).to_numpy(), nan=0) > 0.5
    reached = V.any(axis=1)
    first = V.argmax(axis=1).astype(float); first[~reached] = np.nan
    reach = w[reached].sum() / w.sum() if w.sum() else np.nan
    mf = np.average(first[reached], weights=w[reached]) if reached.any() else np.nan
    return reach * 100, mf


def main():
    print(f"TASK D — inward (shortage, regC) weight sweep: reach% and skills-to-first\n")
    print(f"{'weight':>6} | {'tailored reach':>14} {'transf reach':>13} | "
          f"{'tailored→1st':>13} {'transf→1st':>11} {'gap(t−x)':>9}")
    print("-" * 78)
    missing = []
    for wt in WEIGHTS:
        pt, px = load(PROG[0][0], wt), load(PROG[1][0], wt)
        if pt is None or px is None:
            missing.append(wt)
            print(f"{wt:6} | (pickles not found — run the sweep for this weight)")
            continue
        rt, ft = metrics(pt); rx, fx = metrics(px)
        print(f"{wt:6} | {rt:13.1f}% {rx:12.1f}% | {ft:13.2f} {fx:11.2f} {ft-fx:+9.2f}")
    print("-" * 78)
    print("gap(t−x) = tailored minus transferable skills-to-first. Positive = transferable")
    print("reaches inward sooner (the weight-0.5 finding). Watch whether |gap| shrinks toward 0")
    print("(closes), stays (holds), or flips sign (reverses) as weight goes 0 -> 1.")
    if missing:
        print(f"\n[pending] weights with no pickles yet: {missing} — re-run after the sweep completes.")


if __name__ == "__main__":
    main()
