"""Integrity gate for the 16 parallel-run output pickles (run LOCALLY after the scp/rsync).

For each (scenario, program, regC) it asserts the pickle:
  * exists and unpickles to {scenario: {country: df}}
  * has the expected per-flow step columns (n_viable_transitions_step_0..N; N=20 at_risk,
    30 shortage) — the journey-length contract the figures depend on
  * holds the expected country count (27 no-regC / 21 regC after the 6 XX00-only drops)
  * has no entirely-empty / all-NaN result frame

Prints a PASS/FAIL table and exits non-zero if anything fails. No model import, no network.

    python revision/check_outputs.py
"""
import os, pickle, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results", "figures", "reskilling_simulation")
PROGRAMS = {"optimal": "reskill-optimal", "coreness_ranked": "reskill-coreRanked",
            "digital": "reskill-digital", "green": "reskill-green"}
SCENARIOS = ["at_risk", "shortage"]
JOURNEY = {"at_risk": 20, "shortage": 30}
EXPECT_COUNTRIES = {True: 21, False: 27}   # regC drops the 6 XX00-only countries
STEP_PREFIX = "n_viable_transitions_step_"


def path_for(scenario, sim_name, regc):
    tag = f"{sim_name}_wage-opt_{'regC' if regc else 'no-regC'}_2023"
    return os.path.join(BASE, scenario, tag, f"{tag}.pkl"), tag


def check_one(scenario, sim_name, regc):
    """Return (ok: bool, note: str)."""
    path, tag = path_for(scenario, sim_name, regc)
    if not os.path.exists(path):
        return False, "MISSING file"
    try:
        with open(path, "rb") as h:
            obj = pickle.load(h)
    except Exception as e:
        return False, f"unpickle failed: {e}"
    if scenario not in obj:
        return False, f"top-level key {scenario!r} absent (keys={list(obj)})"
    per_country = obj[scenario]
    n = len(per_country)
    exp_n = EXPECT_COUNTRIES[regc]
    if n != exp_n:
        return False, f"{n} countries, expected {exp_n}"
    # step-column contract + non-empty frames, checked on every country
    exp_last = JOURNEY[scenario]
    bad = []
    for c, df in per_country.items():
        steps = sorted(int(col[len(STEP_PREFIX):]) for col in df.columns
                       if col.startswith(STEP_PREFIX) and col[len(STEP_PREFIX):].isdigit())
        if not steps:
            bad.append(f"{c}:no-step-cols"); continue
        if steps[0] != 0 or steps[-1] != exp_last:
            bad.append(f"{c}:steps {steps[0]}..{steps[-1]}≠0..{exp_last}"); continue
        if len(df) == 0:
            bad.append(f"{c}:empty"); continue
        scol = f"{STEP_PREFIX}{exp_last}"
        if scol in df.columns and df[scol].isna().all():
            bad.append(f"{c}:all-NaN@{exp_last}")
    if bad:
        return False, f"{len(bad)} country issue(s): " + ", ".join(bad[:4]) + ("…" if len(bad) > 4 else "")
    return True, f"{n} countries, steps 0..{exp_last} ✓"


def main():
    print(f"Integrity check — {BASE}\n")
    print(f"{'combo':52} {'result':6} note")
    print("-" * 100)
    all_ok = True
    for scenario in SCENARIOS:
        for prog, sim_name in PROGRAMS.items():
            for regc in (True, False):
                ok, note = check_one(scenario, sim_name, regc)
                all_ok &= ok
                tag = f"{scenario}/{prog}/{'regC' if regc else 'no-regC'}"
                print(f"{tag:52} {'PASS' if ok else 'FAIL':6} {note}")
    print("-" * 100)
    print("ALL 16 PASS — outputs are structurally sound; safe to plot."
          if all_ok else "FAILURES above — inspect before plotting.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
