"""Phase 1 GATE check: does the new code reproduce the frozen review-copy figures?

Reproduces review-copy result frames with the current (Phase-1) code and diffs them
against the frozen review-copy pkls, per (program, regional-constraint) combination.

  * Deterministic programs (coreness_ranked->coreRanked, optimal): compared exactly on
    decision columns and within tight tolerance on continuous columns. These MUST match.
  * green/digital are skipped here: their draws are unseeded and re-randomised every
    step (pre-Task-A), so they can only be compared distributionally, not frame-equal.

Faithful parameters (from the __main__ harness): journey length 30, thresholds
(3.68, 10.80), scenario 'shortage', target_job_availability_coeffy 'COEFFY_mean+sd'.

Usage:
    .venv/bin/python revision/gate_check.py --countries DE          # fast validation
    .venv/bin/python revision/gate_check.py --countries all         # full gate
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVIEW = os.path.join(os.path.dirname(ROOT), "review-copy")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from src.modelling.reskilling import ReskillingPathways  # noqa: E402
from revision.run_sample import assemble_lfs_data  # noqa: E402

PROGRAMS = {"coreness_ranked": "reskill-coreRanked", "optimal": "reskill-optimal"}
SCENARIO = "shortage"
JOURNEY = 30
THRESH = (3.68, 10.80)
DEC_PREFIX = ("n_viable_transitions_step_", "transition_viable_step_",
              "added_skill_idx_step_", "added_skill_label_step_")
ALL_COUNTRIES = ["AT", "BE", "CH", "CY", "CZ", "DE", "DK", "EL", "EE", "ES", "FI",
                 "FR", "HR", "HU", "IE", "IS", "IT", "LT", "LU", "LV", "NL", "NO",
                 "PL", "PT", "RO", "SE", "SK"]


def review_path(sim_name, regc):
    tag = f"{sim_name}_wage-opt_{'regC' if regc else 'no-regC'}_2023"
    return os.path.join(REVIEW, "results", "figures", "reskilling_simulation",
                        "SEP25", SCENARIO, tag, f"{tag}.pkl")


def compare_frame(a, b):
    """Return (n_decision_cols_diff, max_abs_continuous_diff, note)."""
    if a.shape != b.shape:
        # align on common columns/rows for a best-effort delta
        common = [c for c in a.columns if c in b.columns]
        a, b = a[common].reset_index(drop=True), b[common].reset_index(drop=True)
        note = "SHAPE DIFF"
    else:
        a, b = a.reset_index(drop=True), b.reset_index(drop=True)
        note = ""
    dec_diff = 0
    for c in [c for c in a.columns if c.startswith(DEC_PREFIX) and c in b.columns]:
        if not a[c].equals(b[c]):
            dec_diff += 1
    num = [c for c in a.columns if c in b.columns]
    nums = a[num].select_dtypes("number").columns
    maxabs = 0.0
    if len(nums):
        d = (a[nums] - b[nums]).abs().values
        if d.size:
            maxabs = float(np.nanmax(d))
    return dec_diff, maxabs, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="DE")
    ap.add_argument("--out-dir", default="/tmp/gate_out")
    args = ap.parse_args()
    countries = ALL_COUNTRIES if args.countries == "all" else args.countries.split(",")

    lfs = assemble_lfs_data()
    rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc", lfs_data=lfs, year=2023)
    # review-copy was generated before the journey-aware correction, so the regression
    # against the frozen reference must use baseline-only behaviour.
    rp.journey_aware = False

    print(f"GATE: countries={countries} journey={JOURNEY} thresholds={THRESH} scenario={SCENARIO} (journey_aware=False)")
    rows = []
    for prog, sim_name in PROGRAMS.items():
        for regc in (True, False):
            res = rp.simulate_regional(
                level="isco_3_digit", countries=countries, scenarios=[SCENARIO],
                transition_optimisation="wage", reskilling=prog,
                reskilling_journey_length=JOURNEY, region_constraints=regc,
                target_job_availability_coeffy="COEFFY_mean+sd", mask_diagonal=True,
                transition_thresholds=THRESH, out_dir=args.out_dir,
            )
            ref = pd.read_pickle(review_path(sim_name, regc))[SCENARIO]
            got = res[SCENARIO]
            for c in countries:
                if c not in got or c not in ref:
                    rows.append((f"{sim_name}/{'regC' if regc else 'no-regC'}/{c}", "MISSING", "", ""))
                    continue
                dd, mx, note = compare_frame(got[c], ref[c])
                rows.append((f"{sim_name}/{'regC' if regc else 'no-regC'}/{c}", dd, f"{mx:.2e}", note))

    print(f"\n{'case':46} {'dec_diff':9} {'max|Δ|':10} note")
    print("-" * 80)
    ok = True
    for tag, dd, mx, note in rows:
        print(f"{tag:46} {str(dd):9} {str(mx):10} {note}")
        if dd != 0 or note:
            ok = False
    print("-" * 80)
    print("GATE PASS (deterministic programs reproduce review-copy)" if ok
          else "GATE: discrepancies present — investigate")


if __name__ == "__main__":
    main()
