"""Phase 1.2 — small-sample test-mode runner for fast iteration.

A thin, faithful wrapper around ReskillingPathways.simulate_regional that:
  * assembles lfs_data EXACTLY as the __main__ harness in src/modelling/reskilling.py
    does (read clean merged LFS, hyphen->underscore share columns, region fix-ups);
  * runs a bounded subset (default: one country, one scenario, one program, short
    journey) so a Phase-2 A/B change can be checked in minutes instead of hours.

It changes NO model behaviour — it only narrows the inputs. Output is written to a
throwaway directory so the real results/ tree is never touched.

Usage:
    .venv/bin/python revision/run_sample.py
    .venv/bin/python revision/run_sample.py --country DE --program coreness_ranked \
        --scenario shortage --journey 8 --no-regc

Run from the repository root with the environment that can import the model (.venv).
"""
import argparse
import os
import sys

import pandas as pd

# repo root on path (script lives in revision/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.modelling.reskilling import ReskillingPathways  # noqa: E402

LFS_PKL = os.path.join(
    ROOT, "data", "eurostat_data", "interim",
    "clean_eu_lfs_merged_2023_with_final_unweighted_shares_and_earnings_incdecil_imputed.pkl",
)


def assemble_lfs_data():
    """Reproduce the __main__ harness lfs_data assembly (reskilling.py ~3384-3426)."""
    lfs = pd.read_pickle(LFS_PKL)
    rename = {
        c: c.replace("-", "_")
        for c in lfs.columns
        if "-" in c and (c.startswith("share") or c.startswith("COEFFY_share"))
    }
    lfs.rename(columns=rename, inplace=True)

    cats_w = lfs["REGION_2DW"].cat.categories.values
    cats_r = lfs["REGION_2D"].cat.categories.values
    lfs["REGION_2DW"] = lfs["REGION_2DW"].cat.add_categories(list(set(cats_r) - set(cats_w)) + ["99"])
    lfs["REGION_2D"] = lfs["REGION_2D"].cat.add_categories(list(set(cats_w) - set(cats_r)) + ["99"])
    lfs["REGION_2DW"] = lfs["REGION_2DW"].fillna(lfs["REGION_2D"]).fillna("99")
    lfs["REGION_2D"] = lfs["REGION_2D"].fillna("99")
    lfs["NUTS_ID"] = lfs["COUNTRYW"].astype("string") + lfs["REGION_2DW"].astype("string")
    return lfs


def main():
    ap = argparse.ArgumentParser(description="Small-sample reskilling run (Phase 1.2).")
    ap.add_argument("--country", default="DE")
    ap.add_argument("--scenario", default="shortage", choices=["shortage", "at_risk", "high_carbon"])
    ap.add_argument("--program", default="coreness_ranked",
                    choices=["coreness_ranked", "optimal", "green", "digital"])
    ap.add_argument("--journey", type=int, default=8)
    ap.add_argument("--no-regc", action="store_true", help="disable regional constraints")
    ap.add_argument("--thresholds", default="3.68,10.80",
                    help="viable,highly_viable overlap thresholds (paper default)")
    ap.add_argument("--out-dir", default="/tmp/reskilling_sample_out")
    ap.add_argument("--save", default=None, help="optional path to pickle the result dict")
    args = ap.parse_args()

    qv, qhv = (float(x) for x in args.thresholds.split(","))
    lfs = assemble_lfs_data()
    rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc", lfs_data=lfs, year=2023)

    print(f"[run_sample] country={args.country} scenario={args.scenario} program={args.program} "
          f"journey={args.journey} regC={not args.no_regc} thresholds=({qv},{qhv})")
    res = rp.simulate_regional(
        level="isco_3_digit",
        countries=[args.country],
        scenarios=[args.scenario],
        transition_optimisation="wage",
        reskilling=args.program,
        reskilling_journey_length=args.journey,
        region_constraints=not args.no_regc,
        target_job_availability_coeffy="COEFFY_mean+sd",
        mask_diagonal=True,
        transition_thresholds=(qv, qhv),
        out_dir=args.out_dir,
    )
    if args.save:
        pd.to_pickle(res, args.save)
        print(f"[run_sample] saved result dict -> {args.save}")
    print("[run_sample] done.")
    return res


if __name__ == "__main__":
    main()
