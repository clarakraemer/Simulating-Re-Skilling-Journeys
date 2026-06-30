"""Plot-only map regeneration from EXISTING production pickles (no resimulation).

Draws the split EU NUTS-2 choropleths (one transitions PDF + one income PDF per step)
into each PRODUCTION regC folder, using the same visualiser the serial harness used
(visualise_simulation_results_eu), with simulation_results=None so it self-loads each pkl.

Scope (per the revision decision):
  * PRODUCTION regC only  -> 4 programs x 2 flows = 8 folders.
  * SKIP _dwoff/_jaoff (aggregate-only robustness) and no-regC.
Output filenames (split): EU_2023_regional_<scenario>_step_<k>_transitions.pdf
                          EU_2023_regional_<scenario>_step_<k>_income.pdf
The income panel is a REGIONAL POPULATION MEAN (avg income change per at-risk worker),
distinct from the per-switcher metric in R3/A3/A4 (labeled as such on the map).

Usage:
  python revision/regenerate_maps.py            # all 8 production regC combos, all steps
  python revision/regenerate_maps.py --dry-run  # list what would be drawn, draw nothing
  python revision/regenerate_maps.py --modes optimal --scenarios at_risk --steps 8  # subset
"""
import os, sys, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from src import utils
from src.modelling.reskilling import ReskillingPathways

useful_paths = utils.UsefulPaths()

JOURNEY = {"at_risk": 20, "shortage": 30}
MODES = ["optimal", "coreness_ranked", "digital", "green"]  # rp.simulation_name maps -> reskill-*


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODES), help="comma list of mode keys")
    ap.add_argument("--scenarios", default="at_risk,shortage")
    ap.add_argument("--steps", default=None, help="comma list of steps; default = full journey")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    scenarios = [s.strip() for s in a.scenarios.split(",") if s.strip()]
    steps_override = [int(s) for s in a.steps.split(",")] if a.steps else None

    base_root = os.path.join(useful_paths.figure_dir, "reskilling_simulation")
    print(f"[regen] plot-only, PRODUCTION regC only. modes={modes} scenarios={scenarios}")
    if a.dry_run:
        rp = None
    else:
        rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc", lfs_data=None, year=2023)

    n_pdf = 0
    for scenario in scenarios:
        last = JOURNEY[scenario]
        steps = steps_override if steps_override is not None else list(range(last + 1))
        base_dir = os.path.join(base_root, scenario)
        for mode in modes:
            sim_version = ("reskill-" + {"optimal": "optimal", "coreness_ranked": "coreRanked",
                                         "digital": "digital", "green": "green"}[mode])
            tag = f"{sim_version}_wage-opt_regC_2023"
            pkl = os.path.join(base_dir, tag, f"{tag}.pkl")
            if not os.path.exists(pkl):
                print(f"  [skip] missing pkl: {pkl}")
                continue
            print(f"  [{scenario}/{mode}] {len(steps)} steps -> {tag}/  (2 PDFs each)")
            if a.dry_run:
                n_pdf += 2 * len(steps); continue
            for step in steps:
                rp.visualise_simulation_results_eu(
                    simulation_results=None, base_dir=base_dir,
                    transition_optimisation="wage", reskilling_version=sim_version,
                    step=step, regional_constraint=True,
                    vmax_wages=3000, vmax_transitions=8, show_title=False,
                    title_fontsize="small", cbar_fraction=0.025,
                    combine_vars_in_sector_plot=True,
                )
                n_pdf += 2
    print(f"[regen] {'WOULD draw' if a.dry_run else 'drew'} ~{n_pdf} map PDFs "
          f"(2 per step: _transitions + _income).")


if __name__ == "__main__":
    main()
