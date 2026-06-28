"""Country-level parallel driver — same outputs as the serial __main__, faster.

Does NOT modify the model. Calls the UNCHANGED ReskillingPathways.simulate_regional once
per country in parallel worker processes, then REASSEMBLES the per-country slices into the
exact serial structure ({scenario: {country: df}}, same country order) and pickles them to
the exact serial path. Nothing downstream can tell it was run in parallel.

Why it's safe: simulate_regional uses only fixed-seed RNGs created fresh per (country,
region) (RandomState(42), random_state=42) and Task B's order-independent per-draw seeding
— so there is no state that advances across the country loop; per-country == serial.

Config matches the serial harness exactly: per-flow journey (at_risk=20, shortage=30),
journey_aware/destination_weighting from the model defaults, optional_weight=0.5 (a CLI
parameter so the same driver runs the Task-D sweep at 0 and 1.0 without rebuilding),
threshold (3.68,10.80), programs {optimal,coreness_ranked,digital,green}, regC + no-regC.

GATE before any full run (run BOTH; only launch after BIT-IDENTICAL at the REAL journeys):
  # 1) quick assembly check (order / EE-drop / dtypes) — journey-2, seconds:
  python revision/run_parallel.py --test --countries SK,EE,FI --journey 2 --workers 1
  # 2) REAL per-flow journey (at_risk=20 / shortage=30) — exercises the journey-dependent
  #    COEFFY / employment-share aggregates, which is exactly where parallel could diverge:
  python revision/run_parallel.py --test --countries SK,EE,FI --workers 6

FULL RUN (only after you have seen "VERDICT: BIT-IDENTICAL" at the real journeys):
  nohup python revision/run_parallel.py --workers 20 > revision/parallel_run.log 2>&1 &
  # Task-D sweep later, same driver, no rebuild:
  #   python revision/run_parallel.py --workers 20 --optional-weight 0
  #   python revision/run_parallel.py --workers 20 --optional-weight 1.0
"""
import sys, os, argparse, pickle, tempfile, shutil, json
from collections import namedtuple, defaultdict
# Single-thread BLAS BEFORE numpy is imported: each worker process is already a unit of
# parallelism, so per-process BLAS threads would oversubscribe the box; and it removes the
# non-fork-safe thread state that deadlocks fork-based pools. `setdefault` => the caller can
# override (e.g. `OMP_NUM_THREADS=8 python ...` for a fewer-workers x more-BLAS config).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp
import numpy as np, pandas as pd
from pandas.testing import assert_frame_equal
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from src.modelling.reskilling import ReskillingPathways, useful_paths  # noqa
from revision.run_sample import assemble_lfs_data

ALL_COUNTRIES = ["AT","BE","CH","CY","CZ","DE","DK","EL","EE","ES","FI","FR","HR","HU","IE",
                 "IS","IT","LT","LU","LV","NL","NO","PL","PT","RO","SE","SK"]
PROGRAMS = ["optimal", "coreness_ranked", "digital", "green"]
REGCS = [True, False]
SCENARIOS = ["at_risk", "shortage"]
JOURNEY_BY_FLOW = {"at_risk": 20, "high_carbon": 20, "shortage": 30}
THRESH = (3.68, 10.80)   # held fixed across optional_weight (the Task-D "fixed threshold" option)
# scenario -> output-folder suffix, IDENTICAL to the serial __main__ SCENARIO_SUFFIX map.
SCENARIO_SUFFIX = {"shortage": "shortage", "at_risk": "at_risk", "high_carbon": "highcarbon"}
# decision/identity columns that must match EXACTLY (not within a float tolerance).
DEC_PREFIX = ("n_viable_transitions_step_", "transition_viable_step_",
              "added_skill_idx_step_", "added_skill_label_step_")

# Carry EVERY run parameter inside the task: spawn workers re-import the module fresh and do
# NOT inherit parent-side mutations of module globals, so anything that varies per run must
# travel in the picklable task, never via a global the parent mutated.
Task = namedtuple("Task", "scenario program regc country journey weight symmetric threshold")

_RP = None

def _init():
    global _RP
    if _RP is None:
        _RP = ReskillingPathways(osm_version="weighted", sim_metric="cooc",
                                 lfs_data=assemble_lfs_data(), year=2023)
    # Spawn-safe overrides via env (parent sets them in __main__ BEFORE the Pool spawns, so
    # workers inherit them and re-apply on their own _init). Empty/unset => model default.
    dw = os.environ.get("RSJ_DEST_WEIGHTING")
    if dw:
        _RP.destination_weighting = dw
    ja = os.environ.get("RSJ_JOURNEY_AWARE")
    if ja in ("0", "1"):
        _RP.journey_aware = (ja == "1")

def _init_worker():
    """Pool initializer (spawn): each worker loads the model ONCE, fresh — no inherited
    BLAS/lock state from the parent (so no fork deadlock), amortised over its many tasks."""
    _init()

def _simulate(countries, t):
    """Single simulate_regional call with the task's parameters; returns {country: df}."""
    tmp = tempfile.mkdtemp(prefix="rp_")
    try:
        res = _RP.simulate_regional(
            level="isco_3_digit", countries=countries, scenarios=[t.scenario],
            transition_optimisation="wage", reskilling=t.program,
            reskilling_journey_length=t.journey, region_constraints=t.regc,
            target_job_availability_coeffy="COEFFY_mean+sd", mask_diagonal=True,
            transition_thresholds=t.threshold, optional_weight=t.weight,
            symmetric_employment=t.symmetric, out_dir=tmp)
        return res.get(t.scenario, {})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)   # discard the model's throwaway save; use RETURN

def run_one(task):
    """One country. Returns (task, df_or_None). df is None when regC dropped the country."""
    scen = _simulate([task.country], task)
    return task, scen.get(task.country)

def serial_combo(countries, t):
    """Serial reference: one call for ALL countries -> {country: df}, same params as parallel."""
    return _simulate(countries, t)

def assemble(per_country, countries):
    """Build {country: df} in the ORIGINAL country order, skipping regC-dropped (None) —
    identical to how serial's filtered_countries loop inserts them."""
    return {c: per_country[c] for c in countries if per_country.get(c) is not None}

def out_path(scenario, program, regc, weight=0.5, symmetric=False, extra_suffix=""):
    """The EXACT serial output path, reconstructed from the model's OWN attributes (not
    hardcoded), including the `_optw{:g}` / `_symE` variant suffix the model appends for
    non-default robustness runs — so Task-D / Task-E variants self-organise identically."""
    reg = "regC" if regc else "no-regC"
    dirname = _RP.dirname_out_reg.format(sim_version=_RP.simulation_name[program],
                                         opt_target="wage", reg_constraint=reg, year=_RP.year)
    variant = ""
    if abs(float(weight) - 0.5) > 1e-12:
        variant += "_optw{:g}".format(weight)
    if symmetric:
        variant += "_symE"
    dirname += variant + extra_suffix
    sub = SCENARIO_SUFFIX.get(scenario, str(scenario))
    d = os.path.join(useful_paths.figure_dir, "reskilling_simulation", sub, dirname)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{dirname}.pkl"), dirname

def save_like_serial(scenario, program, regc, weight, symmetric, scenario_results,
                     threshold=THRESH, extra_suffix=""):
    """Pickle {scenario: {country: df}} to the exact serial path + a run_metadata.json
    sidecar matching the one simulate_regional writes for the default run."""
    path, dirname = out_path(scenario, program, regc, weight, symmetric, extra_suffix)
    # Atomic: write to a temp file then rename, so a crash mid-write never leaves a
    # half-written pkl that the resume-skip would wrongly trust as complete.
    tmp = path + ".tmp"
    with open(tmp, "wb") as h:
        pickle.dump({scenario: scenario_results}, h, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    try:
        with open(os.path.join(os.path.dirname(path), "run_metadata.json"), "w") as mh:
            json.dump({"program": _RP.simulation_name[program],
                       "transition_optimisation": "wage", "region_constraints": regc,
                       "year": _RP.year, "optional_weight": float(weight),
                       "symmetric_employment": bool(symmetric),
                       "q_viable": float(threshold[0]), "q_highly_viable": float(threshold[1]),
                       "journey_aware": getattr(_RP, "journey_aware", None),
                       "destination_weighting": getattr(_RP, "destination_weighting", None),
                       "produced_by": "revision/run_parallel.py"}, mh, indent=2)
    except Exception as e:
        print(f"[warn] sidecar not written for {dirname}: {e}", flush=True)
    return path

# ---------------------------------------------------------------------------------------
def compare_dicts(ser, par):
    """Compare two FINAL assembled {country: df} dicts across every dimension that matters.
    Returns findings; the verdict is MATCH iff all of them are clean."""
    f = {"order_ok": list(ser.keys()) == list(par.keys()),
         "ser_keys": list(ser.keys()), "par_keys": list(par.keys()),
         "dtype_mismatch": 0, "decision_mismatch": 0, "max_cont_abs": 0.0, "strict_fail": 0}
    for c in ser:
        if c not in par:
            f["strict_fail"] += 1
            continue
        a, b = ser[c].reset_index(), par[c].reset_index()
        if not a.dtypes.equals(b.dtypes):
            f["dtype_mismatch"] += 1
        for col in [x for x in a.columns if x.startswith(DEC_PREFIX) and x in b.columns]:
            if not a[col].equals(b[col]):
                f["decision_mismatch"] += 1
        num = a.select_dtypes("number").columns.intersection(b.columns)
        if len(num):
            d = a[num].to_numpy(dtype=float) - b[num].to_numpy(dtype=float)
            if d.size:
                f["max_cont_abs"] = max(f["max_cont_abs"], float(np.nanmax(np.abs(d))))
        try:   # all-in-one strict check (the contract a downstream reader relies on)
            assert_frame_equal(a, b, check_dtype=True, check_like=False, rtol=0, atol=1e-13)
        except AssertionError:
            f["strict_fail"] += 1
    f["match"] = (f["order_ok"] and f["dtype_mismatch"] == 0 and f["decision_mismatch"] == 0
                  and f["max_cont_abs"] <= 1e-13 and f["strict_fail"] == 0)
    return f

def correctness_test(countries, workers, journey=None, weight=0.5, symmetric=False):
    """serial vs parallel on the FINAL assembled dict. journey=None -> REAL per-flow journey
    (exercises the journey-dependent aggregates); journey=N -> quick assembly check."""
    jdesc = ("override=%d (quick assembly check — does NOT exercise journey-dependent "
             "aggregates)" % journey) if journey is not None else \
            ("REAL per-flow %s (exercises COEFFY/share aggregates)" % JOURNEY_BY_FLOW)
    print(f"=== CORRECTNESS TEST: countries={countries} | journey={jdesc} | "
          f"optional_weight={weight} symmetric={symmetric} ===", flush=True)
    _init()
    # The driver is program-AGNOSTIC: assembly + RNG handling are identical for every program,
    # so optimal across {regC-drop, no-regC, other-scenario} is sufficient — it exercises the
    # EE drop, full-set ordering, and a 2nd flow. (coreness/green/digital assemble identically.)
    combos = [("at_risk", "optimal", True), ("at_risk", "optimal", False),
              ("shortage", "optimal", True)]
    pool = mp.Pool(workers, initializer=_init_worker)
    bad = 0
    try:
        for scen, prog, regc in combos:
            j = journey if journey is not None else JOURNEY_BY_FLOW[scen]
            base = dict(scenario=scen, program=prog, regc=regc, journey=j,
                        weight=weight, symmetric=symmetric, threshold=THRESH)
            ser = assemble(serial_combo(countries, Task(country=None, **base)), countries)
            tasks = [Task(country=c, **base) for c in countries]
            per = {t.country: df for t, df in pool.map(run_one, tasks)}
            par = assemble(per, countries)
            # roundtrip through the ACTUAL pickle so we compare the final ARTIFACT, not just
            # the in-memory dict (guards against any pickle-protocol surprise).
            par = pickle.loads(pickle.dumps({scen: par}, protocol=pickle.HIGHEST_PROTOCOL))[scen]
            f = compare_dicts(ser, par)
            tag = f"{scen}/{prog}/{'regC' if regc else 'no-regC'}"
            if not f["order_ok"]:
                print(f"  {tag:26} ORDER differs: serial={f['ser_keys']} parallel={f['par_keys']}")
            print(f"  {tag:26} n={len(ser):2d}  order={'OK' if f['order_ok'] else 'BAD'}  "
                  f"dtypes={'OK' if f['dtype_mismatch']==0 else f['dtype_mismatch']}  "
                  f"decisions-exact={'OK' if f['decision_mismatch']==0 else f['decision_mismatch']}  "
                  f"max|Δcont|={f['max_cont_abs']:.1e}  roundtrip={'OK' if f['strict_fail']==0 else 'FAIL'}"
                  f"  => {'MATCH' if f['match'] else 'MISMATCH'}", flush=True)
            bad += 0 if f["match"] else 1
    finally:
        pool.close(); pool.join()
    print("VERDICT:", "BIT-IDENTICAL — parallel assembly == serial; licensed"
          if bad == 0 else "FAILED — do NOT run the full parallel job")
    return bad == 0

def full_run(countries, workers, weight=0.5, symmetric=False,
             scenarios=None, regcs=None, threshold=None, extra_suffix=""):
    _init()
    scenarios = scenarios or SCENARIOS
    regcs = REGCS if regcs is None else regcs
    threshold = threshold or THRESH
    # One combo = one (scenario, program, regC) output pkl. We process combos sequentially
    # but parallelise the 27 countries WITHIN each combo, and SAVE as soon as a combo's
    # countries finish. This gives: (a) crash-resilience — a crash loses only the in-flight
    # combo, never finished ones; (b) resume — re-running the same command SKIPS combos whose
    # pkl already exists; (c) live progress — one [SAVED] line per combo across the run.
    # chunksize=1 hands one country at a time, so heavy countries (DE/FR/IT shortage-30)
    # spread across workers instead of clumping into one chunk — shorter tail.
    combos = [(s, p, r) for s in scenarios for p in PROGRAMS for r in regcs]
    print(f"[CONFIG] scenarios={scenarios} regC={regcs} suffix='{extra_suffix}' | per-flow journey {JOURNEY_BY_FLOW} "
          f"| journey_aware={getattr(_RP,'journey_aware',None)} "
          f"| destination_weighting={getattr(_RP,'destination_weighting',None)} "
          f"| optional_weight={weight} | symmetric_employment={symmetric} | thresholds={threshold}", flush=True)
    print(f"[PARALLEL] {len(combos)} combos x {len(countries)} countries on {workers} workers "
          f"| BLAS threads/worker={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    pool = mp.Pool(workers, initializer=_init_worker)
    done = skipped = 0
    try:
        for i, (s, p, r) in enumerate(combos, 1):
            path, dirname = out_path(s, p, r, weight, symmetric, extra_suffix)
            if os.path.exists(path):
                print(f"[SKIP {i}/{len(combos)}] {dirname} (already on disk — resume)", flush=True)
                skipped += 1
                continue
            tasks = [Task(s, p, r, c, JOURNEY_BY_FLOW[s], weight, symmetric, threshold) for c in countries]
            per = {t.country: df for t, df in pool.map(run_one, tasks, chunksize=1)}
            sr = assemble(per, countries)
            save_like_serial(s, p, r, weight, symmetric, sr, threshold=threshold, extra_suffix=extra_suffix)
            done += 1
            print(f"[SAVED {i}/{len(combos)}] {os.path.relpath(path, ROOT)}  ({len(sr)} countries)", flush=True)
    finally:
        pool.close(); pool.join()
    print(f"[DONE] full parallel run complete — {done} saved, {skipped} skipped this invocation", flush=True)

if __name__ == "__main__":
    try: mp.set_start_method("spawn")  # fresh workers (no fork-after-BLAS deadlock)
    except RuntimeError: pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="serial-vs-parallel gate, no real output written")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--countries", default=None, help="comma list; default all 27 (or SK,EE,FI for --test)")
    ap.add_argument("--journey", type=int, default=None,
                    help="test-only: override journey for ALL flows. Omit for the REAL per-flow "
                         "journey (the run that exercises the journey-dependent aggregates).")
    ap.add_argument("--optional-weight", dest="optional_weight", type=float, default=0.5,
                    help="weight of OPTIONAL skills in M_os (default 0.5). Task-D sweep: 0, 1.0.")
    ap.add_argument("--symmetric", action="store_true",
                    help="Task-E: apply the outward unemployment rule to BOTH flows.")
    ap.add_argument("--scenarios", default=None,
                    help="comma list to scope the run, e.g. 'shortage' (Task-D inward) or "
                         "'at_risk,shortage' (default both).")
    ap.add_argument("--regc", choices=["both", "true", "false"], default="both",
                    help="scope regional constraint: 'true' = regC only (Task-D SI), default both.")
    ap.add_argument("--threshold", default=None,
                    help="'q_viable,q_highly_viable' feasibility pair (per-weight for Task-D); "
                         "default is the production (3.68,10.80). MUST be the weight's re-derived pair.")
    ap.add_argument("--destination-weighting", dest="dest_weighting", default=None,
                    help="override rp.destination_weighting (e.g. 'off' for the no-weighting "
                         "counterfactual). 'off' self-organises into a _dwoff/ variant path.")
    ap.add_argument("--journey-aware", dest="journey_aware", choices=["on", "off"], default=None,
                    help="override rp.journey_aware. 'off' (the pre-correction behaviour) "
                         "self-organises into a _jaoff/ variant path; 'on' is the production default.")
    a = ap.parse_args()
    thr = tuple(float(x) for x in a.threshold.split(",")) if a.threshold else None
    scen = a.scenarios.split(",") if a.scenarios else None
    regcs = {"both": None, "true": [True], "false": [False]}[a.regc]
    # Spawn-safe attribute overrides via env (set BEFORE any Pool spawns) + variant suffix so
    # diagnostic runs never collide with the weight-0.5 production pickles.
    suffix = ""
    if a.dest_weighting:
        os.environ["RSJ_DEST_WEIGHTING"] = a.dest_weighting
        if a.dest_weighting == "off":
            suffix += "_dwoff"
    if a.journey_aware is not None:
        os.environ["RSJ_JOURNEY_AWARE"] = "1" if a.journey_aware == "on" else "0"
        if a.journey_aware == "off":
            suffix += "_jaoff"
    if a.test:
        cs = a.countries.split(",") if a.countries else ["SK", "EE", "FI"]
        correctness_test(cs, a.workers, journey=a.journey,
                         weight=a.optional_weight, symmetric=a.symmetric)
    else:
        cs = a.countries.split(",") if a.countries else ALL_COUNTRIES
        full_run(cs, a.workers, weight=a.optional_weight, symmetric=a.symmetric,
                 scenarios=scen, regcs=regcs, threshold=thr, extra_suffix=suffix)
