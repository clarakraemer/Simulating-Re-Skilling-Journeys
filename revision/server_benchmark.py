"""Run this ON THE GPU SERVER (via the PyCharm toggle) to get the real per-step rate.

Two questions it answers:
  (1) Is the GPU even used? The simulation hot path (find_closest, reskill's np.dot, the
      per-worker loop) is plain numpy/pandas — there is NO cupy/torch/numba.cuda dispatch
      in src/. So unless numpy itself is a GPU drop-in here, the GPU sits idle and the
      server is just a faster CPU. This probe reports what's available and whether numpy
      is the stock CPU build.
  (2) The real per-step wall-clock for transferable + tailored, regC, on the server CPU,
      so the full-run projection can be redone on the right hardware.

Edit COUNTRIES / JOURNEY below if you want a heavier/lighter probe. Run and paste the
output back.
"""
import sys, os, time, importlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))

# ---- (1) GPU / backend probe -------------------------------------------------------
print("===== BACKEND PROBE =====", flush=True)
import numpy as np
print("numpy:", np.__version__, "| file:", np.__file__)
print("  (stock CPU numpy if path is site-packages/numpy; a GPU drop-in would differ)")
for mod in ["cupy", "torch", "jax"]:
    try:
        m = importlib.import_module(mod)
        extra = ""
        if mod == "torch":
            extra = f" cuda_available={m.cuda.is_available()} device_count={m.cuda.device_count()}"
        print(f"  {mod}: IMPORTABLE {getattr(m,'__version__','?')}{extra}")
    except Exception:
        print(f"  {mod}: not installed")
os.system("nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv 2>/dev/null || echo '  nvidia-smi: not available'")
print("NOTE: src/ has no cupy/torch/cuda dispatch, so even with a GPU present the hot loop"
      " runs on CPU. Watch nvidia-smi utilization during the run below — if it stays ~0%,"
      " the GPU is idle and the server is a faster CPU only.", flush=True)

# ---- (2) per-step timing -----------------------------------------------------------
from src.modelling.reskilling import ReskillingPathways
from revision.run_sample import assemble_lfs_data

COUNTRIES = ["DE", "SE", "SK"]   # a few; widen to all 27 for a full-pool number
JOURNEY = 8                       # enough steps to estimate a per-step rate
THRESH = (3.68, 10.80)

lfs = assemble_lfs_data()
rp = ReskillingPathways(osm_version="weighted", sim_metric="cooc", lfs_data=lfs, year=2023)
print(f"\n===== TIMING (regC, journey {JOURNEY}, countries={COUNTRIES}) =====", flush=True)
for prog, lab in [("coreness_ranked", "transferable"), ("optimal", "tailored")]:
    t0 = time.time()
    rp.simulate_regional(level="isco_3_digit", countries=COUNTRIES, scenarios=["at_risk"],
        transition_optimisation="wage", reskilling=prog, reskilling_journey_length=JOURNEY,
        region_constraints=True, mask_diagonal=True, transition_thresholds=THRESH,
        out_dir="/tmp/server_bench_out")
    dt = time.time() - t0
    per_step_per_cty = dt / (JOURNEY + 1) / len(COUNTRIES)
    print(f"  {lab:12}: {dt:7.1f}s total  ->  {per_step_per_cty:6.2f} s/step/country", flush=True)
print("\nPaste this whole output back; the s/step/country rates re-project the full run.", flush=True)
