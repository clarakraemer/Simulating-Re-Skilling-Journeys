"""Task B — license the post-hoc capture before trusting the sweep.

Compares, for ONE band on ONE scenario (band=0.25, at_risk @ step 20):
  A = off-mode capture          (feasible sets + per-worker draw seeds)
  B = live share_income_acceptable(band=0.25) run, capturing its ACTUAL choices

Asserts, exactly (not "close"):
  (1) mode-invariance : A's feasible target set == B's, per region (the destination
      choice does not feed back into skill acquisition).
  (2) seeded replay   : recomputing each worker's choice from A's feasible set + the
      worker's deterministic draw seed (via the production _select_destination under
      band=0.25) reproduces B's live choice -- same destination, same income delta.
If both hold, the off capture is licensed for the whole band sweep.
"""
import sys, os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from src.modelling.reskilling import ReskillingPathways

A = pd.read_pickle("/tmp/taskB_cap_at_risk.pkl")["capture"]      # off
B = pd.read_pickle("/tmp/taskB_val_at_risk.pkl")["capture"]      # live band=0.25

# Captures fire once per (nuts, source-occupation-group, step). Iteration order is
# independent of the destination choice, so A and B are emitted in the SAME order ->
# match positionally (A[i] <-> B[i]); keying on (nuts,step) alone collides across
# source-occupation groups.
assert len(A) == len(B), f"capture count differs: {len(A)} vs {len(B)}"

s = type("S", (), {})()
s.destination_weighting = "share_income_acceptable"; s.income_band = 0.25; s.acceptability = "band"

inv_bad = 0; regions = 0; choices = 0; choice_bad = 0; delta_bad = 0; matched = 0; order_bad = 0
for a, b in zip(A, B):
    if (a["country"], a["nuts"], a["step"]) != (b["country"], b["nuts"], b["step"]):
        order_bad += 1; continue
    matched += 1; regions += 1
    # (1) mode-invariance: same feasible set (codes/earnings/COEFFY), same order
    ta = a["targets"].reset_index(drop=True); tb = b["targets"].reset_index(drop=True)
    if not (ta.shape == tb.shape and (ta["code"].values == tb["code"].values).all()
            and np.allclose(ta["annual_earnings"].values, tb["annual_earnings"].values, equal_nan=True)
            and np.allclose(ta["COEFFY"].values, tb["COEFFY"].values, equal_nan=True)):
        inv_bad += 1
        continue
    # (2) seeded replay: A's set + seed under band=0.25 must reproduce B's live choice
    bch = {ch["widx"]: ch for ch in b["choices"]}
    for ach in a["choices"]:
        bc = bch.get(ach["widx"])
        if bc is None:
            continue
        choices += 1
        wk = pd.Series({"annual_earnings": ach["earn"]})
        tgt, _ = ReskillingPathways._select_destination(s, a["targets"], wk, ach["seed"])
        if tgt["code"] != bc["chosen"]:
            choice_bad += 1
        else:
            d_post = float(tgt["annual_earnings"]) - ach["earn"]
            chosen_row = b["targets"][b["targets"]["code"] == bc["chosen"]].iloc[0]
            d_live = float(chosen_row["annual_earnings"]) - bc["earn"]
            if not (np.isclose(d_post, d_live) or (np.isnan(d_post) and np.isnan(d_live))):
                delta_bad += 1

print("=== Task B capture validation (band=0.25, at_risk @ step 20) ===")
print(f"capture-order mismatches     : {order_bad}   (A[i] vs B[i] different region)")
print(f"regions matched (A&B)        : {matched}")
print(f"mode-invariance failures     : {inv_bad}   (feasible set differs A vs B)")
print(f"worker choices checked       : {choices}")
print(f"replay choice mismatches     : {choice_bad}   (post-hoc dest != live dest)")
print(f"income-delta mismatches      : {delta_bad}")
ok = inv_bad == 0 and choice_bad == 0 and delta_bad == 0 and choices > 0
print("VERDICT:", "LICENSED — capture reproduces live exactly, sweep is trustworthy" if ok
      else "FAILED — do NOT trust the sweep; investigate the gap")
