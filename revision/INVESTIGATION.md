# Re-Skilling Journeys — Revision Investigation & Implementation Plan

**Repository:** `Simulating-Re-Skilling-Journeys/` (live working copy)
**Paper:** *Tailored reskilling widens feasible job transitions during Europe's decarbonization* (Krämer, Zaussinger, Egli) — Nature Communications appeal.
**Status:** Investigation complete; plan approved; Phase 1 in progress.
**Scope of this document:** the read-only investigation (CLAUDE.md §7), the dependency audit, the options analysis, and the approved sequenced implementation plan with resolved decisions.

All code references are `path:line` against the live working copy unless noted. Citations were valid at the time of writing (HEAD `af796cb`).

---

## 0. Which copy is which (CLAUDE.md §6)

| | `Simulating-Re-Skilling-Journeys/` | `review-copy/` |
|---|---|---|
| Git remote | `clarakraemer/Simulating-Re-Skilling-Journeys.git` — **matches the published repo** | `clarakraemer/Reskilling-For-Europes-Decarbonization.git` — a different repo |
| History | 29 commits, active dev | 1 squashed commit `"Code for peer review"` |
| Working tree | dirty (4 pending deletions) | clean |
| Extras | tag `v0.3-restructure`, stash, sync branches | extra foreign data artifacts |

**Decision (confirmed by maintainer):** `Simulating-Re-Skilling-Journeys/` is the **live working copy and revision target**; `review-copy/` is the **frozen peer-review snapshot and stays untouched** (it also serves as the golden reference for equality gates — it holds all 18 result `.pkl`s).

The two trees are byte-identical in code; differences are (a) the 4 pending shim/CSV deletions and (b) extra dead/foreign data files in `review-copy`.

---

## 1. Repo map (entry points are notebooks, not a script)

Run order (README §9): **notebooks 01→06**.
- `01-ck-lfs-data-preprocessing.ipynb` — EU-LFS preprocessing (`src/data/lfs.py`).
- `02-fz-esco-preprocessing.ipynb` — ESCO matrices (`src/data/framework.py`).
- `03-ck-esco-tailored-reskilling.ipynb` — tailored upskilling table (**numba**, MCC `compare_nodes_utils`).
- `04-ck-esco-occupation-classification.ipynb` — occupation classification.
- `05-fz-recalculate-skills-centrality.ipynb` — coreness (MCC `cluster_utils`).
- `06-ck-lfs-model-visualisations.ipynb` — **runs the model + figures** (`ReskillingPathways`).

Core modules:
- `src/modelling/reskilling.py` (3,576 ln) — `ReskillingPathways`: `simulate()` (`838`), `simulate_regional()` (`1544`), `reskill()` (`1414`), thresholds (`573`), figure methods (`2252+`).
- `src/modelling/occupation_distance.py` (261 ln) — overlap/shortage/excess; `np.dot(M_os, M_os.T)` at `173`.
- `src/data/framework.py` (2,064 ln) — `Esco`: builds `M_os`, `M_oo`, coreness, ISCO aggregation; caches `.pkl`.
- `src/data/lfs.py` (1,146 ln) — `EuLfs`/`EulfsDs`: EU-LFS preprocessing, income deciles, regional shares.
- Support: `src/{utils,plotting_utils,stats_utils}.py`, `src/patches/seaborn_patch.py`, `src/__init__.py` (`UsefulPaths`).
- **`src/pipeline.py` is broken** (undefined `Target` at line 54) and superseded by nb06 — not a real entry point.

Output layout: `results/figures/reskilling_simulation/SEP25/{at_risk,shortage}/reskill-{coreRanked,digital,green,optimal}_wage-opt_{regC,no-regC}_2023/` = **4 programs × 2 flows × regional-constraint toggle** (the paper's Figures 2–4).

---

## 2. Data flow

```
ESCO v1.1.0 + EU-LFS 2023 + EU-SILC
  nb01 (lfs.py)        -> data/eurostat_data/interim/{eu_lfs_merged_2023, clean_eu_lfs_merged_2023_..._imputed}.{csv,pkl}
  nb02/04 (framework)  -> data/processed/esco/occ_skills_matrix.pkl  (M_os)
                          data/processed/esco/occ_sim_matrix_weighted_coo.pkl (M_oo)
  nb05 (framework/MCC) -> data/processed/esco/skills_network_metrics.pkl (coreness)
  nb03 (numba/MCC)     -> data/interim/upskilling_analysis/upskilling_best_100_skills_per_isco3d_occupation_merged.csv (tailored)
  nb06 ReskillingPathways.__init__ loads all of the above
        -> simulate()/simulate_regional() -> results/.../{program}.pkl + model_data_store.pkl -> figures
```

`data/` and `results/` are **gitignored** (`.gitignore:24-25`); all data lives outside version control. Large cache: `results/figures` ≈ 1.0 GB.

---

## 3. Environment audit

| | `.venv` | `.venv_numba` |
|---|---|---|
| Python | 3.11.11 | 3.12.7 (anaconda) |
| numpy | **2.2.3** | **1.26.4** |
| numba | absent | **0.59.1** |
| scipy / pandas | 1.15.2 / 2.2.3 | 1.15.2 / 2.2.3 |

**The split is forced by exactly one pin:** `numba 0.59.1` requires `numpy < 1.27`, incompatible with the main env's `numpy 2.2.3`. The **live numba surface is small**: only `src/mapping_career_causeways/compare_nodes_utils.py` (`@njit` at 109/230/278/331) and **nb03** use numba; `cluster_utils.py` only mentions it in a comment.

**Single env is feasible** two ways (decided by the nb03 smoke test in Phase 3):
1. Upgrade `numba ≥ 0.61` (supports numpy 2.x) → unify on numpy 2.2.3; or
2. Pin the whole project to numpy 1.26 (numba 0.59.1 already runs pandas 2.2/scipy 1.15 there).

All notebooks declare a generic `python3` kernel — env is chosen manually (a reproducibility trap).

---

## 4. Dead, duplicated, and foreign code (full dependency audit)

Method: AST import graph from the 6 live notebooks + transitive closure (code); literal + config-template file-access scan from live vs dead code (data). The code graph is exact; the data scan under-counts files whose names are built from `{year}`/config templates (flagged below).

### 4.1 Code — 34 `.py` under `src/`, **14 reachable**

**KEEP (reachable):** `src/__init__.py`, `utils.py`, `plotting_utils.py`, `stats_utils.py`, `patches/{__init__,seaborn_patch}.py`, `data/{__init__,framework,lfs}.py`, `modelling/{__init__,reskilling,occupation_distance}.py`, `mapping_career_causeways/{__init__,compare_nodes_utils,cluster_utils}.py`.

**SAFE TO REMOVE (orphan / foreign / broken):** `pipeline.py` (broken), `visualization/occupation_skill_space.py`, `dictionaries.py`, MCC `cluster_profiling_utils.py`, `models/{predict,train}_model.py`, `plotting_utils.py`, `scripts/*` (incl. `upskilling_aws_scripts/*` AWS/S3), `supplementary_online_data/.../download_outputs.py`.

**UNCERTAIN (functionally dead; deferred to Phase 3):** `data/preprocess.py` (only broken `pipeline.py` imports it), `visualization/visualize.py` + `__init__` (`EulfsVis`, only `pipeline.py`), MCC `transitions_utils.py` / `load_data_utils.py` / `text_cleaning_utils.py`.

**Correction to a prior expectation:** the Kanders MCC "transition machinery" (`transitions_utils`, `load_data_utils`) is **not live** — referenced by **0 notebooks**, only by other dead MCC modules. The only live MCC code is `compare_nodes_utils` (nb03) and `cluster_utils` (nb05).

### 4.2 Data — 1,285 files scanned

**KEEP (read by live code):** ESCO v1.1.0 (`occupationSkillRelations.csv`, `ISCOGroups_en.csv`, `greenSkillsCollection_en.csv`, `digCompSkillsCollection_en.csv`, `GreenBrownSkillsValidationETH.xlsx`), ESCO v1.0.3 (`ISCOGroups_en.csv`, `occupationSkillRelations.csv`), `classifications/NACE_REV2_*` (3), `geodata/` NUTS shapefiles **+ sidecars**, `interim/esco/occ_skills_matrix.pkl`, `interim/skills_coreness_measure.csv`, `interim/upskilling_analysis/upskilling_best_100_skills_per_isco3d_occupation_merged.csv`, `processed/esco/{skills_network_metrics,occ_skills_matrix,final_gbn_shares_by_isco_unweighted_new}.pkl`, `processed/esco/ZaussingerSchmidtEgli2025_OccupationClassificationESCOv1.1.csv`, `eurostat_data/interim/{eu_lfs_merged_2023, clean_eu_lfs_merged_2023_..._imputed}.{csv,pkl}` (live via `{year}` template at `lfs.py:413`), `eurostat_data/raw/Yearly_Data/YearlyFiles_2023/`, and **7 MCC inputs (~12 MB)** (`ESCO_ONET_xwalk_full.csv`, `esco_onet_crosswalk_Nov2020.csv`, `ONET_to_US2010SOC.xlsx`, `ESCO_skills_hierarchy.csv`, `ESCO_occupations_{Job_Zones,COVID_Exposure}.csv`, `codebase/.../skills_coreness_measure.csv`).

**SAFE TO REMOVE (high confidence):** committed `data/raw/mapping-career-causeways/.venv/` (807 files, 11 MB); root `data/*.BACKUP_DO_NOT_IMPORT` (7); `data/__pycache__/`; bulk of the MCC data tree except the 7 live files (~175 MB); the 5 `=*` pip-junk files at repo root; review-copy-only foreign data (111 files: KldB, `tobi`, brown/green expert lists, `survey/`, `eu-jtf/`, `_4manuscript/` (2019), German crosswalks) — **0 read by live code, no re-run gap**.

**UNCERTAIN (needs judgement):** `processed/esco/` alt-format siblings (`.csv/.xlsx` of `.pkl`s, ~10); `eurostat_data/**` `*_old-04/052025*` and `summary_review_table_2023.*`; `raw/metadata/` (5) and other config-templated areas the literal scan under-counts; MCC "read-by-dead-only" (5: `ESCO_occupational_hierarchy.csv`, `ESCO_skills_concepts_hierarchy.csv`, `ESCO_occupations_Remote_Labor_Index.csv`, `sim_matrices/OccupationSimilarity_Combined.npy`, `models/feasibility_model.pkl`).

**Foreign-artifact confirmation:** no **live** read path touches KldB / `tobi` / brown-green expert lists / survey / 2019 `_4manuscript` / eu-jtf / German crosswalks / CP2011. The only KldB/CP2011 references are **docstrings inside the dead `framework.Crosswalks` methods** (`esco_de_kldb2010`, `it_cp2011_de_kldb2010`).

**Hardcoded / machine-specific paths:** `notebooks/03` `os.chdir("/Users/go82gax/.../Simulating-Re-Skilling-Journeys/")`; several notebook cells embed absolute `/Users/go82gax/...` data paths. `src/` and `configs/` are clean (route through `useful_paths`).

---

## 5. Reproducibility

- **Seeded:** worker-group order — `np.random.RandomState(42)` + `.sample(random_state=...)` at `reskilling.py:1008-1009` (`simulate()`), `1746`/`1893` (`simulate_regional()`).
- **NOT seeded, re-randomized every step:** green draw (`1520`), digital draw (`1503`), coreness-weighted draw (`1464`) — all `.sample(n=1)` without `random_state`. **Green/digital are not reproducible today.**

**Task A dissolves this:** replacing the random green/digital draws with a deterministic coreness sort makes those programs deterministic (no seed needed). After A, all four standardized programs are reproducible (transferable = `coreRanked` deterministic, tailored = `optimal` deterministic). Only **Task B introduces new stochasticity** (share-weighted destination draw), which is seeded from the start.

---

## 6. Runtime

**`M_oo` is recomputed in full at every skill step** — `reskill()` at `reskilling.py:1524-1532` does a full `occ_skills_mat.copy()` + dense `np.dot(M_os, M_os.T)` per worker × per step (4 programs × 2 flows × 20–30 steps). Only one row of `M_os` changes per call, so an incremental row/column update is exact and far cheaper (see Phase 1.1). Expected ~100× on the dominant op (N ≈ 130 ISCO-3 occupations), full run hours → minutes.

**Sample/test mode:** none explicit, but `simulate(countries=["DE"], reskilling_journey_length=…)` is a de-facto subset switch (`get_occs(lfs_country_subset=…)` at `269`). Phase 1.2 wraps this.

### 6.1 Phase-1 profiling update (record kept honest for the appeal)

The pre-fix paragraph above (and CLAUDE.md §6) assumed `M_oo` recompute was the runtime bottleneck. **Profiling `simulate_regional` (DE/shortage/coreRanked, journey-3) disproved this** and relocated the bottleneck twice:

| Stage | Total runtime (same case) | Dominant cost |
|---|---|---|
| Baseline (before any change) | 241.8 s | `coreness_ranked` selection list-comp `reskilling.py:1483` (~100 s, ~41%); `M_oo` recompute only ~3.5% |
| After Phase 1.1 (incremental `M_oo`) | ~same | unchanged — `M_oo` was never the bottleneck (~0 wall-clock gain) |
| After Phase 1.1b (selection fix) | **77.0 s (3.1× faster)** | now the pandas `.copy()` layer in `find_closest` / share-filtering (`managers.py:557`, ~19.6 s, ~25%) |

**Conclusions recorded:**
- **Phase 1.1 (incremental `M_oo`) is kept** as a correct, free minor win (behaviour-preserving, max\|Δ\|=0), but it is **not** a meaningful speedup here. The earlier "~100×" / "hours → minutes from `M_oo`" estimate was wrong.
- **Phase 1.1b (coreness selection)** delivered the real Phase-1 speedup (**3.1×** so far) by removing ~13,891 per-call pandas `.loc` lookups and memoising the ordered `rem` per `idx_occ`.
- The **next** profiled target is the `find_closest` / `.copy()` layer (~25%), deferred — to be chosen from a fresh profile, not guessed.

### 6.2 Resolved (Task A): journey-aware not-yet-held filter

**Open question (raised in Phase 1):** in the `coreness_ranked` selection the set of skills an occupation already holds (`have`) was read from the **baseline** matrix `self.occ_skills_mat_3d`, not from the per-worker matrix that accumulates skills during the journey — so the "not-yet-held" filter did **not** exclude skills acquired earlier in the same journey. Phase-1.1b preserved this exactly rather than silently changing it.

**Decision (Phase 2):** this baseline-only behaviour is a **deviation from the model's stated assumption** (acquired skills stick and should not be re-offered). It is **corrected to journey-aware** (the filter tracks within-journey acquisitions), made the **default** in commit `0a7aa4a` — a deliberate correctness fix, separate from Task A's core green/digital change (`62dbcd3`). Baseline-only is retained behind `rp.journey_aware = False` for the disable-switch regression and an SI robustness comparison.

**Measured effect** (sample DE/HR/CY/LV, `at_risk`, no-regC, journey-12; COEFFY-weighted mean `n_viable_transitions`):

| country | n | intensity @ step 4 | intensity @ step 12 | workers whose 1st-transition step shifts |
|---|---|---|---|---|
| DE | 620 | 0.666 → 0.666 (+0.0%) | 1.123 → 1.099 (−2.2%) | 13.1% |
| HR | 61 | 0.699 → 0.699 (+0.0%) | 1.279 → 1.206 (−5.6%) | 9.8% |
| CY | 20 | 0.776 → 0.776 (+0.0%) | 1.356 → 1.296 (−4.4%) | 5.0% |
| LV | 18 | 0.651 → 0.651 (+0.0%) | 1.014 → 0.946 (−6.7%) | 0.0% |

**Scope of the change:** confined to **transferable**. The effect is **0% at step 4** and **−2% to −7% at step 12** (journey-aware unlocks slightly fewer transitions; the gap grows with step because that is where the global-coreness walk begins hitting skills the occupation already holds). **Green/digital are bit-identical under either mode** (greedy pick ≡ rank-walk) and **tailored is unaffected by construction**. The shift slightly *widens* tailored's advantage over transferable, consistent with the paper's narrative, and the headline ("at least two-thirds fewer skills") is unaffected. Exact corrected figures come from the single end-of-phase full-country run; the before/after table for the response letter is in `revision/` (Task A deliverable).

### 6.3 Resolved (Task B): employment-share weighting of destinations

**Change.** In the regional (regC) destination pick (`reskilling.py` `_select_destination`), instead of every worker taking the single top-income feasible target, feasible targets are weighted by **NUTS-2 employment share** (`COEFFY`, already merged in the regC path). Draws are **deterministically seeded per draw** (`_draw_seed(share_seed, step, region, worker-index)`) so they are order-independent and a post-hoc replay reproduces a live run exactly. regC-only (no-regC has no employment merge). **Disable switch** `destination_weighting="off"` returns the top-income target — verified to reproduce review-copy exactly (DE coreRanked regC, steps 0–12, max|Δ|=0, 0 decision cols).

**Shipping rule: `above_current` (parameter-free wage floor).** Among targets paying **≥ the worker's current wage** (the income-preference assumption already in the paper), draw by employment share; fall back to all feasible if none qualify. Chosen over a `band=b` rule to avoid an unmotivated "why 25%?" free parameter, and because it **preserves the baseline income-loss share** while delivering ~the reviewer-requested employment-realism gain.

**Dirty-vs-clean income correction (important).** The first sweep used DE+HR, but **both are in the income-exclusion (DROP) list** (DE 25%, HR 0% observed earnings) — so its income column was invalid (it priced imputed earnings as observed). The income exclusion is enforced **only at the reporting stage** (notebook-06 `iso_missing_earnings`), not in the model core, so the standalone sweep didn't inherit it; **published figures are unaffected**. The sweep was re-run on income-**eligible** + regC-surviving countries **SE+SK+DK** (98–99% observed earnings). The employment-share frontier is robust to this (it needs no earnings) and was unchanged in shape.

**List-drift check.** notebook-01 DROP (18 countries) vs notebook-06 `iso_missing_earnings` (15) differ only by **{SI, BG, MT}**, which are excluded from the simulation entirely (ISCO 1–2-digit coverage) and never reach reporting — **benign, but a latent drift** → see Phase-3 codify action.

**Validation (licensed).** Post-hoc capture reproduces a genuine live band run **exactly**: SE+SK+DK, 18,058 worker choices — mode-invariance 0 failures, per-worker destination identity 0 mismatches, income-delta 0 mismatches.

**Committed result (above_current, income-eligible SE+SK+DK; COEFFY-weighted):**

| flow | rule | mean Δ€ | %income-loss | empl. share (rel. to off) |
|---|---|---|---|---|
| AT_RISK (outward) @20 | off (Phase-1) | +7,055 | 20.1% | 1.00× |
| | **above_current** | **+2,483** | **20.1%** | **1.84×** |
| SHORTAGE (inward) @30 | off (Phase-1) | +18,000 | 3.9% | 1.00× |
| | **above_current** | **+15,980** | **3.9%** | **1.25×** |

`above_current` holds income-loss at the Phase-1 baseline (20.1% / 3.9%) while landing workers in 1.84×/1.25× higher-employment destinations. **Effect is outward-dominant** (inward is **modestly affected, not flat** — share 1.00×→1.25×; the earlier "flat inward" was a DE+HR imputed-income artifact). The **`band=b` sweep is retained in the code and reported in the SI as a robustness variant**: results are monotone across bands (off→share_only), conclusions don't hinge on the exact rule, and the parameter-free above-current rule is the conservative reading of the paper's existing income-preference assumption.

### 6.4 Built (Task C): program cost post-processing — `revision/taskC_cost.py`

**Scaffold now, euro numbers later.** Reads per-program steps-to-first-transition (the figures' intensity quantity) and combines with per-participant cost bands → a Results/SI cost table. Bands are **unmistakable placeholders** (`1111/2222/3333/4444` €/skill — German voucher / Spain / OECD / WEF); maintainer supplies real values.

**Inheritance: clean / filter-independent.** Uses ONLY `transition_viable_step_*` (journey count) + `COEFFY` (population). `assert_no_income_columns_used` enforces no earnings column is ever read, so Task C needs **no income KEEP-list**. Flagged in code: an earnings-derived band would require the KEEP list (which lives only in notebook-06 until the Phase-3 codify).

**Framing (maintainer-decided):**
- **Both flows, lead inward.** The cheap-exit vs expensive-entry contrast *is* the inward/outward asymmetry the paper reports, now in euros.
- **Two cost columns, not a blend.** Headline = **€/participant (spend)** + **%-reached (yield)**, kept separate. The blended "€/reached-transition" is reported only as a **secondary** column for high-reach programs (≥50%), because for low-reach programs it is a denominator artifact (green €14.7M, digital undefined). The split lets digital appear honestly (0% reach, real €/participant) and answers R2's ROI ask robustly (spend vs yield, not denominator-driven).
- **Conditioned on needing ≥1 skill;** the **0-skill share** (reach a transition with no reskilling) is reported alongside as context — itself a result.

**Scaffold-test numbers (review-copy, pre-A–E; placeholder bands):** the structure surfaces the headline cleanly — **inward** 0-skill 0%, transferable/tailored ~100% reach at ~23–24 skills, green/digital spend real money per participant at **0.2% / 0% reach**; **outward** **96.3% reach with zero skills**, the few needing reskilling done by tailored in ~5 vs transferable ~10.5 skills. Final table runs on the post-A–E figures.

---

## 7. Change-scope table (Tasks A–E)

| Task | Files / lines | Data ready? | Full re-run? | Risk | Effort |
|---|---|---|---|---|---|
| **A** coreness green/digital | `reskilling.py:1497-1521` (reuse `1466-1495`) | coreness `__init__:221`; URI lists `240-262` — yes | yes (green/digital × 2 flows) | low; removes RNG | ½ d |
| **B** employment-share weighting | `simulate()` `1094-1118` (mask `1076-1080`); `simulate_regional()` `1849-1882` (`1849-1853`) | shares `share_cols` `980/1153/1710` — yes | yes | med; new seeded draw | 1 d |
| **C** cost post-processing | new script; reads `transition_viable_step_{step}` `1142-1197` from pkls | yes (saved pkls) | no | low | ½ d |
| **D** optional-weight sweep | `framework.py:1003` (`weight_optional=0.5`); threshold `reskilling.py:263`, `573` | yes | yes ×3 weights | med; threshold coupling | ½ d + runs |
| **E** symmetric employment | `reskilling.py:1177` (`-1 * annual_earnings`) | yes | only the branch | low | ¼ d |

---

## 8. Options analysis & decision

Four routes were compared (Minimal / Minimal+perf / Fully-clean same-ESCO / Rebuild on newest ESCO).

**Decision: Route 3 (fully clean, ESCO v1.1.0) as the end state, sequenced through Route 2.** Not Route 4 and **no from-scratch rebuild** for the appeal — re-basing onto newer ESCO breaks comparability with the Zaussinger et al. v1.1.0 classification the paper depends on, and forces defending new numbers under deadline. Route 3 keeps the cited science identical while delivering one reproducible environment.

---

## 9. Approved sequenced implementation plan

Three gated phases + a git step 0. Effort excludes long runs. **Resolved decisions are inlined.**

### Step 0 — Git setup
- Archive stale stash → **`~/reskilling-old-wip.patch`** (outside the repo), then `git stash drop`.
- Commit the 3 `.txt` shim deletions.
- **`git mv` `notebooks/isco3_full_summary.csv` → `data/processed/esco/`.** NOTE: `data/` is gitignored, so the file is **preserved on disk but becomes untracked** (consistent with all other data artifacts; nothing reads or regenerates it).
- Tag `pre-revision-2026-06`; branch `revision`.

### Phase 1 — Behaviour-preserving
- **1.0 Baseline equality:** diff current `coreRanked`+`optimal` pkls vs `review-copy` (`assert_frame_equal`). Establishes the golden reference. **Entry gate.**
- **1.1 Incremental `M_oo`** (`reskilling.py:1524-1532`): update one cell, recompute only row/col `idx_occ`, carry `M_oo` across steps. **Bit-exact at weights {0.5,1.0}** (M_os ∈ {0,0.5,1} ⇒ M_oo entries are exact sums of {0,0.25,0.5,1} ≪ 2⁵³, no rounding). Equality test: run old+new in parallel on a small case, assert `np.array_equal` per step; assert the feasible-target set (overlap ≥ threshold) is identical (no 3.7 flip). Under Task D's non-dyadic weights, equality relaxes to `atol≈1e-9`.
- **1.2 Small-sample test mode:** wrapper around `simulate*` (DE, one scenario, short journey).
- **1.3 Prune** high-confidence dead code on/near the model path (defer the 3 uncertain code files to Phase 3).
- **GATE:** full run reproduces the frozen `review-copy` figures — `coreRanked`+`optimal` exact; green/digital **distributional only** (accepted, unavoidable pre-Task-A). Stop here for maintainer review.

### Phase 2 — Behaviour-changing (every step behind a disable switch + regression test back to prior numbers)
- **A** coreness green/digital. Then **regenerate affected figures and produce a before/after table of the manuscript-quoted values** (4-skills & 12-skills intensities; per-country aggregate income losses) for referee reconciliation.
- **B = B2** (seeded share-weighted destination draw at the selection sites). Uniform-weight switch must **collapse exactly to Phase-1 numbers**. Before/after table updated after B runs.
- **C** cost post-processing — cost bands in a clearly-marked config block (**placeholders until maintainer supplies figures**).
- **D** optional-weight sweep {0,0.5,1.0}; **re-derive the threshold per weight** by the same rule (stated explicitly); weight 0.5 must reproduce Phase-1.
- **E** symmetric employment — **reporting-only, default off**.

### Phase 3 — Clean repro environment (gate: identical numbers to end of Phase 2)
- Single env (numba ≥0.61 upgrade vs numpy-1.26 pin — **decided by nb03 smoke test**: diff the regenerated `upskilling_best_100_skills_...csv`).
- Vendor live MCC subset (`compare_nodes_utils`, `cluster_utils`) into `src/`; drop foreign shells.
- Fix hardcoded `/Users/go82gax/...` paths.
- Consolidate requirements → one pinned env; delete `=*` junk.
- End-to-end seeding (one configurable seed).
- **Codify the income-eligibility filter** (NEW, from Task B): the income-subset exclusion currently lives only in notebook-06 (`iso_missing_earnings`) and was found to have drifted from notebook-01's list (by the sim-excluded {SI,BG,MT}). Lift it into `src/` as a single enforced filter (single source of truth) inherited automatically by notebook-06, the Task B sweep, and the cost analysis (Task C). **Codifies the existing reporting filter; changes no published number.**

### Resolved ⚖️ decisions
1. Stash patch → `~/reskilling-old-wip.patch`.
2. CSV → `git mv` to `data/processed/esco/` (preserved on disk, untracked).
3. Green/digital distributional-only check pre-Task-A — accepted.
4. Task B = **B2**, seeded, with uniform-weight regression switch.
5. Task C cost bands — config placeholders, maintainer to supply.
6. Task D — re-derive threshold per weight.
7. Task E — reporting-only, default off.
8. Phase 3 env — decided by nb03 smoke test.
9. Every behaviour-changing step (A/B/D/E) behind a disable switch with regression test to prior numbers.

---

*Investigation was read-only. Implementation begins at Step 0 on branch `revision`; the maintainer reviews at the Phase 1 gate before any Phase 2 change.*
