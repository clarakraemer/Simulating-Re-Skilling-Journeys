# Tailored reskilling widens feasible job transitions during Europe's decarbonization

Krämer, Zaussinger & Egli.

This study examines how the design of reskilling programs can facilitate job switches
during Europe's decarbonization. We develop a data-driven agent-based microsimulation of
*reskilling journeys*: worker groups (EU-LFS data points, carrying population weights)
incrementally acquire skills, and at each step we recompute which occupations become
skill-feasible. We compare four stylized programs — **green**, **digital**, **transferable**,
and **tailored** — across European (NUTS-2) regions, for two transition flows:
*outward* (leaving at-risk high-carbon jobs) and *inward* (into low-carbon shortage
occupations).

## Repository structure

| Path | Purpose |
|---|---|
| `src/modelling/reskilling.py` | Core model (`ReskillingPathways`) — main simulation logic |
| `src/modelling/occupation_distance.py`, `src/utils.py`, `src/plotting_utils.py` | Supporting modules |
| `src/mapping_career_causeways/` | Occupation-similarity utilities adapted from Nesta et al. (2020) |
| `notebooks/01–05` | Data preparation (EU-LFS + ESCO preprocessing, tailored-reskilling table, occupation classification, skill centrality) |
| `notebooks/06-ck-lfs-model-visualisations.ipynb` | All figures and tables |
| `configs/` | Path and parameter configuration (`paths_config.yml`) |
| `revision/` | Reproducibility harness for the revision (see below) |
| `data/`, `results/` | Inputs and outputs — git-ignored (large / restricted access) |

## Environment

Pinned environments are provided:

- `requirements.server.txt` — the exact environment used for the production runs (server).
- `requirements.txt` — local development.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.server.txt
```

## Data

`data/` and `results/` are git-ignored (large files; EU-LFS is restricted-access). To obtain the inputs:

1. Register for **EU-LFS 2023** microdata with Eurostat (restricted access).
2. Download **ESCO v1.1.0** (publicly available from the ESCO portal).
3. Obtain the **occupational classification** from Zaussinger et al. (2025).
4. Place all inputs under the project root following the structure in `configs/paths_config.yml`.

Additional public Eurostat metadata (NUTS-2 region names, EU-SILC) is specified in `configs/`.

## Reproduce

1. Run notebooks `01–05` for data preparation.
2. Run the simulation: either `src/modelling/reskilling.py` (serial), or the parallel driver
   `revision/run_parallel.py` (one worker process per country; bit-identical to serial — see its docstring).
3. Run notebook `06-ck-lfs-model-visualisations.ipynb` for all figures and tables.

## Revision (`revision/`)

Scripts and notes supporting the revision, kept for reproducibility:

- `run_parallel.py` — parallel run driver (per-country worker processes).
- `regenerate_maps.py` — plot-only regeneration of the NUTS-2 choropleths from existing pickles.
- `taskD_*.py` — optional-skill-weight robustness (w = 0 vs 0.5, threshold held fixed).
- `INVESTIGATION.md`, `green_program_incidence.md` — analysis notes.

## License

See `LICENSE`.
