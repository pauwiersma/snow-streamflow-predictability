# Snow–streamflow predictability

Analysis code accompanying:

> Wiersma, P., Lundquist, J. D., & Mariéthoz, G. *Intrinsic limits of predictability in snow–streamflow inference.* Manuscript submitted to *Water Resources Research*.

This repository provides a **clean, paper-scoped** package to:

1. Compute mutual SWE–Q predictability (Spearman \(\rho\)) and skill-transfer asymmetry \(A\)
2. Compute catchment descriptors (\(\tau\), \(f_\mathrm{rain}\), …) and variance partitioning
3. Reproduce the main analysis figures from deposited analysis tables
4. Document the 42 synthetic catchments (16 Dischma variants + 26 Western US) and the 9 metric pairs

## Quick start

```bash
cd snow-streamflow-predictability
python -m venv .venv && source .venv/bin/activate
pip install -e ".[plot]"

# After placing Zenodo tables in data/tables/ (all_pairs.csv, asymmetry.csv):
python scripts/02_make_figures.py
```

Optional: rebuild \(\rho\) / \(A\) tables from Prior ensemble metrics:

```bash
export SSP_DATA_ROOT=/path/to/data   # contains priors/{EXP_ID}/{year}_metrics.csv
python scripts/01_build_analysis_tables.py
python scripts/02_make_figures.py
```

## Repository layout

```text
config/           Experiment IDs, metric pairs, predictors
configs/yml/      wflow experiment YAML used for the 16 Dischma + WUS setups
src/ssp/          Portable analysis library (predictability, descriptors, plots)
scripts/          Figure and table entry points (+ legacy notebook-style script)
data/descriptors/ Catchment-year predictor tables (seed)
data/tables/      Analysis tables (download from Zenodo; see README there)
docs/             Availability statement draft + scope notes
```

## What this repository does **not** include

This is intentionally **not** a dump of the full research codebase. The following are **excluded**:

| Excluded | Why |
|----------|-----|
| Full wflow / eWaterCycle / Julia ensemble runner, SLURM scripts, soil & yearly calibration stack | HPC-specific, multi-TB intermediates, and not needed to evaluate the published \(\rho\)/\(A\) results once analysis tables are deposited |
| Paper 1 synthetic Dischma LOA workflows and related scripts | Different study |
| Swiss station / OSHD / MeteoSwiss ops tooling, Google Drive downloaders, delete/cleanup utils | Operational clutter unrelated to this manuscript |
| Secrets (`credentials.json`, OAuth tokens) | Must never be published |
| Raw 500-member SWE/Q NetCDF ensembles for all catchment-years | Size; analysis-ready metrics and tables are the appropriate AGU deposit |

**If you need any of the excluded materials** (e.g. to regenerate ensembles from scratch, inspect a specific Prior NetCDF, or reuse the calibration pipeline), **contact the corresponding author directly**: Pau Wiersma (`pau.wiersma@unil.ch`). Reasonable research requests will be accommodated where licensing and storage allow.

A path-sanitized copy of the original long-form analysis notebook is kept only as `scripts/legacy_full_analysis_notebook.py` for provenance; it still depends on private Prior outputs and the broader ewc stack and is **not** the supported public interface.

## Data and AGU availability

Processed analysis tables and a software snapshot will be archived on **Zenodo** (DOI to be added) and cited in the manuscript Open Research Statement. Third-party inputs (ERA5-Land, USGS NWIS, wflow_sbm, eWaterCycle) should be cited from their original sources — see [`docs/AVAILABILITY.md`](docs/AVAILABILITY.md).

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite both the paper and the software/data DOIs once available.

## License

MIT — see [`LICENSE`](LICENSE).
