# Snow–streamflow predictability

Analysis code and data for:

> Wiersma, P., Lundquist, J. D., & Mariéthoz, G. *Intrinsic limits of predictability in snow–streamflow inference.* Manuscript submitted to *Water Resources Research*.

This repository contains everything needed to recompute the study’s predictability metrics (\(\rho\), asymmetry \(A\)), catchment descriptors, variance partitioning, and the main analysis figures (Figs. 8 and 10–14). Analysis tables for all 42 catchments and water years 2001–2022 are included under `data/`.

## Quick start

```bash
git clone <this-repo>
cd snow-streamflow-predictability
python -m venv .venv && source .venv/bin/activate
pip install -e ".[plot]"

python scripts/02_make_figures.py
```

Figures are written to `data/figures/`.

## Repository layout

```text
config/            Catchment list, metric pairs, predictors
configs/yml/       wflow experiment configuration snapshots
src/ssp/           Analysis library
scripts/           Figure reproduction
data/tables/       all_pairs.csv, asymmetry.csv (+ variance-partition outputs)
data/descriptors/  Catchment–year descriptors (τ, rainfall fraction, …)
docs/              Availability statement draft and scope notes
```

## What is not included

This archive supports evaluation and reuse of the **published analysis**. It does **not** include the full hydrological ensemble simulation pipeline (wflow / eWaterCycle runners, cluster job scripts, meteorological forcing archives, or raw multi-member SWE/Q NetCDF outputs). Those steps are described in the manuscript Methods; regenerating the ensembles requires substantial compute.

If you need access to simulation intermediates beyond what is deposited here, contact the corresponding author: **Pau Wiersma** (`pau.wiersma@unil.ch`).

## Archiving (GitHub + Zenodo)

Code and the accompanying analysis data live in this Git repository. A versioned snapshot (code + data) will be archived on **Zenodo** with a single DOI for citation — see [`docs/AVAILABILITY.md`](docs/AVAILABILITY.md). Cite third-party inputs (ERA5-Land, USGS, wflow_sbm, eWaterCycle, …) from their original sources.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper and the Zenodo DOI once available.

## License

Code: MIT ([`LICENSE`](LICENSE)).  
Analysis tables in `data/`: Creative Commons Attribution 4.0 (CC-BY-4.0).
