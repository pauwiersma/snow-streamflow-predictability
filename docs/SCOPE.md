# Scope

## Included

- Python package to compute SWE–Q predictability \(\rho\), asymmetry \(A\), catchment descriptors, and variance partitioning
- Scripts to reproduce the main analysis figures from the included tables
- Configuration for the 42 study catchments (16 synthetic Dischma variants + 26 Western US) and nine SWE–Q metric pairs
- Analysis-ready tables (`data/tables/`) and catchment–year descriptors (`data/descriptors/`)
- Example wflow experiment YAML snapshots (`configs/yml/`)

## Not included

- The full model-run stack used to generate ensembles (forcing preparation, calibration, cluster orchestration, raw ensemble NetCDFs)
- Third-party meteorological forcing and gauge archives (cite originals; see [`AVAILABILITY.md`](AVAILABILITY.md))

Ensemble generation is documented in the manuscript Methods. Contact Pau Wiersma (`pau.wiersma@unil.ch`) for research requests involving simulation intermediates not deposited here.
