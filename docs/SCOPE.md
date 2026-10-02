# Repository scope

## Included (public)

- Analysis library for \(\rho\), \(A\), descriptors, and variance partitioning
- Figure scripts that remake the main manuscript analysis figures from tables
- Experiment and metric-pair configuration matching the paper (42 catchments, 9 pairs)
- Dischma / WUS experiment YAML snapshots used for transparency
- Seed catchment-descriptor CSV tables
- Path-sanitized legacy analysis notebook (provenance only)

## Not included (by design)

1. **Hydrological model execution stack** — eWaterCycle wrappers, wflow Julia runners, forcing generation, SLURM job arrays, soil/yearly calibration. These are tied to institutional HPC paths and multi-terabyte outputs. The manuscript Methods already describe the setup; regenerating every ensemble is not required to evaluate the published conclusions once analysis tables are available.

2. **Paper 1 and Swiss operational code** — separate publication and site-ops tooling.

3. **Credentials and personal tokens** — never published.

4. **Raw ensemble NetCDFs** — too large for a typical GitHub/Zenodo software companion; Prefer Prior *metrics* CSVs and aggregated \(\rho\)/\(A\) tables on Zenodo.

## Requesting excluded material

Contact **Pau Wiersma** (`pau.wiersma@unil.ch`) if you need:

- Access to specific Prior ensemble outputs or config JSON for a catchment-year
- The calibration / simulation pipeline used to generate ensembles
- Clarification of experiment ID ↔ manuscript name mappings

Requests for bona fide research reuse will be handled case by case (storage, licenses of third-party inputs, and co-author agreement permitting).
