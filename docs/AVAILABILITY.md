# Data and software availability (draft for WRR / AGU)

Fill in DOIs after the Zenodo deposits are created. Paste a shortened version into the manuscript Open Research Statement.

## Software

Version **0.1.0** of the *snow-streamflow-predictability* analysis code, used to compute snow–streamflow predictability metrics (\(\rho\), \(A\)), catchment descriptors, variance partitioning, and to produce the manuscript analysis figures, is preserved at Zenodo (DOI: **TBD**) under the MIT license, and developed openly at GitHub (**TBD URL**).

Suggested reference citation:

> Wiersma, P., Lundquist, J. D., & Mariéthoz, G. (2026). snow-streamflow-predictability (Version 0.1.0) [Software]. Zenodo. https://doi.org/TBD

## Data

Processed ensemble skill metrics, catchment–year descriptors, and analysis tables supporting the results are available at Zenodo (DOI: **TBD**) under CC-BY-4.0.

Suggested reference citation:

> Wiersma, P., Lundquist, J. D., & Mariéthoz, G. (2026). Analysis tables for intrinsic limits of snow–streamflow predictability [Dataset]. Zenodo. https://doi.org/TBD

## Third-party data and software (cite originals)

- ERA5-Land meteorological forcing (Muñoz-Sabater et al., 2021)
- USGS National Water Information System discharge (U.S. Geological Survey)
- wflow_sbm hydrological model (van Verseveld et al., 2024)
- eWaterCycle platform (Hut et al., 2022)
- HydroMT model building (Eilander et al., 2023)
- SPOTPY sampling / calibration utilities (Houska et al., 2015)
- Observational illustration in Fig. 1 only: WUS-SR SWE reanalysis (Fang et al., 2022); OSHD (Mott et al., 2023); BAFU / USGS gauges

## Scope note

Full ensemble regeneration (500 Latin Hypercube samples × 42 catchments × water years 2001–2022) requires HPC resources and the private simulation stack; it is documented in the manuscript Methods but is not distributed as a turnkey public workflow. See [`SCOPE.md`](SCOPE.md).
