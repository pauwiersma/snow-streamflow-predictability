# Data and software availability (draft for WRR / AGU)

Fill in the DOI and GitHub URL after the Zenodo archive is published. Paste a shortened form into the manuscript Open Research Statement.

## Combined code + data archive (single DOI)

Version **0.1.0** of the *snow-streamflow-predictability* package — analysis code and the supporting catchment–year tables used to produce the manuscript figures — is preserved at Zenodo (DOI: **[add after deposit]**) and developed openly at GitHub (**[add URL]**). Code is available under the MIT license; analysis tables under CC-BY-4.0.

Suggested citation:

> Wiersma, P., Lundquist, J. D., & Mariéthoz, G. (2026). snow-streamflow-predictability: analysis code and data for intrinsic limits of snow–streamflow predictability (Version 0.1.0) [Software]. Zenodo. https://doi.org/[DOI]

The GitHub repository already contains the analysis tables under `data/`. Zenodo provides the persistent, citable snapshot required by AGU (a separate data-only Zenodo record is not required for this package).

## Third-party data and software (cite originals)

- ERA5-Land meteorological forcing (Muñoz-Sabater et al., 2021)
- USGS National Water Information System discharge (U.S. Geological Survey)
- wflow_sbm hydrological model (van Verseveld et al., 2024)
- eWaterCycle platform (Hut et al., 2022)
- HydroMT model building (Eilander et al., 2023)
- SPOTPY sampling utilities (Houska et al., 2015)
- Fig. 1 observational illustration only: WUS-SR SWE reanalysis (Fang et al., 2022); OSHD (Mott et al., 2023); BAFU / USGS gauges

## Note on ensemble regeneration

Full regeneration of the 500-member ensembles for all catchment-years requires HPC resources and is outside this archive; see [`SCOPE.md`](SCOPE.md).
