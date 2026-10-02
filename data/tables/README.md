# Analysis tables

Included with this repository (also part of the Zenodo archive):

| File | Description |
|------|-------------|
| `all_pairs.csv` | Catchment-year Spearman \(\rho\) for nine SWE–Q metric pairs, with catchment descriptors |
| `asymmetry.csv` | Catchment-year asymmetry ratio \(A\) for the same pairs |
| `variance_partition_*.csv` | Written when running `scripts/02_make_figures.py` |
| `wus_catchment_overview*.xlsx` | Western US catchment metadata |

Reproduce figures:

```bash
python scripts/02_make_figures.py
```
