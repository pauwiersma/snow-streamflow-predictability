#!/usr/bin/env python3
"""Build manuscript analysis tables from ensemble skill metrics (if available).

Expects per-experiment yearly metrics CSVs under::

    $SSP_DATA_ROOT/priors/{EXP_ID}/{year}_metrics.csv

Writes::

    $SSP_DATA_ROOT/tables/all_pairs.csv
    $SSP_DATA_ROOT/tables/asymmetry.csv

If Prior metrics are not deposited yet, skip this script and use the Zenodo
analysis tables directly with ``02_make_figures.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

# Allow running without install
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ssp.paths import config_dir, tables_dir, data_root
from ssp.predictability import (
    asymmetry_ratio,
    prepare_skill_series,
    spearman_predictability,
)


def load_configs():
    with open(config_dir() / "experiments.yaml") as f:
        experiments = yaml.safe_load(f)
    with open(config_dir() / "metric_pairs.yaml") as f:
        metrics = yaml.safe_load(f)
    return experiments, metrics


def experiment_map(experiments: dict) -> dict:
    out = {}
    for exp_id, meta in experiments["dischma_variants"].items():
        out[exp_id] = meta["name"]
    for exp_id, meta in experiments["wus_catchments"].items():
        out[exp_id] = meta["name"]
    return out


def main(priors_dir: Path | None = None) -> None:
    experiments, metrics = load_configs()
    names = experiment_map(experiments)
    years = range(experiments["years"]["start"], experiments["years"]["end"] + 1)
    pairs = metrics["pairs"]
    frac = metrics.get("asymmetry_frac_best", 0.1)

    root = Path(priors_dir) if priors_dir else data_root() / "priors"
    if not root.exists():
        print(
            f"Prior metrics directory not found: {root}\n"
            "Place yearly {{year}}_metrics.csv files under priors/{{EXP_ID}}/, "
            "or download analysis tables from Zenodo and run 02_make_figures.py."
        )
        sys.exit(0)

    rho_rows, a_rows = [], []
    for exp_id, display in names.items():
        for year in years:
            path = root / exp_id / f"{year}_metrics.csv"
            if not path.exists():
                # also try flat layout
                path = root / exp_id / f"{year}_metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, index_col=0)
            for pair in pairs:
                swe_m, q_m = pair.split("__", 1)
                if swe_m not in df.columns or q_m not in df.columns:
                    continue
                swe = prepare_skill_series(df[swe_m], swe_m)
                q = prepare_skill_series(df[q_m], q_m)
                rho = spearman_predictability(swe, q)
                a = asymmetry_ratio(swe, q, frac_best=frac)
                basin = display.split("_")[0] if "_" in display else display
                rho_rows.append(
                    {
                        "EXP_ID": display,
                        "BASIN": basin,
                        "year": year,
                        "QSWE_pair": pair,
                        "correlation": rho,
                    }
                )
                a_rows.append(
                    {
                        "EXP_ID": display,
                        "BASIN": basin,
                        "year": year,
                        "QSWE_pair": pair,
                        "asymmetry_ratio": a,
                    }
                )

    out = tables_dir()
    out.mkdir(parents=True, exist_ok=True)
    rho_df = pd.DataFrame(rho_rows)
    a_df = pd.DataFrame(a_rows)
    rho_df.to_csv(out / "all_pairs.csv", index=False)
    a_df.to_csv(out / "asymmetry.csv", index=False)
    print(f"Wrote {len(rho_df)} rho rows and {len(a_df)} asymmetry rows to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--priors", type=Path, default=None, help="Directory of Prior metrics")
    args = p.parse_args()
    main(args.priors)
