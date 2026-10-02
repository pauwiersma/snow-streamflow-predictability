#!/usr/bin/env python3
"""Reproduce manuscript summary figures from analysis tables.

Reads (from ``$SSP_DATA_ROOT/tables`` or ``./data/tables``)::

    all_pairs.csv       # columns: EXP_ID, year, QSWE_pair, correlation, (+ descriptors)
    asymmetry.csv       # columns: EXP_ID, year, QSWE_pair, asymmetry_ratio

Also regenerates the asymmetry schematic (Fig. 8).

Writes PNG/SVG under ``data/figures/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ssp.paths import config_dir, figures_dir, tables_dir
from ssp.plotting.figures import (
    plot_asymmetry_boxen,
    plot_hexbin_rho,
    plot_rho_boxen,
    plot_variance_partition_bars,
)
from ssp.plotting.asymmetry_schematic import plot_schematic
from ssp.variance_partition import variance_partition


def main() -> None:
    with open(config_dir() / "metric_pairs.yaml") as f:
        cfg = yaml.safe_load(f)
    pairs = cfg["pairs"]
    highlighted = [cfg["highlighted"]["dynamics"], cfg["highlighted"]["volume"]]
    predictors = cfg["predictors"]

    tables = tables_dir()
    figs = figures_dir()
    rho_path = tables / "all_pairs.csv"
    a_path = tables / "asymmetry.csv"

    if not rho_path.exists():
        print(
            f"Missing {rho_path}.\n"
            "Download the Zenodo analysis tables into data/tables/, "
            "or run scripts/01_build_analysis_tables.py if you have Prior metrics."
        )
        # Still produce schematic
        plot_schematic(outfile=figs / "fig08_asymmetry_schematic.png", seed=7)
        print(f"Wrote schematic to {figs}")
        return

    rho = pd.read_csv(rho_path)
    plot_rho_boxen(rho, pairs, outfile=figs / "fig11_rho_boxen.png")
    plot_rho_boxen(rho, highlighted, outfile=figs / "fig10_highlighted_pairs_boxen.png")

    # Hexbin for highlighted pairs if descriptor columns present
    for pair, tag in zip(highlighted, ["dynamics", "volume"]):
        sub = rho[rho["QSWE_pair"] == pair]
        xcol = next((c for c in ("temp_storage_spring", "S_dynamic_fraction_spring") if c in sub.columns), None)
        ycol = next((c for c in ("spring_rf_fraction", "rainfall_mixing_ratios") if c in sub.columns), None)
        if xcol and ycol and len(sub):
            plot_hexbin_rho(
                sub, x=xcol, y=ycol, rho="correlation",
                outfile=figs / f"fig13_hexbin_{tag}.png",
            )

        present = [p for p in predictors if p in sub.columns]
        if present and len(sub) > 20:
            vp = variance_partition(sub, "correlation", present)
            plot_variance_partition_bars(
                vp, outfile=figs / f"fig12_vp_{tag}.png", title=tag
            )
            vp.to_csv(tables / f"variance_partition_{tag}.csv", index=False)

    if a_path.exists():
        asym = pd.read_csv(a_path)
        plot_asymmetry_boxen(asym, pairs, outfile=figs / "fig14_asymmetry_boxen.png")

    plot_schematic(outfile=figs / "fig08_asymmetry_schematic.png", seed=7)
    print(f"Figures written to {figs}")


if __name__ == "__main__":
    main()
