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
    # Prefer manuscript predictor names; fall back to aliases present in tables
    predictor_candidates = [
        ("temp_storage_spring", "S_dynamic_fraction_spring", "temp_storage"),
        ("spring_rf_fraction", "rainfall_mixing"),
        ("spring_et_fraction", "et_mixing"),
        ("catchment_area", "area_km2"),
        ("melt_hfi",),
        ("mean_slope_dem", "slope_proxy_relief"),
    ]

    tables = tables_dir()
    figs = figures_dir()
    rho_path = tables / "all_pairs.csv"
    a_path = tables / "asymmetry.csv"

    if not rho_path.exists():
        print(
            f"Missing {rho_path}.\n"
            "Expected data/tables/all_pairs.csv (shipped with this repository)."
        )
        plot_schematic(outfile=figs / "fig08_asymmetry_schematic.png", seed=7)
        print(f"Wrote schematic to {figs}")
        return

    rho = pd.read_csv(rho_path)
    if "correlation" not in rho.columns and "Year" in rho.columns:
        rho = rho.rename(columns={"Year": "year"})

    plot_rho_boxen(rho, pairs, outfile=figs / "fig11_rho_boxen.png")
    print("  wrote fig11_rho_boxen")
    plot_rho_boxen(rho, highlighted, outfile=figs / "fig10_highlighted_pairs_boxen.png")
    print("  wrote fig10_highlighted_pairs_boxen")

    # Resolve which predictor columns exist
    predictors = []
    for candidates in predictor_candidates:
        chosen = next((c for c in candidates if c in rho.columns and rho[c].notna().any()), None)
        if chosen:
            predictors.append(chosen)
    print(f"  predictors available: {predictors}")

    for pair, tag in zip(highlighted, ["dynamics", "volume"]):
        sub = rho[rho["QSWE_pair"] == pair].copy()
        xcol = next(
            (c for c in ("temp_storage_spring", "S_dynamic_fraction_spring", "temp_storage")
             if c in sub.columns and sub[c].notna().any()),
            None,
        )
        ycol = next(
            (c for c in ("spring_rf_fraction", "rainfall_mixing")
             if c in sub.columns and sub[c].notna().any()),
            None,
        )
        if xcol and ycol and len(sub.dropna(subset=[xcol, ycol, "correlation"])) > 5:
            plot_hexbin_rho(
                sub, x=xcol, y=ycol, rho="correlation",
                outfile=figs / f"fig13_hexbin_{tag}.png",
            )
            print(f"  wrote fig13_hexbin_{tag} ({xcol} vs {ycol})")

        present = [p for p in predictors if p in sub.columns]
        if present and len(sub.dropna(subset=present + ["correlation"])) > 20:
            vp = variance_partition(sub, "correlation", present)
            plot_variance_partition_bars(
                vp, outfile=figs / f"fig12_vp_{tag}.png", title=tag
            )
            vp.to_csv(tables / f"variance_partition_{tag}.csv", index=False)
            print(f"  wrote fig12_vp_{tag}")

    if a_path.exists():
        asym = pd.read_csv(a_path)
        plot_asymmetry_boxen(asym, pairs, outfile=figs / "fig14_asymmetry_boxen.png")
        print("  wrote fig14_asymmetry_boxen")
    else:
        print(f"  missing {a_path}; skipping Fig. 14")

    plot_schematic(outfile=figs / "fig08_asymmetry_schematic.png", seed=7)
    print("  wrote fig08_asymmetry_schematic")
    print(f"Figures written to {figs}")
    for f in sorted(figs.glob("fig*.png")):
        print(f"  {f.name} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
