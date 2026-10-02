"""Publication-oriented summary figures from analysis tables."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PAPER_METRIC_NAME_MAP = {
    "SWE_melt_NSE": "Melt-NSE",
    "SWE_NSE": "SWE-NSE",
    "melt_sum_APE": "Melt-bias",
    "NSE_meltseason2": "Q-NSE",
    "KGE_meltseason2": "Q-KGE",
    "Qmean_meltseason2_APE": "Q-bias",
}

PAPER_SWE_BASE_COLORS = {
    "SWE_melt_NSE": "#1b9e77",
    "melt_sum_APE": "#d95f02",
    "SWE_NSE": "#7570b3",
}

PAPER_Q_LIGHTEN = {
    "NSE_meltseason2": 0.05,
    "KGE_meltseason2": 0.25,
    "Qmean_meltseason2_APE": 0.45,
}

HIGHLIGHTED_PAIRS = {
    "SWE_melt_NSE__NSE_meltseason2",
    "melt_sum_APE__Qmean_meltseason2_APE",
}


def _lighten(color, amount: float):
    base = np.array(mcolors.to_rgb(color))
    return tuple((1 - amount) * base + amount * np.array([1.0, 1.0, 1.0]))


def plot_rho_boxen(
    pairs_df: pd.DataFrame,
    pair_selection: Sequence[str],
    outfile: Optional[Path] = None,
    figsize=(5, 2.5),
    fontsize: int = 10,
) -> plt.Figure:
    """Fig. 11-style boxenplot of rho across catchment-years for each metric pair."""
    plot_df = pairs_df.copy()
    plot_df["QSWE_pair"] = pd.Categorical(
        plot_df["QSWE_pair"], categories=list(pair_selection), ordered=True
    )
    plot_df = plot_df.sort_values("QSWE_pair")

    pair_colors = {}
    for pair in pair_selection:
        if "__" in pair:
            swe_m, q_m = pair.split("__", 1)
            pair_colors[pair] = _lighten(
                PAPER_SWE_BASE_COLORS.get(swe_m, "#4c4c4c"),
                PAPER_Q_LIGHTEN.get(q_m, 0.22),
            )
        else:
            pair_colors[pair] = "#808080"

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxenplot(
        data=plot_df,
        y="QSWE_pair",
        x="correlation",
        hue="QSWE_pair",
        palette=pair_colors,
        dodge=False,
        linewidth=0.8,
        k_depth="proportion",
        ax=ax,
        legend=False,
    )
    ax.axvline(0, linestyle="dashed", color="black", linewidth=0.9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel(r"$\rho$ [-]", fontsize=fontsize)
    ax.set_ylabel("")
    ax.set_xlim(-0.1, 1)
    ax.tick_params(axis="y", left=False, labelleft=False)

    labels, bold_flags = [], []
    for pair in pair_selection:
        if "__" in pair:
            swe_m, q_m = pair.split("__", 1)
            labels.append(
                f"{PAPER_METRIC_NAME_MAP.get(swe_m, swe_m)}/"
                f"{PAPER_METRIC_NAME_MAP.get(q_m, q_m)}"
            )
            bold_flags.append(pair in HIGHLIGHTED_PAIRS)
        else:
            labels.append(pair)
            bold_flags.append(False)

    ax_r = ax.twinx()
    ax_r.set_ylim(ax.get_ylim())
    ax_r.set_yticks(ax.get_yticks())
    ax_r.set_yticklabels(labels, fontsize=fontsize - 1)
    for tick, bold in zip(ax_r.get_yticklabels(), bold_flags):
        if bold:
            tick.set_fontweight("bold")
    ax_r.tick_params(axis="y", length=0)
    ax_r.set_ylabel("")
    for spine in ("top", "left", "bottom"):
        ax_r.spines[spine].set_visible(False)

    fig.tight_layout()
    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        fig.savefig(outfile.with_suffix(".svg"), bbox_inches="tight")
    return fig


def plot_asymmetry_boxen(
    asymmetry_df: pd.DataFrame,
    pair_selection: Sequence[str],
    outfile: Optional[Path] = None,
    figsize=(5, 2.5),
    fontsize: int = 10,
) -> plt.Figure:
    """Fig. 14-style boxenplot of asymmetry ratio A."""
    plot_df = asymmetry_df.copy()
    plot_df["QSWE_pair"] = pd.Categorical(
        plot_df["QSWE_pair"], categories=list(pair_selection), ordered=True
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxenplot(
        data=plot_df,
        y="QSWE_pair",
        x="asymmetry_ratio",
        hue="QSWE_pair",
        dodge=False,
        linewidth=0.8,
        k_depth="proportion",
        ax=ax,
        legend=False,
        palette="colorblind",
    )
    ax.axvline(0, linestyle="dashed", color="black", linewidth=0.9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel(r"$A$ [-]", fontsize=fontsize)
    ax.set_ylabel("")
    fig.tight_layout()
    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        fig.savefig(outfile.with_suffix(".svg"), bbox_inches="tight")
    return fig


def plot_variance_partition_bars(
    vp_df: pd.DataFrame,
    outfile: Optional[Path] = None,
    title: str = "",
    figsize=(5, 3),
) -> plt.Figure:
    """Fig. 12-style marginal vs unique R2 bars."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(vp_df))
    width = 0.35
    ax.bar(x - width / 2, vp_df["marginal_R2"], width, label=r"Marginal $R^2$", color="C0")
    ax.bar(x + width / 2, vp_df["unique_delta_R2"], width, label=r"Unique $\Delta R^2$", color="C1")
    ax.axhline(vp_df["full_model_R2"].iloc[0], color="k", lw=1.2, label=r"Full-model $R^2$")
    if np.isfinite(vp_df["loo_R2"].iloc[0]):
        ax.axhline(
            vp_df["loo_R2"].iloc[0],
            color="k",
            ls="--",
            lw=1.0,
            label=r"LOO $R^2$",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(vp_df["predictor"], rotation=30, ha="right")
    ax.set_ylabel(r"Explained variance of $\rho$")
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    return fig


def plot_hexbin_rho(
    df: pd.DataFrame,
    x: str,
    y: str,
    rho: str = "correlation",
    outfile: Optional[Path] = None,
    figsize=(5, 4),
) -> plt.Figure:
    """Fig. 13-style hexbin of rho in tau–frain space."""
    plot_df = df[[x, y, rho]].dropna()
    fig, ax = plt.subplots(figsize=figsize)
    hb = ax.hexbin(
        plot_df[x],
        plot_df[y],
        C=plot_df[rho],
        reduce_C_function=np.mean,
        gridsize=15,
        cmap="viridis",
        mincnt=1,
    )
    ax.scatter(plot_df[x], plot_df[y], c=plot_df[rho], s=8, cmap="viridis", alpha=0.5)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(r"$\rho$")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    return fig
