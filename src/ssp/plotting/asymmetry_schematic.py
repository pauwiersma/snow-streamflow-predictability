"""
Schematic scatterplot explaining the asymmetry calculation.

Generates synthetic, moderately correlated SWE/Q skill ranks (n=500) with
one-sided asymmetry, then highlights the best 50 members of each metric.

A = log2( mean(SWE ranks | best Q) / mean(Q ranks | best SWE) )
Centered at 0; reciprocal mean-rank ratios map to +/- equal magnitude.
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

N = 500
N_BEST = 50
OUTFILE = (
    Path(__file__).resolve().parents[3] / "data" / "figures" / "fig08_asymmetry_schematic.png"
)


def generate_asymmetric_ranks(n=N, seed=7):
    """Correlated ranks where good SWE constrains Q more than the reverse."""
    rng = np.random.default_rng(seed)

    # Higher latent = better skill
    swe = rng.standard_normal(n)
    # Noise grows as SWE worsens → best SWE → tight Q; best Q → spread SWE
    noise_scale = 0.25 + 0.9 * (1.0 / (1.0 + np.exp(1.5 * swe)))
    q = swe + noise_scale * rng.standard_normal(n)

    # Rank 1 = best
    swe_rank = rankdata(-swe, method="ordinal").astype(float)
    q_rank = rankdata(-q, method="ordinal").astype(float)
    return swe_rank, q_rank


def asymmetry_stats(swe_rank, q_rank, n_best=N_BEST):
    best_q = np.argsort(q_rank)[:n_best]
    best_swe = np.argsort(swe_rank)[:n_best]
    mean_swe_given_best_q = np.mean(swe_rank[best_q])
    mean_q_given_best_swe = np.mean(q_rank[best_swe])
    asymmetry = np.log2(mean_swe_given_best_q / mean_q_given_best_swe)
    return asymmetry, mean_swe_given_best_q, mean_q_given_best_swe


def plot_schematic(outfile=OUTFILE, seed=7):
    swe_rank, q_rank = generate_asymmetric_ranks(seed=seed)
    asymmetry, mean_swe_given_best_q, mean_q_given_best_swe = asymmetry_stats(
        swe_rank, q_rank)

    best_swe = swe_rank <= N_BEST
    best_q = q_rank <= N_BEST
    both = best_swe & best_q
    swe_only = best_swe & ~best_q
    q_only = best_q & ~best_swe
    neither = ~(best_swe | best_q)

    fig, ax = plt.subplots(figsize=(4.5,4.5))

    ax.scatter(swe_rank[neither], q_rank[neither], c="0.5", s=10, alpha=0.5,
               label="Ensemble members", zorder=1)
    ax.scatter(swe_rank[swe_only], q_rank[swe_only], c="blue", s=18, alpha=0.85,
               label="50 best SWE", zorder=2)
    ax.scatter(swe_rank[q_only], q_rank[q_only], c="red", s=18, alpha=0.85,
               label="50 best Q", zorder=2)
    ax.scatter(swe_rank[both], q_rank[both], c="purple", s=18, alpha=0.9,
               label="Overlap", zorder=3)

    ax.axvline(N_BEST, color="0.3", linestyle="--", linewidth=1)
    ax.axhline(N_BEST, color="0.3", linestyle="--", linewidth=1)

    ax.set_xlabel("SWE skill ranks")
    ax.set_ylabel("Q skill ranks")
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect("equal")

    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.text(
        0.97, 0.12,
        (f"$\\bar R_{{\\mathrm{{SWE|Q_{{\\mathrm{{q10}}}}}}}}$ = {mean_swe_given_best_q:.1f}\n"
         f"$\\bar R_{{\\mathrm{{Q|SWE_{{\\mathrm{{q10}}}}}}}}$ = {mean_q_given_best_swe:.1f}\n"
         f"$A$ = {asymmetry:.2f}"),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.5", linewidth=0.8),
    )

    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved {outfile}")
    print(f"mean(SWE|best Q): {mean_swe_given_best_q:.3f}")
    print(f"mean(Q|best SWE): {mean_q_given_best_swe:.3f}")
    print(f"Asymmetry A (log2): {asymmetry:.3f}")
    print(f"Seed: {seed}")
    plt.close(fig)


if __name__ == "__main__":
    plot_schematic()
