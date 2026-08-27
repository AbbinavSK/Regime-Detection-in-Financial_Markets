import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from modelling import cluster_kmeans, cluster_hmm, fit_regime_order, hmm_transitions
from plot_style import (
    save_figure, style_axis, shade_periods, format_date_axis, mark_split, annotate_events,
    REGIME_PALETTE, NEUTRAL_COLOURS, CRISIS_EVENTS, EPOCH_LENGTH_COLOURS,
)

K = 3
REGIME_NAMES = ["Calm", "Transitional", "Crisis"]


# Save a figure into its combo's images subfolder and close it. Dashboard diagnostics stay at 150dpi -- see docs/plot_style.md.
def savefig(fig, images_dir, tag, name):
    combo_dir = os.path.join(images_dir, tag)
    os.makedirs(combo_dir, exist_ok=True)
    save_figure(fig, os.path.join(combo_dir, name), dpi=150)
    plt.close(fig)


# Plot a regime timeline shaded by rank, train and test on one continuous axis.
def plot_regime_timeline(train_dates, train_mu, train_labels, test_dates, test_mu, test_labels, rank_of_cluster):
    fig, ax = plt.subplots(figsize=(12, 4))

    def shade(dates, labels, alpha):
        for r in range(K):
            intervals = [(dates[i], dates[min(i + 1, len(dates) - 1)])
                         for i, lbl in enumerate(labels) if rank_of_cluster[lbl] == r]
            shade_periods(ax, intervals, REGIME_PALETTE[r], alpha=alpha)

    shade(train_dates, train_labels, alpha=0.5)
    shade(test_dates, test_labels, alpha=0.25)

    ax.plot(train_dates, train_mu, color="black", linewidth=2)
    ax.plot(test_dates, test_mu, color="black", linewidth=2)

    handles = [Patch(color=REGIME_PALETTE[r], alpha=0.5, label=REGIME_NAMES[r]) for r in range(K)]
    if len(test_dates) > 0:
        handles.append(mark_split(ax, test_dates[0]))

    ax.set_xlabel("Date")
    ax.set_ylabel(r"Average Correlation ($\mu$)")
    format_date_axis(ax)
    style_axis(ax, grid=True)
    ax.legend(handles=handles, loc="upper left", frameon=True, facecolor="white", edgecolor="lightgray")
    plt.tight_layout()
    return fig


# Plot VIX with predicted-crisis periods shaded, train and test on one continuous axis. VIX curve colour marks the epoch length.
def plot_vix_overlay(idx_train, res_train, idx_test, res_test, epoch_length):
    vix_color = EPOCH_LENGTH_COLOURS[epoch_length]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(idx_train, res_train["series_eod"], color=vix_color, linewidth=2.5, zorder=3)
    ax.plot(idx_test, res_test["series_eod"], color=vix_color, linewidth=2.5, zorder=3)
    ax.axhline(30.0, color=NEUTRAL_COLOURS["reference"], linestyle="--", linewidth=1.2, zorder=2)

    def shade_crisis(idx, res, alpha):
        crisis_idx = np.where(res["pred"].astype(bool))[0]
        intervals = [(idx[i], idx[min(i + 1, len(idx) - 1)]) for i in crisis_idx]
        shade_periods(ax, intervals, REGIME_PALETTE[2], alpha=alpha)

    shade_crisis(idx_train, res_train, alpha=0.5)
    shade_crisis(idx_test, res_test, alpha=0.25)

    idx_all = idx_train.append(idx_test)
    annotate_events(ax, CRISIS_EVENTS, idx_all[0], idx_all[-1])

    handles = [
        Line2D([0], [0], color=vix_color, lw=2.5, label=f"VIX (T={epoch_length})"),
        Line2D([0], [0], color=NEUTRAL_COLOURS["reference"], lw=1.2, linestyle="--", label="VIX Threshold"),
        Patch(facecolor=REGIME_PALETTE[2], alpha=0.5, label="Predicted Crisis Regime"),
    ]
    if len(idx_test) > 0:
        handles.append(mark_split(ax, idx_test[0]))

    ax.set_xlabel("Date")
    ax.set_ylabel("VIX")
    ax.set_xlim(idx_all[0], idx_all[-1])
    format_date_axis(ax)
    style_axis(ax, grid=True)
    ax.legend(handles=handles, loc="upper left", frameon=True, facecolor="white", edgecolor="lightgray")
    plt.tight_layout()
    return fig


# Plot train vs test empirical transition matrices side by side, annotated with cell probabilities and occupancy.
def plot_transition_comparison(emp_train, emp_test, occ_train, occ_test, title_tag):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for i, (ax, P, occ, title) in enumerate(((axes[0], emp_train, occ_train, "Train"), (axes[1], emp_test, occ_test, "Test"))):
        im = ax.imshow(P, cmap="viridis", vmin=0, vmax=1)
        style_axis(ax)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(REGIME_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(REGIME_NAMES if i == 0 else [])
        ax.set_title(f"{title} (n={occ.sum()})", fontweight="bold")
        for r in range(K):
            for c in range(K):
                val = P[r, c]
                label = "NaN" if np.isnan(val) else f"{val:.2f}"
                ax.text(c, r, f"{label}\n(occ={occ[r]})", ha="center", va="center", fontsize=8,
                        color="white" if (not np.isnan(val) and val > 0.5) else "black")
    fig.colorbar(im, ax=axes, shrink=0.8, label="transition probability")
    plt.suptitle(f"Empirical transitions: {title_tag}", fontweight="bold")
    return fig


# Fit K-means and HMM (K=3) on train, predict both, mu-rank each set of regimes from train only, and compute HMM dwell times.
def fit_and_rank_regimes(X_train, X_test, train_mu, n_seeds):
    kmeans_train_labels, kmeans_test_labels, kmeans_model = cluster_kmeans(X_train, X_test, K)
    kmeans_order, kmeans_rank_of_cluster, kmeans_crisis_id, kmeans_mean_mu = fit_regime_order(kmeans_train_labels, train_mu, K)

    hmm_train_labels, hmm_test_labels, hmm_model = cluster_hmm(X_train, X_test, K, n_seeds=n_seeds)
    hmm_order, hmm_rank_of_cluster, hmm_crisis_id, hmm_mean_mu = fit_regime_order(hmm_train_labels, train_mu, K)
    _, hmm_dwell = hmm_transitions(hmm_model, hmm_order)

    return {
        "kmeans": {"model": kmeans_model, "train_labels": kmeans_train_labels, "test_labels": kmeans_test_labels,
                   "order": kmeans_order, "rank_of_cluster": kmeans_rank_of_cluster,
                   "crisis_id": kmeans_crisis_id, "mean_mu": kmeans_mean_mu},
        "hmm": {"model": hmm_model, "train_labels": hmm_train_labels, "test_labels": hmm_test_labels,
                "order": hmm_order, "rank_of_cluster": hmm_rank_of_cluster,
                "crisis_id": hmm_crisis_id, "mean_mu": hmm_mean_mu, "dwell": hmm_dwell},
    }
