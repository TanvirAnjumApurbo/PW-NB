"""Visualization: CD diagrams, bar charts, PR distributions, calibration, k-sensitivity."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging

logger = setup_logging()

LOWER_IS_BETTER = {"log_loss", "brier_score", "ece"}

sns.set_style("whitegrid")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 300,
    }
)


def _save_fig(fig, path: Path, name: str):
    """Save figure as both PNG and PDF."""
    for ext in ["png", "pdf"]:
        fpath = path / f"{name}.{ext}"
        fig.savefig(fpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved figure: %s", name)


def plot_cd_diagram(mean_std: pd.DataFrame, metric: str, fig_dir: Path):
    """Plot Critical Difference diagram using average ranks."""
    subset = mean_std[mean_std["metric"] == metric]
    pivot = subset.pivot(index="dataset", columns="classifier", values="mean").dropna()

    ascending = metric in LOWER_IS_BETTER
    ranks = pivot.rank(axis=1, ascending=ascending, method="average")
    avg_ranks = ranks.mean().sort_values()

    n_clf = len(avg_ranks)
    n_datasets = len(pivot)

    # Nemenyi CD: q_alpha * sqrt(k*(k+1) / (6*N))
    # Using q_alpha from studentized range table for alpha=0.05
    q_alpha_table = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    }
    q_alpha = q_alpha_table.get(n_clf, 3.0)
    cd = q_alpha * np.sqrt(n_clf * (n_clf + 1) / (6 * n_datasets))

    fig, ax = plt.subplots(figsize=(10, max(3, n_clf * 0.4)))

    names = list(avg_ranks.index)
    rank_vals = list(avg_ranks.values)

    # Draw horizontal lines for each classifier
    for i, (name, rank) in enumerate(zip(names, rank_vals)):
        y = n_clf - i
        ax.plot(rank, y, "ko", markersize=6)
        side = "left" if rank > np.median(rank_vals) else "right"
        offset = -0.15 if side == "left" else 0.15
        ha = "right" if side == "left" else "left"
        ax.annotate(
            f"{name} ({rank:.2f})",
            xy=(rank, y),
            xytext=(rank + offset, y),
            fontsize=8,
            ha=ha,
            va="center",
        )

    # Draw CD bar
    ax.plot([1, 1 + cd], [n_clf + 0.8, n_clf + 0.8], "k-", linewidth=2)
    ax.text(1 + cd / 2, n_clf + 1.0, f"CD={cd:.2f}", ha="center", fontsize=9)

    # Draw connections for non-significantly different classifiers
    for i in range(n_clf):
        for j in range(i + 1, n_clf):
            if abs(rank_vals[i] - rank_vals[j]) < cd:
                y_i = n_clf - i
                y_j = n_clf - j
                ax.plot(
                    [rank_vals[i], rank_vals[j]],
                    [y_i - 0.1, y_j + 0.1],
                    "gray",
                    alpha=0.3,
                    linewidth=1.5,
                )

    ax.set_xlim(0.5, n_clf + 0.5)
    ax.set_ylim(0, n_clf + 1.5)
    ax.set_xlabel("Average Rank")
    ax.set_title(f"Critical Difference Diagram: {metric}")
    ax.set_yticks([])

    _save_fig(fig, fig_dir, f"cd_diagram_{metric}")


def plot_accuracy_bar_chart(mean_std: pd.DataFrame, fig_dir: Path):
    """Bar chart: PW-NB(k=15) vs GaussianNB per dataset."""
    subset = mean_std[mean_std["metric"] == "accuracy"]
    pivot = subset.pivot(index="dataset", columns="classifier", values="mean")

    pwnb_col = "PW-NB(k=15)"
    gnb_col = "GaussianNB"

    if pwnb_col not in pivot.columns or gnb_col not in pivot.columns:
        logger.warning("Cannot create bar chart: missing PW-NB(k=15) or GaussianNB")
        return

    both = pivot[[gnb_col, pwnb_col]].dropna()
    both["advantage"] = both[pwnb_col] - both[gnb_col]
    both = both.sort_values("advantage")

    fig, ax = plt.subplots(figsize=(12, max(6, len(both) * 0.35)))
    x = np.arange(len(both))
    width = 0.35

    ax.barh(x - width / 2, both[gnb_col], width, label="GaussianNB", color="#2196F3")
    ax.barh(x + width / 2, both[pwnb_col], width, label="PW-NB(k=15)", color="#FF5722")

    ax.set_yticks(x)
    ax.set_yticklabels(both.index, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_title("PW-NB(k=15) vs GaussianNB: Accuracy per Dataset")
    ax.legend(loc="lower right")

    _save_fig(fig, fig_dir, "bar_accuracy_per_dataset")


def plot_pr_distribution(dataset_name: str, X, y, fig_dir: Path, k: int = 15):
    """Plot histogram of PR scores colored by class."""
    from src.proximal_ratio import ProximalRatio
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pr = ProximalRatio(k=k)
    pr.fit(X_scaled, y)
    scores = pr.pr_scores_

    classes = np.unique(y)
    fig, ax = plt.subplots(figsize=(8, 5))

    for c in classes:
        ax.hist(
            scores[y == c],
            bins=20,
            alpha=0.6,
            label=f"Class {c}",
            range=(0, 1),
        )

    ax.set_xlabel("Proximal Ratio")
    ax.set_ylabel("Count")
    ax.set_title(f"PR Score Distribution: {dataset_name} (k={k})")
    ax.legend()

    _save_fig(fig, fig_dir, f"pr_distribution_{dataset_name}")


def plot_reliability_diagram(
    dataset_name: str, mean_std: pd.DataFrame, all_folds: pd.DataFrame, fig_dir: Path
):
    """Plot reliability diagram comparing PW-NB and GaussianNB.
    This is a simplified version using aggregated statistics."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for clf in ["GaussianNB", "PW-NB(k=15)"]:
        subset = mean_std[
            (mean_std["dataset"] == dataset_name) & (mean_std["classifier"] == clf)
        ]
        if subset.empty:
            continue
        ece_row = subset[subset["metric"] == "ece"]
        if not ece_row.empty:
            ece_val = ece_row["mean"].values[0]
            ax.bar(
                clf,
                ece_val,
                alpha=0.7,
                label=f"{clf} (ECE={ece_val:.3f})",
            )

    ax.set_title(f"Calibration: {dataset_name}")
    ax.legend()

    _save_fig(fig, fig_dir, f"reliability_{dataset_name}")


def plot_k_sensitivity(mean_std: pd.DataFrame, fig_dir: Path):
    """Plot accuracy and macro_f1 vs k for PW-NB on representative datasets."""
    representative = ["iris", "wine", "breast_cancer", "glass", "ionosphere", "sonar"]
    k_values = [5, 15, 30, 45]

    for metric in ["accuracy", "macro_f1"]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for idx, ds_name in enumerate(representative):
            ax = axes[idx]
            means = []
            stds = []

            for k in k_values:
                clf_name = f"PW-NB(k={k})"
                row = mean_std[
                    (mean_std["dataset"] == ds_name)
                    & (mean_std["classifier"] == clf_name)
                    & (mean_std["metric"] == metric)
                ]
                if not row.empty:
                    means.append(row["mean"].values[0])
                    stds.append(row["std"].values[0])
                else:
                    means.append(np.nan)
                    stds.append(0)

            ax.errorbar(k_values, means, yerr=stds, marker="o", capsize=3)
            ax.set_title(ds_name, fontsize=10)
            ax.set_xlabel("k")
            ax.set_ylabel(metric)

        fig.suptitle(f"k-Sensitivity: {metric}", fontsize=13)
        fig.tight_layout()
        _save_fig(fig, fig_dir, f"k_sensitivity_{metric}")


def generate_all_figures(results_dir: Path, cache_dir: Path | None = None):
    """Generate all figures."""
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    mean_std = pd.read_csv(results_dir / "summary" / "mean_std.csv")
    all_folds_path = results_dir / "raw" / "all_folds.csv"
    all_folds = pd.read_csv(all_folds_path) if all_folds_path.exists() else None

    # 1. CD diagrams
    for metric in ["accuracy", "macro_f1", "auc_roc", "ece", "brier_score", "log_loss"]:
        try:
            plot_cd_diagram(mean_std, metric, fig_dir)
        except Exception as e:
            logger.error("CD diagram failed for %s: %s", metric, e)

    # 2. Accuracy bar chart
    try:
        plot_accuracy_bar_chart(mean_std, fig_dir)
    except Exception as e:
        logger.error("Bar chart failed: %s", e)

    # 3. PR distribution for representative datasets
    from src.datasets import load_dataset

    for ds_name in ["iris", "breast_cancer", "glass", "ionosphere", "sonar", "yeast"]:
        try:
            X, y, _ = load_dataset(ds_name, cache_dir)
            plot_pr_distribution(ds_name, X, y, fig_dir)
        except Exception as e:
            logger.error("PR distribution failed for %s: %s", ds_name, e)

    # 4. Reliability diagrams
    for ds_name in ["iris", "breast_cancer", "page_blocks", "letter"]:
        try:
            plot_reliability_diagram(ds_name, mean_std, all_folds, fig_dir)
        except Exception as e:
            logger.error("Reliability diagram failed for %s: %s", ds_name, e)

    # 5. k-sensitivity
    try:
        plot_k_sensitivity(mean_std, fig_dir)
    except Exception as e:
        logger.error("k-sensitivity plot failed: %s", e)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "cache"),
    )
    args = parser.parse_args()
    generate_all_figures(Path(args.results_dir), Path(args.cache_dir))


if __name__ == "__main__":
    main()
