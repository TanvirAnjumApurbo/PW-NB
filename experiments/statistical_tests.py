"""Statistical tests: Wilcoxon, Friedman, Nemenyi + CD diagrams."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging

logger = setup_logging()

METRICS = [
    "accuracy",
    "macro_f1",
    "auc_roc",
    "log_loss",
    "brier_score",
    "ece",
    "balanced_accuracy",
    "geometric_mean",
    "mcc",
    "weighted_f1",
]

# For these metrics, lower is better
LOWER_IS_BETTER = {"log_loss", "brier_score", "ece"}


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load mean_std.csv and pivot to get per-dataset mean scores."""
    mean_std = pd.read_csv(results_dir / "summary" / "mean_std.csv")
    return mean_std


def wilcoxon_pairwise(
    mean_std: pd.DataFrame, metric: str, output_dir: Path
) -> pd.DataFrame:
    """Run pairwise Wilcoxon signed-rank tests for a metric."""
    subset = mean_std[mean_std["metric"] == metric]
    pivot = subset.pivot(index="dataset", columns="classifier", values="mean").dropna()

    classifiers = list(pivot.columns)
    n_clf = len(classifiers)
    n_comparisons = n_clf * (n_clf - 1) // 2

    pvals = np.ones((n_clf, n_clf))
    raw_pvals = []

    for i in range(n_clf):
        for j in range(i + 1, n_clf):
            a = pivot[classifiers[i]].values
            b = pivot[classifiers[j]].values
            diff = a - b
            if np.all(diff == 0):
                p = 1.0
            else:
                try:
                    _, p = stats.wilcoxon(a, b, alternative="two-sided")
                except ValueError:
                    p = 1.0
            raw_pvals.append((i, j, p))

    # Holm-Bonferroni correction
    raw_pvals.sort(key=lambda x: x[2])
    for rank, (i, j, p) in enumerate(raw_pvals):
        adjusted = min(p * (n_comparisons - rank), 1.0)
        pvals[i, j] = adjusted
        pvals[j, i] = adjusted

    df_pvals = pd.DataFrame(pvals, index=classifiers, columns=classifiers)
    df_pvals.to_csv(output_dir / f"wilcoxon_{metric}.csv")
    return df_pvals


def friedman_test(mean_std: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Run Friedman test across all classifiers per metric."""
    rows = []
    for metric in METRICS:
        subset = mean_std[mean_std["metric"] == metric]
        pivot = subset.pivot(
            index="dataset", columns="classifier", values="mean"
        ).dropna()

        if pivot.shape[0] < 3 or pivot.shape[1] < 3:
            rows.append(
                {
                    "metric": metric,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "n_datasets": pivot.shape[0],
                }
            )
            continue

        groups = [pivot[col].values for col in pivot.columns]
        try:
            stat, p = stats.friedmanchisquare(*groups)
        except ValueError:
            stat, p = np.nan, np.nan

        rows.append(
            {
                "metric": metric,
                "statistic": stat,
                "p_value": p,
                "n_datasets": pivot.shape[0],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "friedman.csv", index=False)
    return df


def compute_ranks(mean_std: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute average ranks per classifier per metric."""
    rows = []
    for metric in METRICS:
        subset = mean_std[mean_std["metric"] == metric]
        pivot = subset.pivot(
            index="dataset", columns="classifier", values="mean"
        ).dropna()

        ascending = metric in LOWER_IS_BETTER
        ranks = pivot.rank(axis=1, ascending=ascending, method="average")
        avg_ranks = ranks.mean()

        for clf, rank in avg_ranks.items():
            rows.append({"metric": metric, "classifier": clf, "avg_rank": rank})

    df = pd.DataFrame(rows)
    df_pivot = df.pivot(index="metric", columns="classifier", values="avg_rank")
    df_pivot.to_csv(output_dir / "ranks.csv")
    return df_pivot


def run_all_stats(results_dir: Path):
    """Run all statistical tests."""
    stats_dir = results_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    mean_std = load_results(results_dir)

    # Friedman
    friedman_df = friedman_test(mean_std, stats_dir)
    logger.info("Friedman test results:\n%s", friedman_df.to_string())

    # Ranks
    ranks_df = compute_ranks(mean_std, stats_dir)
    logger.info("Average ranks:\n%s", ranks_df.round(2).to_string())

    # Wilcoxon pairwise
    for metric in METRICS:
        try:
            wilcoxon_pairwise(mean_std, metric, stats_dir)
            logger.info("Wilcoxon test done for %s", metric)
        except Exception as e:
            logger.error("Wilcoxon failed for %s: %s", metric, e)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
    )
    args = parser.parse_args()
    run_all_stats(Path(args.results_dir))


if __name__ == "__main__":
    main()
