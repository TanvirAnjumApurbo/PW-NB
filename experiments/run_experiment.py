"""Main experiment runner for PW-NB evaluation."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import adapt_k_for_dataset, get_baselines
from src.datasets import get_dataset_names, load_dataset
from src.metrics import compute_all_metrics
from src.utils import seed_everything, setup_logging

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


def run_single_fold(clf_factory, X_train, y_train, X_test, y_test, classes):
    """Fit and evaluate a classifier on a single fold."""
    clf = clf_factory()

    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    predict_time = time.time() - t0

    # Ensure y_proba has correct shape for all classes
    if y_proba.shape[1] < len(classes):
        full_proba = np.zeros((len(X_test), len(classes)))
        fitted_classes = (
            clf.classes_ if hasattr(clf, "classes_") else np.unique(y_train)
        )
        for i, c in enumerate(fitted_classes):
            idx = np.where(classes == c)[0]
            if len(idx) > 0:
                full_proba[:, idx[0]] = y_proba[:, i]
        y_proba = full_proba

    metrics = compute_all_metrics(
        y_true=y_test, y_pred=y_pred, y_proba=y_proba, classes=classes
    )
    metrics["fit_time"] = fit_time
    metrics["predict_time"] = predict_time

    # Extract PW-NB training metadata — these live only in the fitted object.
    # Must be captured here; once clf is discarded they are gone forever.
    inner_clf = getattr(clf, "model_", clf)   # unwrap AdaptivePWNB → GaussianPWNB
    if hasattr(inner_clf, "pr_scores_"):
        metrics["mean_pr"] = float(inner_clf.pr_scores_.mean())
    if hasattr(clf, "best_k_"):               # AdaptivePWNB only
        metrics["best_k"] = float(clf.best_k_)

    return metrics


def run_experiment(
    datasets: list[str],
    classifiers: dict,
    n_folds: int = 10,
    random_state: int = 42,
    output_dir: Path = Path("results"),
    cache_dir: Path | None = None,
    resume: bool = True,
):
    """Run the full experiment grid."""
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    summary_dir = output_dir / "summary"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    error_log = raw_dir / "errors.log"
    total_start = time.time()

    # Resume: load already-completed data and skip finished datasets
    temp_path = raw_dir / "all_folds_temp.csv"
    all_rows = []
    done_datasets: set[str] = set()
    if resume and temp_path.exists():
        df_existing = pd.read_csv(temp_path)
        all_rows = df_existing.to_dict("records")
        # A dataset is "done" if every classifier appears in it
        clf_names = set(classifiers.keys())
        for ds, grp in df_existing.groupby("dataset"):
            if clf_names <= set(grp["classifier"].unique()):
                done_datasets.add(ds)
        if done_datasets:
            logger.info(
                "Resuming: skipping %d already-completed datasets: %s",
                len(done_datasets),
                sorted(done_datasets),
            )

    total_tasks = len(datasets) * len(classifiers) * n_folds
    completed = len(done_datasets) * len(classifiers) * n_folds

    bar = tqdm(
        total=total_tasks,
        initial=completed,
        unit="fold",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for ds_name in datasets:
        if ds_name in done_datasets:
            continue
        try:
            X, y, meta = load_dataset(ds_name, cache_dir)
        except Exception as e:
            logger.error("Failed to load %s: %s", ds_name, e)
            with open(error_log, "a") as f:
                f.write(f"DATASET LOAD ERROR: {ds_name}\n{traceback.format_exc()}\n\n")
            continue

        classes = np.unique(y)
        min_class_count = np.min(
            np.bincount(y.astype(int))
            if y.dtype.kind == "i"
            else [np.sum(y == c) for c in classes]
        )
        actual_folds = min(n_folds, int(min_class_count))
        if actual_folds < n_folds:
            logger.warning(
                "%s: reducing folds from %d to %d (min class size=%d)",
                ds_name,
                n_folds,
                actual_folds,
                min_class_count,
            )

        skf = StratifiedKFold(
            n_splits=actual_folds, shuffle=True, random_state=random_state
        )

        for clf_name, clf_factory in classifiers.items():
            bar.set_description(f"{ds_name} | {clf_name}")
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                try:
                    # Adapt k for fixed-k PW-NB variants on small datasets.
                    # PW-NB(auto) handles k adaptation internally via inner CV.
                    if "PW-NB" in clf_name and "k=" in clf_name:
                        k_str = clf_name.split("k=")[1].rstrip(")")
                        k_val = int(k_str)
                        adapted_k = adapt_k_for_dataset(k_val, y_train, ds_name)
                        if adapted_k != k_val:
                            from src.pw_nb import GaussianPWNB

                            def clf_factory_adapted(ak=adapted_k):
                                return GaussianPWNB(k=ak, random_state=42)
                        else:
                            clf_factory_adapted = clf_factory
                    else:
                        clf_factory_adapted = clf_factory

                    fold_metrics = run_single_fold(
                        clf_factory_adapted, X_train, y_train, X_test, y_test, classes
                    )

                    for metric_name, value in fold_metrics.items():
                        all_rows.append(
                            {
                                "dataset": ds_name,
                                "classifier": clf_name,
                                "fold": fold_idx,
                                "metric": metric_name,
                                "value": value,
                            }
                        )

                except Exception:
                    tb = traceback.format_exc()
                    logger.error(
                        "FAILED: %s / %s / fold %d\n%s",
                        ds_name,
                        clf_name,
                        fold_idx,
                        tb,
                    )
                    with open(error_log, "a") as f:
                        f.write(
                            f"FOLD ERROR: {ds_name} / {clf_name} / fold {fold_idx}\n"
                            f"{tb}\n\n"
                        )

                completed += 1
                bar.update(1)

        # Save intermediate results after each dataset
        if all_rows:
            df_temp = pd.DataFrame(all_rows)
            temp_path = raw_dir / "all_folds_temp.csv"
            df_temp.to_csv(temp_path, index=False)
        bar.write(f"[saved] {ds_name} done ({completed}/{total_tasks} folds)")

    total_time = time.time() - total_start
    logger.info(
        "Total runtime: %.1f seconds (%.1f minutes)", total_time, total_time / 60
    )
    bar.close()

    # Save final results
    df = pd.DataFrame(all_rows)
    df.to_csv(raw_dir / "all_folds.csv", index=False)

    # Aggregate: mean +/- std
    metric_rows = df[~df["metric"].isin(["fit_time", "predict_time"])]
    agg = (
        metric_rows.groupby(["dataset", "classifier", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.to_csv(summary_dir / "mean_std.csv", index=False)

    # Pivot tables per metric (separate mean and std files for journal reporting)
    for metric in METRICS:
        subset = agg[agg["metric"] == metric]
        pivot_mean = subset.pivot(index="dataset", columns="classifier", values="mean")
        pivot_mean.to_csv(summary_dir / f"{metric}_table_mean.csv")
        pivot_std = subset.pivot(index="dataset", columns="classifier", values="std")
        pivot_std.to_csv(summary_dir / f"{metric}_table_std.csv")

    # Print summary
    print("\n" + "=" * 80)
    print("HEADLINE RESULTS: Mean Accuracy across datasets")
    print("=" * 80)
    acc_pivot = agg[agg["metric"] == "accuracy"].pivot(
        index="dataset", columns="classifier", values="mean"
    )
    print(acc_pivot.round(4).to_markdown())
    print()

    return df


def main():
    parser = argparse.ArgumentParser(description="PW-NB Experiment Runner")
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset names or 'all'",
    )
    parser.add_argument(
        "--classifiers",
        type=str,
        default="all",
        help="Comma-separated classifier names or 'all'",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="5,15,30,45",
        help="Comma-separated k values for PW-NB",
    )
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--cache-dir", type=str, default=str(PROJECT_ROOT / "data" / "cache")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only iris, wine, breast_cancer",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing all_folds_temp.csv",
    )
    parser.add_argument(
        "--low-priority",
        action="store_true",
        help="Run at below-normal process priority to reduce system lag",
    )

    args = parser.parse_args()

    if args.low_priority:
        try:
            import psutil
            p = psutil.Process()
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            logger.info("Process priority set to BELOW_NORMAL")
        except Exception as e:
            logger.warning("Could not set low priority: %s", e)

    seed_everything(args.seed)

    k_values = [int(k) for k in args.k_values.split(",")]
    all_classifiers = get_baselines(k_values=k_values, random_state=args.seed)

    if args.quick:
        dataset_list = ["iris", "wine", "breast_cancer"]
    elif args.datasets == "all":
        dataset_list = get_dataset_names()
    else:
        dataset_list = [s.strip() for s in args.datasets.split(",")]

    if args.classifiers != "all":
        selected = [s.strip() for s in args.classifiers.split(",")]
        all_classifiers = {k: v for k, v in all_classifiers.items() if k in selected}

    cache_dir = None if args.no_cache else Path(args.cache_dir)

    logger.info(
        "Running experiment: %d datasets x %d classifiers x %d folds",
        len(dataset_list),
        len(all_classifiers),
        args.n_folds,
    )

    run_experiment(
        datasets=dataset_list,
        classifiers=all_classifiers,
        n_folds=args.n_folds,
        random_state=args.seed,
        output_dir=Path(args.output_dir),
        cache_dir=cache_dir,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
