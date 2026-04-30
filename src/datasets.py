"""Dataset loading and caching for the PW-NB experiment."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("pwnb.datasets")

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_CSV_PATH = _PROJECT_ROOT / "datasets_selected.csv"


def _did_map() -> dict[str, int]:
    """Return name → OpenML DID mapping from datasets_selected.csv."""
    df = pd.read_csv(_CSV_PATH)
    return dict(zip(df["name"], df["did"]))


def _preprocess(
    X: np.ndarray, y: np.ndarray, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply preprocessing rules per spec."""
    # Encode target if needed
    if y.dtype.kind in ("U", "S", "O"):
        le = LabelEncoder()
        y = le.fit_transform(y)
    y = np.asarray(y, dtype=int)

    # Convert X to float, handling categorical via get_dummies
    if isinstance(X, pd.DataFrame):
        cat_cols = X.select_dtypes(include=["category", "object"]).columns
        if len(cat_cols) > 0:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
        X = X.values.astype(np.float64)
    else:
        X = np.asarray(X, dtype=np.float64)

    # Handle missing values
    nan_mask = np.isnan(X)
    if nan_mask.any():
        nan_rows = nan_mask.any(axis=1)
        frac_nan = nan_rows.sum() / len(X)
        if frac_nan < 0.05:
            keep = ~nan_rows
            X = X[keep]
            y = y[keep]
            logger.info(
                "%s: dropped %d rows with NaN (%.1f%%)",
                name,
                nan_rows.sum(),
                frac_nan * 100,
            )
        else:
            col_medians = np.nanmedian(X, axis=0)
            for j in range(X.shape[1]):
                mask_j = np.isnan(X[:, j])
                X[mask_j, j] = col_medians[j]
            logger.info("%s: median-imputed NaN values", name)

    return X, y


def _compute_metadata(name: str, X: np.ndarray, y: np.ndarray, source: str) -> dict:
    """Compute dataset metadata."""
    classes, counts = np.unique(y, return_counts=True)
    return {
        "name": name,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": len(classes),
        "imbalance_ratio": float(counts.max() / counts.min()),
        "source": source,
    }


def _load_openml_by_did(
    name: str, did: int, cache_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load an OpenML dataset by its pinned DID."""
    if cache_dir is not None:
        openml.config.cache_directory = str(cache_dir)

    ds = openml.datasets.get_dataset(
        did,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X_df, y_series, _, _ = ds.get_data(target=ds.default_target_attribute)

    X = X_df if isinstance(X_df, pd.DataFrame) else pd.DataFrame(X_df)
    y_arr = y_series.values if hasattr(y_series, "values") else np.asarray(y_series)

    X, y_arr = _preprocess(X, y_arr, name)
    meta = _compute_metadata(name, X, y_arr, f"OpenML(DID={did})")
    return X, y_arr, meta


def load_dataset(
    name: str, cache_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a single dataset by name."""
    dm = _did_map()
    if name not in dm:
        raise ValueError(
            f"Unknown dataset: {name!r}. "
            f"Available: {sorted(dm)}"
        )
    return _load_openml_by_did(name, dm[name], cache_dir)


def load_all_datasets(
    cache_dir: Path | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict]]:
    """Load all 60 datasets."""
    results = {}
    for name in get_dataset_names():
        try:
            X, y, meta = load_dataset(name, cache_dir)
            results[name] = (X, y, meta)
            logger.info(
                "Loaded %s: n=%d, d=%d, classes=%d, IR=%.2f",
                name,
                meta["n_samples"],
                meta["n_features"],
                meta["n_classes"],
                meta["imbalance_ratio"],
            )
        except Exception as e:
            logger.error("Failed to load %s: %s", name, e)
    return results


def get_dataset_names() -> list[str]:
    """Return the ordered list of all 60 dataset names."""
    df = pd.read_csv(_CSV_PATH)
    return list(df["name"])
