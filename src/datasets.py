"""Dataset loading and caching for the PW-NB experiment."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("pwnb.datasets")

SKLEARN_DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}

OPENML_DATASETS = {
    "glass": {"name": "glass", "version": 1},
    "vehicle": {"name": "vehicle", "version": 1},
    "ionosphere": {"name": "ionosphere", "version": 1},
    "sonar": {"name": "sonar", "version": 1},
    "ecoli": {"name": "ecoli", "version": 1},
    "yeast": {"name": "yeast", "version": 4},
    "segment": {"name": "segment", "version": 1},
    "waveform": {"name": "waveform-5000", "version": 1},
    "optdigits": {"name": "optdigits", "version": 1},
    "satellite": {"name": "satimage", "version": 1},
    "pendigits": {"name": "pendigits", "version": 1},
    "vowel": {"name": "vowel", "version": 1},
    "balance_scale": {"name": "balance-scale", "version": 1},
    "page_blocks": {"name": "page-blocks", "version": 1},
    "spambase": {"name": "spambase", "version": 1},
    "banknote": {"name": "banknote-authentication", "version": 1},
    "robot_navigation": {"name": "wall-robot-navigation", "version": 1},
    "letter": {"name": "letter", "version": 1},
    "transfusion": {"name": "blood-transfusion-service-center", "version": 1},
    "parkinsons": {"name": "parkinsons", "version": 1},
}


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
        # Identify categorical columns
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


def _load_sklearn_dataset(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a sklearn built-in dataset."""
    loader = SKLEARN_DATASETS[name]
    data = loader()
    X, y = data.data, data.target
    X, y = _preprocess(X, y, name)
    meta = _compute_metadata(name, X, y, "sklearn")
    return X, y, meta


def _load_openml_dataset(
    name: str, cache_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load an OpenML dataset."""
    spec = OPENML_DATASETS[name]
    oml_name = spec["name"]
    version = spec["version"]

    if cache_dir is not None:
        openml.config.cache_directory = str(cache_dir)

    try:
        datasets = openml.datasets.list_datasets(
            data_name=oml_name, output_format="dataframe"
        )
        if datasets.empty:
            raise ValueError(f"No OpenML dataset found with name={oml_name}")

        if version in datasets["version"].values:
            did = int(datasets.loc[datasets["version"] == version].index[0])
        else:
            logger.warning("%s: version %d not available, using latest", name, version)
            did = int(datasets.index[0])

        ds = openml.datasets.get_dataset(did, download_data=True)
    except Exception:
        logger.warning("%s: list_datasets failed, trying direct fetch", name)
        try:
            ds = openml.datasets.get_dataset(
                oml_name, version=version, download_data=True
            )
        except Exception:
            ds = openml.datasets.get_dataset(
                oml_name, version="active", download_data=True
            )

    X_df, y_series, _, _ = ds.get_data(target=ds.default_target_attribute)

    X = X_df if isinstance(X_df, pd.DataFrame) else pd.DataFrame(X_df)
    y_arr = y_series.values if hasattr(y_series, "values") else np.asarray(y_series)

    X, y_arr = _preprocess(X, y_arr, name)
    meta = _compute_metadata(name, X, y_arr, f"OpenML({oml_name})")
    return X, y_arr, meta


def load_dataset(
    name: str, cache_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a single dataset by name."""
    if name in SKLEARN_DATASETS:
        return _load_sklearn_dataset(name)
    elif name in OPENML_DATASETS:
        return _load_openml_dataset(name, cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def load_all_datasets(
    cache_dir: Path | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict]]:
    """Load all 24 datasets.

    Returns
    -------
    dict mapping dataset name to (X, y, metadata).
    """
    all_names = list(SKLEARN_DATASETS.keys()) + list(OPENML_DATASETS.keys())
    results = {}
    for name in all_names:
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
    """Return the ordered list of all dataset names."""
    return list(SKLEARN_DATASETS.keys()) + list(OPENML_DATASETS.keys())
