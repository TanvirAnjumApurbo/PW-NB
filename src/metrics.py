"""Evaluation metrics for classification including calibration metrics.

Provides a unified compute_all_metrics() function that returns all 10
metrics used in the PW-NB evaluation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


def brier_score(y_true: NDArray, y_proba: NDArray, classes: NDArray) -> float:
    """Multi-class Brier score.

    BS = (1/N) * sum_i sum_c (p_{i,c} - 1[y_i == c])^2

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
    y_proba : ndarray of shape (n_samples, n_classes)
    classes : ndarray of shape (n_classes,)

    Returns
    -------
    float
    """
    n = len(y_true)
    n_classes = len(classes)
    one_hot = np.zeros((n, n_classes))
    for ci, c in enumerate(classes):
        one_hot[y_true == c, ci] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: NDArray,
    y_pred: NDArray,
    y_proba: NDArray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error with equal-width bins.

    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Bins are over the max-class predicted probability in [0, 1].

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
    y_pred : ndarray of shape (n_samples,)
    y_proba : ndarray of shape (n_samples, n_classes)
    n_bins : int, default=15

    Returns
    -------
    float
    """
    confidences = np.max(y_proba, axis=1)
    correct = (y_pred == y_true).astype(float)
    n = len(y_true)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for m in range(n_bins):
        lo, hi = bin_boundaries[m], bin_boundaries[m + 1]
        if m == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        bin_size = mask.sum()
        if bin_size == 0:
            continue

        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def geometric_mean_score(y_true: NDArray, y_pred: NDArray, classes: NDArray) -> float:
    """Per-class geometric mean of recall (macro-averaged).

    GM = (prod_c recall_c) ^ (1 / n_classes)

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
    y_pred : ndarray of shape (n_samples,)
    classes : ndarray of shape (n_classes,)

    Returns
    -------
    float
    """
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            recalls.append(0.0)
        else:
            recalls.append((y_pred[mask] == c).mean())

    recalls = np.array(recalls)
    if np.any(recalls == 0):
        return 0.0
    return float(np.prod(recalls) ** (1.0 / len(recalls)))


def compute_all_metrics(
    y_true: NDArray,
    y_pred: NDArray,
    y_proba: NDArray,
    classes: NDArray,
) -> dict[str, float]:
    """Compute all 10 evaluation metrics.

    Safe to return NaN for undefined metrics (e.g., AUC with single
    class in a fold).

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True labels.
    y_pred : ndarray of shape (n_samples,)
        Predicted labels.
    y_proba : ndarray of shape (n_samples, n_classes)
        Predicted class probabilities.
    classes : ndarray of shape (n_classes,)
        All possible class labels.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: accuracy, macro_f1, auc_roc, log_loss,
        brier_score, ece, balanced_accuracy, geometric_mean, mcc,
        weighted_f1.
    """
    results: dict[str, float] = {}

    # 1. Accuracy
    results["accuracy"] = float(accuracy_score(y_true, y_pred))

    # 2. Macro F1
    results["macro_f1"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # 3. AUC-ROC
    try:
        unique_in_fold = np.unique(y_true)
        if len(unique_in_fold) < 2:
            results["auc_roc"] = float("nan")
        elif len(classes) == 2:
            # Binary: use positive class probability
            pos_idx = 1
            results["auc_roc"] = float(roc_auc_score(y_true, y_proba[:, pos_idx]))
        else:
            results["auc_roc"] = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    average="macro",
                    multi_class="ovr",
                    labels=classes,
                )
            )
    except ValueError:
        results["auc_roc"] = float("nan")

    # 4. Log loss
    y_proba_clipped = np.clip(y_proba, 1e-15, 1 - 1e-15)
    try:
        results["log_loss"] = float(log_loss(y_true, y_proba_clipped, labels=classes))
    except ValueError:
        results["log_loss"] = float("nan")

    # 5. Brier score
    results["brier_score"] = brier_score(y_true, y_proba, classes)

    # 6. ECE
    results["ece"] = expected_calibration_error(y_true, y_pred, y_proba)

    # 7. Balanced accuracy
    results["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    # 8. Geometric mean
    results["geometric_mean"] = geometric_mean_score(y_true, y_pred, classes)

    # 9. MCC
    try:
        results["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        results["mcc"] = float("nan")

    # 10. Weighted F1
    results["weighted_f1"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    return results
