"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import brier_score, compute_all_metrics, expected_calibration_error


class TestECE:
    """Tests for Expected Calibration Error."""

    def test_perfect_calibration_small_ece(self):
        """Synthetic dataset with near-perfect calibration => ECE < 0.05."""
        rng = np.random.RandomState(42)
        n = 10000
        n_classes = 3

        # Generate random probabilities and sample labels from them
        raw = rng.dirichlet(np.ones(n_classes), size=n)
        y_true = np.array([rng.choice(n_classes, p=p) for p in raw])
        y_proba = raw
        y_pred = np.argmax(y_proba, axis=1)

        ece = expected_calibration_error(y_true, y_pred, y_proba, n_bins=15)
        assert ece < 0.05, f"ECE = {ece:.4f} (expected < 0.05)"

    def test_maximally_miscalibrated(self):
        """Always predict probability 1.0 for class 0 but half are class 1.
        ECE should be ~0.5."""
        n = 1000
        y_true = np.array([0] * (n // 2) + [1] * (n // 2))
        y_proba = np.zeros((n, 2))
        y_proba[:, 0] = 1.0  # Always confident class 0
        y_pred = np.zeros(n, dtype=int)

        ece = expected_calibration_error(y_true, y_pred, y_proba, n_bins=15)
        assert abs(ece - 0.5) < 0.05, f"ECE = {ece:.4f} (expected ~0.5)"


class TestBrierScore:
    """Tests for multi-class Brier score."""

    def test_perfect_classifier(self):
        """A perfect classifier has BS = 0."""
        classes = np.array([0, 1, 2])
        y_true = np.array([0, 1, 2, 0, 1])
        y_proba = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=float,
        )

        bs = brier_score(y_true, y_proba, classes)
        assert bs == pytest.approx(0.0, abs=1e-10)

    def test_uniform_random_classifier(self):
        """Uniform random classifier: BS ~ (L-1)/L for L balanced classes."""
        rng = np.random.RandomState(42)
        for L in [2, 3, 5, 10]:
            n = 10000
            classes = np.arange(L)
            y_true = rng.choice(L, size=n)
            y_proba = np.full((n, L), 1.0 / L)

            bs = brier_score(y_true, y_proba, classes)
            expected = (L - 1) / L
            assert bs == pytest.approx(
                expected, abs=0.01
            ), f"L={L}: BS={bs:.4f}, expected ~{expected:.4f}"

    def test_worst_case_binary(self):
        """Always predict wrong class with certainty: BS = 2."""
        classes = np.array([0, 1])
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array(
            [
                [0, 1],
                [0, 1],
                [0, 1],
                [1, 0],
                [1, 0],
                [1, 0],
            ],
            dtype=float,
        )
        bs = brier_score(y_true, y_proba, classes)
        assert bs == pytest.approx(2.0, abs=1e-10)


class TestComputeAllMetrics:
    """Test the unified metrics function."""

    def test_returns_all_keys(self):
        classes = np.array([0, 1])
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])

        result = compute_all_metrics(y_true, y_pred, y_proba, classes)
        expected_keys = {
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
        }
        assert set(result.keys()) == expected_keys

    def test_perfect_predictions(self):
        classes = np.array([0, 1])
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)

        result = compute_all_metrics(y_true, y_pred, y_proba, classes)
        assert result["accuracy"] == 1.0
        assert result["macro_f1"] == 1.0
        assert result["brier_score"] == pytest.approx(0.0)
        assert result["mcc"] == pytest.approx(1.0)
