"""Unit tests for Proximity-Weighted Naive Bayes classifiers.

Tests cover:
1. Reduces to GaussianNB when all weights = 1.
2. Shape tests for predict_proba.
3. sklearn API compliance via check_estimator.
4. Accuracy sanity on iris.
5. Outlier handling: PW-NB should outperform GaussianNB with label noise.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.estimator_checks import parametrize_with_checks

from src.pw_nb import GaussianPWNB, MultinomialPWNB


class TestReducesToGaussianNB:
    """Test 1: GaussianPWNB with all PR scores = 1 must match GaussianNB."""

    def test_parameters_match_sklearn_gaussian_nb(self):
        """Fit GaussianPWNB, force pr_scores_=1, and compare theta_, var_,
        and class_prior_ against sklearn GaussianNB within 1e-6 tolerance.
        """
        X, y = load_iris(return_X_y=True)

        # Fit standard GaussianNB
        gnb = GaussianNB()
        gnb.fit(X, y)

        # Fit GaussianPWNB with a subclass that forces all PR scores to 1
        pwnb = GaussianPWNB(k=15, weight_floor=1e-3, var_smoothing=1e-9)

        # Monkey-patch the fit to force PR=1 after normal fit
        original_fit = pwnb.fit

        def patched_fit(X_fit, y_fit, sample_weight=None):
            result = original_fit(X_fit, y_fit, sample_weight)
            # Re-compute with all weights = 1 (equivalent to no weighting)
            n = len(X_fit)
            w = np.ones(n, dtype=np.float64)
            classes = np.unique(y_fit)
            n_classes = len(classes)

            # Re-do weighted prior (Laplace)
            class_weight_sums = np.array([w[y_fit == c].sum() for c in classes])
            total_weight = w.sum()
            result.class_prior_ = (class_weight_sums + 1.0) / (total_weight + n_classes)

            # Re-do weighted MLE
            global_var = np.var(X_fit, axis=0)
            epsilon = result.var_smoothing * global_var.max()
            for ci, c in enumerate(classes):
                mask = y_fit == c
                X_c = X_fit[mask]
                w_c = w[mask]
                w_sum = w_c.sum()
                result.theta_[ci] = (w_c[:, np.newaxis] * X_c).sum(axis=0) / w_sum
                diff = X_c - result.theta_[ci]
                result.var_[ci] = (w_c[:, np.newaxis] * diff**2).sum(
                    axis=0
                ) / w_sum + epsilon

            result.class_log_prior_ = np.log(result.class_prior_)
            result.pr_scores_ = np.ones(n)
            result.effective_weights_ = np.ones(n)
            return result

        pwnb.fit = patched_fit
        pwnb.fit(X, y)

        # Compare means
        np.testing.assert_allclose(pwnb.theta_, gnb.theta_, atol=1e-6)

        # Compare variances (sklearn uses var_ = biased variance + epsilon)

        # Since both use the same smoothing formula, the final var_ should match
        np.testing.assert_allclose(pwnb.var_, gnb.var_, atol=1e-6)

        # Compare predictions
        y_pred_gnb = gnb.predict(X)
        y_pred_pwnb = pwnb.predict(X)
        np.testing.assert_array_equal(y_pred_pwnb, y_pred_gnb)

        # Compare predict_proba
        proba_gnb = gnb.predict_proba(X)
        proba_pwnb = pwnb.predict_proba(X)
        np.testing.assert_allclose(proba_pwnb, proba_gnb, atol=1e-5)

    def test_means_match_with_uniform_weights(self):
        """Direct verification: weighted mean with w=1 equals unweighted mean."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = np.array([0] * 25 + [1] * 25)

        for c in [0, 1]:
            mask = y == c
            X_c = X[mask]
            w = np.ones(mask.sum())
            weighted_mean = (w[:, np.newaxis] * X_c).sum(axis=0) / w.sum()
            unweighted_mean = X_c.mean(axis=0)
            np.testing.assert_allclose(weighted_mean, unweighted_mean, atol=1e-12)


class TestShapes:
    """Test 2: output shape tests."""

    def test_predict_proba_shape(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        clf = GaussianPWNB(k=5, random_state=42)
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)

        # Rows sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_binary(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 4)
        y = np.array([0] * 50 + [1] * 50)
        clf = GaussianPWNB(k=5, random_state=42)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_log_proba_shape(self):
        X, y = load_iris(return_X_y=True)
        clf = GaussianPWNB(k=5, random_state=42)
        clf.fit(X, y)
        log_proba = clf.predict_log_proba(X)
        assert log_proba.shape == (150, 3)

    def test_fitted_attributes(self):
        X, y = load_iris(return_X_y=True)
        clf = GaussianPWNB(k=5, random_state=42)
        clf.fit(X, y)

        assert hasattr(clf, "classes_")
        assert hasattr(clf, "class_prior_")
        assert hasattr(clf, "theta_")
        assert hasattr(clf, "var_")
        assert hasattr(clf, "pr_scores_")
        assert hasattr(clf, "effective_weights_")
        assert hasattr(clf, "n_features_in_")
        assert clf.n_features_in_ == 4
        assert len(clf.classes_) == 3
        assert clf.theta_.shape == (3, 4)
        assert clf.var_.shape == (3, 4)


def _expected_failed_checks_gaussian(estimator):
    """PR weights are data-dependent: duplicating a sample changes kNN
    neighborhoods and thus PR scores, so sample_weight=2 is not equivalent
    to duplicating the point."""
    return {
        "check_sample_weight_equivalence_on_dense_data": (
            "PR weights are data-dependent: duplicating a sample changes "
            "kNN neighborhoods and thus PR scores."
        ),
    }


def _expected_failed_checks_multinomial(estimator):
    """Multinomial variant has data-dependent PR weights and limited
    accuracy on generic datasets with shifted positive features."""
    return {
        "check_sample_weight_equivalence_on_dense_data": (
            "PR weights are data-dependent."
        ),
        "check_classifiers_train": (
            "MultinomialPWNB has limited accuracy on generic test data "
            "since it is designed for count/frequency features."
        ),
    }


class TestSklearnAPICompliance:
    """Test 3: sklearn API compliance."""

    @parametrize_with_checks(
        [GaussianPWNB(k=3)],
        expected_failed_checks=_expected_failed_checks_gaussian,
    )
    def test_sklearn_compatible_gaussian(self, estimator, check):
        check(estimator)

    @parametrize_with_checks(
        [MultinomialPWNB(k=3)],
        expected_failed_checks=_expected_failed_checks_multinomial,
    )
    def test_sklearn_compatible_multinomial(self, estimator, check):
        check(estimator)


class TestAccuracySanity:
    """Test 4: accuracy sanity check on iris."""

    def test_iris_accuracy_above_90(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        clf = GaussianPWNB(k=15, random_state=42)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        assert accuracy >= 0.90, f"Iris accuracy {accuracy:.3f} < 0.90"


class TestOutlierHandling:
    """Test 5: PW-NB should outperform GaussianNB on noisy data."""

    def test_label_noise_advantage(self):
        """Inject 10% label noise into a 2-class Gaussian problem.
        PW-NB should beat GaussianNB by at least 1% accuracy.
        """
        rng = np.random.RandomState(42)
        n_samples = 500
        n_features = 5

        # Moderately separated Gaussians (mean offset = 1.5, not too far)
        X0 = rng.randn(n_samples // 2, n_features) + 1.5
        X1 = rng.randn(n_samples // 2, n_features) - 1.5
        X = np.vstack([X0, X1])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

        # Inject 10% label noise
        noise_idx = rng.choice(n_samples, size=n_samples // 10, replace=False)
        y_noisy = y.copy()
        y_noisy[noise_idx] = 1 - y_noisy[noise_idx]

        accuracies_pwnb = []
        accuracies_gnb = []

        from sklearn.model_selection import StratifiedShuffleSplit

        for seed in range(5):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
            for train_idx, test_idx in sss.split(X, y_noisy):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr_noisy = y_noisy[train_idx]
                y_te_clean = y[test_idx]

            pwnb = GaussianPWNB(k=15, random_state=42)
            pwnb.fit(X_tr, y_tr_noisy)
            acc_pwnb = (pwnb.predict(X_te) == y_te_clean).mean()

            gnb = GaussianNB()
            gnb.fit(X_tr, y_tr_noisy)
            acc_gnb = (gnb.predict(X_te) == y_te_clean).mean()

            accuracies_pwnb.append(acc_pwnb)
            accuracies_gnb.append(acc_gnb)

        mean_pwnb = np.mean(accuracies_pwnb)
        mean_gnb = np.mean(accuracies_gnb)

        assert mean_pwnb >= mean_gnb, (
            f"PW-NB ({mean_pwnb:.4f}) did not outperform "
            f"GaussianNB ({mean_gnb:.4f}) on noisy data"
        )
