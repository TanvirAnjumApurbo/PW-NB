"""Proximity-Weighted Naive Bayes classifiers.

Implements GaussianPWNB and MultinomialPWNB, sklearn-compatible classifiers
that weight training instances by their Proximal Ratio (PR) scores during
parameter estimation.

Core idea: outliers (PR=0) vanish from the estimator, boundary/overlap
points (0 < PR < 1) contribute partially, clean interior points (PR=1)
contribute fully.

References
----------
Amer, A.A., Ravana, S.D. & Habeeb, R.A.A. (2025). Effective k-nearest
neighbor models for data classification enhancement. Journal of Big Data,
12, 86.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from src.proximal_ratio import ProximalRatio


class GaussianPWNB(ClassifierMixin, BaseEstimator):
    """Gaussian Proximity-Weighted Naive Bayes classifier.

    A Naive Bayes variant where each training instance contributes to
    the likelihood/prior estimation with a weight equal to its Proximal
    Ratio (PR) score. Features are modeled with Gaussian distributions
    using weighted MLE.

    Parameters
    ----------
    k : int, default=15
        Number of nearest neighbors for PR computation.
    metric : str, default="euclidean"
        Distance metric for PR computation.
    weight_floor : float, default=1e-3
        Minimum weight for any training instance. Prevents parameter
        collapse when a minority class has many PR=0 points.
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features added to
        variances for numerical stability. Same convention as
        sklearn GaussianNB.
    empty_neighborhood_value : float, default=1.0
        PR value when |S|=0. See ProximalRatio docs.
    radius_estimator : str, default="mean"
        Radius estimation method. See ProximalRatio docs.
    random_state : int or None, default=None
        Random state for reproducibility.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    class_prior_ : ndarray of shape (n_classes,)
        Weighted prior probability of each class.
    theta_ : ndarray of shape (n_classes, n_features)
        Weighted mean of each feature per class.
    var_ : ndarray of shape (n_classes, n_features)
        Weighted variance of each feature per class (with smoothing).
    pr_scores_ : ndarray of shape (n_samples,)
        PR scores of training points.
    effective_weights_ : ndarray of shape (n_samples,)
        Weights after flooring.
    n_features_in_ : int
        Number of features seen during fit.

    References
    ----------
    Amer, A.A., Ravana, S.D. & Habeeb, R.A.A. (2025). Effective
    k-nearest neighbor models for data classification enhancement.
    Journal of Big Data, 12, 86.
    """

    def __init__(
        self,
        k: int = 15,
        metric: str = "euclidean",
        weight_floor: float = 1e-3,
        var_smoothing: float = 1e-9,
        empty_neighborhood_value: float = 1.0,
        radius_estimator: str = "mean",
        random_state: int | None = None,
    ):
        self.k = k
        self.metric = metric
        self.weight_floor = weight_floor
        self.var_smoothing = var_smoothing
        self.empty_neighborhood_value = empty_neighborhood_value
        self.radius_estimator = radius_estimator
        self.random_state = random_state

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.integer],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> GaussianPWNB:
        """Fit Gaussian PW-NB according to X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Additional sample weights multiplied with PR weights.

        Returns
        -------
        self
        """
        X, y = self._validate_data(X, y, dtype=np.float64, reset=True)
        self.classes_ = unique_labels(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Standardize features for PR computation (training data only)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Compute PR scores on standardized data
        pr_computer = ProximalRatio(
            k=self.k,
            metric=self.metric,
            empty_neighborhood_value=self.empty_neighborhood_value,
            radius_estimator=self.radius_estimator,
            random_state=self.random_state,
        )
        pr_computer.fit(X_scaled, y)
        self.pr_scores_ = pr_computer.pr_scores_
        self.class_radii_ = pr_computer.class_radii_

        # Apply weight floor
        w = np.maximum(self.pr_scores_, self.weight_floor)

        # Multiply with external sample_weight if given
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            w = w * sample_weight

        self.effective_weights_ = w

        # Weighted prior with Laplace smoothing
        class_weight_sums = np.zeros(n_classes)
        for ci, c in enumerate(self.classes_):
            class_weight_sums[ci] = w[y == c].sum()

        total_weight = w.sum()
        self.class_prior_ = (class_weight_sums + 1.0) / (total_weight + n_classes)

        # Weighted Gaussian MLE per class and feature
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))

        # Variance smoothing: fraction of largest global variance
        global_var = np.var(X, axis=0)
        epsilon = self.var_smoothing * global_var.max()

        for ci, c in enumerate(self.classes_):
            mask = y == c
            X_c = X[mask]
            w_c = w[mask]
            w_sum = w_c.sum()

            # Weighted mean
            self.theta_[ci] = (w_c[:, np.newaxis] * X_c).sum(axis=0) / w_sum

            # Weighted variance
            diff = X_c - self.theta_[ci]
            self.var_[ci] = (w_c[:, np.newaxis] * diff**2).sum(axis=0) / w_sum + epsilon

        self.class_log_prior_ = np.log(self.class_prior_)

        return self

    def _joint_log_likelihood(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute unnormalized log posterior for each class.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Already validated input.

        Returns
        -------
        jll : ndarray of shape (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        jll = np.zeros((n_samples, n_classes))

        for ci in range(n_classes):
            log_prior = self.class_log_prior_[ci]
            mu = self.theta_[ci]
            var = self.var_[ci]
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * var) + (X - mu) ** 2 / var, axis=1
            )
            jll[:, ci] = log_prior + log_likelihood

        return jll

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.integer]:
        """Predict class labels for X."""
        check_is_fitted(self)
        X = self._validate_data(X, dtype=np.float64, reset=False)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict class probabilities for X."""
        check_is_fitted(self)
        X = self._validate_data(X, dtype=np.float64, reset=False)
        jll = self._joint_log_likelihood(X)
        log_prob_norm = jll - logsumexp(jll, axis=1, keepdims=True)
        return np.exp(log_prob_norm)

    def predict_log_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict normalized log class probabilities for X."""
        check_is_fitted(self)
        X = self._validate_data(X, dtype=np.float64, reset=False)
        jll = self._joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)


class AdaptivePWNB(ClassifierMixin, BaseEstimator):
    """PW-NB with inner cross-validation for automatic k selection.

    Wraps GaussianPWNB and selects the best k from k_candidates using
    stratified inner CV on each outer training fold. This avoids manual
    k tuning while ensuring no information leakage from test data.

    k candidates that exceed the minimum class size are filtered out
    before inner CV. If fewer than two valid candidates remain, or the
    data is too small for inner splits, the smallest valid k is used
    directly.

    Parameters
    ----------
    k_candidates : tuple of int, default=(5, 15, 30, 45)
        Candidate k values to evaluate.
    inner_folds : int, default=3
        Number of inner CV folds for k selection.
    metric : str, default="euclidean"
        Distance metric passed to GaussianPWNB.
    weight_floor : float, default=1e-3
        Passed to GaussianPWNB.
    var_smoothing : float, default=1e-9
        Passed to GaussianPWNB.
    empty_neighborhood_value : float, default=1.0
        Passed to GaussianPWNB.
    radius_estimator : str, default="mean"
        Passed to GaussianPWNB.
    random_state : int or None, default=None
        Random state for inner CV splits and GaussianPWNB.

    Attributes
    ----------
    best_k_ : int
        k value selected by inner CV.
    model_ : GaussianPWNB
        Final model fitted on the full training set with best_k_.
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        k_candidates: tuple = (5, 15, 30, 45),
        inner_folds: int = 3,
        metric: str = "euclidean",
        weight_floor: float = 1e-3,
        var_smoothing: float = 1e-9,
        empty_neighborhood_value: float = 1.0,
        radius_estimator: str = "mean",
        random_state: int | None = None,
    ):
        self.k_candidates = k_candidates
        self.inner_folds = inner_folds
        self.metric = metric
        self.weight_floor = weight_floor
        self.var_smoothing = var_smoothing
        self.empty_neighborhood_value = empty_neighborhood_value
        self.radius_estimator = radius_estimator
        self.random_state = random_state

    def _make_model(self, k: int) -> GaussianPWNB:
        return GaussianPWNB(
            k=k,
            metric=self.metric,
            weight_floor=self.weight_floor,
            var_smoothing=self.var_smoothing,
            empty_neighborhood_value=self.empty_neighborhood_value,
            radius_estimator=self.radius_estimator,
            random_state=self.random_state,
        )

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.integer],
    ) -> "AdaptivePWNB":
        """Select k via inner CV then fit on the full training set."""
        X, y = self._validate_data(X, y, dtype=np.float64, reset=True)
        self.classes_ = unique_labels(y)

        _, counts = np.unique(y, return_counts=True)
        min_count = int(counts.min())

        # Keep only candidates strictly less than the minimum class size
        # (mirrors adapt_k_for_dataset logic used for fixed-k variants)
        valid_k = [k for k in self.k_candidates if k < min_count]
        if not valid_k:
            valid_k = [max(1, min_count - 1)]

        # Reduce inner folds if data is too small for the requested split
        actual_inner_folds = min(self.inner_folds, min_count)

        if actual_inner_folds < 2 or len(valid_k) == 1:
            # Not enough data for inner CV — use the smallest valid k
            self.best_k_ = valid_k[0]
        else:
            inner_cv = StratifiedKFold(
                n_splits=actual_inner_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            best_k, best_score = valid_k[0], -np.inf

            for k in valid_k:
                fold_scores = []
                for tr_idx, val_idx in inner_cv.split(X, y):
                    X_tr, y_tr = X[tr_idx], y[tr_idx]
                    X_val, y_val = X[val_idx], y[val_idx]

                    # Clip k to the inner fold's minimum class size
                    _, inner_counts = np.unique(y_tr, return_counts=True)
                    effective_k = min(k, max(1, int(inner_counts.min()) - 1))

                    m = self._make_model(effective_k)
                    m.fit(X_tr, y_tr)
                    fold_scores.append(float(np.mean(m.predict(X_val) == y_val)))

                mean_score = float(np.mean(fold_scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_k = k

            self.best_k_ = best_k

        self.model_ = self._make_model(self.best_k_)
        self.model_.fit(X, y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.integer]:
        """Predict class labels for X."""
        check_is_fitted(self)
        return self.model_.predict(X)

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict class probabilities for X."""
        check_is_fitted(self)
        return self.model_.predict_proba(X)

    def predict_log_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict normalized log class probabilities for X."""
        check_is_fitted(self)
        return self.model_.predict_log_proba(X)


class MultinomialPWNB(ClassifierMixin, BaseEstimator):
    """Multinomial Proximity-Weighted Naive Bayes classifier.

    Same weighted-count approach as GaussianPWNB but applied to
    multinomial (count/frequency) likelihoods. Features must be
    non-negative.

    Parameters
    ----------
    k : int, default=15
        Number of nearest neighbors for PR computation.
    metric : str, default="euclidean"
        Distance metric for PR computation.
    weight_floor : float, default=1e-3
        Minimum weight for any training instance.
    alpha : float, default=1.0
        Laplace smoothing parameter.
    empty_neighborhood_value : float, default=1.0
        PR value when |S|=0.
    radius_estimator : str, default="mean"
        Radius estimation method.
    random_state : int or None, default=None
        Random state for reproducibility.
    """

    def __init__(
        self,
        k: int = 15,
        metric: str = "euclidean",
        weight_floor: float = 1e-3,
        alpha: float = 1.0,
        empty_neighborhood_value: float = 1.0,
        radius_estimator: str = "mean",
        random_state: int | None = None,
    ):
        self.k = k
        self.metric = metric
        self.weight_floor = weight_floor
        self.alpha = alpha
        self.empty_neighborhood_value = empty_neighborhood_value
        self.radius_estimator = radius_estimator
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        return tags

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.integer],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> MultinomialPWNB:
        """Fit Multinomial PW-NB according to X, y."""
        X, y = self._validate_data(X, y, dtype=np.float64, reset=True)
        if np.any(X < 0):
            raise ValueError(
                "Negative values in data passed to MultinomialPWNB (input X). "
                "MultinomialPWNB requires non-negative feature values."
            )

        self.classes_ = unique_labels(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Standardize for PR computation
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Compute PR scores
        pr_computer = ProximalRatio(
            k=self.k,
            metric=self.metric,
            empty_neighborhood_value=self.empty_neighborhood_value,
            radius_estimator=self.radius_estimator,
            random_state=self.random_state,
        )
        pr_computer.fit(X_scaled, y)
        self.pr_scores_ = pr_computer.pr_scores_

        # Apply weight floor
        w = np.maximum(self.pr_scores_, self.weight_floor)
        if sample_weight is not None:
            w = w * np.asarray(sample_weight, dtype=np.float64)
        self.effective_weights_ = w

        # Weighted prior
        class_weight_sums = np.zeros(n_classes)
        for ci, c in enumerate(self.classes_):
            class_weight_sums[ci] = w[y == c].sum()
        total_weight = w.sum()
        self.class_prior_ = (class_weight_sums + 1.0) / (total_weight + n_classes)

        # Weighted feature log-probabilities per class
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        for ci, c in enumerate(self.classes_):
            mask = y == c
            X_c = X[mask]
            w_c = w[mask]
            weighted_counts = (w_c[:, np.newaxis] * X_c).sum(axis=0)
            smoothed = weighted_counts + self.alpha
            total = smoothed.sum()
            self.feature_log_prob_[ci] = np.log(smoothed / total)

        self.class_log_prior_ = np.log(self.class_prior_)
        return self

    def _joint_log_likelihood(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute unnormalized log posterior for each class."""
        return self.class_log_prior_ + X @ self.feature_log_prob_.T

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.integer]:
        """Predict class labels for X."""
        check_is_fitted(self)
        X = self._validate_data(X, dtype=np.float64, reset=False)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict class probabilities for X."""
        check_is_fitted(self)
        X = self._validate_data(X, dtype=np.float64, reset=False)
        jll = self._joint_log_likelihood(X)
        log_prob_norm = jll - logsumexp(jll, axis=1, keepdims=True)
        return np.exp(log_prob_norm)

    def predict_log_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict normalized log class probabilities for X."""
        check_is_fitted(self)
        X = self._validate_data(X, dtype=np.float64, reset=False)
        jll = self._joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)


class AdaptivePWNB(ClassifierMixin, BaseEstimator):
    """PW-NB with inner cross-validation for automatic k selection.

    Wraps GaussianPWNB and selects the best k from k_candidates using
    stratified inner CV on each outer training fold. This avoids manual
    k tuning while ensuring no information leakage from test data.

    k candidates that exceed the minimum class size are filtered out
    before inner CV. If fewer than two valid candidates remain, or the
    data is too small for inner splits, the smallest valid k is used
    directly.

    Parameters
    ----------
    k_candidates : tuple of int, default=(5, 15, 30, 45)
        Candidate k values to evaluate.
    inner_folds : int, default=3
        Number of inner CV folds for k selection.
    metric : str, default="euclidean"
        Distance metric passed to GaussianPWNB.
    weight_floor : float, default=1e-3
        Passed to GaussianPWNB.
    var_smoothing : float, default=1e-9
        Passed to GaussianPWNB.
    empty_neighborhood_value : float, default=1.0
        Passed to GaussianPWNB.
    radius_estimator : str, default="mean"
        Passed to GaussianPWNB.
    random_state : int or None, default=None
        Random state for inner CV splits and GaussianPWNB.

    Attributes
    ----------
    best_k_ : int
        k value selected by inner CV.
    model_ : GaussianPWNB
        Final model fitted on the full training set with best_k_.
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        k_candidates: tuple = (5, 15, 30, 45),
        inner_folds: int = 3,
        metric: str = "euclidean",
        weight_floor: float = 1e-3,
        var_smoothing: float = 1e-9,
        empty_neighborhood_value: float = 1.0,
        radius_estimator: str = "mean",
        random_state: int | None = None,
    ):
        self.k_candidates = k_candidates
        self.inner_folds = inner_folds
        self.metric = metric
        self.weight_floor = weight_floor
        self.var_smoothing = var_smoothing
        self.empty_neighborhood_value = empty_neighborhood_value
        self.radius_estimator = radius_estimator
        self.random_state = random_state

    def _make_model(self, k: int) -> GaussianPWNB:
        return GaussianPWNB(
            k=k,
            metric=self.metric,
            weight_floor=self.weight_floor,
            var_smoothing=self.var_smoothing,
            empty_neighborhood_value=self.empty_neighborhood_value,
            radius_estimator=self.radius_estimator,
            random_state=self.random_state,
        )

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.integer],
    ) -> "AdaptivePWNB":
        """Select k via inner CV then fit on the full training set."""
        X, y = self._validate_data(X, y, dtype=np.float64, reset=True)
        self.classes_ = unique_labels(y)

        _, counts = np.unique(y, return_counts=True)
        min_count = int(counts.min())

        # Keep only candidates strictly less than the minimum class size
        # (mirrors adapt_k_for_dataset logic used for fixed-k variants)
        valid_k = [k for k in self.k_candidates if k < min_count]
        if not valid_k:
            valid_k = [max(1, min_count - 1)]

        # Reduce inner folds if data is too small for the requested split
        actual_inner_folds = min(self.inner_folds, min_count)

        if actual_inner_folds < 2 or len(valid_k) == 1:
            # Not enough data for inner CV — use the smallest valid k
            self.best_k_ = valid_k[0]
        else:
            inner_cv = StratifiedKFold(
                n_splits=actual_inner_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            best_k, best_score = valid_k[0], -np.inf

            for k in valid_k:
                fold_scores = []
                for tr_idx, val_idx in inner_cv.split(X, y):
                    X_tr, y_tr = X[tr_idx], y[tr_idx]
                    X_val, y_val = X[val_idx], y[val_idx]

                    # Clip k to the inner fold's minimum class size
                    _, inner_counts = np.unique(y_tr, return_counts=True)
                    effective_k = min(k, max(1, int(inner_counts.min()) - 1))

                    m = self._make_model(effective_k)
                    m.fit(X_tr, y_tr)
                    fold_scores.append(float(np.mean(m.predict(X_val) == y_val)))

                mean_score = float(np.mean(fold_scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_k = k

            self.best_k_ = best_k

        self.model_ = self._make_model(self.best_k_)
        self.model_.fit(X, y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.integer]:
        """Predict class labels for X."""
        check_is_fitted(self)
        return self.model_.predict(X)

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict class probabilities for X."""
        check_is_fitted(self)
        return self.model_.predict_proba(X)

    def predict_log_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict normalized log class probabilities for X."""
        check_is_fitted(self)
        return self.model_.predict_log_proba(X)
