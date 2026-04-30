"""Proximal Ratio (PR) computation for training instances.

Implements the Proximal Ratio technique from:
    Amer, Ravana & Habeeb (2025), "Effective k-nearest neighbor models for
    data classification enhancement", Journal of Big Data, 12:86.

The PR score measures the local class-consistency of a training point
within its k-nearest neighborhood, filtered by a class-wise radius.

    PR(t) = val / |S|

where S is the set of k-nearest neighbors of t that fall within the
class radius R_{y_t}, and val is the count of same-class points in S.

References
----------
Equations 1-4, Figure 2, and the "Proximal ratio computation" section
of the above paper.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_consistent_length


def _compute_class_radius_mean(X_c: NDArray[np.floating], metric: str) -> float:
    """Compute class radius using the paper's Eq. 1 (mean formulation).

    Parameters
    ----------
    X_c : ndarray of shape (n_c, n_features)
        Feature matrix for a single class.
    metric : str
        Distance metric (passed to sklearn.metrics.pairwise_distances).

    Returns
    -------
    float
        Class radius R_c.

    Notes
    -----
    The paper defines R_c = sum(all pairwise distances) / N_c (Eq. 1).
    This divides by N_c, NOT by N_c*(N_c-1) or N_c^2, which is
    unconventional. It makes R_c approximately equal to the average
    total distance from a point to all others in its class, yielding
    a much larger value than the conventional mean pairwise distance.
    We follow the paper literally.
    """
    n_c = len(X_c)
    if n_c < 2:
        return np.inf
    # pdist computes only the upper triangle: n_c*(n_c-1)/2 entries
    # Full symmetric matrix sum = 2 * upper-triangle sum (diagonal is 0)
    d = pdist(X_c, metric=metric)
    return float(2.0 * d.sum() / n_c)


def _compute_class_radius_median(X_c: NDArray[np.floating], metric: str) -> float:
    """Compute class radius using median of per-point distance sums."""
    n_c = len(X_c)
    if n_c < 2:
        return np.inf
    D = squareform(pdist(X_c, metric=metric))
    per_point_sums = D.sum(axis=1)
    return float(np.median(per_point_sums))


def _compute_class_radius_trimmed_mean(
    X_c: NDArray[np.floating], metric: str, trim_fraction: float
) -> float:
    """Compute class radius using trimmed mean of per-point distance sums."""
    n_c = len(X_c)
    if n_c < 2:
        return np.inf
    D = squareform(pdist(X_c, metric=metric))
    per_point_sums = D.sum(axis=1)
    sorted_sums = np.sort(per_point_sums)
    trim_count = int(n_c * trim_fraction)
    if trim_count > 0 and 2 * trim_count < n_c:
        trimmed = sorted_sums[trim_count:-trim_count]
    else:
        trimmed = sorted_sums
    return float(trimmed.mean())


class ProximalRatio:
    """Compute Proximal Ratio (PR) scores per training instance.

    Follows Amer, Ravana & Habeeb (2025), J. Big Data, Equations 1-4.

    Parameters
    ----------
    k : int, default=15
        Number of nearest neighbors to consider.
    metric : str, default="euclidean"
        Distance metric (any metric accepted by sklearn NearestNeighbors).
    empty_neighborhood_value : float, default=1.0
        PR value assigned when no k-nearest neighbors fall within the
        class radius (|S| = 0). Default is 1.0 because an isolated
        point with no radius-local neighbors carries no evidence of
        class overlap, so it should not be down-weighted. Set to 0.0
        to instead treat such points as outliers.
    radius_estimator : str, default="mean"
        Method for computing class radii. "mean" follows the paper's
        Eq. 1 exactly. "trimmed_mean" and "median" are alternatives
        for robustness ablations.
    trim_fraction : float, default=0.1
        Fraction of points to trim from each end when
        radius_estimator="trimmed_mean". Ignored otherwise.
    random_state : int or None, default=None
        Not used directly (PR is deterministic given X, y, k), but
        stored for API consistency.

    Attributes
    ----------
    pr_scores_ : ndarray of shape (n_samples,)
        PR score per training point, in [0, 1].
    class_radii_ : dict[int, float]
        Computed radius R_c for each class label c.

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
        empty_neighborhood_value: float = 1.0,
        radius_estimator: str = "mean",
        trim_fraction: float = 0.1,
        random_state: int | None = None,
    ):
        self.k = k
        self.metric = metric
        self.empty_neighborhood_value = empty_neighborhood_value
        self.radius_estimator = radius_estimator
        self.trim_fraction = trim_fraction
        self.random_state = random_state

    def fit(self, X: NDArray[np.floating], y: NDArray[np.integer]) -> ProximalRatio:
        """Compute class radii and PR score for every training point.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Training labels (integer-encoded).

        Returns
        -------
        self
        """
        X = check_array(X, dtype=np.float64)
        y = np.asarray(y)
        check_consistent_length(X, y)

        n = len(X)
        classes = np.unique(y)

        # Step 1: Compute class-wise radii (Eq. 1 / Eq. 4)
        radii: dict = {}
        for c in classes:
            X_c = X[y == c]
            if self.radius_estimator == "mean":
                radii[c] = _compute_class_radius_mean(X_c, self.metric)
            elif self.radius_estimator == "median":
                radii[c] = _compute_class_radius_median(X_c, self.metric)
            elif self.radius_estimator == "trimmed_mean":
                radii[c] = _compute_class_radius_trimmed_mean(
                    X_c, self.metric, self.trim_fraction
                )
            else:
                raise ValueError(
                    f"Unknown radius_estimator: {self.radius_estimator!r}. "
                    "Must be 'mean', 'trimmed_mean', or 'median'."
                )

        # Step 2: k-NN across full training set (k+1 to exclude self)
        effective_k = min(self.k + 1, n)
        nn = NearestNeighbors(n_neighbors=effective_k, metric=self.metric)
        nn.fit(X)
        dists, idxs = nn.kneighbors(X)

        # Step 3: Vectorized PR computation (Eq. 2)
        neigh_dists = dists[:, 1:]   # (n, k) — exclude self
        neigh_idxs = idxs[:, 1:]

        # Per-point class radius lookup
        point_radii = np.empty(n)
        for c in classes:
            point_radii[y == c] = radii[c]

        # Boolean masks over the (n, k) neighbor matrix
        in_radius = neigh_dists <= point_radii[:, np.newaxis]
        neigh_labels = y[neigh_idxs]
        same_class = neigh_labels == y[:, np.newaxis]

        n_in_radius = in_radius.sum(axis=1)
        n_same_in_radius = (same_class & in_radius).sum(axis=1)

        pr = np.where(
            n_in_radius == 0,
            self.empty_neighborhood_value,
            n_same_in_radius / np.maximum(n_in_radius, 1),
        )

        self._pr_scores = pr
        self._class_radii = radii
        self._X = X
        self._y = y
        return self

    @property
    def pr_scores_(self) -> NDArray[np.floating]:
        """PR score per training point, shape (n_samples,), in [0, 1]."""
        return self._pr_scores

    @property
    def class_radii_(self) -> dict[int, float]:
        """R_c for each class."""
        return self._class_radii
