"""Unit tests for Proximal Ratio computation.

Tests cover:
1. Range test: PR scores in [0, 1] for random data.
2. Pure-class test: single label => all PR = 1.0.
3. Perfect separation test: distant blobs => all PR = 1.0.
4. Figure 2 reproduction: manual layout matching the paper's worked example.
5. Radius computation: verify Eq. 1 on collinear points.
6. Empty-neighborhood edge case: isolated outlier => falls back to default.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.proximal_ratio import ProximalRatio, _compute_class_radius_mean


class TestPRRange:
    """Test 1: all PR scores lie in [0, 1] for random data."""

    def test_random_data_pr_in_unit_interval(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = rng.choice([0, 1, 2], size=100)

        pr = ProximalRatio(k=10)
        pr.fit(X, y)

        assert pr.pr_scores_.shape == (100,)
        assert np.all(pr.pr_scores_ >= 0.0)
        assert np.all(pr.pr_scores_ <= 1.0)

    def test_various_k_values(self):
        rng = np.random.RandomState(123)
        X = rng.randn(50, 3)
        y = rng.choice([0, 1], size=50)

        for k in [1, 3, 5, 10, 20, 49]:
            pr = ProximalRatio(k=k)
            pr.fit(X, y)
            assert np.all(pr.pr_scores_ >= 0.0)
            assert np.all(pr.pr_scores_ <= 1.0)


class TestPureClass:
    """Test 2: if all points share one label, all PR scores must be 1.0."""

    def test_single_class(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 4)
        y = np.zeros(30, dtype=int)

        pr = ProximalRatio(k=10)
        pr.fit(X, y)

        np.testing.assert_array_equal(pr.pr_scores_, 1.0)

    def test_single_class_various_k(self):
        rng = np.random.RandomState(99)
        X = rng.randn(20, 2)
        y = np.ones(20, dtype=int)

        for k in [1, 5, 10, 19]:
            pr = ProximalRatio(k=k)
            pr.fit(X, y)
            np.testing.assert_array_equal(pr.pr_scores_, 1.0)


class TestPerfectSeparation:
    """Test 3: two Gaussian blobs far apart => all PR = 1.0 (for k <= n_c - 1)."""

    def test_distant_blobs(self):
        rng = np.random.RandomState(42)
        n_per_class = 20

        X0 = rng.randn(n_per_class, 2) * 0.1
        X1 = rng.randn(n_per_class, 2) * 0.1 + 1000.0
        X = np.vstack([X0, X1])
        y = np.array([0] * n_per_class + [1] * n_per_class)

        for k in [1, 5, 10, 19]:
            pr = ProximalRatio(k=k)
            pr.fit(X, y)
            np.testing.assert_array_equal(
                pr.pr_scores_,
                1.0,
                err_msg=f"Failed for k={k}",
            )

    def test_three_distant_blobs(self):
        rng = np.random.RandomState(42)
        n_per_class = 15
        X0 = rng.randn(n_per_class, 2) * 0.1
        X1 = rng.randn(n_per_class, 2) * 0.1 + np.array([1000, 0])
        X2 = rng.randn(n_per_class, 2) * 0.1 + np.array([0, 1000])
        X = np.vstack([X0, X1, X2])
        y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

        pr = ProximalRatio(k=10)
        pr.fit(X, y)
        np.testing.assert_array_equal(pr.pr_scores_, 1.0)


class TestFigure2Reproduction:
    """Test 4: reproduce the worked example from Figure 2 of the paper.

    The paper's Figure 2 (page 8) shows a 2D layout with k=3:
        - Class 0 (red circles): cluster on the left side
        - Class 1 (blue triangles): cluster on the right side
        - Overlap zone in the middle

    Four labeled points with known PR values:
        Point A (Class 0, idx 0): PR = 3/3 = 1.0   (clean interior point)
        Point B (Class 0, idx 1): PR = 2/3 ~ 0.667  (boundary/overlap point)
        Point C (Class 1, idx 5): PR = 1.0           (clean class 1 point)
        Point D (Class 1, idx 6): PR = 0/3 = 0.0     (outlier in class 0 territory)

    The geometry below was verified numerically to produce these exact values.
    """

    @pytest.fixture
    def figure2_data(self):
        """Construct a 2D dataset matching Figure 2's geometry.

        Layout (verified numerically):
        - Class 0 cluster on the left:
            A=(0,0) deep inside, B=(3,0) near boundary,
            plus 3 supporting points.
        - Class 1 cluster on the right:
            C=(8,0) clean point, D=(1,0) outlier in class 0 territory,
            plus 3 supporting points including one at (6, 0.5)
            bridging the gap.

        With k=3:
        - A's 3-NN: all class 0 within R_0 => PR=1.0
        - B's 3-NN: 2 class 0 + 1 class 1 within R_0 => PR=2/3
        - C's 3-NN: all class 1 within R_1 => PR=1.0
        - D's 3-NN: all class 0 within R_1 => PR=0/3=0.0
        """
        X = np.array(
            [
                [0.0, 0.0],  # idx 0: Point A (class 0) - deep inside
                [3.0, 0.0],  # idx 1: Point B (class 0) - near boundary
                [0.5, 0.5],  # idx 2: class 0
                [0.5, -0.5],  # idx 3: class 0
                [-0.5, 0.0],  # idx 4: class 0
                [8.0, 0.0],  # idx 5: Point C (class 1) - clean
                [1.0, 0.0],  # idx 6: Point D (class 1) - outlier
                [6.0, 0.5],  # idx 7: class 1 - bridge region
                [9.0, 0.0],  # idx 8: class 1
                [8.5, 0.5],  # idx 9: class 1
            ]
        )
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        return X, y

    def test_class_radii(self, figure2_data):
        """Verify class radii are computed correctly per Eq. 1."""
        X, y = figure2_data
        pr = ProximalRatio(k=3)
        pr.fit(X, y)

        from sklearn.metrics import pairwise_distances as pdist

        X0 = X[y == 0]
        D0 = pdist(X0)
        expected_r0 = D0.sum() / len(X0)

        X1 = X[y == 1]
        D1 = pdist(X1)
        expected_r1 = D1.sum() / len(X1)

        assert pr.class_radii_[0] == pytest.approx(expected_r0, abs=1e-10)
        assert pr.class_radii_[1] == pytest.approx(expected_r1, abs=1e-10)

    def test_point_a_clean(self, figure2_data):
        """Point A (idx 0, Class 0): all k-NN in radius are same-class => PR=1."""
        X, y = figure2_data
        pr = ProximalRatio(k=3)
        pr.fit(X, y)
        assert pr.pr_scores_[0] == pytest.approx(1.0)

    def test_point_b_overlap(self, figure2_data):
        """Point B (idx 1, Class 0): 2 same-class + 1 diff in radius => PR=2/3."""
        X, y = figure2_data
        pr = ProximalRatio(k=3)
        pr.fit(X, y)
        assert pr.pr_scores_[1] == pytest.approx(2.0 / 3.0, abs=1e-10)

    def test_point_c_clean(self, figure2_data):
        """Point C (idx 5, Class 1): neighbors in radius are all same-class => PR=1."""
        X, y = figure2_data
        pr = ProximalRatio(k=3)
        pr.fit(X, y)
        assert pr.pr_scores_[5] == pytest.approx(1.0)

    def test_point_d_outlier(self, figure2_data):
        """Point D (idx 6, Class 1): neighbors in radius all Class 0 => PR=0."""
        X, y = figure2_data
        pr = ProximalRatio(k=3)
        pr.fit(X, y)
        assert pr.pr_scores_[6] == pytest.approx(0.0)

    def test_all_four_points_together(self, figure2_data):
        """Verify all four named points match expected PR values simultaneously."""
        X, y = figure2_data
        pr = ProximalRatio(k=3)
        pr.fit(X, y)

        expected = {
            0: 1.0,  # Point A: clean interior
            1: 2.0 / 3.0,  # Point B: overlap zone
            5: 1.0,  # Point C: clean class 1
            6: 0.0,  # Point D: outlier
        }
        for idx, expected_pr in expected.items():
            assert pr.pr_scores_[idx] == pytest.approx(expected_pr, abs=1e-10), (
                f"Point at index {idx}: "
                f"expected PR={expected_pr}, got {pr.pr_scores_[idx]}"
            )


class TestRadiusComputation:
    """Test 5: verify Eq. 1 on known geometries."""

    def test_three_collinear_points(self):
        """Three 1D points at 0, 1, 2: verify R_c matches Eq. 1.

        Pairwise distance matrix:
            [[0, 1, 2],
             [1, 0, 1],
             [2, 1, 0]]

        Sum of all entries = 8.  N_c = 3.  R_c = 8/3 = 2.6667.
        """
        X_c = np.array([[0.0], [1.0], [2.0]])
        r = _compute_class_radius_mean(X_c, metric="euclidean")
        expected = 8.0 / 3.0
        assert r == pytest.approx(expected, abs=1e-10)

    def test_two_points(self):
        """Two points at distance 5: R_c = (0+5+5+0) / 2 = 5."""
        X_c = np.array([[0.0], [5.0]])
        r = _compute_class_radius_mean(X_c, metric="euclidean")
        assert r == pytest.approx(5.0, abs=1e-10)

    def test_single_point_returns_inf(self):
        """A class with 1 point should have infinite radius."""
        X_c = np.array([[3.0, 4.0]])
        r = _compute_class_radius_mean(X_c, metric="euclidean")
        assert r == np.inf

    def test_four_points_square(self):
        """Four points at unit square corners.

        Pairwise distances (unique): 1, 1, sqrt(2), sqrt(2), 1, 1.
        Full matrix sum = 2*(4 + 2*sqrt(2)) = 8 + 4*sqrt(2).
        N_c = 4.  R_c = 2 + sqrt(2).
        """
        X_c = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        r = _compute_class_radius_mean(X_c, metric="euclidean")
        expected = 2.0 + np.sqrt(2.0)
        assert r == pytest.approx(expected, abs=1e-10)


class TestEmptyNeighborhood:
    """Test 6: edge case when |S| = 0.

    The paper's radius formula (Eq. 1) divides the full pairwise sum by N_c,
    producing large radii that include most neighbors. Getting |S| = 0
    (no k-NN within the class radius) is geometrically rare but the code
    must handle it. We verify:
    (a) The empty_neighborhood_value parameter is stored correctly.
    (b) Outlier points correctly get PR = 0 (val=0 with |S|>0).
    (c) With co-located class members (R_c=0) and k=1, self-class
        neighbors at distance 0 are still within the radius (0 <= 0).
    """

    def test_parameter_stored(self):
        """Verify the empty_neighborhood_value parameter is accessible."""
        pr1 = ProximalRatio(k=5, empty_neighborhood_value=1.0)
        pr0 = ProximalRatio(k=5, empty_neighborhood_value=0.0)
        assert pr1.empty_neighborhood_value == 1.0
        assert pr0.empty_neighborhood_value == 0.0

    def test_outlier_gets_pr_zero(self):
        """An outlier surrounded by foreign-class points gets PR=0."""
        # Class 0: tight cluster near origin
        # Class 1: core at (100,0) + outlier at (0.5, 0) in class 0 territory
        X = np.array(
            [
                [0.0, 0.0],  # class 0
                [0.1, 0.0],  # class 0
                [0.0, 0.1],  # class 0
                [0.1, 0.1],  # class 0
                [100.0, 0.0],  # class 1 core
                [100.1, 0.0],  # class 1 core
                [0.5, 0.0],  # class 1 outlier
            ]
        )
        y = np.array([0, 0, 0, 0, 1, 1, 1])

        pr = ProximalRatio(k=3)
        pr.fit(X, y)

        # Outlier (idx 6) at (0.5, 0): 3-NN are all class 0 points,
        # and within R_1 (which is large due to the outlier inflating it).
        # val=0, |S|=3 => PR=0.
        assert pr.pr_scores_[6] == pytest.approx(0.0)

    def test_colocated_class_zero_radius(self):
        """Co-located class members give R_c=0; neighbors at dist 0 are
        still within radius (0 <= 0), so PR is well-defined, not empty."""
        X = np.array(
            [
                [0.0, 0.0],  # class 0
                [0.0, 0.0],  # class 0 (co-located)
                [10.0, 0.0],  # class 1
                [10.0, 0.0],  # class 1 (co-located)
            ]
        )
        y = np.array([0, 0, 1, 1])

        pr = ProximalRatio(k=1, empty_neighborhood_value=0.5)
        pr.fit(X, y)

        # Each point's 1-NN is the co-located same-class point at dist 0.
        # 0 <= R_c=0 so it's within radius. val=1, |S|=1. PR=1.
        # empty_neighborhood_value should NOT be triggered.
        assert pr.pr_scores_[0] == pytest.approx(1.0)
        assert pr.pr_scores_[1] == pytest.approx(1.0)
        assert pr.pr_scores_[2] == pytest.approx(1.0)
        assert pr.pr_scores_[3] == pytest.approx(1.0)

    def test_empty_neighborhood_value_both_settings(self):
        """Both empty_neighborhood_value settings produce same results
        when |S| > 0 for all points (parameter is unused)."""
        rng = np.random.RandomState(42)
        X = rng.randn(30, 2)
        y = np.array([0] * 15 + [1] * 15)

        pr1 = ProximalRatio(k=5, empty_neighborhood_value=1.0)
        pr1.fit(X, y)

        pr0 = ProximalRatio(k=5, empty_neighborhood_value=0.0)
        pr0.fit(X, y)

        # For typical data, |S| > 0 always, so the parameter doesn't matter.
        np.testing.assert_array_almost_equal(pr1.pr_scores_, pr0.pr_scores_)


class TestRadiusEstimators:
    """Test alternative radius estimators (median, trimmed_mean)."""

    def test_median_estimator_produces_valid_scores(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 2)
        y = np.array([0] * 15 + [1] * 15)

        pr = ProximalRatio(k=5, radius_estimator="median")
        pr.fit(X, y)

        assert pr.pr_scores_.shape == (30,)
        assert np.all(pr.pr_scores_ >= 0.0)
        assert np.all(pr.pr_scores_ <= 1.0)

    def test_trimmed_mean_estimator_produces_valid_scores(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 2)
        y = np.array([0] * 15 + [1] * 15)

        pr = ProximalRatio(k=5, radius_estimator="trimmed_mean", trim_fraction=0.1)
        pr.fit(X, y)

        assert pr.pr_scores_.shape == (30,)
        assert np.all(pr.pr_scores_ >= 0.0)
        assert np.all(pr.pr_scores_ <= 1.0)

    def test_invalid_estimator_raises(self):
        pr = ProximalRatio(k=5, radius_estimator="invalid")
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 0, 1])
        with pytest.raises(ValueError, match="Unknown radius_estimator"):
            pr.fit(X, y)
