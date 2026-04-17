"""Tests for quadrature rules: Gauss-Hermite, sigma points, cubature."""

import jax.numpy as jnp
import lineax as lx

from gaussx._quadrature._quadrature import (
    cubature_points,
    gauss_hermite_points,
    sigma_points,
)


class TestGaussHermitePoints:
    def test_1d_shapes(self):
        """1D should return (order, 1) points and (order,) weights."""
        pts, wts = gauss_hermite_points(5, 1)
        assert pts.shape == (5, 1)
        assert wts.shape == (5,)

    def test_2d_shapes(self):
        """2D with order 3 should return (9, 2) points and (9,) weights."""
        pts, wts = gauss_hermite_points(3, 2)
        assert pts.shape == (9, 2)
        assert wts.shape == (9,)

    def test_weights_sum(self):
        """Weights should sum to sqrt(2*pi)^dim for probabilists' convention."""
        _pts, wts = gauss_hermite_points(5, 1)
        # hermegauss weights sum to sqrt(2*pi)
        assert jnp.allclose(jnp.sum(wts), jnp.sqrt(2 * jnp.pi), rtol=1e-6)

    def test_integrate_constant(self):
        """Should integrate a constant exactly."""
        _pts, wts = gauss_hermite_points(3, 2)
        # E[1] = 1 under N(0,I), weights include the Gaussian measure
        # For hermegauss: sum(w_i) = (2*pi)^{d/2}
        integral = jnp.sum(wts)
        assert jnp.allclose(integral, (2 * jnp.pi), rtol=1e-6)

    def test_integrate_x_squared_1d(self):
        """Should integrate x^2 under N(0,1) = 1."""
        pts, wts = gauss_hermite_points(5, 1)
        # E[x^2] under N(0,1) = 1
        # With hermegauss: sum(w_i * x_i^2) / sum(w_i) = E[x^2] = 1
        integral = jnp.sum(wts * pts[:, 0] ** 2) / jnp.sum(wts)
        assert jnp.allclose(integral, 1.0, rtol=1e-6)


class TestSigmaPoints:
    def test_shapes(self):
        """Should produce 2N+1 points."""
        N = 3
        mean = jnp.zeros(N)
        cov = lx.MatrixLinearOperator(jnp.eye(N), lx.positive_semidefinite_tag)

        chi, w_m, w_c = sigma_points(mean, cov)
        assert chi.shape == (2 * N + 1, N)
        assert w_m.shape == (2 * N + 1,)
        assert w_c.shape == (2 * N + 1,)

    def test_mean_weights_sum_one(self):
        """Mean weights should sum to 1."""
        N = 4
        mean = jnp.zeros(N)
        cov = lx.MatrixLinearOperator(jnp.eye(N), lx.positive_semidefinite_tag)

        _, w_m, _ = sigma_points(mean, cov)
        assert jnp.allclose(jnp.sum(w_m), 1.0, atol=1e-10)

    def test_weighted_mean_matches(self):
        """Weighted mean of sigma points should equal the input mean."""
        N = 3
        mean = jnp.array([1.0, 2.0, 3.0])
        cov = lx.MatrixLinearOperator(0.5 * jnp.eye(N), lx.positive_semidefinite_tag)

        chi, w_m, _ = sigma_points(mean, cov)
        recovered_mean = jnp.sum(w_m[:, None] * chi, axis=0)
        assert jnp.allclose(recovered_mean, mean, atol=1e-6)

    def test_first_point_is_mean(self):
        """First sigma point should be the mean."""
        N = 3
        mean = jnp.array([1.0, -1.0, 0.5])
        cov = lx.MatrixLinearOperator(jnp.eye(N), lx.positive_semidefinite_tag)

        chi, _, _ = sigma_points(mean, cov)
        assert jnp.allclose(chi[0], mean, atol=1e-10)


class TestCubaturePoints:
    def test_shapes(self):
        """Should produce 2N points."""
        N = 4
        mean = jnp.zeros(N)
        cov = lx.MatrixLinearOperator(jnp.eye(N), lx.positive_semidefinite_tag)

        chi, wts = cubature_points(mean, cov)
        assert chi.shape == (2 * N, N)
        assert wts.shape == (2 * N,)

    def test_weights_sum_one(self):
        """Weights should sum to 1."""
        N = 3
        mean = jnp.zeros(N)
        cov = lx.MatrixLinearOperator(jnp.eye(N), lx.positive_semidefinite_tag)

        _, wts = cubature_points(mean, cov)
        assert jnp.allclose(jnp.sum(wts), 1.0, atol=1e-10)

    def test_equal_weights(self):
        """All weights should be 1/(2N)."""
        N = 5
        mean = jnp.zeros(N)
        cov = lx.MatrixLinearOperator(jnp.eye(N), lx.positive_semidefinite_tag)

        _, wts = cubature_points(mean, cov)
        assert jnp.allclose(wts, 1.0 / (2 * N), atol=1e-10)

    def test_weighted_mean_matches(self):
        """Weighted mean should equal the input mean."""
        N = 3
        mean = jnp.array([2.0, -1.0, 0.5])
        cov = lx.MatrixLinearOperator(jnp.eye(N), lx.positive_semidefinite_tag)

        chi, wts = cubature_points(mean, cov)
        recovered_mean = jnp.sum(wts[:, None] * chi, axis=0)
        assert jnp.allclose(recovered_mean, mean, atol=1e-6)
