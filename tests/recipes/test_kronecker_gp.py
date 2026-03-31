"""Tests for Kronecker GP recipes."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._recipes._kronecker_gp import (
    kronecker_mll,
    kronecker_posterior_predictive,
)


def _make_pd(key, n):
    """Make a positive definite matrix."""
    A = jax.random.normal(key, (n, n))
    return A @ A.T + 0.1 * jnp.eye(n)


def _rbf_kernel(x, y, lengthscale=1.0):
    """Scalar RBF kernel for exact-reference tests."""
    return jnp.exp(-0.5 * ((x - y) / lengthscale) ** 2)


class TestKroneckerMLL:
    def test_matches_dense(self, getkey):
        """Kronecker MLL should match dense computation."""
        n1, n2 = 4, 3
        N = n1 * n2
        K1 = _make_pd(getkey(), n1)
        K2 = _make_pd(getkey(), n2)
        noise_var = 0.1
        y = jax.random.normal(getkey(), (N,))

        K1_op = lx.MatrixLinearOperator(K1, lx.positive_semidefinite_tag)
        K2_op = lx.MatrixLinearOperator(K2, lx.positive_semidefinite_tag)

        mll = kronecker_mll([K1_op, K2_op], y, noise_var, (n1, n2))

        # Dense reference
        K_full = jnp.kron(K1, K2) + noise_var * jnp.eye(N)
        _, logdet = jnp.linalg.slogdet(K_full)
        alpha = jnp.linalg.solve(K_full, y)
        mll_ref = -0.5 * (y @ alpha + logdet + N * jnp.log(2 * jnp.pi))

        assert jnp.allclose(mll, mll_ref, atol=1e-4)

    def test_scalar(self, getkey):
        """Should return a scalar."""
        n1, n2 = 3, 3
        K1 = _make_pd(getkey(), n1)
        K2 = _make_pd(getkey(), n2)
        y = jax.random.normal(getkey(), (n1 * n2,))

        K1_op = lx.MatrixLinearOperator(K1, lx.positive_semidefinite_tag)
        K2_op = lx.MatrixLinearOperator(K2, lx.positive_semidefinite_tag)

        mll = kronecker_mll([K1_op, K2_op], y, 0.1, (n1, n2))
        assert mll.shape == ()
        assert jnp.isfinite(mll)

    def test_negative(self, getkey):
        """MLL should be negative (log of probability density)."""
        n1, n2 = 4, 3
        K1 = _make_pd(getkey(), n1)
        K2 = _make_pd(getkey(), n2)
        y = jax.random.normal(getkey(), (n1 * n2,))

        K1_op = lx.MatrixLinearOperator(K1, lx.positive_semidefinite_tag)
        K2_op = lx.MatrixLinearOperator(K2, lx.positive_semidefinite_tag)

        mll = kronecker_mll([K1_op, K2_op], y, 0.1, (n1, n2))
        assert mll < 0

    def test_three_factors(self, getkey):
        """Should work with three Kronecker factors."""
        n1, n2, n3 = 3, 3, 2
        N = n1 * n2 * n3
        K1 = _make_pd(getkey(), n1)
        K2 = _make_pd(getkey(), n2)
        K3 = _make_pd(getkey(), n3)
        y = jax.random.normal(getkey(), (N,))

        ops = [
            lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
            for K in [K1, K2, K3]
        ]
        mll = kronecker_mll(ops, y, 0.1, (n1, n2, n3))

        # Dense reference
        K_full = jnp.kron(jnp.kron(K1, K2), K3) + 0.1 * jnp.eye(N)
        alpha = jnp.linalg.solve(K_full, y)
        _, logdet = jnp.linalg.slogdet(K_full)
        mll_ref = -0.5 * (y @ alpha + logdet + N * jnp.log(2 * jnp.pi))

        assert jnp.allclose(mll, mll_ref, atol=1e-3)


class TestKroneckerPosteriorPredictive:
    def test_matches_dense(self, getkey):
        """Kronecker posterior mean and variance should match dense GP."""
        n1, n2 = 4, 3
        N = n1 * n2
        n1_test, n2_test = 2, 2
        noise_var = 0.1

        x1 = jnp.linspace(-1.0, 1.0, n1)
        x2 = jnp.linspace(-1.5, 1.5, n2)
        x1_test = jnp.linspace(-0.75, 0.75, n1_test)
        x2_test = jnp.linspace(-1.0, 1.0, n2_test)

        K1 = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(x1))(x1)
        K2 = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(x2))(x2)
        y = jax.random.normal(getkey(), (N,))

        # Test cross-covariance factors
        K_cross_1 = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(x1))(
            x1_test
        )
        K_cross_2 = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(x2))(
            x2_test
        )
        K_test_diag_1 = jnp.ones(n1_test)
        K_test_diag_2 = jnp.ones(n2_test)

        K1_op = lx.MatrixLinearOperator(K1, lx.positive_semidefinite_tag)
        K2_op = lx.MatrixLinearOperator(K2, lx.positive_semidefinite_tag)

        mean, var = kronecker_posterior_predictive(
            [K1_op, K2_op],
            y,
            noise_var,
            (n1, n2),
            [K_cross_1, K_cross_2],
            K_test_diag_factors=[K_test_diag_1, K_test_diag_2],
        )

        # Dense reference
        K_full = jnp.kron(K1, K2) + noise_var * jnp.eye(N)
        K_cross_full = jnp.kron(K_cross_1, K_cross_2)
        alpha = jnp.linalg.solve(K_full, y)
        mean_ref = K_cross_full @ alpha
        solve_cross = jnp.linalg.solve(K_full, K_cross_full.T)
        var_ref = jnp.ones(n1_test * n2_test) - jnp.sum(
            K_cross_full * solve_cross.T,
            axis=1,
        )

        assert jnp.allclose(mean, mean_ref, atol=1e-4)
        assert jnp.allclose(var, var_ref, atol=1e-4)

    def test_shapes(self, getkey):
        """Output shapes should match test grid."""
        n1, n2 = 4, 3
        n1_test, n2_test = 2, 3
        N_test = n1_test * n2_test

        K1 = _make_pd(getkey(), n1)
        K2 = _make_pd(getkey(), n2)
        y = jax.random.normal(getkey(), (n1 * n2,))
        K_cross_1 = jax.random.normal(getkey(), (n1_test, n1))
        K_cross_2 = jax.random.normal(getkey(), (n2_test, n2))

        K1_op = lx.MatrixLinearOperator(K1, lx.positive_semidefinite_tag)
        K2_op = lx.MatrixLinearOperator(K2, lx.positive_semidefinite_tag)

        mean, var = kronecker_posterior_predictive(
            [K1_op, K2_op],
            y,
            0.1,
            (n1, n2),
            [K_cross_1, K_cross_2],
            K_test_diag_factors=[jnp.ones(n1_test), jnp.ones(n2_test)],
        )

        assert mean.shape == (N_test,)
        assert var.shape == (N_test,)

    def test_nonnegative_variance(self, getkey):
        """Predictive variances should be non-negative."""
        n1, n2 = 4, 3

        K1 = _make_pd(getkey(), n1)
        K2 = _make_pd(getkey(), n2)
        y = jax.random.normal(getkey(), (n1 * n2,))

        # Use submatrix of eigenvectors as cross-covariance for well-conditioned test
        Q1 = jnp.linalg.eigh(K1)[1][:, :2]
        Q2 = jnp.linalg.eigh(K2)[1][:, :2]

        K1_op = lx.MatrixLinearOperator(K1, lx.positive_semidefinite_tag)
        K2_op = lx.MatrixLinearOperator(K2, lx.positive_semidefinite_tag)

        _, var = kronecker_posterior_predictive(
            [K1_op, K2_op],
            y,
            0.1,
            (n1, n2),
            [Q1.T, Q2.T],
            K_test_diag_factors=[jnp.ones(Q1.shape[1]), jnp.ones(Q2.shape[1])],
        )
        assert jnp.all(var >= -1e-6)

    def test_requires_test_prior_diagonals(self, getkey):
        """Variance computation should require exact test prior diagonals."""
        n1, n2 = 2, 2
        K1 = _make_pd(getkey(), n1)
        K2 = _make_pd(getkey(), n2)
        y = jax.random.normal(getkey(), (n1 * n2,))
        K_cross_1 = jax.random.normal(getkey(), (1, n1))
        K_cross_2 = jax.random.normal(getkey(), (1, n2))

        K1_op = lx.MatrixLinearOperator(K1, lx.positive_semidefinite_tag)
        K2_op = lx.MatrixLinearOperator(K2, lx.positive_semidefinite_tag)

        try:
            kronecker_posterior_predictive(
                [K1_op, K2_op],
                y,
                0.1,
                (n1, n2),
                [K_cross_1, K_cross_2],
            )
        except TypeError:
            pass
        else:
            raise AssertionError("Expected missing K_test_diag_factors to fail")
