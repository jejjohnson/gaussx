"""Tests for the collapsed ELBO (Titsias bound)."""

import jax
import jax.numpy as jnp

from gaussx import collapsed_elbo


def _exact_mll(K, y, noise_var):
    """Compute exact log marginal likelihood: log N(y | 0, K + sigma^2 I)."""
    N = y.shape[0]
    Ky = K + noise_var * jnp.eye(N)
    L = jnp.linalg.cholesky(Ky)
    alpha = jnp.linalg.solve(Ky, y)
    log_2pi = jnp.log(2.0 * jnp.pi)
    return -0.5 * (y @ alpha + 2.0 * jnp.sum(jnp.log(jnp.diag(L))) + N * log_2pi)


class TestCollapsedELBO:
    def test_m_equals_n_recovers_mll(self, getkey):
        """When M=N (all points are inducing), ELBO equals exact MLL."""
        N = 10
        X = jax.random.normal(getkey(), (N, 2))
        # RBF kernel
        dists = jnp.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
        K = jnp.exp(-0.5 * dists)
        noise_var = 0.1
        y = jax.random.normal(getkey(), (N,))

        K_diag = jnp.diag(K)
        K_xz = K  # M=N, all points are inducing
        K_zz = K

        elbo_val = collapsed_elbo(y, K_diag, K_xz, K_zz, noise_var)
        mll_val = _exact_mll(K, y, noise_var)

        assert jnp.allclose(elbo_val, mll_val, atol=1e-4)

    def test_elbo_leq_mll(self, getkey):
        """ELBO is a lower bound on the MLL."""
        N, M = 30, 10
        X = jax.random.normal(getkey(), (N, 2))
        Z = X[:M]  # subset of data as inducing points
        noise_var = 0.1

        dists_ff = jnp.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
        K_ff = jnp.exp(-0.5 * dists_ff)
        dists_xz = jnp.sum((X[:, None] - Z[None, :]) ** 2, axis=-1)
        K_xz = jnp.exp(-0.5 * dists_xz)
        dists_zz = jnp.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
        K_zz = jnp.exp(-0.5 * dists_zz)

        y = jax.random.normal(getkey(), (N,))
        K_diag = jnp.diag(K_ff)

        elbo_val = collapsed_elbo(y, K_diag, K_xz, K_zz, noise_var)
        mll_val = _exact_mll(K_ff, y, noise_var)

        assert elbo_val <= mll_val + 1e-5

    def test_trace_penalty_nonnegative(self, getkey):
        """The trace penalty is nonnegative (it only reduces the ELBO)."""
        N, M = 20, 5
        X = jax.random.normal(getkey(), (N, 2))
        Z = X[:M]
        dists_xz = jnp.sum((X[:, None] - Z[None, :]) ** 2, axis=-1)
        K_xz = jnp.exp(-0.5 * dists_xz)
        dists_zz = jnp.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
        K_zz = jnp.exp(-0.5 * dists_zz)

        L_zz = jnp.linalg.cholesky(K_zz)
        V = jax.scipy.linalg.solve_triangular(L_zz, K_xz.T, lower=True)

        dists_ff = jnp.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
        K_diag = jnp.diag(jnp.exp(-0.5 * dists_ff))

        trace_diff = jnp.sum(K_diag) - jnp.sum(V**2)
        assert trace_diff >= -1e-6

    def test_jit_compatible(self, getkey):
        """Works under jax.jit."""
        N, M = 15, 5
        noise_var = 0.1
        y = jax.random.normal(getkey(), (N,))
        K_diag = jnp.ones(N)
        K_xz = jax.random.normal(getkey(), (N, M)) * 0.3
        K_zz = jnp.eye(M)

        val = jax.jit(collapsed_elbo)(y, K_diag, K_xz, K_zz, noise_var)
        assert jnp.isfinite(val)

    def test_increasing_m_tightens_bound(self, getkey):
        """More inducing points yields a tighter (higher) ELBO."""
        N = 30
        X = jax.random.normal(getkey(), (N, 2))
        noise_var = 0.1
        y = jax.random.normal(getkey(), (N,))

        dists_ff = jnp.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
        K_ff = jnp.exp(-0.5 * dists_ff)
        K_diag = jnp.diag(K_ff)

        elbos = []
        for M in [5, 10, 20]:
            Z = X[:M]
            dists_xz = jnp.sum((X[:, None] - Z[None, :]) ** 2, axis=-1)
            K_xz = jnp.exp(-0.5 * dists_xz)
            dists_zz = jnp.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
            K_zz = jnp.exp(-0.5 * dists_zz)
            elbos.append(collapsed_elbo(y, K_diag, K_xz, K_zz, noise_var))

        # ELBO should increase (or stay same) with more inducing points
        assert elbos[1] >= elbos[0] - 1e-4
        assert elbos[2] >= elbos[1] - 1e-4
