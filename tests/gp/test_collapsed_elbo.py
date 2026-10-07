"""Tests for the collapsed ELBO (Titsias bound)."""

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gaussx import collapsed_elbo
from gaussx._testing import psd_operator, random_kronecker_pd


def _exact_mll(K, y, noise_var):
    """Compute exact log marginal likelihood: log N(y | 0, K + sigma^2 I)."""
    N = y.shape[0]
    Ky = K + noise_var * jnp.eye(N)
    L = jnp.linalg.cholesky(Ky)
    alpha = jnp.linalg.solve(Ky, y)
    log_2pi = jnp.log(2.0 * jnp.pi)
    return -0.5 * (y @ alpha + 2.0 * jnp.sum(jnp.log(jnp.diag(L))) + N * log_2pi)


class TestCollapsedELBO:
    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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


# gh-353: K_xx_diag name, operator K_zz, deprecated solver=.


def _small_problem():
    x = jnp.linspace(0.0, 5.0, 12)
    z = jnp.linspace(0.5, 4.5, 4)

    def k(a, b):
        return jnp.exp(-0.5 * einx.subtract("i, j -> i j", a, b) ** 2)

    return jnp.sin(x), jnp.ones(12), k(x, z), k(z, z)


def test_k_diag_keyword_is_deprecated():
    y, K_xx_diag, K_xz, K_zz = _small_problem()
    new = collapsed_elbo(y, K_xx_diag=K_xx_diag, K_xz=K_xz, K_zz=K_zz, noise_var=0.3)
    with pytest.warns(DeprecationWarning, match="K_xx_diag"):
        old = collapsed_elbo(y, K_diag=K_xx_diag, K_xz=K_xz, K_zz=K_zz, noise_var=0.3)
    assert old == new
    with pytest.raises(TypeError, match="both"):
        collapsed_elbo(
            y,
            K_diag=K_xx_diag,
            K_xx_diag=K_xx_diag,
            K_xz=K_xz,
            K_zz=K_zz,
            noise_var=0.3,
        )


def test_solver_is_deprecated():
    from gaussx import DenseSolver

    y, K_xx_diag, K_xz, K_zz = _small_problem()
    with pytest.warns(DeprecationWarning, match="solver"):
        collapsed_elbo(y, K_xx_diag, K_xz, K_zz, 0.3, solver=DenseSolver())


def test_operator_k_zz_matches_array():
    y, K_xx_diag, K_xz, K_zz = _small_problem()
    ref = collapsed_elbo(y, K_xx_diag, K_xz, K_zz, 0.3)
    assert jnp.allclose(
        collapsed_elbo(y, K_xx_diag, K_xz, psd_operator(K_zz), 0.3), ref
    )


def test_kronecker_k_zz_is_factorised_per_factor():
    """Only B = I + V Vᵀ / σ² is factorised at (16, 16); K_zz per factor."""
    K_zz = random_kronecker_pd(jr.key(0), (4, 4), jitter=1.0)
    K_xz = 0.3 * jr.normal(jr.key(1), (20, 16))
    y = jr.normal(jr.key(2), (20,))
    K_xx_diag = 4.0 * jnp.ones(20)
    structured = collapsed_elbo(y, K_xx_diag, K_xz, K_zz, 0.3, jitter=0.0)
    dense = collapsed_elbo(y, K_xx_diag, K_xz, K_zz.as_matrix(), 0.3, jitter=0.0)
    assert jnp.allclose(structured, dense, rtol=1e-10)
    jaxpr = str(
        jax.make_jaxpr(lambda yy: collapsed_elbo(yy, K_xx_diag, K_xz, K_zz, 0.3))(y)
    )
    chol = [ln for ln in jaxpr.splitlines() if "= cholesky" in ln]
    assert sum("[16,16]" in ln for ln in chol) == 1
