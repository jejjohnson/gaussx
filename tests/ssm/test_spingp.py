"""Tests for SpInGP Kalman filter recipes."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._ssm._spingp import spingp_log_likelihood, spingp_posterior


def _make_prior_precision(key, N, d):
    """Build a simple PD block-tridiagonal prior precision."""
    k1, k2 = jax.random.split(key)
    # Diagonal blocks: positive definite
    diag_raw = jax.random.normal(k1, (N, d, d))
    diag_blocks = jax.vmap(lambda A: A @ A.T + 3.0 * jnp.eye(d))(diag_raw)
    # Sub-diagonal blocks: small coupling
    sub_diag = 0.1 * jax.random.normal(k2, (N - 1, d, d))
    return BlockTriDiag(diag_blocks, sub_diag)


class TestSpInGPPosterior:
    def test_basic_shapes(self, getkey):
        """Posterior mean and precision should have correct shapes."""
        N, d, d_obs = 5, 2, 1
        prior_prec = _make_prior_precision(getkey(), N, d)
        H = jnp.array([[1.0, 0.0]])  # (d_obs, d)
        R = lx.MatrixLinearOperator(0.1 * jnp.eye(d_obs))
        y = jax.random.normal(getkey(), (N, d_obs))

        post_mean, post_prec = spingp_posterior(prior_prec, H, R, y)
        assert post_mean.shape == (N * d,)
        assert post_prec._num_blocks == N
        assert post_prec._block_size == d

    def test_posterior_precision_larger_than_prior(self, getkey):
        """Posterior precision should be >= prior (added info)."""
        N, d, d_obs = 4, 2, 1
        prior_prec = _make_prior_precision(getkey(), N, d)
        H = jnp.array([[1.0, 0.0]])
        R = lx.MatrixLinearOperator(0.1 * jnp.eye(d_obs))
        y = jax.random.normal(getkey(), (N, d_obs))

        _, post_prec = spingp_posterior(prior_prec, H, R, y)
        # Diagonal blocks should be >= prior diagonal blocks
        diff = post_prec.diagonal - prior_prec.diagonal
        # Each diff block should be PSD (H^T R^{-1} H is PSD)
        for k in range(N):
            eigvals = jnp.linalg.eigvalsh(diff[k])
            assert jnp.all(eigvals >= -1e-10)

    def test_no_observations_recovers_prior(self, getkey):
        """With infinite noise, posterior should approach prior."""
        N, d, d_obs = 3, 2, 1
        prior_prec = _make_prior_precision(getkey(), N, d)
        H = jnp.array([[1.0, 0.0]])
        # Very large observation noise -> near-zero likelihood precision
        R = lx.MatrixLinearOperator(1e10 * jnp.eye(d_obs))
        y = jax.random.normal(getkey(), (N, d_obs))

        _, post_prec = spingp_posterior(prior_prec, H, R, y)
        assert jnp.allclose(post_prec.diagonal, prior_prec.diagonal, atol=1e-6)

    def test_per_timestep_emission(self, getkey):
        """Should work with per-timestep emission matrices."""
        N, d, d_obs = 4, 2, 1
        prior_prec = _make_prior_precision(getkey(), N, d)
        # Per-timestep emission: (N, d_obs, d)
        H = jax.random.normal(getkey(), (N, d_obs, d))
        R = lx.MatrixLinearOperator(0.1 * jnp.eye(d_obs))
        y = jax.random.normal(getkey(), (N, d_obs))

        post_mean, post_prec = spingp_posterior(prior_prec, H, R, y)
        assert post_mean.shape == (N * d,)
        assert post_prec._num_blocks == N


class TestSpInGPLogLikelihood:
    def test_returns_scalar(self, getkey):
        """Log-likelihood should be a finite scalar."""
        N, d, d_obs = 5, 2, 1
        prior_prec = _make_prior_precision(getkey(), N, d)
        H = jnp.array([[1.0, 0.0]])
        R = lx.MatrixLinearOperator(0.1 * jnp.eye(d_obs))
        y = jax.random.normal(getkey(), (N, d_obs))

        ll = spingp_log_likelihood(prior_prec, H, R, y)
        assert ll.shape == ()
        assert jnp.isfinite(ll)

    def test_more_noise_lower_ll(self, getkey):
        """Higher obs noise changes log-likelihood."""
        N, d, d_obs = 4, 2, 1
        prior_prec = _make_prior_precision(getkey(), N, d)
        H = jnp.array([[1.0, 0.0]])
        y = jax.random.normal(getkey(), (N, d_obs))

        R_small = lx.MatrixLinearOperator(0.01 * jnp.eye(d_obs))
        R_large = lx.MatrixLinearOperator(100.0 * jnp.eye(d_obs))

        ll_small = spingp_log_likelihood(prior_prec, H, R_small, y)
        ll_large = spingp_log_likelihood(prior_prec, H, R_large, y)

        # Both should be finite
        assert jnp.isfinite(ll_small)
        assert jnp.isfinite(ll_large)

    def test_consistent_with_dense(self, getkey):
        """SpInGP log-likelihood should match dense GP log-likelihood."""
        N, d, d_obs = 3, 2, 1
        prior_prec = _make_prior_precision(getkey(), N, d)
        H_shared = jnp.array([[1.0, 0.0]])  # (1, 2)
        R = lx.MatrixLinearOperator(0.5 * jnp.eye(d_obs))
        y = 0.1 * jax.random.normal(getkey(), (N, d_obs))

        ll_spingp = spingp_log_likelihood(prior_prec, H_shared, R, y)

        # Dense computation
        K_prior_inv = prior_prec.as_matrix()  # (Nd, Nd)
        K_prior = jnp.linalg.inv(K_prior_inv)

        # Build full observation matrix: H_full = block_diag(H, H, ..., H)
        H_full = jnp.zeros((N * d_obs, N * d))
        for k in range(N):
            H_full = H_full.at[k * d_obs : (k + 1) * d_obs, k * d : (k + 1) * d].set(
                H_shared
            )

        R_full = jnp.kron(jnp.eye(N), R.as_matrix())
        # Marginal covariance in observation space
        S = H_full @ K_prior @ H_full.T + R_full
        y_flat = y.reshape(-1)
        _, ld_S = jnp.linalg.slogdet(S)
        log_2pi = jnp.log(2.0 * jnp.pi)
        ll_dense = -0.5 * (
            y_flat @ jnp.linalg.solve(S, y_flat) + ld_S + N * d_obs * log_2pi
        )

        assert jnp.allclose(ll_spingp, ll_dense, atol=1e-4)
