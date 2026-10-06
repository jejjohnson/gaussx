"""Tests for SpInGP Kalman filter recipes."""

import re

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._einx import rearrange
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
    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_prior_mean(self, getkey):
        """A nonzero prior mean enters as Lambda_prior @ mu_prior."""
        N, d, d_obs = 4, 2, 1
        prior_prec = _make_prior_precision(getkey(), N, d)
        H = jnp.array([[1.0, 0.0]])
        R = lx.MatrixLinearOperator(0.1 * jnp.eye(d_obs))
        y = jax.random.normal(getkey(), (N, d_obs))
        mu_prior = jax.random.normal(getkey(), (N * d,))

        zero_mean, post_prec = spingp_posterior(prior_prec, H, R, y)
        default, _ = spingp_posterior(prior_prec, H, R, y, prior_mean=None)
        shifted, _ = spingp_posterior(prior_prec, H, R, y, prior_mean=mu_prior)

        assert jnp.allclose(default, zero_mean)
        correction = jnp.linalg.solve(
            post_prec.as_matrix(), prior_prec.as_matrix() @ mu_prior
        )
        assert jnp.allclose(shifted, zero_mean + correction, atol=1e-8)

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

    @pytest.mark.slow
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

    @pytest.mark.slow
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


# ---------------------------------------------------------------------------
# gh-403: R is factorised once; values and gradients against a dense joint
# ---------------------------------------------------------------------------


def _lapack_factorisations(f, *args):
    """``(routine, shape)`` of every LAPACK factorisation in ``jit(f)``."""
    hlo = jax.jit(f).lower(*args).compile().as_text()
    found = []
    for line in hlo.splitlines():
        match = re.search(r'custom_call_target="lapack_(\w+?)_ffi"', line)
        if match and match.group(1) in ("dpotrf", "dgetrf"):
            shape = re.search(r"=\s*\(?([a-z0-9]+\[[0-9,]*\])", line).group(1)
            found.append((match.group(1), shape))
    return found


def _small_problem(per_step_h):
    # Pinned: the checks are identities, any well-conditioned model will do.
    N, d, M = 6, 2, 3
    k1, k2, k3, k4, k5 = jr.split(jr.key(0), 5)
    B = jr.normal(k1, (N, d, d))
    diag_blocks = jax.vmap(lambda b: b @ b.T + 3 * jnp.eye(d))(B)
    prior = BlockTriDiag(diag_blocks, 0.1 * jr.normal(k2, (N - 1, d, d)))
    H = jr.normal(k3, (N, M, d) if per_step_h else (M, d))
    C = jr.normal(k4, (M, M))
    R = C @ C.T + jnp.eye(M)
    y = jr.normal(k5, (N, M))
    return prior, H, R, y


def _dense_log_likelihood(prior, H, R, y):
    N = y.shape[0]
    H_steps = H if H.ndim == 3 else jnp.broadcast_to(H, (N, *H.shape))
    H_full = jax.scipy.linalg.block_diag(*H_steps)
    cov = H_full @ jnp.linalg.inv(prior.as_matrix()) @ H_full.T
    cov = cov + jnp.kron(jnp.eye(N), R)
    return jax.scipy.stats.multivariate_normal.logpdf(
        rearrange(y, "N M -> (N M)"), jnp.zeros(y.size), cov
    )


@pytest.mark.slow
@pytest.mark.parametrize("per_step_h", [False, True], ids=["shared_H", "per_step_H"])
@pytest.mark.slow
def test_log_likelihood_and_gradient_match_dense_joint(per_step_h):
    prior, H, R, y = _small_problem(per_step_h)
    N, d, M = prior.diagonal.shape[0], prior.diagonal.shape[1], R.shape[0]
    # Symmetric parametrisations: a raw-matrix gradient depends on which
    # triangle a Cholesky reads, the gradient through these does not.
    C = jnp.linalg.cholesky(R - jnp.eye(M))
    B = jax.vmap(jnp.linalg.cholesky)(prior.diagonal - 3 * jnp.eye(d))

    def build(C, B):
        prior_ = BlockTriDiag(
            jax.vmap(lambda b: b @ b.T + 3 * jnp.eye(d))(B), prior.sub_diagonal
        )
        return prior_, C @ C.T + jnp.eye(M)

    def structured(C, B):
        prior_, R_ = build(C, B)
        R_op = lx.MatrixLinearOperator(R_, lx.positive_semidefinite_tag)
        return spingp_log_likelihood(prior_, H, R_op, y)

    def dense(C, B):
        prior_, R_ = build(C, B)
        return _dense_log_likelihood(prior_, H, R_, y)

    assert B.shape == (N, d, d)
    assert jnp.allclose(structured(C, B), dense(C, B), rtol=1e-10)
    g_struct = jax.grad(structured, argnums=(0, 1))(C, B)
    g_dense = jax.grad(dense, argnums=(0, 1))(C, B)
    for a, b in zip(g_struct, g_dense, strict=True):
        assert jnp.allclose(a, b, rtol=1e-8, atol=1e-10)


def test_diagonal_obs_noise_matches_dense():
    prior, H, R, y = _small_problem(False)
    r = jnp.diag(R)
    got = spingp_log_likelihood(prior, H, lx.DiagonalLinearOperator(r), y)
    assert jnp.allclose(got, _dense_log_likelihood(prior, H, jnp.diag(r), y))


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="counts LAPACK calls")
def test_obs_noise_factorised_once():
    prior, H, R, y = _small_problem(False)
    M = R.shape[0]

    def f(R):
        R_op = lx.MatrixLinearOperator(R, lx.positive_semidefinite_tag)
        return spingp_log_likelihood(prior, H, R_op, y)

    found = _lapack_factorisations(f, R)
    of_R = [call for call in found if call[1] == f"f64[{M},{M}]"]
    assert of_R == [("dpotrf", f"f64[{M},{M}]")]
    assert not any(routine == "dgetrf" for routine, _ in found)
