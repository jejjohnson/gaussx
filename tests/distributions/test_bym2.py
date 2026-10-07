"""BYM2GMRF: the BYM2 pair with its exact constrained density (gh-508).

References are dense and in closed form: the density on the constraint
surface (Rue & Held, eq. 2.30), the joint covariance of ``(b, u*)`` and the
exact Gaussian marginal likelihood. Keys are pinned; the one sampling test
bounds the estimator by its own sampling distribution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import gaussx as gx
from gaussx._testing import assert_sample_moments


pytest.importorskip("numpyro")

THETAS = [(1.0, 0.5), (3.0, 0.2), (0.4, 0.9)]


def _graph():
    """Five areas: a path 0-1-2-3-4 plus a chord 0-2."""
    s, r = np.array([0, 1, 2, 3, 0]), np.array([1, 2, 3, 4, 2])
    n = 5
    W = np.zeros((n, n))
    W[s, r] = W[r, s] = 1.0
    L = np.diag(W.sum(1)) - W
    R = gx.SparseOperator.from_coo(
        np.r_[np.arange(n), r],
        np.r_[np.arange(n), s],
        jnp.asarray(np.r_[np.diag(L), -np.ones(len(s))]),
        (n, n),
        symmetric=True,
    )
    V = jnp.ones(n) / jnp.sqrt(n)
    R_star = gx.generalized_variance_scale(R, V) * R
    return R_star, V, n


def _dense(R_star, n, tau, phi):
    """Exact constrained covariance of (b, u*) and the log-density constant."""
    Rd = np.asarray(R_star.as_matrix())
    Rp = np.linalg.pinv(Rd, hermitian=True, rtol=None)
    cov_u = Rp
    cov_b = ((1 - phi) * np.eye(n) + phi * Rp) / tau
    cross = np.sqrt(phi / tau) * Rp
    cov = np.block([[cov_b, cross], [cross.T, cov_u]])
    ev = np.linalg.eigvalsh(Rd)
    half_pdet = 0.5 * np.sum(np.log(ev[ev > 1e-9]))
    return cov, half_pdet


def _on_surface(n, seed):
    x = np.random.default_rng(seed).normal(size=2 * n)
    x[n:] -= x[n:].mean()
    return x


@pytest.mark.parametrize(("tau", "phi"), THETAS)
def test_log_prob_is_the_constrained_density(tau, phi):
    R_star, V, n = _graph()
    Q = np.asarray(gx.bym2_precision(R_star, tau, phi).as_matrix())
    _, half_pdet = _dense(R_star, n, tau, phi)
    x = _on_surface(n, 0)
    reference = (
        -0.5 * x @ Q @ x
        + 0.5 * n * np.log(tau / (1 - phi))
        + half_pdet
        - 0.5 * (2 * n - 1) * np.log(2 * np.pi)
    )
    prior = gx.BYM2GMRF(R_star, tau, phi, V, include_normalizer=True)
    assert np.isclose(float(prior.log_prob(jnp.asarray(x))), reference, atol=1e-10)


def test_default_density_has_the_right_theta_dependence():
    # Without the constant, differences across theta must still be exact:
    # that is what a marginal likelihood or a hyperparameter posterior sees.
    R_star, V, n = _graph()
    x = jnp.asarray(_on_surface(n, 1))

    def default(theta):
        return float(gx.BYM2GMRF(R_star, *theta, V).log_prob(x))

    def exact(theta):
        return float(
            gx.BYM2GMRF(R_star, *theta, V, include_normalizer=True).log_prob(x)
        )

    for a, b in [(THETAS[0], THETAS[1]), (THETAS[1], THETAS[2])]:
        assert np.isclose(default(a) - default(b), exact(a) - exact(b), atol=1e-10)

    # IntrinsicGMRF's default drops a theta-dependent term for BYM2 (the bug).
    def intrinsic(theta):
        Q = gx.bym2_precision(R_star, *theta)
        padded = jnp.concatenate([jnp.zeros((n, 1)), V[:, None]])
        return float(gx.IntrinsicGMRF(jnp.zeros(2 * n), 1.0, Q, padded).log_prob(x))

    a, b = THETAS[0], THETAS[1]
    assert not np.isclose(intrinsic(a) - intrinsic(b), exact(a) - exact(b), atol=1e-3)


def test_log_pdet_override_matches_the_matrix_tree_value():
    R_star, V, n = _graph()
    x = jnp.asarray(_on_surface(n, 2))
    _, half_pdet = _dense(R_star, n, 1.0, 0.5)
    computed = gx.BYM2GMRF(R_star, 1.0, 0.5, V, include_normalizer=True)
    given = gx.BYM2GMRF(
        R_star, 1.0, 0.5, V, include_normalizer=True, log_pdet=2.0 * half_pdet
    )
    assert np.isclose(computed.log_prob(x), given.log_prob(x), atol=1e-10)


@pytest.mark.parametrize(("tau", "phi"), THETAS)
def test_marginal_variances_are_exact(tau, phi):
    R_star, V, n = _graph()
    cov, _ = _dense(R_star, n, tau, phi)
    variances = gx.BYM2GMRF(R_star, tau, phi, V).marginal_variances()
    assert np.allclose(variances, np.diag(cov), rtol=1e-6)


@pytest.mark.slow
def test_samples_satisfy_the_constraint_and_the_covariance():
    R_star, V, n = _graph()
    tau, phi = 2.0, 0.6
    cov, _ = _dense(R_star, n, tau, phi)
    x = gx.BYM2GMRF(R_star, tau, phi, V).sample(jax.random.PRNGKey(0), (20_000,))
    assert x.shape == (20_000, 2 * n)
    assert np.allclose(np.asarray(x[:, n:]).sum(axis=1), 0.0, atol=1e-8)
    # 7 sigma of each moment estimator's own sampling distribution.
    assert_sample_moments(x, jnp.zeros(2 * n), jnp.asarray(cov))


@pytest.mark.slow
def test_gradients_in_theta_match_finite_differences():
    R_star, V, n = _graph()
    x = jnp.asarray(_on_surface(n, 3))

    def lp(log_tau, logit_phi):
        tau, phi = jnp.exp(log_tau), jax.nn.sigmoid(logit_phi)
        return gx.BYM2GMRF(R_star, tau, phi, V, include_normalizer=True).log_prob(x)

    g = jax.grad(lp, argnums=(0, 1))(0.3, -0.2)
    eps = 1e-6
    fd_tau = (lp(0.3 + eps, -0.2) - lp(0.3 - eps, -0.2)) / (2 * eps)
    fd_phi = (lp(0.3, -0.2 + eps) - lp(0.3, -0.2 - eps)) / (2 * eps)
    assert np.isclose(g[0], fd_tau, rtol=1e-6)
    assert np.isclose(g[1], fd_phi, rtol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize(("tau", "phi"), THETAS)
def test_laplace_log_marginal_is_the_exact_gaussian_marginal(tau, phi):
    # y = b + noise: Laplace is exact, and p(y | theta) is N(0, cov_b + s2 I).
    R_star, V, n = _graph()
    noise_var = 0.3
    y = jnp.asarray(np.random.default_rng(4).normal(size=n))
    cov, _ = _dense(R_star, n, tau, phi)
    marginal = cov[:n, :n] + noise_var * np.eye(n)
    _, logdet = np.linalg.slogdet(marginal)
    reference = -0.5 * (
        n * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(marginal, y)
    )
    select_b = gx.SparseOperator.from_coo(
        np.arange(n), np.arange(n), jnp.ones(n), (n, 2 * n)
    )
    prior = gx.BYM2GMRF(R_star, tau, phi, V, include_normalizer=True)
    result = gx.laplace_mode(
        prior, gx.GaussianLikelihood(y, noise_var), projector=select_b
    )
    assert np.isclose(float(result.log_marginal), reference, atol=1e-8)


def test_validation():
    R_star, _, n = _graph()
    with pytest.raises(ValueError, match="rows"):
        gx.BYM2GMRF(R_star, 1.0, 0.5, jnp.ones(n + 1))
    assert isinstance(gx.BYM2GMRF(R_star, 1.0, 0.5, jnp.ones(n)), gx.IntrinsicGMRF)
    assert isinstance(
        gx.BYM2GMRF(R_star, 1.0, 0.5, jnp.ones(n)).structure, lx.AbstractLinearOperator
    )
