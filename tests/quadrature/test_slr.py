"""Tests for statistical linear regression."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from gaussx import (
    FifthOrderCubatureIntegrator,
    GaussHermiteIntegrator,
    GaussianState,
    TaylorIntegrator,
    UnscentedIntegrator,
    statistical_linear_regression,
)
from gaussx._testing import random_pd_matrix, tree_allclose


_POINT_RULES = [
    FifthOrderCubatureIntegrator(),
    GaussHermiteIntegrator(order=8),
    UnscentedIntegrator(alpha=1.0),
]


def _linear_problem(getkey, dim=4, out=3):
    """A linear-Gaussian conditional whose SLR surrogate is known exactly."""
    cov_mat = random_pd_matrix(getkey(), dim)
    state = GaussianState(
        mean=jax.random.normal(getkey(), (dim,)),
        cov=lx.MatrixLinearOperator(cov_mat, lx.positive_semidefinite_tag),
    )
    gain = jax.random.normal(getkey(), (out, dim))
    offset = jax.random.normal(getkey(), (out,))
    noise = random_pd_matrix(getkey(), out)
    return state, gain, offset, noise


@pytest.mark.parametrize("integrator", _POINT_RULES)
def test_linear_conditional_is_recovered_exactly(getkey, integrator):
    """For E[y|f] = H f + c and constant Var[y|f] = R, SLR returns (H, c, R)."""
    state, gain, offset, noise = _linear_problem(getkey)

    result = statistical_linear_regression(
        lambda f: gain @ f + offset,
        lambda f: noise,
        state,
        integrator,
    )

    assert tree_allclose(result.A, gain, atol=1e-10)
    assert tree_allclose(result.b, offset, atol=1e-10)
    assert tree_allclose(result.omega, noise, atol=1e-10)
    assert tree_allclose(result.mu, gain @ state.mean + offset, atol=1e-10)


def test_noiseless_linear_conditional_has_zero_residual(getkey):
    """With no conditional noise the surrogate is exact, so Ω vanishes."""
    state, gain, offset, _ = _linear_problem(getkey)
    out = gain.shape[0]

    result = statistical_linear_regression(
        lambda f: gain @ f + offset,
        lambda f: jnp.zeros((out, out)),
        state,
        FifthOrderCubatureIntegrator(),
    )

    assert tree_allclose(result.A, gain, atol=1e-10)
    assert tree_allclose(result.omega, jnp.zeros((out, out)), atol=1e-10)


def test_diagonal_conditional_variance_matches_full(getkey):
    """An (M,) conditional variance is read as the diagonal of an (M, M) one."""
    state, gain, offset, noise = _linear_problem(getkey)
    diag = jnp.diag(jnp.diag(noise))

    full = statistical_linear_regression(
        lambda f: gain @ f + offset,
        lambda f: diag,
        state,
        FifthOrderCubatureIntegrator(),
    )
    vector = statistical_linear_regression(
        lambda f: gain @ f + offset,
        lambda f: jnp.diag(diag),
        state,
        FifthOrderCubatureIntegrator(),
    )

    assert tree_allclose(full.omega, vector.omega, atol=1e-10)
    assert tree_allclose(full.A, vector.A, atol=1e-10)


def test_rejects_badly_shaped_conditional_variance(getkey):
    """A conditional variance that is neither (M,) nor (M, M) is an error."""
    state, gain, offset, _ = _linear_problem(getkey)

    with pytest.raises(ValueError, match="must return an"):
        statistical_linear_regression(
            lambda f: gain @ f + offset,
            lambda f: jnp.zeros((3, 3, 3)),
            state,
            FifthOrderCubatureIntegrator(),
        )


def test_rejects_non_point_based_integrator(getkey):
    """Taylor linearises rather than sampling, so it has no points."""
    state, gain, offset, noise = _linear_problem(getkey)

    with pytest.raises(NotImplementedError, match="not a point-based rule"):
        statistical_linear_regression(
            lambda f: gain @ f + offset,
            lambda f: noise,
            state,
            TaylorIntegrator(),
        )


def test_slr_plus_kalman_update_matches_exact_posterior(getkey):
    """SLR sites fed to a linear-Gaussian update reproduce the exact posterior."""
    state, gain, offset, noise = _linear_problem(getkey)
    cov_mat = state.cov.as_matrix()
    y = jax.random.normal(getkey(), (gain.shape[0],))

    slr = statistical_linear_regression(
        lambda f: gain @ f + offset,
        lambda f: noise,
        state,
        FifthOrderCubatureIntegrator(),
    )

    # Kalman update through the surrogate.
    innov_cov = slr.A @ cov_mat @ slr.A.T + slr.omega
    kalman = cov_mat @ slr.A.T @ jnp.linalg.inv(innov_cov)
    post_mean = state.mean + kalman @ (y - (slr.A @ state.mean + slr.b))
    post_cov = cov_mat - kalman @ innov_cov @ kalman.T

    # The exact linear-Gaussian posterior.
    exact_innov = gain @ cov_mat @ gain.T + noise
    exact_kalman = cov_mat @ gain.T @ jnp.linalg.inv(exact_innov)
    exact_mean = state.mean + exact_kalman @ (y - (gain @ state.mean + offset))
    exact_cov = cov_mat - exact_kalman @ exact_innov @ exact_kalman.T

    assert tree_allclose(post_mean, exact_mean, atol=1e-8)
    assert tree_allclose(post_cov, exact_cov, atol=1e-8)


def test_nonlinear_rules_agree():
    """On a mildly nonlinear conditional the rules give the same surrogate.

    The covariance is fixed rather than drawn: the residual here is
    quadrature truncation error, whose size depends on the spread of the
    state, so a random draw would make the tolerance meaningless.
    """
    cov = jnp.array(
        [
            [0.05, 0.01, 0.00],
            [0.01, 0.04, 0.01],
            [0.00, 0.01, 0.06],
        ]
    )
    state = GaussianState(
        mean=jnp.zeros(3),
        cov=lx.MatrixLinearOperator(cov, lx.positive_semidefinite_tag),
    )

    def cond_mean(f):
        return jax.nn.sigmoid(f)

    def cond_var(f):
        p = cond_mean(f)
        return p * (1 - p)

    reference = statistical_linear_regression(
        cond_mean, cond_var, state, GaussHermiteIntegrator(order=20)
    )
    cubature = statistical_linear_regression(
        cond_mean, cond_var, state, FifthOrderCubatureIntegrator()
    )

    # The degree-5 rule is not exact on a sigmoid, so what is being checked
    # is that both rules resolve the same surrogate to quadrature accuracy.
    assert tree_allclose(cubature.A, reference.A, atol=1e-3)
    assert tree_allclose(cubature.omega, reference.omega, atol=1e-4)


def test_jit_vmap_grad(getkey):
    """SLR is traceable and differentiable end to end."""
    dim, out = 3, 2
    cov_mat = random_pd_matrix(getkey(), dim)
    gain = jax.random.normal(getkey(), (out, dim))
    noise = random_pd_matrix(getkey(), out)

    def gain_trace(mean):
        state = GaussianState(
            mean=mean,
            cov=lx.MatrixLinearOperator(cov_mat, lx.positive_semidefinite_tag),
        )
        result = statistical_linear_regression(
            lambda f: jnp.tanh(gain @ f),
            lambda f: noise,
            state,
            FifthOrderCubatureIntegrator(),
        )
        return jnp.trace(result.A @ result.A.T)

    mean = jax.random.normal(getkey(), (dim,))

    assert tree_allclose(jax.jit(gain_trace)(mean), gain_trace(mean))

    means = jax.random.normal(getkey(), (3, dim))
    assert jax.vmap(gain_trace)(means).shape == (3,)

    grad = jax.grad(gain_trace)(mean)
    assert grad.shape == (dim,)
    assert bool(jnp.all(jnp.isfinite(grad)))
