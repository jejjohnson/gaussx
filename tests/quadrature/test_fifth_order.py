"""Tests for the fifth-order fully symmetric cubature rule."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from gaussx import (
    FifthOrderCubatureIntegrator,
    GaussHermiteIntegrator,
    GaussianState,
    UnscentedIntegrator,
    fifth_order_cubature_points,
)
from gaussx._testing import random_pd_matrix, tree_allclose


def _standard_state(dim):
    """Standard normal state of the given dimension."""
    return GaussianState(
        mean=jnp.zeros(dim),
        cov=lx.MatrixLinearOperator(jnp.eye(dim), lx.positive_semidefinite_tag),
    )


@pytest.mark.parametrize("dim", [1, 2, 3, 5, 10])
def test_point_count_is_two_d_squared_plus_one(dim):
    """The rule uses exactly 2 D^2 + 1 points."""
    state = _standard_state(dim)
    points, weights = fifth_order_cubature_points(state.mean, state.cov)

    assert points.shape == (2 * dim**2 + 1, dim)
    assert weights.shape == (2 * dim**2 + 1,)


@pytest.mark.parametrize("dim", [1, 2, 3, 5, 10])
def test_weights_sum_to_one(dim):
    """A quadrature rule for a normalised measure has unit total weight."""
    state = _standard_state(dim)
    _, weights = fifth_order_cubature_points(state.mean, state.cov)

    assert tree_allclose(jnp.sum(weights), jnp.asarray(1.0), rtol=1e-12)


@pytest.mark.parametrize(
    ("power", "expected"),
    [(1, 0.0), (2, 1.0), (3, 0.0), (4, 3.0), (5, 0.0)],
)
def test_univariate_moments_are_exact(power, expected):
    """Exact through degree 5 for the standard normal moments."""
    state = _standard_state(1)
    points, weights = fifth_order_cubature_points(state.mean, state.cov)

    moment = jnp.sum(weights * points[:, 0] ** power)
    assert tree_allclose(moment, jnp.asarray(expected), atol=1e-12)


def test_multivariate_quartic_is_exact(getkey):
    """Exact for a degree-4 polynomial under a correlated Gaussian."""
    dim = 5
    cov_mat = random_pd_matrix(getkey(), dim)
    mean = jax.random.normal(getkey(), (dim,))
    cov = lx.MatrixLinearOperator(cov_mat, lx.positive_semidefinite_tag)

    points, weights = fifth_order_cubature_points(mean, cov)

    # E[(aᵀ(x − μ))⁴] = 3 (aᵀ Σ a)² for x ~ N(μ, Σ).
    a = jax.random.normal(getkey(), (dim,))
    centred = (points - mean) @ a
    quartic = jnp.sum(weights * centred**4)
    expected = 3.0 * (a @ cov_mat @ a) ** 2
    assert tree_allclose(quartic, expected, rtol=1e-10)

    # E[(x_i − μ_i)²(x_j − μ_j)²] = Σ_ii Σ_jj + 2 Σ_ij².
    d = points - mean
    cross = jnp.sum(weights * d[:, 0] ** 2 * d[:, 2] ** 2)
    expected_cross = cov_mat[0, 0] * cov_mat[2, 2] + 2 * cov_mat[0, 2] ** 2
    assert tree_allclose(cross, expected_cross, rtol=1e-10)


def test_integrator_beats_unscented_on_a_quartic():
    """The motivating case from the issue: E[Σ x_i⁴] under N(0, 0.5 I)."""
    dim = 10
    state = GaussianState(
        mean=jnp.zeros(dim),
        cov=lx.MatrixLinearOperator(0.5 * jnp.eye(dim), lx.positive_semidefinite_tag),
    )

    def quartic(x):
        return jnp.array([jnp.sum(x**4)])

    result = FifthOrderCubatureIntegrator().integrate(quartic, state)

    # E[Σ x_i⁴] = 3 σ⁴ D = 3 · 0.25 · 10.
    assert tree_allclose(result.state.mean[0], jnp.asarray(7.5), rtol=1e-10)


# Fixed covariance for the truncation-error tests. A degree-5 rule is not
# exact on a non-polynomial integrand, and the size of the residual depends
# on the spread of the state, so a random draw would make the tolerances
# below meaningless.
_SMOOTH_COV = jnp.array([[1.0, 0.25], [0.25, 0.8]])


def _smooth(x):
    return jnp.array([jnp.sin(x[0]) + x[1] ** 2, jnp.cos(x[0] * x[1])])


def _smooth_state(scale):
    return GaussianState(
        mean=jnp.zeros(2),
        cov=lx.MatrixLinearOperator(scale * _SMOOTH_COV, lx.positive_semidefinite_tag),
    )


def test_matches_gauss_hermite_on_a_smooth_function():
    """Tracks a high-order tensor-product rule on a smooth, non-polynomial
    integrand, and does so more closely than the degree-3 unscented rule.
    """
    state = _smooth_state(0.05)

    reference = GaussHermiteIntegrator(order=30).integrate(_smooth, state)
    cubature = FifthOrderCubatureIntegrator().integrate(_smooth, state)
    unscented = UnscentedIntegrator(alpha=1.0).integrate(_smooth, state)

    cubature_err = jnp.abs(cubature.state.mean - reference.state.mean).max()
    unscented_err = jnp.abs(unscented.state.mean - reference.state.mean).max()

    assert cubature_err < 1e-2
    assert cubature_err < unscented_err


def test_truncation_error_falls_at_the_expected_rate():
    """Degree-5 exactness means the leading error term scales as sigma^6."""

    def error(scale):
        state = _smooth_state(scale)
        reference = GaussHermiteIntegrator(order=30).integrate(_smooth, state)
        cubature = FifthOrderCubatureIntegrator().integrate(_smooth, state)
        return jnp.abs(cubature.state.mean - reference.state.mean).max()

    # Halving the covariance scales sigma by sqrt(2), so a sigma^6 error
    # term should shrink by roughly 2^3 = 8.
    ratio = error(0.04) / error(0.02)
    assert 4.0 < ratio < 16.0


def test_linear_function_recovers_exact_moments(getkey):
    """Linear maps are degree 1, so mean, covariance and cross are exact."""
    dim = 3
    cov_mat = random_pd_matrix(getkey(), dim)
    mean = jax.random.normal(getkey(), (dim,))
    state = GaussianState(
        mean=mean,
        cov=lx.MatrixLinearOperator(cov_mat, lx.positive_semidefinite_tag),
    )
    matrix = jax.random.normal(getkey(), (2, dim))

    result = FifthOrderCubatureIntegrator().integrate(lambda x: matrix @ x, state)

    assert tree_allclose(result.state.mean, matrix @ mean, rtol=1e-10)
    assert tree_allclose(
        result.state.cov.as_matrix(), matrix @ cov_mat @ matrix.T, rtol=1e-10
    )
    assert tree_allclose(result.cross_cov, cov_mat @ matrix.T, rtol=1e-10)


def test_points_and_weights_matches_the_rule():
    """The integrator exposes the same points its ``integrate`` uses."""
    state = _standard_state(3)
    points, w_m, w_c = FifthOrderCubatureIntegrator().points_and_weights(state)
    expected_points, expected_weights = fifth_order_cubature_points(
        state.mean, state.cov
    )

    assert tree_allclose(points, expected_points)
    assert tree_allclose(w_m, expected_weights)
    assert tree_allclose(w_c, expected_weights)


def test_jit_vmap_grad(getkey):
    """The rule is a plain JAX computation: traceable and differentiable."""
    dim = 2

    def integrate_quartic(mean):
        state = GaussianState(
            mean=mean,
            cov=lx.MatrixLinearOperator(jnp.eye(dim), lx.positive_semidefinite_tag),
        )
        result = FifthOrderCubatureIntegrator().integrate(
            lambda x: jnp.array([jnp.sum(x**4)]), state
        )
        return result.state.mean[0]

    mean = jax.random.normal(getkey(), (dim,))

    assert tree_allclose(jax.jit(integrate_quartic)(mean), integrate_quartic(mean))

    means = jax.random.normal(getkey(), (4, dim))
    assert jax.vmap(integrate_quartic)(means).shape == (4,)

    # E[Σ (μ_i + z_i)⁴] = Σ (μ_i⁴ + 6 μ_i² + 3), so d/dμ_i = 4 μ_i³ + 12 μ_i.
    grad = jax.grad(integrate_quartic)(mean)
    assert tree_allclose(grad, 4 * mean**3 + 12 * mean, rtol=1e-8)
