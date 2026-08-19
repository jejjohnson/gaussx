"""Tests for the third-order spherical-radial cubature integrator."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from gaussx import (
    CubatureIntegrator,
    GaussianState,
    UnscentedIntegrator,
    cubature_points,
)
from gaussx._testing import random_pd_matrix, tree_allclose


@pytest.mark.parametrize("dim", [1, 2, 5])
def test_point_count_and_weights(dim):
    """The rule uses 2N equally weighted points."""
    state = GaussianState(
        mean=jnp.zeros(dim),
        cov=lx.MatrixLinearOperator(jnp.eye(dim), lx.positive_semidefinite_tag),
    )
    points, w_m, w_c = CubatureIntegrator().points_and_weights(state)

    assert points.shape == (2 * dim, dim)
    assert tree_allclose(jnp.sum(w_m), jnp.asarray(1.0), rtol=1e-12)
    assert tree_allclose(w_m, w_c)
    # Every weight is positive, unlike the scaled unscented transform.
    assert bool(jnp.all(w_m > 0))


@pytest.mark.parametrize(("power", "expected"), [(1, 0.0), (2, 1.0), (3, 0.0)])
def test_exact_through_degree_three(power, expected):
    """Exact for the standard normal moments up to degree 3."""
    state = GaussianState(
        mean=jnp.zeros(1),
        cov=lx.MatrixLinearOperator(jnp.eye(1), lx.positive_semidefinite_tag),
    )
    points, weights = cubature_points(state.mean, state.cov)

    moment = jnp.sum(weights * points[:, 0] ** power)
    assert tree_allclose(moment, jnp.asarray(expected), atol=1e-12)


def test_not_exact_at_degree_four():
    """Degree 3 means the fourth moment is missed -- that is the CKF's limit."""
    state = GaussianState(
        mean=jnp.zeros(1),
        cov=lx.MatrixLinearOperator(jnp.eye(1), lx.positive_semidefinite_tag),
    )
    points, weights = cubature_points(state.mean, state.cov)

    fourth = jnp.sum(weights * points[:, 0] ** 4)
    assert not bool(jnp.allclose(fourth, 3.0, atol=1e-6))


def test_linear_function_is_exact(getkey):
    """Affine maps are degree 1, so all three moments are exact."""
    dim = 3
    cov_mat = random_pd_matrix(getkey(), dim)
    mean = jax.random.normal(getkey(), (dim,))
    state = GaussianState(
        mean=mean, cov=lx.MatrixLinearOperator(cov_mat, lx.positive_semidefinite_tag)
    )
    matrix = jax.random.normal(getkey(), (2, dim))

    result = CubatureIntegrator().integrate(lambda x: matrix @ x, state)

    assert tree_allclose(result.state.mean, matrix @ mean, rtol=1e-10)
    assert tree_allclose(
        result.state.cov.as_matrix(), matrix @ cov_mat @ matrix.T, rtol=1e-10
    )
    assert tree_allclose(result.cross_cov, cov_mat @ matrix.T, rtol=1e-10)


def test_matches_unscented_without_a_centre_point(getkey):
    """The 2N non-centre points coincide with the unscented ones at alpha=1.

    ``UnscentedIntegrator(alpha=1, beta=0, kappa=0)`` puts zero weight on
    its centre point and scales the rest identically, so the two rules must
    agree exactly.
    """
    dim = 3
    cov_mat = random_pd_matrix(getkey(), dim)
    state = GaussianState(
        mean=jax.random.normal(getkey(), (dim,)),
        cov=lx.MatrixLinearOperator(cov_mat, lx.positive_semidefinite_tag),
    )

    def quadratic(x):
        return jnp.array([jnp.sum(x**2), x[0] * x[1]])

    cubature = CubatureIntegrator().integrate(quadratic, state)
    unscented = UnscentedIntegrator(alpha=1.0, beta=0.0, kappa=0.0).integrate(
        quadratic, state
    )

    assert tree_allclose(cubature.state.mean, unscented.state.mean, atol=1e-10)
    assert tree_allclose(cubature.cross_cov, unscented.cross_cov, atol=1e-10)


def test_jit_vmap_grad(getkey):
    """Traceable and differentiable."""
    dim = 2

    def integrate_quadratic(mean):
        state = GaussianState(
            mean=mean,
            cov=lx.MatrixLinearOperator(jnp.eye(dim), lx.positive_semidefinite_tag),
        )
        result = CubatureIntegrator().integrate(
            lambda x: jnp.array([jnp.sum(x**2)]), state
        )
        return result.state.mean[0]

    mean = jax.random.normal(getkey(), (dim,))

    assert tree_allclose(jax.jit(integrate_quadratic)(mean), integrate_quadratic(mean))
    assert jax.vmap(integrate_quadratic)(
        jax.random.normal(getkey(), (4, dim))
    ).shape == (4,)

    # E[|mu + z|^2] = |mu|^2 + D, so the gradient is 2 mu.
    assert tree_allclose(jax.grad(integrate_quadratic)(mean), 2 * mean, rtol=1e-8)
