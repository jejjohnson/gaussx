"""Tests for the diagonal fast paths of cavity_distribution / newton_update."""

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import cavity_distribution, newton_update
from gaussx._testing import random_pd_matrix, tree_allclose


def _site(getkey, n=6):
    """Posterior marginals and site naturals for ``n`` scalar latents."""
    post_mean = jr.normal(getkey(), (n,))
    post_var = jnp.exp(jr.normal(getkey(), (n,)))  # positive
    site_nat1 = jr.normal(getkey(), (n,))
    # Keep the cavity precision positive: site precision below 1 / post_var.
    site_nat2 = 0.4 / post_var
    return post_mean, post_var, site_nat1, site_nat2


@pytest.mark.parametrize("power", [1.0, 0.5, 0.1])
def test_cavity_diagonal_matches_operator_path(getkey, power):
    """The elementwise path agrees with the operator path it replaces."""
    post_mean, post_var, site_nat1, site_nat2 = _site(getkey)

    diag_mean, diag_var = cavity_distribution(
        post_mean, post_var, site_nat1, site_nat2, power=power
    )
    op_mean, op_cov = cavity_distribution(
        post_mean,
        lx.DiagonalLinearOperator(post_var),
        site_nat1,
        lx.DiagonalLinearOperator(site_nat2),
        power=power,
    )

    assert tree_allclose(diag_mean, op_mean, atol=1e-10)
    assert tree_allclose(diag_var, jnp.diag(op_cov.as_matrix()), atol=1e-10)


def test_cavity_diagonal_satisfies_the_precision_identity(getkey):
    """post_prec = cav_prec + power * site_prec, elementwise."""
    post_mean, post_var, site_nat1, site_nat2 = _site(getkey)
    power = 0.5

    _, cav_var = cavity_distribution(
        post_mean, post_var, site_nat1, site_nat2, power=power
    )

    assert tree_allclose(1.0 / post_var, 1.0 / cav_var + power * site_nat2, atol=1e-10)


def test_cavity_rejects_mixed_argument_types(getkey):
    """An array variance with an operator site is a programming error."""
    post_mean, post_var, site_nat1, site_nat2 = _site(getkey)

    with pytest.raises(TypeError, match="both be arrays or both be operators"):
        cavity_distribution(
            post_mean, post_var, site_nat1, lx.DiagonalLinearOperator(site_nat2)
        )

    with pytest.raises(TypeError, match="both be arrays or both be operators"):
        cavity_distribution(
            post_mean, lx.DiagonalLinearOperator(post_var), site_nat1, site_nat2
        )


def test_newton_diagonal_matches_dense_path(getkey):
    """For a log-concave site the floor is inactive and the paths agree."""
    n = 5
    mean = jr.normal(getkey(), (n,))
    grad = jr.normal(getkey(), (n,))
    hess_diag = -jnp.exp(jr.normal(getkey(), (n,)))  # strictly negative

    diag_nat1, diag_nat2 = newton_update(mean, grad, hess_diag)
    dense_nat1, dense_nat2 = newton_update(mean, grad, jnp.diag(hess_diag))

    assert tree_allclose(diag_nat1, dense_nat1, atol=1e-10)
    assert tree_allclose(diag_nat2, jnp.diag(dense_nat2), atol=1e-10)


def test_newton_precision_floor_engages(getkey):
    """A non-log-concave site is clipped to the floor rather than going negative."""
    mean = jnp.array([0.5, -1.0, 0.25, 2.0])
    grad = jnp.array([0.1, -0.3, 0.7, 0.0])
    hess_diag = jnp.array([-2.0, 0.5, 0.0, 3.0])  # last three are not log-concave
    floor = 1e-3

    nat1, nat2 = newton_update(mean, grad, hess_diag, precision_floor=floor)

    assert tree_allclose(nat2, jnp.maximum(-hess_diag, floor), atol=1e-12)
    assert bool(jnp.all(nat2 > 0))
    # nat1 is built from the floored precision, not the raw Hessian.
    assert tree_allclose(nat1, grad + nat2 * mean, atol=1e-12)


def test_newton_dense_path_ignores_the_floor(getkey):
    """The full-matrix path returns -hessian unmodified, as documented."""
    n = 4
    mean = jr.normal(getkey(), (n,))
    grad = jr.normal(getkey(), (n,))
    hess = random_pd_matrix(getkey(), n)  # positive definite: not log-concave

    _, nat2 = newton_update(mean, grad, hess, precision_floor=1.0)

    assert tree_allclose(nat2, -hess, atol=1e-12)


def test_round_trip_through_a_newton_site(getkey):
    """A Newton site removed by the cavity returns the original marginals."""
    n = 5
    post_mean = jr.normal(getkey(), (n,))
    post_var = jnp.exp(jr.normal(getkey(), (n,)))
    grad = jr.normal(getkey(), (n,))
    hess_diag = -2.0 / post_var  # log-concave, and small enough to remove

    site_nat1, site_nat2 = newton_update(post_mean, grad, hess_diag)
    cav_mean, cav_var = cavity_distribution(
        post_mean, post_var, site_nat1, site_nat2, power=1.0
    )

    # Adding the site back to the cavity restores the posterior marginals.
    restored_prec = 1.0 / cav_var + site_nat2
    restored_mean = (cav_mean / cav_var + site_nat1) / restored_prec

    assert tree_allclose(1.0 / restored_prec, post_var, atol=1e-8)
    assert tree_allclose(restored_mean, post_mean, atol=1e-8)


def test_jit_vmap_grad(getkey):
    """Both diagonal paths are traceable and differentiable."""
    post_mean, post_var, site_nat1, site_nat2 = _site(getkey)

    def cavity_sum(variance):
        mean, var = cavity_distribution(
            post_mean, variance, site_nat1, site_nat2, power=0.5
        )
        return jnp.sum(mean) + jnp.sum(var)

    assert tree_allclose(jax.jit(cavity_sum)(post_var), cavity_sum(post_var))

    variances = jnp.stack([post_var, post_var * 1.5])
    assert jax.vmap(cavity_sum)(variances).shape == (2,)
    assert bool(jnp.all(jnp.isfinite(jax.grad(cavity_sum)(post_var))))

    def newton_sum(hess_diag):
        nat1, nat2 = newton_update(post_mean, site_nat1, hess_diag)
        return jnp.sum(nat1) + jnp.sum(nat2)

    hess_diag = -jnp.exp(jr.normal(getkey(), (post_mean.shape[0],)))
    assert tree_allclose(jax.jit(newton_sum)(hess_diag), newton_sum(hess_diag))
    assert bool(jnp.all(jnp.isfinite(jax.grad(newton_sum)(hess_diag))))
