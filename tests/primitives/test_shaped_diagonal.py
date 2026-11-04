"""Primitives on a diagonal whose array or pytree has a shape (#663, #682).

``lx.DiagonalLinearOperator(d)`` with d of shape (T, N) acts on (T, N) arrays,
and its matrix is diag(vec d), of size T N. ``lx.diagonal`` flattens d, so a
fast path that rebuilds or divides by it used to break on the shaped vector.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._primitives._scale import scale_factor
from gaussx._testing import assert_sample_moments, default_tolerances


def _diagonal() -> jax.Array:
    return jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.result_type(float))


def _operators():
    plain = lx.DiagonalLinearOperator(_diagonal())
    return [
        pytest.param(plain, id="plain"),
        pytest.param(
            lx.TaggedLinearOperator(plain, lx.positive_semidefinite_tag), id="tagged"
        ),
    ]


def _assert_close(actual, expected) -> None:
    rtol, atol = default_tolerances(expected)
    assert actual.shape == expected.shape
    assert jnp.allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("operator", _operators())
def test_solve_keeps_the_shape(operator):
    vector = jnp.ones((2, 3), dtype=jnp.result_type(float))
    _assert_close(gaussx.solve(operator, vector), vector / _diagonal())
    # the structural path wins over a solver, as in the issue's reproduction
    solved = gaussx.solve(operator, vector, solver=lx.CG(rtol=1e-5, atol=1e-5))
    _assert_close(solved, vector / _diagonal())


def test_solve_still_takes_a_flat_vector():
    operator = lx.DiagonalLinearOperator(_diagonal())
    vector = jnp.ones(6, dtype=jnp.result_type(float))
    _assert_close(gaussx.solve(operator, vector), vector / jnp.ravel(_diagonal()))


@pytest.mark.parametrize("operator", _operators())
def test_inv_acts_on_the_shape(operator):
    inverse = gaussx.inv(operator)
    vector = jnp.ones((2, 3), dtype=jnp.result_type(float))
    assert inverse.in_structure().shape == (2, 3)
    _assert_close(inverse.mv(vector), vector / _diagonal())


@pytest.mark.parametrize("operator", _operators())
@pytest.mark.parametrize("factor", [gaussx.sqrt, gaussx.cholesky])
def test_square_roots_act_on_the_shape(operator, factor):
    root = factor(operator)
    vector = jnp.ones((2, 3), dtype=jnp.result_type(float))
    assert root.in_structure().shape == (2, 3)
    _assert_close(root.mv(vector), vector * jnp.sqrt(_diagonal()))
    _assert_close(gaussx.solve(root, vector), vector / jnp.sqrt(_diagonal()))


def test_scale_factor_keeps_the_shape():
    scaled = scale_factor(lx.DiagonalLinearOperator(_diagonal()), 2.0)
    assert scaled.in_structure().shape == (2, 3)
    _assert_close(scaled.diagonal, 2.0 * _diagonal())
    identity = lx.IdentityLinearOperator(
        jax.ShapeDtypeStruct((2, 3), jnp.result_type(float))
    )
    assert scale_factor(identity, 2.0).in_structure().shape == (2, 3)


def test_sandwich_keeps_the_shape():
    operator = lx.DiagonalLinearOperator(_diagonal())
    result = gaussx.sandwich(operator, operator)
    assert result.in_structure().shape == (2, 3)
    _assert_close(result.mv(jnp.ones_like(_diagonal())), _diagonal() ** 3)


def test_gaussian_log_prob_takes_the_flattened_value():
    """log N(y; 0, diag(d)) = −½ (yᵀ D⁻¹ y + ln det D + M ln 2π)."""
    operator = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(_diagonal()), lx.positive_semidefinite_tag
    )
    value = jnp.ones(6, dtype=jnp.result_type(float))
    actual = gaussx.gaussian_log_prob(jnp.zeros_like(value), operator, value)
    expected = -0.5 * (
        jnp.sum(1.0 / _diagonal())
        + jnp.sum(jnp.log(_diagonal()))
        + 6 * jnp.log(2.0 * jnp.pi)
    )
    _assert_close(actual, expected)


def test_sample_mvn_stays_flat_with_the_right_moments():
    operator = lx.DiagonalLinearOperator(_diagonal())
    draws = gaussx.sample_mvn(
        jnp.zeros(6, dtype=_diagonal().dtype),
        operator,
        key=jr.key(0),
        num_samples=20_000,
    )
    assert draws.shape == (20_000, 6)
    # the moment bound comes from the estimator's own sampling distribution
    assert_sample_moments(draws, jnp.zeros(6), operator.as_matrix())


def test_a_pytree_diagonal_keeps_its_structure():
    tree = {"a": jnp.arange(1.0, 3.0), "b": jnp.array([[3.0, 4.0], [5.0, 6.0]])}
    operator = lx.DiagonalLinearOperator(tree)
    vector = jax.tree.map(jnp.ones_like, tree)
    solved = gaussx.solve(operator, vector)
    inverted = gaussx.inv(operator).mv(vector)
    rooted = gaussx.sqrt(operator).mv(vector)
    for name, leaf in tree.items():
        _assert_close(solved[name], 1.0 / leaf)
        _assert_close(inverted[name], 1.0 / leaf)
        _assert_close(rooted[name], jnp.sqrt(leaf))


def test_transforms_of_a_shaped_solve():
    vector = jnp.ones((2, 3), dtype=jnp.result_type(float))

    def solve_sum(diagonal, rhs):
        return jnp.sum(gaussx.solve(lx.DiagonalLinearOperator(diagonal), rhs))

    value = eqx.filter_jit(solve_sum)(_diagonal(), vector)
    _assert_close(value, jnp.sum(1.0 / _diagonal()))
    gradient = jax.grad(solve_sum)(_diagonal(), vector)
    _assert_close(gradient, -1.0 / _diagonal() ** 2)
    batch = jnp.stack([vector, 2.0 * vector])
    solved = jax.vmap(
        lambda rhs: gaussx.solve(lx.DiagonalLinearOperator(_diagonal()), rhs)
    )(batch)
    _assert_close(solved[1], 2.0 / _diagonal())
