"""Tests for CGSolver strategy."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import Kronecker
from gaussx._strategies import CGSolver
from gaussx._testing import random_pd_matrix, tree_allclose


def test_solve_psd(getkey):
    cg = CGSolver(rtol=1e-8, atol=1e-8)
    mat = random_pd_matrix(getkey(), 5)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (5,))
    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(cg.solve(op, v), expected, rtol=1e-4)


def test_solve_diagonal(getkey):
    cg = CGSolver()
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(d), lx.positive_semidefinite_tag
    )
    v = jr.normal(getkey(), (4,))
    expected = v / d
    assert tree_allclose(cg.solve(op, v), expected, rtol=1e-4)


def test_logdet_psd(getkey):
    """Stochastic logdet should be within ~10% for moderate-size PSD."""
    cg = CGSolver(num_probes=50, lanczos_order=20)
    mat = random_pd_matrix(getkey(), 20)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    key = jr.PRNGKey(42)
    estimated = cg.logdet(op, key=key)
    true_ld = jnp.linalg.slogdet(mat)[1]
    # Stochastic estimate — allow generous tolerance
    assert jnp.abs(estimated - true_ld) < 0.1 * jnp.abs(true_ld) + 1.0


def test_logdet_diagonal(getkey):
    """Stochastic logdet on diagonal should be reasonably accurate."""
    cg = CGSolver(num_probes=50, lanczos_order=10)
    d = jnp.abs(jr.normal(getkey(), (10,))) + 0.5
    op = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(d), lx.positive_semidefinite_tag
    )
    key = jr.PRNGKey(123)
    estimated = cg.logdet(op, key=key)
    true_ld = jnp.sum(jnp.log(d))
    assert jnp.abs(estimated - true_ld) < 0.1 * jnp.abs(true_ld) + 1.0


def test_filter_jit_solve(getkey):
    cg = CGSolver(rtol=1e-6, atol=1e-6)
    mat = random_pd_matrix(getkey(), 4)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (4,))

    @eqx.filter_jit
    def f(op, v):
        return cg.solve(op, v)

    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(f(op, v), expected, rtol=1e-4)


def test_solve_structured_operator():
    """CG drives a gaussx operator's matrix-free ``mv``.

    ``lineax.CG`` calls ``lineax.linearise``, which lineax registers only
    for its own operator classes — so this raised ``NotImplementedError``
    on every gaussx operator until the registration in
    ``gaussx._operators``. It is the route for structured covariances with
    no closed-form solve, e.g. a `SumOfKroneckers` of three or more terms.
    """
    factor = lx.MatrixLinearOperator(
        random_pd_matrix(jr.key(0), 3),
        (lx.symmetric_tag, lx.positive_semidefinite_tag),
    )
    op = Kronecker(factor, factor, tags=lx.positive_semidefinite_tag)
    v = jr.normal(jr.key(1), (9,))
    expected = jnp.linalg.solve(op.as_matrix(), v)
    assert tree_allclose(
        CGSolver(rtol=1e-10, atol=1e-10).solve(op, v), expected, rtol=1e-5
    )
