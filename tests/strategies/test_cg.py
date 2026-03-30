"""Tests for CGSolver strategy."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._strategies import CGSolver


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def _make_pd(getkey, n):
    A = jr.normal(getkey(), (n, n))
    return A @ A.T + 1.0 * jnp.eye(n)


def test_solve_psd(getkey):
    cg = CGSolver(rtol=1e-8, atol=1e-8)
    mat = _make_pd(getkey, 5)
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
    mat = _make_pd(getkey, 20)
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
    mat = _make_pd(getkey, 4)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (4,))

    @eqx.filter_jit
    def f(op, v):
        return cg.solve(op, v)

    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(f(op, v), expected, rtol=1e-4)
