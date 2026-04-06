"""Tests for ComposedSolver strategy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._strategies import CGSolver, ComposedSolver, DenseSolver
from gaussx._testing import tree_allclose


def _make_pd_operator(getkey, n=4):
    """Create a random positive-definite operator."""
    A = jr.normal(getkey(), (n, n))
    M = A @ A.T + n * jnp.eye(n)
    return lx.MatrixLinearOperator(M, lx.positive_semidefinite_tag)


def test_solve_delegates_to_solve_strategy(getkey):
    """solve() should use the solve strategy, not the logdet strategy."""
    op = _make_pd_operator(getkey)
    v = jr.normal(getkey(), (4,))
    dense = DenseSolver()
    composed = ComposedSolver(solve_strategy=dense, logdet_strategy=CGSolver())
    assert tree_allclose(composed.solve(op, v), dense.solve(op, v))


def test_logdet_delegates_to_logdet_strategy(getkey):
    """logdet() should use the logdet strategy, not the solve strategy."""
    op = _make_pd_operator(getkey)
    dense = DenseSolver()
    composed = ComposedSolver(solve_strategy=CGSolver(), logdet_strategy=dense)
    assert tree_allclose(composed.logdet(op), dense.logdet(op))


def test_dense_solve_cg_logdet(getkey):
    """Dense solve + CG logdet should each match their standalone strategy."""
    op = _make_pd_operator(getkey)
    v = jr.normal(getkey(), (4,))
    dense = DenseSolver()
    cg = CGSolver()
    composed = ComposedSolver(solve_strategy=dense, logdet_strategy=cg)

    assert tree_allclose(composed.solve(op, v), dense.solve(op, v))
    assert tree_allclose(composed.logdet(op), cg.logdet(op))


def test_cg_solve_dense_logdet(getkey):
    """CG solve + dense logdet should each match their standalone strategy."""
    op = _make_pd_operator(getkey)
    v = jr.normal(getkey(), (4,))
    dense = DenseSolver()
    cg = CGSolver()
    composed = ComposedSolver(solve_strategy=cg, logdet_strategy=dense)

    assert tree_allclose(composed.solve(op, v), cg.solve(op, v))
    assert tree_allclose(composed.logdet(op), dense.logdet(op))


def test_same_strategy_both(getkey):
    """ComposedSolver(Dense, Dense) should match DenseSolver exactly."""
    op = _make_pd_operator(getkey)
    v = jr.normal(getkey(), (4,))
    dense = DenseSolver()
    composed = ComposedSolver(
        solve_strategy=DenseSolver(), logdet_strategy=DenseSolver()
    )

    assert tree_allclose(composed.solve(op, v), dense.solve(op, v))
    assert tree_allclose(composed.logdet(op), dense.logdet(op))


def test_filter_jit(getkey):
    """ComposedSolver should be JIT-compatible."""
    op = _make_pd_operator(getkey)
    v = jr.normal(getkey(), (4,))
    composed = ComposedSolver(
        solve_strategy=DenseSolver(), logdet_strategy=DenseSolver()
    )

    result_eager = composed.solve(op, v)
    result_jit = jax.jit(composed.solve)(op, v)
    assert tree_allclose(result_eager, result_jit)


def test_pytree_roundtrip():
    """Strategy should survive flatten/unflatten."""
    composed = ComposedSolver(
        solve_strategy=DenseSolver(), logdet_strategy=DenseSolver()
    )
    leaves, treedef = jax.tree.flatten(composed)
    restored = jax.tree.unflatten(treedef, leaves)
    assert isinstance(restored, ComposedSolver)
