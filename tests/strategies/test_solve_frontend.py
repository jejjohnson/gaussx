"""Tests for the unified solve front door.

Covers ``as_linear_operator`` and ``linear_solve``.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import as_linear_operator, linear_solve
from gaussx._strategies import CGSolver, MINRESSolver
from gaussx._testing import random_pd_matrix, tree_allclose


def test_as_linear_operator_matches_matrix(getkey):
    """A wrapped matvec reproduces the underlying matrix action."""
    mat = random_pd_matrix(getkey(), 6)
    op = as_linear_operator(lambda v: mat @ v, shape=(6, 6), positive_semidefinite=True)
    v = jr.normal(getkey(), (6,))
    assert tree_allclose(op.mv(v), mat @ v, rtol=1e-5)
    assert lx.is_positive_semidefinite(op)
    assert lx.is_symmetric(op)


def test_as_linear_operator_requires_shape():
    with pytest.raises(ValueError, match="shape"):
        as_linear_operator(lambda v: v)


def test_in_structure_int_and_tuple(getkey):
    """`in_structure` accepts an int or a shape tuple."""
    mat = random_pd_matrix(getkey(), 4)
    op_int = as_linear_operator(
        lambda v: mat @ v, in_structure=4, positive_semidefinite=True
    )
    op_tuple = as_linear_operator(
        lambda v: mat @ v, in_structure=(4,), positive_semidefinite=True
    )
    v = jr.normal(getkey(), (4,))
    assert tree_allclose(op_int.mv(v), op_tuple.mv(v), rtol=1e-6)


def test_linear_solve_psd_operator(getkey):
    """PSD operator solves via the default CG path."""
    mat = random_pd_matrix(getkey(), 8)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    b = jr.normal(getkey(), (8,))
    x = linear_solve(op, b, solver=CGSolver(rtol=1e-8, atol=1e-8))
    assert tree_allclose(x, jnp.linalg.solve(mat, b), rtol=1e-4)


def test_linear_solve_matvec_tuple(getkey):
    """A bare `(matvec, shape)` pair is coerced and solved.

    The tuple path carries no structural tags, so a solver that works from the
    raw matvec (MINRES) is supplied explicitly.
    """
    mat = random_pd_matrix(getkey(), 7)
    b = jr.normal(getkey(), (7,))
    x = linear_solve(
        (lambda v: mat @ v, (7, 7)), b, solver=MINRESSolver(rtol=1e-10, atol=1e-10)
    )
    assert tree_allclose(x, jnp.linalg.solve(mat, b), rtol=1e-4)


def test_linear_solve_negative_definite(getkey):
    """A negative-definite operator is solved via the negated PSD system.

    This mirrors how elliptic (Laplacian-like) operators are handed over by
    finite-volume / spectral callers.
    """
    pd = random_pd_matrix(getkey(), 10)
    neg = -pd  # symmetric negative definite
    op = as_linear_operator(lambda v: neg @ v, shape=(10, 10), negative_definite=True)
    b = jr.normal(getkey(), (10,))
    x = linear_solve(op, b, solver=CGSolver(rtol=1e-8, atol=1e-8))
    assert tree_allclose(x, jnp.linalg.solve(neg, b), rtol=1e-4)


def test_default_solver_symmetric_indefinite(getkey):
    """A symmetric indefinite operator defaults to MINRES."""
    a = jr.normal(getkey(), (12, 12))
    sym = 0.5 * (a + a.T)  # symmetric, generally indefinite
    op = as_linear_operator(lambda v: sym @ v, shape=(12, 12), symmetric=True)
    b = jr.normal(getkey(), (12,))
    x = linear_solve(op, b)  # no solver -> default MINRES
    assert tree_allclose(sym @ x, b, rtol=1e-3, atol=1e-4)


def test_default_solver_nonsymmetric_raises(getkey):
    """An untagged (non-symmetric) operator has no safe default solver."""
    a = jr.normal(getkey(), (5, 5))
    op = as_linear_operator(lambda v: a @ v, shape=(5, 5))
    b = jr.normal(getkey(), (5,))
    with pytest.raises(ValueError, match="non-symmetric"):
        linear_solve(op, b)


def test_preconditioner_callable(getkey):
    """A callable Jacobi preconditioner is accepted and yields the right solve."""
    mat = random_pd_matrix(getkey(), 30)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    b = jr.normal(getkey(), (30,))
    inv_diag = 1.0 / jnp.diag(mat)
    x = linear_solve(
        op,
        b,
        solver=CGSolver(rtol=1e-8, atol=1e-8),
        preconditioner=lambda v: inv_diag * v,
    )
    assert tree_allclose(x, jnp.linalg.solve(mat, b), rtol=1e-4)


def test_preconditioner_operator(getkey):
    """A lineax operator preconditioner is accepted."""
    mat = random_pd_matrix(getkey(), 20)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    b = jr.normal(getkey(), (20,))
    precond = lx.DiagonalLinearOperator(1.0 / jnp.diag(mat))
    x = linear_solve(
        op, b, solver=CGSolver(rtol=1e-8, atol=1e-8), preconditioner=precond
    )
    assert tree_allclose(x, jnp.linalg.solve(mat, b), rtol=1e-4)
