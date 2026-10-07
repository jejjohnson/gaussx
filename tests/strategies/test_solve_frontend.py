"""Tests for the unified solve front door.

Covers ``as_linear_operator`` and ``linear_solve``.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx import as_linear_operator, linear_solve
from gaussx._einx import einsum
from gaussx._linalg._symmetrize import symmetrize
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


# -- solver= takes a gaussx strategy or a lineax solver (gh-376) ------------


@pytest.mark.parametrize("kind", ["gaussx", "lineax"])
def test_linear_solve_accepts_both_solver_kinds(kind):
    A = jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = jnp.array([1.0, 2.0, 3.0])
    op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
    if kind == "gaussx":
        solver = CGSolver(rtol=1e-8, atol=1e-8)
    else:
        solver = lx.CG(rtol=1e-8, atol=1e-8)
    x = linear_solve(op, b, solver=solver)
    assert tree_allclose(x, jnp.array([2.0, 1.0, 13.0]) / 9.0, rtol=1e-6)


def test_linear_solve_rejects_other_solver_types():
    op = lx.MatrixLinearOperator(jnp.eye(3), lx.positive_semidefinite_tag)
    with pytest.raises(TypeError, match=r"gaussx\.AbstractSolveStrategy.*lineax"):
        linear_solve(op, jnp.ones(3), solver="cg")


def test_dispatch_solve_wraps_a_lineax_solver():
    from gaussx._strategies._dispatch import dispatch_solve

    A = jnp.array([[2.0, 0.0], [0.0, 4.0]])
    x = dispatch_solve(lx.MatrixLinearOperator(A), jnp.array([2.0, 4.0]), lx.LU())
    assert tree_allclose(x, jnp.ones(2))


# -- Default solver and preconditioner attachment (gh-390) ------------------


def test_default_solver_for_a_small_psd_matrix_is_auto():
    from gaussx import AutoSolver
    from gaussx._solve_frontend import _default_solver

    A = jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = jnp.array([1.0, 2.0, 3.0])
    op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
    assert isinstance(_default_solver(op), AutoSolver)
    # AutoSolver picks a direct solve at n = 3: exact, not CG's 1e-5.
    assert tree_allclose(linear_solve(op, b), jnp.linalg.solve(A, b), rtol=1e-12)


def test_negative_definite_diagonal_is_solved_structurally():
    # _negate keeps -A a NegLinearOperator, so a diagonal stays diagonal.
    from gaussx import AutoSolver, DenseSolver
    from gaussx._solve_frontend import _negate

    d = jnp.array([1.0, 2.0, 4.0])
    op = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(-d),
        (lx.symmetric_tag, lx.negative_semidefinite_tag),
    )
    b = jnp.array([1.0, 1.0, 1.0])
    assert isinstance(AutoSolver()._get_strategy(_negate(op)), DenseSolver)
    assert tree_allclose(linear_solve(op, b), -b / d, rtol=1e-12)


def _kappa_1e3_system(n=60):
    q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (n, n)))
    scale = jnp.logspace(0, 3, n)
    # Badly scaled on the diagonal so that Jacobi helps.
    A = einsum(q * scale, q, "i k, j k -> i j")
    D = jnp.logspace(0, 1.5, n)
    A = einx.multiply("i j, i, j -> i j", A, D, D)
    return symmetrize(A), jr.normal(jr.key(1), (n,))


def _counting(A, count):
    def mv(v):
        jax.debug.callback(lambda: count.__setitem__(0, count[0] + 1))
        return A @ v

    return lx.FunctionLinearOperator(
        mv, jax.ShapeDtypeStruct((A.shape[0],), A.dtype), lx.positive_semidefinite_tag
    )


@pytest.mark.parametrize("wrapper", ["composed", "auto"])
def test_preconditioner_reaches_the_cg_of_composed_and_auto(wrapper):
    from gaussx import (
        AutoSolver,
        ComposedSolver,
        JacobiPreconditioner,
        SLQLogdet,
    )

    A, b = _kappa_1e3_system()
    pre = JacobiPreconditioner(diagonal=jnp.diag(A))
    if wrapper == "composed":
        solver = ComposedSolver(CGSolver(rtol=1e-8, atol=1e-8), SLQLogdet())
    else:
        solver = AutoSolver(size_threshold=10)

    def matvecs(preconditioner):
        count = [0]
        x = linear_solve(
            _counting(A, count), b, solver=solver, preconditioner=preconditioner
        )
        jax.effects_barrier()
        return count[0], x

    plain, _ = matvecs(None)
    preconditioned, x = matvecs(pre)
    assert preconditioned < plain
    rtol = 1e-4 if wrapper == "auto" else 1e-6
    assert tree_allclose(A @ x, b, rtol=rtol, atol=rtol * float(jnp.max(jnp.abs(b))))


@pytest.mark.parametrize("name", ["preconditioned_cg", "bbmm", "minres"])
def test_preconditioner_on_other_strategies_raises_clearly(name):
    from gaussx import BBMMSolver, JacobiPreconditioner, PreconditionedCGSolver

    solver = {
        "preconditioned_cg": PreconditionedCGSolver(),
        "bbmm": BBMMSolver(),
        "minres": MINRESSolver(),
    }[name]
    op = lx.MatrixLinearOperator(jnp.eye(3), lx.positive_semidefinite_tag)
    match = (
        "its own preconditioner" if name == "preconditioned_cg" else "not implemented"
    )
    with pytest.raises(ValueError, match=match):
        linear_solve(
            op, jnp.ones(3), solver=solver, preconditioner=JacobiPreconditioner()
        )


def test_negative_definite_keeps_structure_with_explicit_strategy(monkeypatch):
    """Negation keeps the spectral solve instead of a matvec wrapper (gh-391)."""
    n = 8
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n)
    symbol = -(3.0 - 2.0 * jnp.cos(k))  # conjugate-even, negative definite
    op = gaussx.circulant_from_symbol(symbol, tags=lx.negative_semidefinite_tag)
    b = jnp.arange(1.0, n + 1.0)
    expected = jnp.linalg.solve(op.as_matrix(), b)

    def _forbidden(self):
        raise AssertionError("DiagonalizedOperator.as_matrix called")

    monkeypatch.setattr(gaussx.DiagonalizedOperator, "as_matrix", _forbidden)
    x = linear_solve(op, b, solver=gaussx.DenseSolver())
    monkeypatch.undo()
    assert tree_allclose(x, expected)
    assert tree_allclose(x, -gaussx.solve(-op, b))
