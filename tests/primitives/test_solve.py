"""Tests for gaussx.solve with structural dispatch."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import BlockDiag, Kronecker, KroneckerSum, LowRankUpdate
from gaussx._primitives import solve
from gaussx._testing import dense_solve, random_pd_matrix, tree_allclose


class LazyDiagonal(lx.DiagonalLinearOperator):
    def as_matrix(self):
        raise NotImplementedError("dense materialization unavailable")


def test_solve_diagonal(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    v = jr.normal(getkey(), (4,))
    assert tree_allclose(solve(op, v), dense_solve(op, v))


@pytest.mark.slow
def test_solve_block_diag(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 2 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    bd = BlockDiag(A, B)
    v = jr.normal(getkey(), (5,))
    assert tree_allclose(solve(bd, v), dense_solve(bd, v))


def test_solve_kronecker(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    K = Kronecker(A, B)
    v = jr.normal(getkey(), (6,))
    assert tree_allclose(solve(K, v), dense_solve(K, v), rtol=1e-4)


def test_solve_kronecker_lazy_factors(getkey):
    a_diag = jnp.abs(jr.normal(getkey(), (2,))) + 0.5
    b_diag = jnp.abs(jr.normal(getkey(), (3,))) + 0.5
    K = Kronecker(LazyDiagonal(a_diag), LazyDiagonal(b_diag))
    v = jr.normal(getkey(), (6,))
    expected = jnp.linalg.solve(jnp.kron(jnp.diag(a_diag), jnp.diag(b_diag)), v)
    assert tree_allclose(solve(K, v), expected, rtol=1e-4)


def test_solve_low_rank(getkey):
    d = jnp.abs(jr.normal(getkey(), (5,))) + 1.0
    base = lx.DiagonalLinearOperator(d)
    U = jr.normal(getkey(), (5, 2)) * 0.1
    lr = LowRankUpdate(base, U)
    v = jr.normal(getkey(), (5,))
    assert tree_allclose(solve(lr, v), dense_solve(lr, v), rtol=1e-4)


def test_solve_dense_fallback(getkey):
    mat = jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3)
    op = lx.MatrixLinearOperator(mat)
    v = jr.normal(getkey(), (3,))
    assert tree_allclose(solve(op, v), dense_solve(op, v))


def test_solve_filter_jit(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    v = jr.normal(getkey(), (4,))

    @eqx.filter_jit
    def f(op, v):
        return solve(op, v)

    assert tree_allclose(f(op, v), dense_solve(op, v))


def test_solve_kronecker_sum_grad_with_repeated_eigenvalue():
    # gh-295: differentiating through the factors' eigh gave NaN when a
    # factor has a repeated eigenvalue (here A = s I). The implicit JVP
    # must match the dense reference.
    B = random_pd_matrix(jr.key(0), 2)
    psd = lx.positive_semidefinite_tag
    v = jnp.arange(1.0, 7.0)

    def structured(s):
        K = KroneckerSum(
            lx.MatrixLinearOperator(s * jnp.eye(3), psd),
            lx.MatrixLinearOperator(B, psd),
        )
        return solve(K, v).sum()

    def dense(s):
        K = jnp.kron(s * jnp.eye(3), jnp.eye(2)) + jnp.kron(jnp.eye(3), B)
        return jnp.linalg.solve(K, v).sum()

    grad = jax.grad(structured)(1.5)
    assert jnp.isfinite(grad)
    assert jnp.allclose(grad, jax.grad(dense)(1.5), rtol=1e-8, atol=1e-8)
    assert jnp.allclose(jax.jit(jax.grad(structured))(1.5), grad, rtol=1e-12)


# -- solver= takes a lineax solver or a gaussx strategy (gh-376) ------------

_A3 = jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
_B3 = jnp.array([1.0, 2.0, 3.0])
_X3 = jnp.array([2.0, 1.0, 13.0]) / 9.0


@pytest.mark.x64_only(reason="rtol=1e-6 against CG at tolerance 1e-8")
@pytest.mark.parametrize("kind", ["lineax", "gaussx"])
def test_solve_accepts_both_solver_kinds(kind):
    from gaussx import CGSolver

    op = lx.MatrixLinearOperator(_A3, lx.positive_semidefinite_tag)
    if kind == "lineax":
        solver = lx.CG(rtol=1e-8, atol=1e-8)
    else:
        solver = CGSolver(rtol=1e-8, atol=1e-8)
    assert tree_allclose(solve(op, _B3, solver=solver), _X3, rtol=1e-6)


def test_solve_strategy_owns_a_structured_solve():
    # A gaussx strategy takes the whole solve, Kronecker or not; a lineax
    # solver keeps being threaded into the per-factor solves.
    from gaussx import DenseSolver

    factor = lx.MatrixLinearOperator(_A3, lx.positive_semidefinite_tag)
    op = Kronecker(factor, factor)
    b = jnp.arange(9.0)
    expected = dense_solve(op, b)
    assert tree_allclose(solve(op, b, solver=DenseSolver()), expected)
    assert tree_allclose(solve(op, b, solver=lx.LU()), expected)


def test_solve_rejects_other_solver_types():
    op = lx.MatrixLinearOperator(_A3)
    with pytest.raises(TypeError, match=r"lineax\.AbstractLinearSolver.*gaussx"):
        solve(op, _B3, solver=object())
