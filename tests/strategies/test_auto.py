"""Tests for AutoSolver strategy."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._operators import BlockDiag, Kronecker, LowRankUpdate
from gaussx._strategies import AutoSolver
from gaussx._testing import random_pd_matrix, tree_allclose


def test_solve_diagonal(getkey):
    auto = AutoSolver()
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    v = jr.normal(getkey(), (4,))
    expected = v / d
    assert tree_allclose(auto.solve(op, v), expected)


def test_solve_block_diag(getkey):
    auto = AutoSolver()
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    bd = BlockDiag(A, B)
    v = jr.normal(getkey(), (5,))
    expected = jnp.linalg.solve(bd.as_matrix(), v)
    assert tree_allclose(auto.solve(bd, v), expected)


def test_solve_kronecker(getkey):
    auto = AutoSolver()
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    K = Kronecker(A, B)
    v = jr.normal(getkey(), (6,))
    expected = jnp.linalg.solve(K.as_matrix(), v)
    assert tree_allclose(auto.solve(K, v), expected, rtol=1e-4)


def test_solve_low_rank(getkey):
    auto = AutoSolver()
    d = jnp.abs(jr.normal(getkey(), (5,))) + 1.0
    base = lx.DiagonalLinearOperator(d)
    U = jr.normal(getkey(), (5, 2)) * 0.1
    lr = LowRankUpdate(base, U)
    v = jr.normal(getkey(), (5,))
    expected = jnp.linalg.solve(lr.as_matrix(), v)
    assert tree_allclose(auto.solve(lr, v), expected, rtol=1e-4)


def test_solve_dense_psd(getkey):
    """Small dense PSD should use DenseSolver path."""
    auto = AutoSolver(size_threshold=1000)
    mat = random_pd_matrix(getkey(), 5)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (5,))
    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(auto.solve(op, v), expected)


def test_logdet_diagonal(getkey):
    auto = AutoSolver()
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    expected = jnp.sum(jnp.log(jnp.abs(d)))
    assert tree_allclose(auto.logdet(op), expected)


def test_logdet_kronecker(getkey):
    auto = AutoSolver()
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    K = Kronecker(A, B)
    expected = jnp.linalg.slogdet(K.as_matrix())[1]
    assert tree_allclose(auto.logdet(K), expected, rtol=1e-4)


def test_logdet_dense_small(getkey):
    """Small dense matrix should use DenseSolver (exact logdet)."""
    auto = AutoSolver(size_threshold=1000)
    mat = random_pd_matrix(getkey(), 8)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    expected = jnp.linalg.slogdet(mat)[1]
    assert tree_allclose(auto.logdet(op), expected, rtol=1e-5)


def test_filter_jit(getkey):
    auto = AutoSolver()
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    v = jr.normal(getkey(), (4,))

    @eqx.filter_jit
    def f(op, v):
        return auto.solve(op, v), auto.logdet(op)

    sol, ld = f(op, v)
    assert tree_allclose(sol, jnp.linalg.solve(op.as_matrix(), v))
    assert tree_allclose(ld, jnp.linalg.slogdet(op.as_matrix())[1])


# -- gh-321: structured operators past the size threshold keep their exact
# structural solve and logdet instead of CG + stochastic SLQ. Keys pinned:
# the routing is a property of the operator type, not of the draw.


_PSD = lx.positive_semidefinite_tag


def _factor(n, seed):
    k = jr.normal(jr.key(seed), (n, n))
    return lx.MatrixLinearOperator(k @ k.T + n * jnp.eye(n), _PSD)


def _btd():
    N, d = 600, 2  # n = 1200 > 1000
    return gaussx.BlockTriDiag(
        jnp.tile(4.0 * jnp.eye(d), (N, 1, 1)),
        jnp.tile(-1.0 * jnp.eye(d), (N - 1, 1, 1)),
        tags=_PSD,
    )


def _structured():
    K1 = _factor(40, 0)  # every operator below has n = 1200 or 1600
    return {
        "block_tridiag": _btd(),
        "kronecker": Kronecker(K1, K1),
        "kronecker_sum": gaussx.KroneckerSum(K1, K1, tags=_PSD),
        "diagonalised": gaussx.DiagonalisedOperator(
            jnp.linspace(1.0, 2.0, 1200),
            lambda x: x,
            lambda x: x,
            (1200,),
            normal=True,
            tags=_PSD,
        ),
        "diagonal": lx.DiagonalLinearOperator(jnp.linspace(1.0, 2.0, 1200)),
    }


_WRAPPERS = {
    "bare": lambda op: op,
    "tagged": lambda op: lx.TaggedLinearOperator(op, _PSD),
    "scaled": lambda op: op * 2.0,
    "divided": lambda op: op / 2.0,
    "negated": lambda op: -op,
}


@pytest.mark.parametrize("wrapper", list(_WRAPPERS))
@pytest.mark.parametrize("name", list(_structured()))
def test_structured_operators_route_to_dense(name, wrapper):
    op = _WRAPPERS[wrapper](_structured()[name])
    assert op.in_size() > AutoSolver().size_threshold
    assert type(AutoSolver()._get_strategy(op)).__name__ == "DenseSolver"


def test_large_unstructured_psd_still_routes_to_cg():
    op = _factor(1100, 1)
    assert type(AutoSolver()._get_strategy(op)).__name__ == "CGSolver"


@pytest.mark.slow
def test_block_tridiag_logdet_and_log_prob_are_exact():
    btd = _btd()
    assert jnp.allclose(AutoSolver().logdet(btd), gaussx.logdet(btd), atol=1e-10)
    z = jr.normal(jr.key(0), (btd.in_size(),))
    loc = jnp.zeros(btd.in_size())
    default = gaussx.MultivariateNormalPrecision(loc, btd).log_prob(z)
    dense = gaussx.MultivariateNormalPrecision(
        loc, btd, solver=gaussx.DenseSolver()
    ).log_prob(z)
    assert jnp.allclose(default, dense, atol=1e-8)
