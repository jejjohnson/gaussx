"""Coverage tests for structured primitive dispatch added in the consolidation.

Each test checks a specialized (non-materializing) dispatch path against
the dense reference computed from ``as_matrix()``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import (
    BlockDiag,
    Kronecker,
    KroneckerSum,
    LowRankUpdate,
    SumKronecker,
)
from gaussx._operators._block_tridiag import LowerBlockTriDiag, UpperBlockTriDiag
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._diag import diag
from gaussx._primitives._inv import InverseOperator, inv
from gaussx._primitives._logdet import logdet
from gaussx._primitives._solve import solve
from gaussx._primitives._sqrt import sqrt
from gaussx._primitives._submatrix import submatrix
from gaussx._primitives._svd import svd
from gaussx._primitives._trace import trace
from gaussx._testing import tree_allclose


def _psd_matrix(key, n):
    m = jr.normal(key, (n, n))
    return m @ m.T + n * jnp.eye(n)


def _psd_operator(key, n):
    return lx.MatrixLinearOperator(
        _psd_matrix(key, n),
        (lx.symmetric_tag, lx.positive_semidefinite_tag),
    )


@pytest.fixture
def low_rank(getkey):
    base = lx.DiagonalLinearOperator(jnp.abs(jr.normal(getkey(), (6,))) + 1.0)
    U = jr.normal(getkey(), (6, 2))
    d = jnp.abs(jr.normal(getkey(), (2,))) + 0.5
    return LowRankUpdate(base, U, d)


class TestLowRankUpdateDispatch:
    def test_diag(self, low_rank):
        assert tree_allclose(diag(low_rank), jnp.diag(low_rank.as_matrix()))

    def test_trace(self, low_rank):
        assert tree_allclose(trace(low_rank), jnp.trace(low_rank.as_matrix()))

    def test_inv_stays_low_rank(self, low_rank):
        inverse = inv(low_rank)
        assert isinstance(inverse, LowRankUpdate)
        assert tree_allclose(inverse.as_matrix(), jnp.linalg.inv(low_rank.as_matrix()))

    def test_inv_nonsymmetric_falls_back(self, getkey):
        base = lx.DiagonalLinearOperator(jnp.abs(jr.normal(getkey(), (5,))) + 1.0)
        U = jr.normal(getkey(), (5, 2))
        V = jr.normal(getkey(), (5, 2))
        op = LowRankUpdate(base, U, jnp.ones(2), V)
        inverse = inv(op)
        assert isinstance(inverse, InverseOperator)
        assert tree_allclose(inverse.as_matrix(), jnp.linalg.inv(op.as_matrix()))


class TestKroneckerSumDispatch:
    @pytest.fixture
    def kron_sum(self, getkey):
        return KroneckerSum(_psd_operator(getkey(), 3), _psd_operator(getkey(), 4))

    def test_diag(self, kron_sum):
        assert tree_allclose(diag(kron_sum), jnp.diag(kron_sum.as_matrix()))

    def test_trace(self, kron_sum):
        assert tree_allclose(trace(kron_sum), jnp.trace(kron_sum.as_matrix()))

    def test_logdet_non_diagonal_factors(self, kron_sum):
        _, expected = jnp.linalg.slogdet(kron_sum.as_matrix())
        assert tree_allclose(logdet(kron_sum), expected)


class TestSumKroneckerDispatch:
    @pytest.fixture
    def sum_kron(self, getkey):
        k1 = Kronecker(_psd_operator(getkey(), 3), _psd_operator(getkey(), 4))
        k2 = Kronecker(_psd_operator(getkey(), 3), _psd_operator(getkey(), 4))
        return SumKronecker(k1, k2)

    def test_diag(self, sum_kron):
        assert tree_allclose(diag(sum_kron), jnp.diag(sum_kron.as_matrix()))

    def test_trace(self, sum_kron):
        assert tree_allclose(trace(sum_kron), jnp.trace(sum_kron.as_matrix()))


class TestBlockBidiagonalDispatch:
    @pytest.fixture
    def upper(self, getkey):
        diag_blocks = jax.vmap(lambda k: jnp.linalg.cholesky(_psd_matrix(k, 2)))(
            jr.split(getkey(), 3)
        )
        super_blocks = jr.normal(getkey(), (2, 2, 2))
        return UpperBlockTriDiag(jnp.swapaxes(diag_blocks, -1, -2), super_blocks)

    def test_logdet_upper(self, upper):
        _, expected = jnp.linalg.slogdet(upper.as_matrix())
        assert tree_allclose(logdet(upper), expected)

    def test_logdet_lower(self, upper):
        lower = upper.T
        assert isinstance(lower, LowerBlockTriDiag)
        _, expected = jnp.linalg.slogdet(lower.as_matrix())
        assert tree_allclose(logdet(lower), expected)

    def test_diag_and_trace(self, upper):
        mat = upper.as_matrix()
        assert tree_allclose(diag(upper), jnp.diag(mat))


class TestLineaxLazyOperatorDispatch:
    """solve/logdet/diag/trace/inv on lineax-native Add/Mul/Div/Neg/Composed."""

    @pytest.fixture
    def diag_op(self, getkey):
        return lx.DiagonalLinearOperator(jnp.abs(jr.normal(getkey(), (4,))) + 1.0)

    def test_solve_mul_div_neg(self, diag_op, getkey):
        v = jr.normal(getkey(), (4,))
        for op in (2.5 * diag_op, diag_op / 2.5, -diag_op):
            expected = jnp.linalg.solve(op.as_matrix(), v)
            assert tree_allclose(solve(op, v), expected)

    def test_solve_composed(self, diag_op, getkey):
        op = diag_op @ diag_op
        v = jr.normal(getkey(), (4,))
        assert tree_allclose(solve(op, v), jnp.linalg.solve(op.as_matrix(), v))

    def test_solve_tagged_structured_unwraps(self, diag_op, getkey):
        tagged = lx.TaggedLinearOperator(diag_op, lx.positive_semidefinite_tag)
        v = jr.normal(getkey(), (4,))
        assert tree_allclose(solve(tagged, v), v / lx.diagonal(diag_op))

    def test_solve_identity(self, getkey):
        ident = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((4,), jnp.float64))
        v = jr.normal(getkey(), (4,))
        assert tree_allclose(solve(ident, v), v)

    def test_logdet_mul_neg_composed(self, diag_op):
        for op in (2.5 * diag_op, -diag_op, diag_op @ diag_op, diag_op / 2.5):
            _, expected = jnp.linalg.slogdet(op.as_matrix())
            assert tree_allclose(logdet(op), expected)

    def test_diag_add_mul_neg(self, diag_op, getkey):
        dense = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4)))
        for op in (dense + diag_op, 2.5 * dense, -dense, dense / 2.5):
            assert tree_allclose(diag(op), jnp.diag(op.as_matrix()))
            assert tree_allclose(trace(op), jnp.trace(op.as_matrix()))

    def test_inv_mul(self, diag_op):
        op = 2.5 * diag_op
        assert tree_allclose(inv(op).as_matrix(), jnp.linalg.inv(op.as_matrix()))

    def test_cholesky_tagged_unwraps(self, getkey):
        inner = BlockDiag(_psd_operator(getkey(), 2), _psd_operator(getkey(), 3))
        tagged = lx.TaggedLinearOperator(inner, lx.positive_semidefinite_tag)
        L = cholesky(tagged)
        assert isinstance(L, BlockDiag)
        assert tree_allclose(L.as_matrix() @ L.as_matrix().T, tagged.as_matrix())

    def test_sqrt_tagged_unwraps(self, getkey):
        inner = lx.DiagonalLinearOperator(jnp.abs(jr.normal(getkey(), (4,))) + 1.0)
        tagged = lx.TaggedLinearOperator(inner, lx.positive_semidefinite_tag)
        S = sqrt(tagged)
        assert isinstance(S, lx.DiagonalLinearOperator)


class TestStructuredSVD:
    def test_kronecker(self, getkey):
        K = Kronecker(
            lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3))),
            lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4))),
        )
        U, s, Vt = svd(K)
        mat = K.as_matrix()
        assert tree_allclose(U @ jnp.diag(s) @ Vt, mat)
        assert jnp.all(jnp.diff(s) <= 1e-12)
        assert tree_allclose(s, jnp.linalg.svd(mat, compute_uv=False))

    def test_block_diag(self, getkey):
        bd = BlockDiag(
            lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3))),
            lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2))),
        )
        U, s, Vt = svd(bd)
        mat = bd.as_matrix()
        assert tree_allclose(U @ jnp.diag(s) @ Vt, mat)
        assert tree_allclose(s, jnp.linalg.svd(mat, compute_uv=False))


class TestStructuredSubmatrix:
    def test_kronecker(self, getkey):
        K = Kronecker(
            lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3))),
            lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4))),
        )
        rows = jnp.array([0, 5, 11, 3])
        cols = jnp.array([2, 2, 7, 10])
        expected = K.as_matrix()[jnp.ix_(rows, cols)]
        assert tree_allclose(submatrix(K, rows, cols), expected)

    def test_tagged_delegates(self, getkey):
        inner = lx.DiagonalLinearOperator(jr.normal(getkey(), (5,)))
        tagged = lx.TaggedLinearOperator(inner, lx.symmetric_tag)
        rows = jnp.array([0, 2, 4])
        cols = jnp.array([1, 2, 3])
        expected = tagged.as_matrix()[jnp.ix_(rows, cols)]
        assert tree_allclose(submatrix(tagged, rows, cols), expected)


class TestRequiredLineaxPredicates:
    """lineax 0.1.1 requires every predicate for every operator.

    Regression test: ``has_unit_diagonal`` was unregistered, so
    ``lx.AutoLinearSolver`` crashed on triangular-tagged gaussx
    operators (the Triangular solver queries it).
    """

    def test_has_unit_diagonal_registered(self, getkey):
        dense = _psd_operator(getkey(), 2)
        operators = [
            BlockDiag(dense, dense),
            Kronecker(dense, dense),
            KroneckerSum(dense, dense),
            LowRankUpdate(lx.DiagonalLinearOperator(jnp.ones(2)), jnp.ones((2, 1))),
        ]
        for op in operators:
            assert lx.has_unit_diagonal(op) is False

    def test_auto_solver_on_lower_block_tridiag(self, getkey):
        diag_blocks = jax.vmap(lambda k: jnp.linalg.cholesky(_psd_matrix(k, 2)))(
            jr.split(getkey(), 3)
        )
        sub_blocks = jr.normal(getkey(), (2, 2, 2))
        L = LowerBlockTriDiag(diag_blocks, sub_blocks)
        b = jr.normal(getkey(), (6,))
        out = lx.linear_solve(L, b, lx.AutoLinearSolver(well_posed=True))
        assert tree_allclose(out.value, jnp.linalg.solve(L.as_matrix(), b))
