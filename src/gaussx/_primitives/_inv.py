"""Structured lazy inverse with dispatch on operator type."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx

from gaussx._operators._block_diag import BlockDiag, _resolve_dtype
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._low_rank_update import LowRankUpdate, _arrays_match


def inv(
    operator: lx.AbstractLinearOperator,
    *,
    solver: lx.AbstractLinearSolver | None = None,
) -> lx.AbstractLinearOperator:
    """Return a lazy inverse operator A^{-1}.

    The returned operator computes A^{-1} v via ``solve(A, v)``
    when ``mv`` is called. For structured operators, the inverse
    preserves structure.

    Related to ``lineax.invert`` (lineax >= 0.1.1), which wraps
    ``lx.linear_solve`` in a ``FunctionLinearOperator``. The gaussx
    fallback ``InverseOperator`` differs in that its matvec routes
    through the *structured* gaussx ``solve`` dispatch, and its
    ``as_matrix`` uses a Cholesky path for PSD operators.

    Args:
        operator: An invertible linear operator.
        solver: Optional lineax solver for the fallback InverseOperator.

    Returns:
        An operator representing A^{-1}.
    """
    if isinstance(operator, lx.IdentityLinearOperator):
        return operator
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _inv_diagonal(operator)
    if isinstance(operator, BlockDiag):
        return _inv_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _inv_kronecker(operator)
    if (
        isinstance(operator, LowRankUpdate)
        and lx.is_symmetric(operator)
        and _arrays_match(operator.U, operator.V)
    ):
        return _inv_low_rank_symmetric(operator, solver)
    if isinstance(operator, lx.MulLinearOperator):
        return (1.0 / operator.scalar) * inv(operator.operator, solver=solver)
    if isinstance(operator, lx.DivLinearOperator):
        return operator.scalar * inv(operator.operator, solver=solver)
    if isinstance(operator, lx.NegLinearOperator):
        return -inv(operator.operator, solver=solver)
    if isinstance(operator, lx.ComposedLinearOperator) and (
        operator.operator1.in_size() == operator.operator1.out_size()
        and operator.operator2.in_size() == operator.operator2.out_size()
    ):
        # (A B)^{-1} = B^{-1} A^{-1}
        return inv(operator.operator2, solver=solver) @ inv(
            operator.operator1, solver=solver
        )
    return InverseOperator(operator, solver)


def _inv_diagonal(
    operator: lx.DiagonalLinearOperator,
) -> lx.DiagonalLinearOperator:
    diag = lx.diagonal(operator)
    return lx.DiagonalLinearOperator(1.0 / diag)


def _inv_block_diag(operator: BlockDiag) -> BlockDiag:
    return BlockDiag(*(inv(op) for op in operator.operators))


def _inv_kronecker(operator: Kronecker) -> Kronecker:
    return Kronecker(*(inv(op) for op in operator.operators))


def _inv_low_rank_symmetric(
    operator: LowRankUpdate,
    solver: lx.AbstractLinearSolver | None,
) -> LowRankUpdate:
    """Woodbury inverse of a symmetric low-rank update, kept low-rank.

    (L + U D U^T)^{-1} = L^{-1} - L^{-1} U C^{-1} U^T L^{-1} with the
    symmetric capacitance C = D^{-1} + U^T L^{-1} U. Eigendecomposing
    C = W diag(w) W^T turns the correction into a diagonal-middle
    low-rank update, so the result is again a ``LowRankUpdate``:

        (L + U D U^T)^{-1} = L^{-1} + Z diag(-1/w) Z^T,  Z = L^{-1} U W.

    Only the k x k capacitance is ever eigendecomposed.
    """
    from gaussx._primitives._solve import _low_rank_capacitance

    Linv_U, C = _low_rank_capacitance(operator, solver)
    w, W = jnp.linalg.eigh(C)
    Z = Linv_U @ W
    return LowRankUpdate(inv(operator.base, solver=solver), Z, -1.0 / w, Z)


class InverseOperator(lx.AbstractLinearOperator):
    """Lazy inverse: ``mv`` computes A^{-1} v via solve."""

    original: lx.AbstractLinearOperator
    _solver: lx.AbstractLinearSolver | None = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)

    def __init__(
        self,
        original: lx.AbstractLinearOperator,
        solver: lx.AbstractLinearSolver | None = None,
    ) -> None:
        self.original = original
        self._solver = solver
        self._dtype = _resolve_dtype(original)

    def mv(self, vector):
        from gaussx._primitives._solve import solve

        return solve(self.original, vector, solver=self._solver)

    def as_matrix(self):
        # PSD path: A = L L^T => A^{-1} = L^{-T} L^{-1}, computed via two
        # triangular solves. More stable and faster than jnp.linalg.inv.
        # Handles a leading batch shape (..., n, n) — ``L.shape[-1]`` and a
        # broadcast identity keep the cho_solve call rank-correct.
        if lx.is_positive_semidefinite(self.original):
            from gaussx._primitives._cholesky import cholesky

            L = cholesky(self.original).as_matrix()
            n = L.shape[-1]
            identity = jnp.broadcast_to(
                jnp.eye(n, dtype=L.dtype), (*L.shape[:-2], n, n)
            )
            return jax.scipy.linalg.cho_solve((L, True), identity)
        return jnp.linalg.inv(self.original.as_matrix())

    def transpose(self):
        return InverseOperator(self.original.T, self._solver)

    def in_structure(self):
        # Inverse swaps in/out, but for square operators they're the same
        return self.original.out_structure()

    def out_structure(self):
        return self.original.in_structure()


# Register InverseOperator with lineax's singledispatch tag queries.
# Inverse preserves symmetry but not triangularity direction.

for _check in (
    lx.is_symmetric,
    lx.is_diagonal,
    lx.is_positive_semidefinite,
    lx.is_negative_semidefinite,
):

    @_check.register(InverseOperator)
    def _(operator, check=_check):
        return check(operator.original)


@lx.is_lower_triangular.register(InverseOperator)
def _(operator):
    return False


@lx.is_upper_triangular.register(InverseOperator)
def _(operator):
    return False


@lx.is_tridiagonal.register(InverseOperator)
def _(operator):
    return False


@lx.has_unit_diagonal.register(InverseOperator)
def _(operator):
    return False
