"""Structured lazy inverse with dispatch on operator type."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import lineax as lx

from gaussx._operators._block_diag import BlockDiag, _resolve_dtype
from gaussx._operators._kronecker import Kronecker


def inv(
    operator: lx.AbstractLinearOperator,
    *,
    solver: lx.AbstractLinearSolver | None = None,
) -> lx.AbstractLinearOperator:
    """Return a lazy inverse operator A^{-1}.

    The returned operator computes A^{-1} v via ``solve(A, v)``
    when ``mv`` is called. For structured operators, the inverse
    preserves structure.

    Args:
        operator: An invertible linear operator.
        solver: Optional lineax solver for the fallback InverseOperator.

    Returns:
        An operator representing A^{-1}.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _inv_diagonal(operator)
    if isinstance(operator, BlockDiag):
        return _inv_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _inv_kronecker(operator)
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
