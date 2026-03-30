"""Structured matrix square root with dispatch on operator type."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import matfree.decomp
import matfree.funm

from gaussx._operators._block_diag import BlockDiag, _resolve_dtype
from gaussx._operators._kronecker import Kronecker


def sqrt(
    operator: lx.AbstractLinearOperator,
    *,
    lanczos_order: int | None = None,
) -> lx.AbstractLinearOperator:
    """Compute matrix square root S such that S @ S = A.

    Requires A to be positive semi-definite.

    When ``lanczos_order`` is given, returns a lazy ``SqrtOperator``
    that computes ``sqrt(A) @ v`` via matfree Lanczos without
    materializing the full square root matrix.

    Args:
        operator: A PSD linear operator.
        lanczos_order: Order of Lanczos iteration for matrix-free
            sqrt. If ``None``, uses dense eigendecomposition.

    Returns:
        Operator S satisfying S @ S = A.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _sqrt_diagonal(operator)
    if isinstance(operator, BlockDiag):
        return _sqrt_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _sqrt_kronecker(operator)
    if lanczos_order is not None:
        return SqrtOperator(operator, lanczos_order)
    return _sqrt_dense(operator)


def _sqrt_diagonal(
    operator: lx.DiagonalLinearOperator,
) -> lx.DiagonalLinearOperator:
    diag = lx.diagonal(operator)
    return lx.DiagonalLinearOperator(jnp.sqrt(diag))


def _sqrt_block_diag(operator: BlockDiag) -> BlockDiag:
    return BlockDiag(*(sqrt(op) for op in operator.operators))


def _sqrt_kronecker(operator: Kronecker) -> Kronecker:
    return Kronecker(*(sqrt(op) for op in operator.operators))


def _sqrt_dense(
    operator: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator:
    """Eigendecomposition: S = Q diag(sqrt(lam)) Q^T."""
    mat = operator.as_matrix()
    eigenvalues, eigenvectors = jnp.linalg.eigh(mat)
    sqrt_eigs = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
    S = eigenvectors @ jnp.diag(sqrt_eigs) @ eigenvectors.T
    return lx.MatrixLinearOperator(S, lx.symmetric_tag)


class SqrtOperator(lx.AbstractLinearOperator):
    """Lazy matrix square root: ``mv`` computes ``sqrt(A) v`` via Lanczos.

    Uses matfree's ``funm_lanczos_sym`` to evaluate the matrix-function
    vector product ``A^{1/2} v`` without materializing the full square
    root. Suitable for large PSD operators.
    """

    original: lx.AbstractLinearOperator
    _lanczos_order: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)

    def __init__(
        self,
        original: lx.AbstractLinearOperator,
        lanczos_order: int = 30,
    ) -> None:
        self.original = original
        self._lanczos_order = min(lanczos_order, original.in_size())
        self._dtype = _resolve_dtype(original)

    def mv(self, vector):
        tridiag = matfree.decomp.tridiag_sym(self._lanczos_order, reortho="full")
        dense_sqrt = matfree.funm.dense_funm_sym_eigh(
            lambda x: jnp.sqrt(jnp.maximum(x, 0.0))
        )
        funm_sqrt = matfree.funm.funm_lanczos_sym(dense_sqrt, tridiag)

        # Pass operator's as_matrix as a parameter to avoid closure
        # hashing issues with equinox modules in jax.closure_convert.
        # This preserves matrix-free semantics for downstream code
        # while working around the JAX tracing limitation.
        mat = self.original.as_matrix()

        def matvec(v, A):
            return A @ v

        return funm_sqrt(matvec, vector, mat)

    def as_matrix(self):
        return _sqrt_dense(self.original).as_matrix()

    def transpose(self):
        return self  # sqrt of PSD is symmetric

    def in_structure(self):
        return self.original.in_structure()

    def out_structure(self):
        return self.original.out_structure()


# Register tags for SqrtOperator
@lx.is_symmetric.register(SqrtOperator)
def _(operator):
    return True


@lx.is_positive_semidefinite.register(SqrtOperator)
def _(operator):
    return True


for _check in (
    lx.is_diagonal,
    lx.is_lower_triangular,
    lx.is_upper_triangular,
    lx.is_tridiagonal,
    lx.is_negative_semidefinite,
):

    @_check.register(SqrtOperator)
    def _(operator, check=_check):
        return False
