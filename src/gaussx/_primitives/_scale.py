"""Scale a factor by a scalar without losing its structured type (gh-326).

A is a square operator, c a scalar and R a root of A (a Cholesky factor L
with A = L Lᵀ, or a square root A^½). A scalar multiple c A has the root
√c R, which `scale_factor` builds without wrapping R in a lineax
``MulLinearOperator``, so downstream dispatch still sees R's type. A
Kronecker product is A = A_1 ⊗ … ⊗ A_D.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import lineax as lx
from jax.core import Tracer
from jaxtyping import Array, ArrayLike

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import LowerBlockTriDiag
from gaussx._operators._diagonalised import DiagonalizedOperator
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._sparse import SparseOperator


def scale_factor(
    operator: lx.AbstractLinearOperator, scale: ArrayLike | Array
) -> lx.AbstractLinearOperator:
    """Scalar multiple s R of a factor that keeps R's structured type.

        s R,   with s folded into R's own storage

    Used by `cholesky` and `sqrt` on c A and A / c: they factor A and scale
    the factor by s = √c (or 1/√c). Downstream code dispatches on the
    factor's type (``solve(L, ·)``, ``logdet(L)``), so the scale is folded
    into one Kronecker factor, every block of a `BlockDiag`, the entries of
    a diagonal, dense or block-bidiagonal factor, or the eigenvalues of a
    `DiagonalizedOperator`; anything else becomes a lineax
    ``MulLinearOperator``.

    Args:
        operator: The factor R.
        scale: The scalar s.

    Returns:
        The operator s R, of R's structured type where a rule applies.

    Examples:
        A Kronecker factor scales its first factor only:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> from gaussx._primitives._scale import scale_factor
        >>> R_1 = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> R = gaussx.Kronecker(R_1, R_1)
        >>> scaled = scale_factor(R, 3.0)
        >>> type(scaled).__name__
        'Kronecker'
        >>> bool(jnp.allclose(scaled.as_matrix(), 3.0 * R.as_matrix()))
        True
    """
    scale = jnp.asarray(scale)
    if isinstance(operator, lx.DiagonalLinearOperator):
        # s diag(a) = diag(s a)
        return lx.DiagonalLinearOperator(scale * lx.diagonal(operator))
    if isinstance(operator, lx.IdentityLinearOperator):
        dtype = operator.in_structure().dtype
        ones = jnp.ones(operator.in_size(), dtype=dtype)
        # s I = diag(s 1)
        return lx.DiagonalLinearOperator(scale * ones)
    if isinstance(operator, lx.MatrixLinearOperator):
        # s R, entry by entry
        return lx.MatrixLinearOperator(scale * operator.matrix, operator.tags)
    if isinstance(operator, lx.TaggedLinearOperator):
        return lx.TaggedLinearOperator(
            scale_factor(operator.operator, scale), operator.tags
        )
    if isinstance(operator, Kronecker):
        first, *rest = operator.operators
        # s (R_1 ⊗ … ⊗ R_D) = (s R_1) ⊗ R_2 ⊗ … ⊗ R_D
        return Kronecker(scale_factor(first, scale), *rest)
    if isinstance(operator, BlockDiag):
        # s blockdiag(R_1, …, R_D) = blockdiag(s R_1, …, s R_D)
        return BlockDiag(*(scale_factor(block, scale) for block in operator.operators))
    if isinstance(operator, LowerBlockTriDiag):
        # s R, block by block
        return LowerBlockTriDiag(
            scale * operator.diagonal,
            scale * operator.sub_diagonal,
            tags=operator.tags,
        )
    if isinstance(operator, DiagonalizedOperator):
        # s W⁻¹ diag(γ) W = W⁻¹ diag(s γ) W
        return operator.with_eigenvalues(scale * operator.eigenvalues)
    return scale * operator


def split_scalar(
    operator: lx.AbstractLinearOperator,
) -> tuple[ArrayLike | Array, lx.AbstractLinearOperator]:
    """Peel nested ``Mul``, ``Div`` and ``Neg`` wrappers into A = c B.

        A = c B,   B not a scalar wrapper

    Args:
        operator: The operator A, possibly wrapped in scalar operations.

    Returns:
        Tuple ``(scale, base)`` holding the accumulated scalar c and the
        innermost operator B.
    """
    scale: ArrayLike | Array = 1.0
    while True:
        if isinstance(operator, lx.MulLinearOperator):
            # c ← c s, for s B
            scale = scale * operator.scalar
        elif isinstance(operator, lx.DivLinearOperator):
            # c ← c / s, for B / s
            scale = scale / operator.scalar
        elif isinstance(operator, lx.NegLinearOperator):
            # c ← −c, for −B
            scale = -scale
        else:
            return scale, operator
        operator = operator.operator


def _concretely_negative(scale: ArrayLike | Array) -> bool:
    """Whether a concrete (untraced) scalar has a negative real part."""
    if isinstance(scale, Tracer):
        return False
    return bool(jnp.real(jnp.asarray(scale)) < 0)


def scaled_root(
    operator: lx.MulLinearOperator | lx.DivLinearOperator | lx.NegLinearOperator,
    root: Callable[[lx.AbstractLinearOperator], Any],
    dense: Callable[[lx.AbstractLinearOperator], lx.AbstractLinearOperator],
    name: str,
) -> Any:
    """Root of a scalar multiple, root(c A) = √c root(A), for `cholesky`/`sqrt`.

        root(c A) = √c root(A),   c > 0

    The nested scalar wrappers are peeled into one c and the root of the
    base A is scaled by √c with `scale_factor`, so its structured type
    survives (gh-326). A concretely negative c (e.g. −A) is not PSD and
    raises, unless A is tagged negative semi-definite, which keeps the
    previous dense route. A traced negative c gives NaN, as for any non-PSD
    input.

    Args:
        operator: The scalar multiple c A, as lineax ``Mul``, ``Div`` or
            ``Neg`` wrappers.
        root: The structured root, `cholesky` or `sqrt`.
        dense: The dense fallback root, applied to the whole of c A.
        name: The root's name, for the error message.

    Returns:
        The root of c A: √c root(A) of root(A)'s type, or the dense root
        when A contains a sparse operator or is NSD with c < 0.

    Raises:
        ValueError: If c is concretely negative and A is not tagged
            negative semi-definite.
    """
    scale, base = split_scalar(operator)
    if _concretely_negative(scale):
        if lx.is_negative_semidefinite(base):
            return dense(operator)
        raise ValueError(
            f"{name} requires a positive semi-definite operator; got a "
            f"negative multiple of a {type(base).__name__}."
        )
    if _contains_sparse(base):
        # A sparse Cholesky is a permuted `SparseCholeskyFactor`, not an
        # operator, so it can neither be scaled nor sit in a BlockDiag.
        return dense(operator)
    # √c
    root_scale = jnp.sqrt(scale)
    # √c root(A)
    return scale_factor(root(base), root_scale)


def _contains_sparse(operator: lx.AbstractLinearOperator) -> bool:
    """Whether a `SparseOperator` sits anywhere inside tags, blocks or factors."""
    if isinstance(operator, SparseOperator):
        return True
    if isinstance(operator, lx.TaggedLinearOperator):
        return _contains_sparse(operator.operator)
    if isinstance(operator, BlockDiag | Kronecker):
        return any(_contains_sparse(part) for part in operator.operators)
    return False
