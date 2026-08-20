"""Structured matrix square root with dispatch on operator type."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import matfree.decomp
import matfree.funm
from jaxtyping import Array, Float

from gaussx._operators._block_diag import BlockDiag, _resolve_dtype
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum, KroneckerSumSqrt
from gaussx._operators._sum_kronecker import SumOfKroneckers


_DEFAULT_LANCZOS_ORDER = 50


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
            sqrt. If ``None``, uses dense eigendecomposition for most
            operators, *except* `SumOfKroneckers` — where ``None``
            falls back to a Lanczos sqrt with the module default order
            (no closed-form sqrt exists for the sum-of-Kronecker
            structure). Pass an explicit ``lanczos_order`` to override
            the default rank.

    Returns:
        Operator S satisfying S @ S = A.
    """
    if isinstance(operator, lx.IdentityLinearOperator):
        return operator
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _sqrt_diagonal(operator)
    if isinstance(operator, lx.TaggedLinearOperator):
        return sqrt(operator.operator, lanczos_order=lanczos_order)
    if isinstance(operator, BlockDiag):
        return _sqrt_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _sqrt_kronecker(operator)
    if isinstance(operator, KroneckerSum):
        return _sqrt_kronecker_sum(operator)
    if isinstance(operator, SumOfKroneckers):
        return _sqrt_sum_kronecker(
            operator,
            lanczos_order=(
                _DEFAULT_LANCZOS_ORDER if lanczos_order is None else lanczos_order
            ),
        )
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


def _sqrt_kronecker_sum(operator: KroneckerSum) -> KroneckerSumSqrt:
    return KroneckerSumSqrt(operator.A, operator.B)


def _sqrt_sum_kronecker(
    operator: SumOfKroneckers,
    *,
    lanczos_order: int = _DEFAULT_LANCZOS_ORDER,
) -> SumKroneckerSqrt:
    return SumKroneckerSqrt(operator, lanczos_order=lanczos_order)


@jax.custom_jvp
def dense_symmetric_sqrt(matrix: Float[Array, "N N"]) -> Float[Array, "N N"]:
    """Symmetric square root of a PSD matrix: ``S = Q diag(sqrt(lam)) Q^T``.

    Array in, array out -- `sqrt` is the operator-level entry point. Unlike a
    Cholesky factor this is symmetric and defined for a *semi*-definite
    matrix, so ``S @ S.T == matrix`` holds even when ``matrix`` is singular.

    The custom JVP exists because the naive derivative -- differentiating
    straight through `jax.numpy.linalg.eigh` -- is non-finite whenever an
    eigenvalue is repeated, and a repeated eigenvalue is the *common* case
    here: an isotropic covariance ``sigma^2 I`` is nothing but repeated
    eigenvalues. See `_dense_symmetric_sqrt_jvp` for why the true derivative
    has no such singularity.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    sqrt_eigs = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
    return (eigenvectors * sqrt_eigs[None, :]) @ eigenvectors.T


@dense_symmetric_sqrt.defjvp
def _dense_symmetric_sqrt_jvp(primals, tangents):
    r"""Sylvester-equation derivative of the symmetric square root.

    Differentiating the defining identity ``S S = A`` gives the Sylvester
    equation ``dS S + S dS = dA``. In the eigenbasis of ``A`` -- which is
    also the eigenbasis of ``S`` -- with ``s_i = sqrt(lam_i)`` this is
    diagonalised entrywise into
    ``(V^T dS V)_ij (s_i + s_j) = (V^T dA V)_ij``,
    so the derivative divides by *sums* of square-rooted eigenvalues. The
    eigendecomposition's own derivative divides by eigenvalue *gaps*
    ``lam_i - lam_j``, which is why chaining through `eigh` blows up at a
    repeated eigenvalue while the square root itself stays perfectly smooth
    there -- ``s_i + s_j`` is bounded away from zero for any ``A`` with at
    most one zero eigenvalue.

    The one genuinely singular case is a doubly-degenerate zero eigenvalue,
    where ``s_i + s_j = 0`` and the square root really is non-differentiable
    (scalar ``sqrt`` at the origin). Those entries are returned as zero
    rather than ``NaN``, so a rank-deficient covariance still yields a
    usable gradient for the directions that are identified.
    """
    (matrix,) = primals
    (tangent,) = tangents
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    sqrt_eigs = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
    primal_out = (eigenvectors * sqrt_eigs[None, :]) @ eigenvectors.T

    # eigh reads one triangle, so the primal only ever sees the symmetric
    # part of its argument; the tangent must be projected the same way.
    tangent = 0.5 * (tangent + tangent.T)
    rotated = eigenvectors.T @ tangent @ eigenvectors
    denominator = sqrt_eigs[:, None] + sqrt_eigs[None, :]
    # Double `where` so the unused branch cannot contribute a NaN to any
    # higher-order tangent, the standard JAX safe-divide idiom.
    safe = jnp.where(denominator > 0.0, denominator, 1.0)
    scaled = jnp.where(denominator > 0.0, rotated / safe, 0.0)
    tangent_out = eigenvectors @ scaled @ eigenvectors.T
    return primal_out, tangent_out


def _sqrt_dense(
    operator: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator:
    """Eigendecomposition: S = Q diag(sqrt(lam)) Q^T."""
    S = dense_symmetric_sqrt(operator.as_matrix())
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

        def matvec(v, operator):
            return operator.mv(v)

        return funm_sqrt(matvec, vector, self.original)

    def as_matrix(self):
        return _sqrt_dense(self.original).as_matrix()

    def transpose(self):
        return self  # sqrt of PSD is symmetric

    def in_structure(self):
        return self.original.in_structure()

    def out_structure(self):
        return self.original.out_structure()


class SumKroneckerSqrt(SqrtOperator):
    """Lazy Lanczos square-root operator for ``SumOfKroneckers`` covariances.

    Specialization of `SqrtOperator` that narrows ``original`` to a
    `SumOfKroneckers` operator. `mv` computes ``sqrt(A) v`` via
    matfree's Lanczos matrix-function product without materializing the full
    square root.

    Args:
        original: The ``SumOfKroneckers`` covariance to take the square root of.
        lanczos_order: Number of Lanczos iterations; clamped to the operator
            size.
    """

    original: SumOfKroneckers

    def __init__(
        self,
        original: SumOfKroneckers,
        lanczos_order: int = _DEFAULT_LANCZOS_ORDER,
    ) -> None:
        super().__init__(original, lanczos_order=lanczos_order)


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
    lx.has_unit_diagonal,
):

    @_check.register(SqrtOperator)
    def _(operator, check=_check):
        return False
