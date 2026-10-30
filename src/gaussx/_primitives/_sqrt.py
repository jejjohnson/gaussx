"""Structured matrix square root A^½, dispatched on the operator type of A.

A is a square N × N positive-semidefinite operator and A^½ its principal
square root, the PSD S with S S = A. A symmetric A has the eigendecomposition
A = Q diag(γ) Qᵀ, with orthonormal eigenvectors Q and eigenvalues γ, so
A^½ = Q diag(√γ) Qᵀ; a diagonalised operator is A = W⁻¹ diag(γ) W for a
forward transform W (an FFT, say). A Kronecker product is A = A_1 ⊗ … ⊗ A_D
and a Kronecker sum A_1 ⊕ A_2 = A_1 ⊗ I + I ⊗ A_2. A low-rank update on a
scaled identity is A = c I + U D Uᵀ, with orthonormal N × k factor U and
D = diag(d).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import matfree.decomp
import matfree.funm
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._operators._block_diag import BlockDiag, _resolve_dtype
from gaussx._operators._diagonalised import DiagonalizedOperator
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum, KroneckerSumSqrt
from gaussx._operators._low_rank_update import (
    LowRankUpdate,
    orthonormal_scaled_identity,
    scaled_identity_like,
)
from gaussx._operators._sum_kronecker import SumOfKroneckers
from gaussx._operators._utils import register_lineax_structure_functions
from gaussx._primitives._scale import scaled_root


_DEFAULT_LANCZOS_ORDER = 50


def sqrt(
    operator: lx.AbstractLinearOperator,
    *,
    lanczos_order: int | None = None,
) -> lx.AbstractLinearOperator:
    """Principal matrix square root A^½ with structural dispatch, lazy where possible.

        A^½ A^½ = A,   A^½ = Q diag(√γ) Qᵀ

    Requires A to be positive semi-definite.

    When ``lanczos_order`` is given, returns a lazy `SqrtOperator` that
    computes A^½ v by matfree Lanczos without materialising A^½.

    A scalar multiple c A or A / c (c > 0) takes the root of A and scales
    it by √c, keeping its structured type. A concretely negative multiple
    such as −A is not PSD and raises.

    Algorithm:
        1. Take the first structural rule that matches A: an identity is its
           own root, a diagonal takes elementwise roots, a diagonalised
           operator takes roots of its eigenvalues, block-diagonal and
           Kronecker operators take one root per block or factor, a
           Kronecker sum and a sum of Kroneckers return lazy root operators,
           and an orthonormal low-rank update on c I stays a low-rank
           update.
        2. Tags and scalar multiples recurse into their parts.
        3. Fall back to a lazy Lanczos root when ``lanczos_order`` is given,
           else a dense ``eigh``.

    Args:
        operator: The PSD operator A.
        lanczos_order: Order of the Lanczos iteration for a matrix-free
            root. If ``None``, uses a dense eigendecomposition for most
            operators, *except* `SumOfKroneckers`, where ``None`` falls back
            to a Lanczos root with the module default order (no closed-form
            root exists for the sum-of-Kroneckers structure). Pass an
            explicit ``lanczos_order`` to override the default.

    Returns:
        The operator A^½, with A^½ A^½ = A.

    Raises:
        ValueError: If ``operator`` is a concretely negative multiple of a
            non-NSD operator, such as ``-A``.

    Examples:
        A block-diagonal operator gets one root per block:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]), psd)
        >>> B = lx.MatrixLinearOperator(jnp.array([[3.0, 0.5], [0.5, 1.0]]), psd)
        >>> S = gaussx.sqrt(gaussx.BlockDiag(A, B))
        >>> type(S).__name__
        'BlockDiag'
        >>> M = gaussx.BlockDiag(A, B).as_matrix()
        >>> bool(jnp.allclose((S @ S).as_matrix(), M, atol=1e-5))
        True

        A scalar multiple keeps the diagonal structure, √(4 D) = 2 √D:

        >>> D = lx.DiagonalLinearOperator(jnp.array([1.0, 4.0, 9.0]))
        >>> S = gaussx.sqrt(4.0 * D)
        >>> type(S).__name__, [round(float(v), 4) for v in lx.diagonal(S)]
        ('DiagonalLinearOperator', [2.0, 4.0, 6.0])

        ``lanczos_order`` returns a lazy, matrix-free root:

        >>> S = gaussx.sqrt(A, lanczos_order=2)
        >>> type(S).__name__
        'SqrtOperator'
        >>> twice = S.mv(S.mv(jnp.array([1.0, 0.0])))
        >>> bool(jnp.allclose(twice, jnp.array([2.0, 1.0]), atol=1e-4))
        True
    """
    if isinstance(operator, lx.IdentityLinearOperator):
        # I^½ = I
        return operator
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _sqrt_diagonal(operator)
    if isinstance(operator, lx.TaggedLinearOperator):
        return sqrt(operator.operator, lanczos_order=lanczos_order)
    if isinstance(
        operator, lx.MulLinearOperator | lx.DivLinearOperator | lx.NegLinearOperator
    ):
        # (c A)^½ = √c A^½ for c > 0; c < 0 raises
        return scaled_root(
            operator,
            lambda base: sqrt(base, lanczos_order=lanczos_order),
            # The unscaled route keeps a requested matrix-free Lanczos root.
            _sqrt_dense
            if lanczos_order is None
            else lambda base: SqrtOperator(base, lanczos_order),
            "sqrt",
        )
    if isinstance(operator, DiagonalizedOperator):
        # A^½ = W⁻¹ diag(√γ) W satisfies A^½ A^½ = A for any diagonalisable A
        return operator.with_eigenvalues(jnp.sqrt(operator.eigenvalues))
    if isinstance(operator, BlockDiag):
        return _sqrt_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _sqrt_kronecker(operator)
    if isinstance(operator, KroneckerSum):
        return _sqrt_kronecker_sum(operator)
    if isinstance(operator, LowRankUpdate) and operator.rank > 0:
        shift = orthonormal_scaled_identity(operator)
        if shift is not None:
            return _sqrt_low_rank_orthonormal(operator, shift)
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
    """Square root of a diagonal operator diag(a), entry by entry.

        diag(a)^½ = diag(√a)

    Args:
        operator: Diagonal operator diag(a).

    Returns:
        The diagonal operator diag(√a).
    """
    diagonal = lx.diagonal(operator)
    # √a_i
    root_diagonal = jnp.sqrt(diagonal)
    return lx.DiagonalLinearOperator(root_diagonal)


def _sqrt_block_diag(operator: BlockDiag) -> BlockDiag:
    """Square root of a block-diagonal operator, block by block.

        blockdiag(A_1, …, A_D)^½ = blockdiag(A_1^½, …, A_D^½)

    Args:
        operator: Block-diagonal operator with PSD blocks A_d.

    Returns:
        The block-diagonal root.
    """
    # A_d^½, per block
    return BlockDiag(*(sqrt(block) for block in operator.operators))


def _sqrt_kronecker(operator: Kronecker) -> Kronecker:
    """Square root of a Kronecker product, factor by factor.

        (A_1 ⊗ … ⊗ A_D)^½ = A_1^½ ⊗ … ⊗ A_D^½

    Args:
        operator: Kronecker product of PSD factors A_d.

    Returns:
        The Kronecker root.
    """
    # A_d^½, per factor
    return Kronecker(*(sqrt(factor) for factor in operator.operators))


def _sqrt_low_rank_orthonormal(
    operator: LowRankUpdate, shift: Float[Array, ""]
) -> LowRankUpdate:
    """Square root of c I + U D Uᵀ in closed form for orthonormal U.

        (c I + U D Uᵀ)^½ = √c I + U diag(√(c + d) − √c) Uᵀ,   Uᵀ U = I

    The principal root of the PSD operator, again an orthonormal
    ``LowRankUpdate`` on a scaled-identity base (gh-333).

    Args:
        operator: Low-rank update c I + U D Uᵀ with orthonormal U.
        shift: The scalar shift c of the base.

    Returns:
        The root, a PSD-tagged orthonormal low-rank update on √c I.
    """
    # √c
    root_shift = jnp.sqrt(shift)
    psd = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    # c + d
    shifted_weights = shift + operator.d
    # √(c + d)
    root_shifted_weights = jnp.sqrt(shifted_weights)
    # √(c + d) − √c
    root_weights = root_shifted_weights - root_shift
    return LowRankUpdate(
        scaled_identity_like(operator.base, root_shift, psd),
        operator.U,
        root_weights,
        tags=psd,
        orthonormal=True,
    )


def _sqrt_kronecker_sum(operator: KroneckerSum) -> KroneckerSumSqrt:
    """Lazy square root (A_1 ⊕ A_2)^½ of a Kronecker sum, in the joint eigenbasis.

    Args:
        operator: Kronecker sum A_1 ⊕ A_2.

    Returns:
        The lazy `KroneckerSumSqrt` of the same factors.
    """
    return KroneckerSumSqrt(operator.A, operator.B)


def _sqrt_sum_kronecker(
    operator: SumOfKroneckers,
    *,
    lanczos_order: int = _DEFAULT_LANCZOS_ORDER,
) -> SumOfKroneckersSqrt:
    """Lazy Lanczos square root of a sum of Kronecker products.

    Args:
        operator: Sum of Kronecker products A.
        lanczos_order: Number of Lanczos iterations.

    Returns:
        The lazy `SumOfKroneckersSqrt` of A.
    """
    return SumOfKroneckersSqrt(operator, lanczos_order=lanczos_order)


@jax.custom_jvp
def dense_symmetric_sqrt(matrix: Float[Array, "N N"]) -> Float[Array, "N N"]:
    """Symmetric square root of a dense PSD matrix, A^½ = Q diag(√γ) Qᵀ.

        A = Q diag(γ) Qᵀ,   A^½ = Q diag(√max(γ, 0)) Qᵀ

    Array in, array out: `sqrt` is the operator-level entry point. Unlike a
    Cholesky factor this is symmetric and defined for a *semi*-definite
    matrix, so A^½ (A^½)ᵀ = A holds even when A is singular.

    The custom JVP exists because the naive derivative, differentiating
    straight through `jax.numpy.linalg.eigh`, is non-finite whenever an
    eigenvalue is repeated, and a repeated eigenvalue is the *common* case
    here: an isotropic covariance σ² I is nothing but repeated eigenvalues.
    See `_dense_symmetric_sqrt_jvp` for why the true derivative has no such
    singularity.

    Algorithm:
        1. Eigendecompose A = Q diag(γ) Qᵀ.
        2. Clip the eigenvalues at zero and take their square roots.
        3. Rotate back: A^½ = Q diag(√γ) Qᵀ.

    Args:
        matrix: Dense symmetric PSD matrix A, N × N.

    Returns:
        The symmetric root A^½, N × N.

    Examples:
        A diagonal matrix with a zero eigenvalue:

        >>> import jax.numpy as jnp
        >>> from gaussx._primitives._sqrt import dense_symmetric_sqrt
        >>> root = dense_symmetric_sqrt(jnp.diag(jnp.array([4.0, 0.0])))
        >>> [round(float(v), 4) for v in jnp.diag(root)]
        [2.0, 0.0]

        The root squares back to the matrix:

        >>> matrix = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        >>> root = dense_symmetric_sqrt(matrix)
        >>> bool(jnp.allclose(root @ root, matrix, atol=1e-5))
        True
    """
    # A = Q diag(γ) Qᵀ
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    # max(γ, 0)
    clipped_eigenvalues = jnp.maximum(eigenvalues, 0.0)
    # √γ
    sqrt_eigs = jnp.sqrt(clipped_eigenvalues)
    # jnp, not einx: einx cannot resolve an empty matrix (N = 0), which sqrt accepts.
    # Q diag(√γ)
    scaled_eigenvectors = eigenvectors * sqrt_eigs[None, :]
    # A^½ = Q diag(√γ) Qᵀ
    return scaled_eigenvectors @ eigenvectors.T


@dense_symmetric_sqrt.defjvp
def _dense_symmetric_sqrt_jvp(primals, tangents):
    """Sylvester-equation derivative of the symmetric square root.

        dS S + S dS = dA,   (Qᵀ dS Q)_ij (s_i + s_j) = (Qᵀ dA Q)_ij

    Differentiating the defining identity S S = A, with S = A^½, gives the
    Sylvester equation above. In the eigenbasis Q of A, which is also the
    eigenbasis of S, with s_i = √γ_i it is diagonalised entrywise, so the
    derivative divides by *sums* of square-rooted eigenvalues. The
    eigendecomposition's own derivative divides by eigenvalue *gaps*
    γ_i − γ_j, which is why chaining through `eigh` blows up at a repeated
    eigenvalue while the square root itself stays perfectly smooth there:
    s_i + s_j is bounded away from zero for any A with at most one zero
    eigenvalue.

    The one genuinely singular case is a doubly-degenerate zero eigenvalue,
    where s_i + s_j = 0 and the square root really is non-differentiable
    (scalar √ at the origin). Those entries are returned as zero rather
    than ``NaN``, so a rank-deficient covariance still yields a usable
    gradient for the directions that are identified.

    Args:
        primals: Tuple ``(matrix,)`` holding A.
        tangents: Tuple ``(tangent,)`` holding dA.

    Returns:
        Tuple ``(primal_out, tangent_out)`` holding S and dS.
    """
    (matrix,) = primals
    (tangent,) = tangents
    # A = Q diag(γ) Qᵀ
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    # max(γ, 0)
    clipped_eigenvalues = jnp.maximum(eigenvalues, 0.0)
    # s = √γ
    sqrt_eigs = jnp.sqrt(clipped_eigenvalues)
    # jnp, not einx: einx cannot resolve an empty matrix (N = 0), which sqrt accepts.
    # Q diag(s)
    scaled_eigenvectors = eigenvectors * sqrt_eigs[None, :]
    # S = Q diag(s) Qᵀ
    primal_out = scaled_eigenvectors @ eigenvectors.T

    # eigh reads one triangle, so the primal only ever sees the symmetric
    # part of its argument; the tangent must be projected the same way.
    # dA + dAᵀ
    tangent_sum = tangent + tangent.T
    # ½ (dA + dAᵀ)
    tangent = 0.5 * tangent_sum
    # Qᵀ dA
    rotated_left = eigenvectors.T @ tangent
    # Qᵀ dA Q
    rotated = rotated_left @ eigenvectors
    # s_i + s_j
    denominator = sqrt_eigs[:, None] + sqrt_eigs[None, :]
    # Double `where` so the unused branch cannot contribute a NaN to any
    # higher-order tangent, the standard JAX safe-divide idiom.
    is_positive = denominator > 0.0
    safe = jnp.where(is_positive, denominator, 1.0)
    # (Qᵀ dS Q)_ij = (Qᵀ dA Q)_ij / (s_i + s_j), zero where s_i + s_j = 0
    scaled = jnp.where(is_positive, rotated / safe, 0.0)
    # Q (Qᵀ dS Q)
    unrotated_left = eigenvectors @ scaled
    # dS = Q (Qᵀ dS Q) Qᵀ
    tangent_out = unrotated_left @ eigenvectors.T
    return primal_out, tangent_out


def _sqrt_dense(
    operator: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator:
    """Dense square root by eigendecomposition, A^½ = Q diag(√γ) Qᵀ.

    Args:
        operator: The PSD operator A, densified here.

    Returns:
        A symmetric-tagged dense operator holding A^½.
    """
    # A^½ = Q diag(√γ) Qᵀ
    root = dense_symmetric_sqrt(operator.as_matrix())
    return lx.MatrixLinearOperator(root, lx.symmetric_tag)


class SqrtOperator(lx.AbstractLinearOperator):
    """Lazy matrix square root A^½: ``mv`` computes A^½ v by Lanczos.

        A^½ v ≈ ‖v‖₂ V_m T_m^½ e₁,   A V_m ≈ V_m T_m

    Uses matfree's ``funm_lanczos_sym`` to evaluate the matrix-function
    vector product A^½ v from an order-m Lanczos decomposition (basis V_m,
    tridiagonal T_m, first unit vector e₁) without materialising A^½.
    Suitable for large PSD operators.

    Args:
        original: The PSD operator A.
        lanczos_order: Number of Lanczos iterations m; clamped to N.

    Examples:
        The lazy root applied twice recovers A v:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from gaussx._primitives._sqrt import SqrtOperator
        >>> matrix = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> S = SqrtOperator(A, lanczos_order=2)
        >>> v = jnp.array([1.0, 0.0])
        >>> bool(jnp.allclose(S.mv(S.mv(v)), matrix @ v, atol=1e-4))
        True
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
        # f(γ) = √max(γ, 0), applied to T_m's eigenvalues
        dense_sqrt = matfree.funm.dense_funm_sym_eigh(
            lambda eigenvalue: jnp.sqrt(jnp.maximum(eigenvalue, 0.0))
        )
        funm_sqrt = matfree.funm.funm_lanczos_sym(dense_sqrt, tridiag)

        def matvec(v, operator):
            return operator.mv(v)

        # A^½ v ≈ ‖v‖₂ V_m T_m^½ e₁
        return funm_sqrt(matvec, vector, self.original)

    def as_matrix(self):
        return _sqrt_dense(self.original).as_matrix()

    def transpose(self):
        # (A^½)ᵀ = A^½ for a PSD A
        return self

    def in_structure(self):
        return self.original.in_structure()

    def out_structure(self):
        return self.original.out_structure()


class SumOfKroneckersSqrt(SqrtOperator):
    """Lazy Lanczos square root A^½ of a `SumOfKroneckers` covariance.

        A = ∑_i A_i ⊗ B_i,   A^½ v by Lanczos

    What `gaussx.sqrt` returns for a `SumOfKroneckers`. Not to be confused
    with `KroneckerSumSqrt`, the closed-form root of a `KroneckerSum`.

    Specialization of `SqrtOperator` that narrows ``original`` to a
    `SumOfKroneckers` operator. `mv` computes A^½ v with matfree's Lanczos
    matrix-function product without materialising the full square root.

    Args:
        original: The `SumOfKroneckers` covariance A to take the root of.
        lanczos_order: Number of Lanczos iterations; clamped to the operator
            size.

    Examples:
        The lazy root of a two-term sum, applied twice:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> def psd(matrix):
        ...     return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> A_1 = psd(jnp.array([[2.0, 0.5], [0.5, 1.0]]))
        >>> I_2 = psd(jnp.eye(2))
        >>> A = gaussx.SumOfKroneckers(
        ...     gaussx.Kronecker(A_1, I_2), gaussx.Kronecker(I_2, A_1)
        ... )
        >>> S = gaussx.SumOfKroneckersSqrt(A)
        >>> v = jnp.arange(1.0, 5.0)
        >>> bool(jnp.allclose(S.mv(S.mv(v)), A.mv(v), atol=1e-4))
        True
    """

    original: SumOfKroneckers

    def __init__(
        self,
        original: SumOfKroneckers,
        lanczos_order: int = _DEFAULT_LANCZOS_ORDER,
    ) -> None:
        super().__init__(original, lanczos_order=lanczos_order)


class SumKroneckerSqrt(SumOfKroneckersSqrt):
    """Deprecated alias for `SumOfKroneckersSqrt` (gh-297).

    Subclasses `SumOfKroneckersSqrt` so ``isinstance`` checks keep working,
    and emits a `DeprecationWarning` on construction. `gaussx.sqrt` returns
    the new class, which is *not* an instance of this alias. Will be removed
    in gaussx 0.7.0.
    """

    def __init__(
        self,
        original: SumOfKroneckers,
        lanczos_order: int = _DEFAULT_LANCZOS_ORDER,
    ) -> None:
        warn_deprecated(
            "SumKroneckerSqrt is deprecated and will be removed in gaussx 0.7.0; "
            "use SumOfKroneckersSqrt."
        )
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


# lineax.linearise / materialise / diagonal / conj (gh-410).
register_lineax_structure_functions(SqrtOperator)
