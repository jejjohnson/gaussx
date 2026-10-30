"""Covariance propagation, trace of a product and matrix right-hand-side solves.

A and B are generic square operators (B is also the right-hand side of the
solves, X their solution), J a linear map and Σ a covariance. A low-rank
update is A = A₀ + U_A D_A V_Aᵀ, with base A₀, factors U_A and V_A and the
diagonal D_A = diag(d_A). D = diag(d) is a diagonal operator. The diagonal
conditional variance uses the kernel-matrix names of `gaussx._linalg._schur`.
"""

from __future__ import annotations

from typing import overload

import einx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange, reduce
from gaussx._linalg._schur import conditional_variance as _conditional_variance
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._diagonalised import DiagonalizedOperator
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._operators._sparse import SparseOperator
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._diag import diag
from gaussx._primitives._solve import solve
from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._strategies._dispatch import dispatch_solve


@overload
def cov_transform(
    J: Float[Array, "M N"],
    cov_operator: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator: ...


@overload
def cov_transform(
    J: lx.AbstractLinearOperator,
    cov_operator: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator: ...


def cov_transform(
    J: Float[Array, "M N"] | lx.AbstractLinearOperator,
    cov_operator: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """Covariance propagation through a linear map, J Σ Jᵀ.

        cov(J x) = J Σ Jᵀ,   Σ = cov(x)

    Used in error propagation, Kalman filter updates and first-order
    uncertainty propagation. Structure is exploited where it can be:

    - An operator J routes through `sandwich`, which keeps matched
      ``Kronecker`` and ``BlockDiag`` structure and avoids materialising Σ
      when J or Σ is diagonal.
    - A diagonal Σ = diag(d) with a dense J forms (J diag(d)) Jᵀ directly,
      without the N × N materialisation of Σ.

    Otherwise Σ is materialised and the dense product formed. The result
    is tagged symmetric (and positive semi-definite when Σ is).

    Args:
        J: Jacobian or linear map J, M × N, as an array or an operator.
        cov_operator: Input covariance Σ, N × N.

    Returns:
        The propagated covariance J Σ Jᵀ, M × M. For an operator J its
        class follows `sandwich`; otherwise it is a
        `lineax.MatrixLinearOperator`.

    Examples:
        A dense map and a diagonal covariance:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> J = jnp.array([[1.0, 2.0], [0.0, 1.0], [1.0, 1.0]])
        >>> covariance = lx.DiagonalLinearOperator(jnp.array([1.0, 0.5]))
        >>> propagated = gaussx.cov_transform(J, covariance)
        >>> propagated.as_matrix().shape
        (3, 3)
        >>> dense = J @ jnp.diag(jnp.array([1.0, 0.5])) @ J.T
        >>> bool(jnp.allclose(propagated.as_matrix(), dense))
        True
    """
    if isinstance(J, lx.AbstractLinearOperator):
        # J Σ Jᵀ, structure-preserving
        return sandwich(J, cov_operator)

    tags = _sandwich_tags(cov_operator)

    if isinstance(cov_operator, lx.DiagonalLinearOperator):
        variances = lx.diagonal(cov_operator)
        # J diag(d)
        scaled_map = einx.multiply("out inp, inp -> out inp", J, variances)
        # J diag(d) Jᵀ
        propagated = einsum(scaled_map, J, "out inp, other inp -> out other")
        return lx.MatrixLinearOperator(propagated, tags)

    covariance = cov_operator.as_matrix()
    # J Σ
    map_covariance = einsum(J, covariance, "out inp, inp col -> out col")
    # J Σ Jᵀ
    propagated = einsum(map_covariance, J, "out col, other col -> out other")
    return lx.MatrixLinearOperator(propagated, tags)


def sandwich(
    A: lx.AbstractLinearOperator,
    P: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """The sandwich A Σ Aᵀ of two operators, keeping compatible structure.

        A Σ Aᵀ

    Dispatches on structure before any dense fallback:

    - Matched `gaussx.Kronecker` factors: (A_1 ⊗ A_2) (Σ_1 ⊗ Σ_2)
      (A_1 ⊗ A_2)ᵀ = A_1 Σ_1 A_1ᵀ ⊗ A_2 Σ_2 A_2ᵀ, factor by factor.
    - Matched `gaussx.BlockDiag` blocks: block by block.
    - A diagonal A = diag(d): diag(d) Σ diag(d), which stays diagonal when
      Σ is.
    - A diagonal Σ = diag(d): (A diag(d)) Aᵀ, without materialising Σ.

    The result is tagged symmetric (and positive semi-definite when Σ is).

    Args:
        A: Linear map A, M × N.
        P: Covariance Σ, N × N.

    Returns:
        The operator A Σ Aᵀ, M × M.

    Raises:
        ValueError: If ``P`` is not square or its size does not match the
            input size of ``A``.

    Examples:
        Matched Kronecker factors stay a Kronecker product:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> def psd(matrix):
        ...     return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> A = gaussx.Kronecker(psd(2.0 * jnp.eye(2)), psd(jnp.eye(3)))
        >>> covariance = gaussx.Kronecker(psd(jnp.eye(2)), psd(0.5 * jnp.eye(3)))
        >>> result = gaussx.sandwich(A, covariance)
        >>> type(result).__name__
        'Kronecker'
        >>> bool(jnp.allclose(result.as_matrix(), 2.0 * jnp.eye(6)))
        True

        A diagonal map scales the rows and columns of a dense covariance:

        >>> D = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> covariance = psd(jnp.array([[1.0, 0.5], [0.5, 1.0]]))
        >>> gaussx.sandwich(D, covariance).as_matrix().tolist()
        [[1.0, 1.0], [1.0, 4.0]]
    """
    _check_sandwich_shapes(A, P)
    tags = _sandwich_tags(P)

    if (
        isinstance(A, Kronecker)
        and isinstance(P, Kronecker)
        and len(A.operators) == len(P.operators)
        and all(
            a.in_size() == p.in_size() and p.in_size() == p.out_size()
            for a, p in zip(A.operators, P.operators, strict=True)
        )
    ):
        # A_1 Σ_1 A_1ᵀ ⊗ A_2 Σ_2 A_2ᵀ ⊗ …
        return Kronecker(
            *(sandwich(a, p) for a, p in zip(A.operators, P.operators, strict=True)),
            tags=tags,
        )

    if (
        isinstance(A, BlockDiag)
        and isinstance(P, BlockDiag)
        and len(A.operators) == len(P.operators)
        and all(
            a.in_size() == p.in_size() and p.in_size() == p.out_size()
            for a, p in zip(A.operators, P.operators, strict=True)
        )
    ):
        # blockdiag(A_1 Σ_1 A_1ᵀ, A_2 Σ_2 A_2ᵀ, …)
        return BlockDiag(
            *(sandwich(a, p) for a, p in zip(A.operators, P.operators, strict=True)),
            tags=tags,
        )

    if isinstance(A, lx.DiagonalLinearOperator):
        scale = lx.diagonal(A)
        if isinstance(P, lx.DiagonalLinearOperator):
            # diag(d) diag(s) diag(d) = diag(d ⊙ s ⊙ d)
            return lx.TaggedLinearOperator(
                lx.DiagonalLinearOperator(scale * lx.diagonal(P) * scale),
                tags,
            )
        covariance = P.as_matrix()
        # diag(d) Σ
        row_scaled = einx.multiply("row, row col -> row col", scale, covariance)
        # diag(d) Σ diag(d)
        scaled = einx.multiply("row col, col -> row col", row_scaled, scale)
        return lx.MatrixLinearOperator(scaled, tags)

    if isinstance(P, lx.DiagonalLinearOperator):
        variances = lx.diagonal(P)
        linear_map = A.as_matrix()
        # A diag(d)
        scaled_map = einx.multiply("out inp, inp -> out inp", linear_map, variances)
        # A diag(d) Aᵀ
        propagated = einsum(scaled_map, linear_map, "out inp, other inp -> out other")
        return lx.MatrixLinearOperator(propagated, tags)

    linear_map = A.as_matrix()
    covariance = P.as_matrix()
    # A Σ
    map_covariance = einsum(linear_map, covariance, "out inp, inp col -> out col")
    # A Σ Aᵀ
    propagated = einsum(map_covariance, linear_map, "out col, other col -> out other")
    return lx.MatrixLinearOperator(propagated, tags)


def _sandwich_tags(P: lx.AbstractLinearOperator) -> frozenset[object]:
    """The tags of A Σ Aᵀ: symmetric, and PSD when Σ is.

    Args:
        P: The covariance Σ.

    Returns:
        The lineax tags for the sandwich.
    """
    tags: set[object] = set()
    if lx.is_positive_semidefinite(P):
        tags.update({lx.symmetric_tag, lx.positive_semidefinite_tag})
    elif lx.is_symmetric(P):
        tags.add(lx.symmetric_tag)
    return frozenset(tags)


def _check_sandwich_shapes(
    A: lx.AbstractLinearOperator,
    P: lx.AbstractLinearOperator,
) -> None:
    """Check that Σ is square and A Σ Aᵀ is defined.

    Args:
        A: The linear map A.
        P: The covariance Σ.

    Raises:
        ValueError: If the shapes are incompatible.
    """
    if P.in_size() != P.out_size():
        raise ValueError(
            f"P must be square; got shape ({P.out_size()}, {P.in_size()})."
        )
    if A.in_size() != P.in_size():
        raise ValueError(
            "A and P shapes are incompatible for A @ P @ A.T; got "
            f"A shape ({A.out_size()}, {A.in_size()}) and "
            f"P shape ({P.out_size()}, {P.in_size()})."
        )


def trace_product(
    A: lx.AbstractLinearOperator,
    B: lx.AbstractLinearOperator,
) -> Float[Array, ""]:
    """Trace of a product of two operators, tr(A B), with structural dispatch.

        tr(A B) = ∑_ij A_ij B_ji

    Uses structure where it can to avoid materialising either operator:

    - Both diagonal: ∑_i A_ii B_ii, O(N).
    - One diagonal: the same sum against the other's diagonal, without
      materialising it.
    - Matched `gaussx.BlockDiag` (same block sizes): ∑_b tr(A_b B_b).
    - Matched `gaussx.Kronecker` (same factor sizes): ∏_f tr(A_f B_f).
    - Both `gaussx.LowRankUpdate` (A = A₀ + U_A D_A V_Aᵀ): the base term
      recurses, the two cross terms need one matrix-vector product per
      low-rank column, and the low-rank term costs O(N k_A k_B). For a pure
      low-rank operator (a zero diagonal base, as Nyström and random
      Fourier features produce) this replaces the O(N²) dense fallback.
    - Otherwise ∑_ij A_ij B_ji on the materialised matrices, O(N²).

    Args:
        A: Square operator A, N × N.
        B: Square operator B, N × N.

    Returns:
        The scalar tr(A B).

    Examples:
        Matched Kronecker factors multiply their traces:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A_1 = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]))
        >>> A_2 = lx.MatrixLinearOperator(jnp.eye(3))
        >>> A = gaussx.Kronecker(A_1, A_2)
        >>> B = gaussx.Kronecker(A_1, A_2)
        >>> bool(jnp.isclose(gaussx.trace_product(A, B), 10.0 * 3.0))
        True

        A diagonal operator against a dense one:

        >>> D = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> bool(jnp.isclose(gaussx.trace_product(D, A_1), 6.0))
        True
    """

    # Both diagonal: tr(A B) = ∑_i A_ii B_ii, an O(N) inner product.
    if isinstance(A, lx.DiagonalLinearOperator) and isinstance(
        B, lx.DiagonalLinearOperator
    ):
        # ∑_i A_ii B_ii
        return jnp.sum(lx.diagonal(A) * lx.diagonal(B))

    # One diagonal: tr(D B) = ∑_i D_ii B_ii.
    if isinstance(A, lx.DiagonalLinearOperator):
        # ∑_i A_ii B_ii
        return jnp.sum(lx.diagonal(A) * diag(B))
    if isinstance(B, lx.DiagonalLinearOperator):
        # ∑_i A_ii B_ii
        return jnp.sum(diag(A) * lx.diagonal(B))

    if (
        isinstance(A, BlockDiag)
        and isinstance(B, BlockDiag)
        and len(A.operators) == len(B.operators)
        and all(
            a.in_size() == b.in_size()
            for a, b in zip(A.operators, B.operators, strict=True)
        )
    ):
        # tr(A_b B_b) per block
        parts = [
            trace_product(a, b) for a, b in zip(A.operators, B.operators, strict=True)
        ]
        # tr(A B) = ∑_b tr(A_b B_b)
        return jnp.sum(jnp.stack(parts))

    if (
        isinstance(A, Kronecker)
        and isinstance(B, Kronecker)
        and len(A.operators) == len(B.operators)
        and all(
            a.in_size() == b.in_size()
            for a, b in zip(A.operators, B.operators, strict=True)
        )
    ):
        # tr(A_f B_f) per factor
        parts = [
            trace_product(a, b) for a, b in zip(A.operators, B.operators, strict=True)
        ]
        # tr((A_1 ⊗ A_2 ⊗ …) (B_1 ⊗ B_2 ⊗ …)) = ∏_f tr(A_f B_f)
        return jnp.prod(jnp.stack(parts))

    if isinstance(A, LowRankUpdate) and isinstance(B, LowRankUpdate):
        return _trace_product_low_rank(A, B)

    return _trace_product_dense(A, B)


def _trace_product_low_rank(A: LowRankUpdate, B: LowRankUpdate) -> Float[Array, ""]:
    """tr(A B) for two low-rank updates, term by term.

        tr(A B) = tr(A₀ B₀) + tr(A₀ U_B D_B V_Bᵀ) + tr(U_A D_A V_Aᵀ B₀)
                  + tr(U_A D_A V_Aᵀ U_B D_B V_Bᵀ)

    with A = A₀ + U_A D_A V_Aᵀ and B = B₀ + U_B D_B V_Bᵀ, where the rank of A
    is k_A and that of B is k_B.

    Algorithm:
        1. Recurse on the bases for tr(A₀ B₀) (O(N) for diagonal ones).
        2. Cross terms: tr(A₀ U_B D_B V_Bᵀ) = ∑_l d_B,l (V_Bᵀ A₀ U_B)_ll, one
           matrix-vector product with A₀ per column of U_B, and alike for
           tr(U_A D_A V_Aᵀ B₀).
        3. Low-rank term: ∑_kl d_A,k M_kl d_B,l N_lk with M = V_Aᵀ U_B and
           N = V_Bᵀ U_A.

    Args:
        A: The low-rank update A.
        B: The low-rank update B.

    Returns:
        The scalar tr(A B).
    """
    base_A, base_B = A.base, B.base
    # M = V_Aᵀ U_B
    cross_factor_AB = einsum(A.V, B.U, "row rank_a, row rank_b -> rank_a rank_b")
    # N = V_Bᵀ U_A
    cross_factor_BA = einsum(B.V, A.U, "row rank_b, row rank_a -> rank_b rank_a")
    # D_A M
    scaled_rows = einx.multiply(
        "rank_a rank_b, rank_a -> rank_a rank_b", cross_factor_AB, A.d
    )
    # D_A M D_B
    scaled = einx.multiply("rank_a rank_b, rank_b -> rank_a rank_b", scaled_rows, B.d)
    # ∑_kl d_A,k M_kl d_B,l N_lk
    low_low = einsum(scaled, cross_factor_BA, "rank_a rank_b, rank_b rank_a -> ")
    # A₀ U_B, one structural mv per column
    base_A_U_B = jax.vmap(base_A.mv, in_axes=1, out_axes=1)(B.U)
    # B₀ U_A, one structural mv per column
    base_B_U_A = jax.vmap(base_B.mv, in_axes=1, out_axes=1)(A.U)
    # (V_Bᵀ A₀ U_B)_ll = ∑_n V_B,nl (A₀ U_B)_nl
    diag_A = reduce(B.V * base_A_U_B, "row rank_b -> rank_b", "sum")
    # tr(A₀ U_B D_B V_Bᵀ) = ∑_l d_B,l (V_Bᵀ A₀ U_B)_ll
    cross_A = einsum(diag_A, B.d, "rank_b, rank_b -> ")
    # (V_Aᵀ B₀ U_A)_kk = ∑_n V_A,nk (B₀ U_A)_nk
    diag_B = reduce(A.V * base_B_U_A, "row rank_a -> rank_a", "sum")
    # tr(U_A D_A V_Aᵀ B₀) = ∑_k d_A,k (V_Aᵀ B₀ U_A)_kk
    cross_B = einsum(diag_B, A.d, "rank_a, rank_a -> ")
    # tr(A B) = tr(A₀ B₀) + both cross terms + the low-rank term
    return trace_product(base_A, base_B) + cross_A + cross_B + low_low


def _trace_product_dense(
    A: lx.AbstractLinearOperator, B: lx.AbstractLinearOperator
) -> Float[Array, ""]:
    """Dense fallback for tr(A B) = ∑_ij A_ij B_ji, elementwise.

    Args:
        A: The operator A.
        B: The operator B.

    Returns:
        The scalar tr(A B).
    """
    dense_A = A.as_matrix()
    dense_B = B.as_matrix()
    # Bᵀ
    dense_B_transpose = rearrange(dense_B, "row col -> col row")
    # tr(A B) = ∑_ij A_ij B_ji
    return reduce(dense_A * dense_B_transpose, "row col -> ", "sum")


def diag_conditional_variance(
    K_XX_diag: Float[Array, " N"],
    K_XZ: Float[Array, "N M"],
    A_X: Float[Array, "N M"],
) -> Float[Array, " N"]:
    """Diagonal of the Schur complement, max([K_XX]_nn − [A_X K_ZX]_nn, 0).

        σ_n² = max([K_XX]_nn − [A_X K_ZX]_nn, 0),   A_X = K_XZ K_ZZ⁻¹

    A thin wrapper around `gaussx.conditional_variance` without a
    variational covariance; call that directly when the variational term
    with Σ_Z is needed too.

    Args:
        K_XX_diag: Prior variances [K_XX]_nn, the diagonal of K_XX.
        K_XZ: Cross-kernel matrix K_XZ between the inputs and the inducing
            inputs.
        A_X: Projection A_X = K_XZ K_ZZ⁻¹.

    Returns:
        The conditional variances σ_n².

    Examples:
        Two inputs and one inducing input with K_ZZ = 2:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> K_XZ = jnp.array([[1.0], [0.5]])
        >>> variance = gaussx.diag_conditional_variance(
        ...     jnp.array([2.0, 2.0]), K_XZ, K_XZ / 2.0
        ... )
        >>> bool(jnp.allclose(variance, jnp.array([1.5, 1.875])))
        True
    """
    return _conditional_variance(K_XX_diag, K_XZ, A_X)


def solve_columns(
    operator: lx.AbstractLinearOperator,
    matrix: Float[Array, "N K"],
    *,
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, "N K"]:
    """Solve A X = B column by column, with structured dispatch per column.

        X = A⁻¹ B,   x_j = A⁻¹ b_j

    Use it when A is an operator with a cheap per-vector solve (Cholesky,
    diagonal, block-diagonal, …).

    Args:
        operator: Square operator A, N × N.
        matrix: Right-hand side B, one column b_j per system.
        solver: Optional solver strategy; ``None`` uses structural dispatch.

    Returns:
        The solution X = A⁻¹ B.

    Examples:
        A diagonal operator divides each row:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> B = jnp.array([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
        >>> gaussx.solve_columns(A, B).tolist()
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    """
    if solver is not None:
        # x_j = A⁻¹ b_j through the strategy
        return jax.vmap(
            lambda col: dispatch_solve(operator, col, solver),
            in_axes=1,
            out_axes=1,
        )(matrix)
    # x_j = A⁻¹ b_j through structural dispatch
    return jax.vmap(
        lambda col: solve(operator, col),
        in_axes=1,
        out_axes=1,
    )(matrix)


def solve_matrix(
    operator: lx.AbstractLinearOperator,
    matrix: Float[Array, "N K"],
    *,
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, "N K"]:
    """Solve A X = B with one factorisation for the whole matrix right-hand side.

        X = A⁻¹ B = L⁻ᵀ L⁻¹ B,   A = L Lᵀ

    When ``solver`` is ``None`` and A is positive semi-definite without a
    structured solve (dense, possibly behind tagged or scalar wrappers, or
    a composition or sum), this factors A = L Lᵀ once with
    `gaussx.cholesky` and runs one ``cho_solve`` on all of B, avoiding the
    per-column refactorisation of `solve_columns`.

    Structured operators (``LowRankUpdate``, ``Kronecker``, ``BlockDiag``,
    …), operators that are not PSD and a custom ``solver`` go to
    `solve_columns`, whose per-column structured `gaussx.solve` (Woodbury,
    Roth's lemma, per block) is cheaper than densifying a Cholesky factor
    (gh-356).

    Args:
        operator: Square operator A, N × N.
        matrix: Right-hand side B, one column per system.
        solver: Optional solver strategy; when given, the solve runs column
            by column through `solve_columns`.

    Returns:
        The solution X = A⁻¹ B.

    Examples:
        A dense PSD operator is factored once:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> dense = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(dense, lx.positive_semidefinite_tag)
        >>> B = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        >>> X = gaussx.solve_matrix(A, B)
        >>> bool(jnp.allclose(dense @ X, B, atol=1e-5))
        True
    """
    if (
        solver is None
        and lx.is_positive_semidefinite(operator)
        and not _has_structured_solve(operator)
    ):
        # A = L Lᵀ, once
        factor = cholesky(operator).as_matrix()
        # X = L⁻ᵀ L⁻¹ B
        return jax.scipy.linalg.cho_solve((factor, True), matrix)
    return solve_columns(operator, matrix, solver=solver)


_STRUCTURED_SOLVE = (
    lx.IdentityLinearOperator,
    lx.DiagonalLinearOperator,
    BlockDiag,
    Kronecker,
    LowRankUpdate,
    KroneckerSum,
    BlockTriDiag,
    DiagonalizedOperator,
    SparseOperator,
)


def _has_structured_solve(operator: lx.AbstractLinearOperator) -> bool:
    """Whether `gaussx.solve` has a structured rule under the scalar wrappers.

    Dense operators, compositions (whose factors may be rectangular) and
    sums keep the single full-system factorisation (gh-356).

    Args:
        operator: The operator A.

    Returns:
        ``True`` when A, unwrapped, has a structured solve.
    """
    while isinstance(
        operator,
        lx.TaggedLinearOperator
        | lx.MulLinearOperator
        | lx.DivLinearOperator
        | lx.NegLinearOperator,
    ):
        operator = operator.operator
    if not isinstance(operator, _STRUCTURED_SOLVE):
        return False
    if isinstance(operator, BlockDiag | Kronecker):
        return all(op.in_size() == op.out_size() for op in operator.operators)
    if isinstance(operator, LowRankUpdate):
        return operator.base.in_size() == operator.base.out_size()
    return True


def solve_rows(
    operator: lx.AbstractLinearOperator,
    matrix: Float[Array, "K N"],
    *,
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, "K N"]:
    """Solve A xᵢ = bᵢ for each row bᵢ of a matrix B, with one solve per row.

        X = B A⁻ᵀ,   xᵢ = A⁻¹ bᵢ

    For a symmetric A this is B A⁻¹, as in the Kalman gain
    G = Σ_xy Σ_y⁻¹, the Schur complement and GP prediction.

    Args:
        operator: Square operator A, N × N.
        matrix: Right-hand sides B, one row bᵢ per system.
        solver: Optional solver strategy; ``None`` uses structural dispatch.

    Returns:
        The solutions X, one row xᵢ per row of B.

    Examples:
        A symmetric operator, so X = B A⁻¹:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> dense = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(dense, lx.positive_semidefinite_tag)
        >>> B = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> X = gaussx.solve_rows(A, B)
        >>> X.shape
        (3, 2)
        >>> bool(jnp.allclose(X @ dense, B, atol=1e-5))
        True
    """
    if solver is not None:
        # xᵢ = A⁻¹ bᵢ through the strategy
        return jax.vmap(
            lambda row: dispatch_solve(operator, row, solver),
        )(matrix)
    # xᵢ = A⁻¹ bᵢ through structural dispatch
    return jax.vmap(lambda row: solve(operator, row))(matrix)
