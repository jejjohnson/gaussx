"""Randomized interpolative (column ID) and CUR decompositions of an M × N operator A.

The column ID A ≈ A_:,J X keeps k actual columns J of A with an interpolation matrix
X, X_:,J = I_k. CUR A ≈ C U R adds k actual rows I, with C = A_:,J, R = A_I,: and a
linking matrix U. P is the column permutation of a pivoted QR, whose triangular factor
has the blocks R₁₁ (k × k) and R₁₂; these are not the CUR rows R.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float, Int

from gaussx._einx import einsum, rearrange
from gaussx._randomized._range_finder import qb


class ColumnID(NamedTuple):
    """Column interpolative decomposition A ≈ A_:,J X.

    Attributes:
        columns: The k column indices J.
        interpolation: The k × N matrix X, with X_:,J = I_k.
    """

    columns: Int[Array, " k"]
    interpolation: Float[Array, "k n"]


class CUR(NamedTuple):
    """CUR decomposition A ≈ C U R, with C = A_:,J and R = A_I,:.

    Attributes:
        columns: The k column indices J.
        rows: The k row indices I.
        C: The selected columns A_:,J, M × k.
        U: The linking matrix, k × k.
        R: The selected rows A_I,:, k × N.
    """

    columns: Int[Array, " k"]
    rows: Int[Array, " k"]
    C: Float[Array, "m k"]
    U: Float[Array, "k k"]
    R: Float[Array, "k n"]


def _interpolative(
    projection: Float[Array, "l n"], rank: int
) -> tuple[Int[Array, " k"], Float[Array, "k n"]]:
    """Column ID of a short, wide B by column-pivoted QR.

        B P = Q_B [R₁₁ R₁₂; 0 R₂₂],   J = P_1:k,   X = [I_k  R₁₁⁻¹ R₁₂] Pᵀ

    B must carry A's singular values (e.g. B = Qᵀ A), not merely span its row space:
    the truncation at k is not invariant under an invertible left factor.

    Args:
        projection: The ℓ × N matrix B to decompose.
        rank: Number of columns k.

    Returns:
        Tuple ``(columns, interpolation)`` holding J and X.
    """
    in_size = projection.shape[1]
    # B P = Q_B [R₁₁ R₁₂; 0 R₂₂]
    _, triangular, permutation = jax.scipy.linalg.qr(
        projection, mode="economic", pivoting=True
    )
    # J = P_1:k, and the rest of P
    skeleton, redundant = permutation[:rank], permutation[rank:]
    # 0, k × N
    interpolation = jnp.zeros((rank, in_size), dtype=projection.dtype)
    # I_k
    identity = jnp.eye(rank)
    # X_:,J = I_k
    interpolation = interpolation.at[:, skeleton].set(identity)
    if rank < in_size:
        # R₁₁⁻¹ R₁₂
        coefficients = jax.scipy.linalg.solve_triangular(
            triangular[:rank, :rank], triangular[:rank, rank:], lower=False
        )
        # X_:,P_k+1:N = R₁₁⁻¹ R₁₂
        interpolation = interpolation.at[:, redundant].set(coefficients)
    return skeleton, interpolation


def _check_rank(op: lx.AbstractLinearOperator, rank: int) -> None:
    """Reject a rank k outside [1, min(M, N)].

    Args:
        op: Operator A of size M × N.
        rank: Number of columns k.

    Raises:
        ValueError: If k is outside [1, min(M, N)].
    """
    largest_rank = min(op.out_size(), op.in_size())
    if not 1 <= rank <= largest_rank:
        raise ValueError(
            f"rank must be in [1, min(m, n)] = [1, {largest_rank}], got {rank}."
        )


def column_id(
    op: lx.AbstractLinearOperator,
    rank: int,
    *,
    oversample: int = 10,
    n_power_iter: int = 2,
    key: jax.Array | None = None,
) -> ColumnID:
    """Randomized column interpolative decomposition (Voronin & Martinsson, 2017).

        A ≈ A_:,J X,   X_:,J = I_k

    The randomized QB factorisation A ≈ Q B, B = Qᵀ A (`gaussx.qb`), keeps A's dominant
    singular values in B, so the column ID of B by column-pivoted QR serves for A:

        B P = Q_B [R₁₁ R₁₂; 0 R₂₂],   J = P_1:k,   X = [I_k  R₁₁⁻¹ R₁₂] Pᵀ,
        A ≈ Q B ≈ Q B_:,J X ≈ A_:,J X

    With a strong rank-revealing QR, ‖A − A_:,J X‖₂ ≤ (1 + √(1 + 4k(N − k)))
    ‖A − Q Qᵀ A‖₂ (Halko, Martinsson & Tropp, 2011, §5.2); LAPACK's column pivoting
    meets it in practice.

    Unlike a truncated SVD, the factors are actual columns of A, so they keep its
    sparsity, non-negativity and units (e.g. representative stations or time steps).

    Algorithm:
        1. Factorise A ≈ Q B with ℓ = k + p rows, `gaussx.qb`.
        2. Take the column-pivoted QR of B and read J from its permutation P.
        3. Set X_:,J = I_k and fill the rest with R₁₁⁻¹ R₁₂.

    Args:
        op: Operator A of size M × N; may be matrix-free (touched through
            (k + p)(1 + 2q) matvecs with A or Aᵀ).
        rank: Number of columns k, at most min(M, N).
        oversample: Extra sketch rows p (see `gaussx.qb`).
        n_power_iter: Power iterations q for slowly decaying spectra.
        key: PRNG key for the sketch. ``None`` means ``jax.random.PRNGKey(0)``.

    Returns:
        A `ColumnID` ``(columns, interpolation)``.

    Raises:
        ValueError: If ``rank`` is outside [1, min(M, N)], or on the `gaussx.qb`
            errors.

    Examples:
        A rank-five matrix is exact on five of its columns:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> left = jr.normal(jr.key(0), (60, 5))
        >>> matrix = left @ jr.normal(jr.key(1), (5, 40))
        >>> columns, interpolation = gaussx.column_id(
        ...     lx.MatrixLinearOperator(matrix), 5
        ... )
        >>> columns.shape, interpolation.shape
        ((5,), (5, 40))
        >>> reconstruction = matrix[:, columns] @ interpolation
        >>> bool(jnp.allclose(reconstruction, matrix, atol=1e-3))
        True

        A ``rank`` above min(M, N) is rejected:

        >>> gaussx.column_id(lx.MatrixLinearOperator(matrix), 41)
        Traceback (most recent call last):
        ...
        ValueError: rank must be in [1, min(m, n)] = [1, 40], got 41.

    References:
        Voronin, S. & Martinsson, P.-G. (2017). Efficient algorithms for CUR and
        interpolative matrix decompositions. *Advances in Computational Mathematics*,
        43(3), 495-516.
    """
    _check_rank(op, rank)
    # A ≈ Q B
    _, projection = qb(
        op, rank, oversample=oversample, n_power_iter=n_power_iter, key=key
    )
    # B ≈ B_:,J X
    columns, interpolation = _interpolative(projection, rank)
    return ColumnID(columns, interpolation)


def _select(
    op: lx.AbstractLinearOperator, indices: Int[Array, " k"]
) -> Float[Array, "k m"]:
    """Columns of A at the given indices, as rows, through matvecs.

        A e_i, for i ∈ indices

    Args:
        op: Operator A.
        indices: Indices i of the unit vectors e_i.

    Returns:
        The vectors A e_i, one per row.
    """
    dtype = op.in_structure().dtype
    # e_i
    unit = jax.nn.one_hot(indices, op.in_size(), dtype=dtype)
    # A e_i
    return jax.vmap(op.mv)(unit)


def cur(
    op: lx.AbstractLinearOperator,
    rank: int,
    *,
    oversample: int = 10,
    n_power_iter: int = 2,
    key: jax.Array | None = None,
) -> CUR:
    """Randomized CUR decomposition (Voronin & Martinsson, 2017, §4).

        A ≈ C U R,   C = A_:,J,   R = A_I,:,   U = X R⁺

    C U R = C X R⁺ R is the column ID with X replaced by its projection onto the row
    space of R. The error is that of the column ID, amplified by a factor that depends
    on the row selection's conditioning, ‖R⁺‖ (Voronin & Martinsson, 2017, §4); it is
    exact when rank(A) ≤ k.

    Algorithm:
        1. Take the column ID A ≈ A_:,J X with `column_id`.
        2. Read C = A_:,J (k matvecs).
        3. Take the column-pivoted QR of Cᵀ, whose permutation gives I = P_1:k.
        4. Read R = A_I,: (k transpose matvecs) and set U = X R⁺.

    Args:
        op: Operator A of size M × N; may be matrix-free.
        rank: Number of columns and rows k, at most min(M, N).
        oversample: Extra sketch rows p for the column ID.
        n_power_iter: Power iterations q for the column ID.
        key: PRNG key for the sketch. ``None`` means ``jax.random.PRNGKey(0)``.

    Returns:
        A `CUR` ``(columns, rows, C, U, R)``.

    Raises:
        ValueError: If ``rank`` is outside [1, min(M, N)], or on the `gaussx.qb`
            errors.

    Examples:
        A rank-five matrix is exact on five of its columns and rows:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> left = jr.normal(jr.key(0), (60, 5))
        >>> matrix = left @ jr.normal(jr.key(1), (5, 40))
        >>> decomposition = gaussx.cur(lx.MatrixLinearOperator(matrix), 5)
        >>> reconstruction = decomposition.C @ decomposition.U @ decomposition.R
        >>> bool(jnp.allclose(reconstruction, matrix, atol=1e-3))
        True
        >>> bool(jnp.array_equal(decomposition.R, matrix[decomposition.rows]))
        True

        A non-positive ``rank`` is rejected:

        >>> gaussx.cur(lx.MatrixLinearOperator(matrix), 0)
        Traceback (most recent call last):
        ...
        ValueError: rank must be in [1, min(m, n)] = [1, 40], got 0.

    References:
        Voronin, S. & Martinsson, P.-G. (2017). Efficient algorithms for CUR and
        interpolative matrix decompositions. *Advances in Computational Mathematics*,
        43(3), 495-516.
    """
    # A ≈ A_:,J X
    columns, interpolation = column_id(
        op, rank, oversample=oversample, n_power_iter=n_power_iter, key=key
    )
    # Cᵀ, the selected columns as rows, k × M
    columns_as_rows = _select(op, columns)
    # Cᵀ P = Q_C R_C
    _, _, permutation = jax.scipy.linalg.qr(
        columns_as_rows, mode="economic", pivoting=True
    )
    # I = P_1:k
    rows = permutation[:rank]
    op_transpose = op.transpose()
    # R = A_I,:
    selected_rows = _select(op_transpose, rows)
    # R⁺
    pseudo_inverse = jnp.linalg.pinv(selected_rows)
    # U = X R⁺
    linking = einsum(interpolation, pseudo_inverse, "k n, n row -> k row")
    # C
    selected_columns = rearrange(columns_as_rows, "k m -> m k")
    return CUR(columns, rows, selected_columns, linking, selected_rows)
