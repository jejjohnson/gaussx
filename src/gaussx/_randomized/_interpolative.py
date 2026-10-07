"""Randomized interpolative (column ID) and CUR decompositions (G17)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lineax as lx
from jaxtyping import Array, Float, Int

from gaussx._einx import einsum, rearrange
from gaussx._randomized._range_finder import qb


class ColumnID(NamedTuple):
    r"""Column interpolative decomposition $A \approx A_{:,J}\,X$.

    Attributes:
        columns: The $k$ column indices $J$, shape ``(k,)``.
        interpolation: $X$, shape ``(k, n)``, with $X_{:,J} = I_k$.
    """

    columns: Int[Array, " k"]
    interpolation: Float[Array, "k n"]


class CUR(NamedTuple):
    r"""CUR decomposition $A \approx C\,U\,R$, $C = A_{:,J}$, $R = A_{I,:}$.

    Attributes:
        columns: The $k$ column indices $J$, shape ``(k,)``.
        rows: The $k$ row indices $I$, shape ``(k,)``.
        C: The selected columns $A_{:,J}$, shape ``(m, k)``.
        U: The linking matrix, shape ``(k, k)``.
        R: The selected rows $A_{I,:}$, shape ``(k, n)``.
    """

    columns: Int[Array, " k"]
    rows: Int[Array, " k"]
    C: Float[Array, "m k"]
    U: Float[Array, "k k"]
    R: Float[Array, "k n"]


def _interpolative(
    Y: Float[Array, "l n"], rank: int
) -> tuple[Int[Array, " k"], Float[Array, "k n"]]:
    """Column ID of a short, wide ``Y`` by column-pivoted QR.

    ``Y`` must carry ``A``'s singular values (e.g. ``B = QᵀA``), not merely
    span its row space: the truncation at ``rank`` is not invariant under an
    invertible left factor.
    """
    n = Y.shape[1]
    _, R, perm = jsl.qr(Y, mode="economic", pivoting=True)
    skeleton, redundant = perm[:rank], perm[rank:]
    X = jnp.zeros((rank, n), dtype=Y.dtype).at[:, skeleton].set(jnp.eye(rank))
    if rank < n:
        T = jsl.solve_triangular(R[:rank, :rank], R[:rank, rank:], lower=False)
        X = X.at[:, redundant].set(T)
    return skeleton, X


def _check_rank(op: lx.AbstractLinearOperator, rank: int) -> None:
    m, n = op.out_size(), op.in_size()
    if not 1 <= rank <= min(m, n):
        raise ValueError(
            f"rank must be in [1, min(m, n)] = [1, {min(m, n)}], got {rank}."
        )


def column_id(
    op: lx.AbstractLinearOperator,
    rank: int,
    *,
    oversample: int = 10,
    n_power_iter: int = 2,
    key: jax.Array | None = None,
) -> ColumnID:
    r"""Randomized column interpolative decomposition (Voronin & Martinsson, 2017).

    Finds $k$ actual columns $J$ of $A$ and an interpolation matrix $X$
    with $X_{:,J} = I_k$ such that $A \approx A_{:,J}X$. The randomized QB
    factorisation $A \approx QB$, $B = Q^\top A \in \mathbb R^{\ell \times
    n}$ with $\ell = k + p$ (`gaussx.qb`), keeps $A$'s dominant singular
    values in $B$, so the column ID of $B$ by column-pivoted QR,

    $$
    B P = Q_B\,\begin{bmatrix} R_{11} & R_{12} \\ 0 & R_{22}\end{bmatrix},
    \qquad J = P_{1:k}, \qquad
    X = [\,I_k\ \ R_{11}^{-1}R_{12}\,]\,P^\top,
    $$

    serves for $A$: $A \approx QB \approx QB_{:,J}X \approx A_{:,J}X$. With a
    strong rank-revealing QR, $\|A - A_{:,J}X\|_2 \le (1 + \sqrt{1 +
    4k(n-k)})\,\|A - QQ^\top A\|_2$ (Halko, Martinsson & Tropp, 2011,
    §5.2); LAPACK's column pivoting meets it in practice.

    ```text
    Q, B = qb(A, k; oversample, n_power_iter, key)      # B = Qᵀ A, (k + p) × n
    _, R, P = qr(B, pivoting=True)
    J = P[:k];  X[:, P[:k]] = I;  X[:, P[k:]] = R₁₁⁻¹ R₁₂
    ```

    Unlike a truncated SVD, the factors are actual columns of $A$, so they
    keep its sparsity, non-negativity and units (e.g. representative
    stations or time steps).

    Args:
        op: Operator $A$ of shape ``(m, n)``; may be matrix-free (touched
            through ``(k + p)(1 + 2q)`` matvecs with $A$ or $A^\top$).
        rank: Number of columns $k$, at most ``min(m, n)``.
        oversample: Extra sketch rows $p$ (see `gaussx.qb`).
        n_power_iter: Power iterations $q$ for slowly decaying spectra.
        key: PRNG key for the sketch. ``None`` means
            ``jax.random.PRNGKey(0)``.

    Returns:
        A `ColumnID` ``(columns, interpolation)``.

    Raises:
        ValueError: If ``rank`` is outside ``[1, min(m, n)]``, or on the
            ``qb`` errors.

    References:
        Voronin, S. & Martinsson, P.-G. (2017). Efficient algorithms for CUR
        and interpolative matrix decompositions. *Advances in Computational
        Mathematics*, 43(3), 495-516.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr, lineax as lx
        >>> import gaussx as gx
        >>> A = jr.normal(jr.key(0), (60, 5)) @ jr.normal(jr.key(1), (5, 40))
        >>> cols, X = gx.column_id(lx.MatrixLinearOperator(A), 5)
        >>> bool(jnp.allclose(A[:, cols] @ X, A, atol=1e-3))
        True
    """
    _check_rank(op, rank)
    _, B = qb(op, rank, oversample=oversample, n_power_iter=n_power_iter, key=key)
    columns, interpolation = _interpolative(B, rank)
    return ColumnID(columns, interpolation)


def _select(
    op: lx.AbstractLinearOperator, indices: Int[Array, " k"]
) -> Float[Array, "k m"]:
    """``op`` applied to the unit vectors ``e_i``, ``i ∈ indices`` (rows out)."""
    dtype = op.in_structure().dtype
    unit = jax.nn.one_hot(indices, op.in_size(), dtype=dtype)
    return jax.vmap(op.mv)(unit)


def cur(
    op: lx.AbstractLinearOperator,
    rank: int,
    *,
    oversample: int = 10,
    n_power_iter: int = 2,
    key: jax.Array | None = None,
) -> CUR:
    r"""Randomized CUR decomposition (Voronin & Martinsson, 2017, §4).

    $A \approx C U R$ with $C = A_{:,J}$ and $R = A_{I,:}$ actual columns
    and rows of $A$:

    1. column ID, $A \approx A_{:,J}X$ (`column_id`);
    2. row ID of $C = A_{:,J}$, by column-pivoted QR of $C^\top$: $I =
       P_{1:k}$, and $R = A_{I,:}$;
    3. $U = X R^{+}$, so $CUR = C X R^{+}R$, the column ID with $X$ replaced
       by its projection onto the row space of $R$.

    ```text
    J, X = column_id(A, k)
    C = A[:, J]                          # k matvecs
    _, _, P = qr(Cᵀ, pivoting=True);  I = P[:k]
    R = A[I, :]                          # k transpose-matvecs
    U = X R⁺
    ```

    The error is that of the column ID, amplified by a factor that depends
    on the row selection's conditioning, $\|R^{+}\|$ (Voronin & Martinsson,
    2017, §4); it is exact when $\operatorname{rank}(A) \le k$.

    Args:
        op: Operator $A$ of shape ``(m, n)``; may be matrix-free.
        rank: Number of columns and rows $k$, at most ``min(m, n)``.
        oversample: Extra sketch rows $p$ for the column ID.
        n_power_iter: Power iterations $q$ for the column ID.
        key: PRNG key for the sketch. ``None`` means
            ``jax.random.PRNGKey(0)``.

    Returns:
        A `CUR` ``(columns, rows, C, U, R)``.

    Raises:
        ValueError: If ``rank`` is outside ``[1, min(m, n)]``.

    References:
        Voronin, S. & Martinsson, P.-G. (2017). Efficient algorithms for CUR
        and interpolative matrix decompositions. *Advances in Computational
        Mathematics*, 43(3), 495-516.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr, lineax as lx
        >>> import gaussx as gx
        >>> A = jr.normal(jr.key(0), (60, 5)) @ jr.normal(jr.key(1), (5, 40))
        >>> d = gx.cur(lx.MatrixLinearOperator(A), 5)
        >>> bool(jnp.allclose(d.C @ d.U @ d.R, A, atol=1e-3))
        True
        >>> bool(jnp.array_equal(d.R, A[d.rows]))
        True
    """
    columns, X = column_id(
        op, rank, oversample=oversample, n_power_iter=n_power_iter, key=key
    )
    Ct = _select(op, columns)  # (k, m): the selected columns, as rows
    _, _, perm = jsl.qr(Ct, mode="economic", pivoting=True)
    rows = perm[:rank]
    R = _select(op.transpose(), rows)  # (k, n)
    U = einsum(X, jnp.linalg.pinv(R), "k n, n j -> k j")
    return CUR(columns, rows, rearrange(Ct, "k m -> m k"), U, R)
