"""Randomly pivoted partial Cholesky (RPCholesky)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from gaussx._einx import reduce


def rp_cholesky(
    diagonal: Float[Array, " N"],
    column: Callable[[Int[Array, ""]], Float[Array, " N"]],
    rank: int,
    *,
    pivoting: Literal["random", "greedy"] = "random",
    block_size: int = 1,
    key: jax.Array | None = None,
) -> tuple[Float[Array, "N k"], Int[Array, " k"]]:
    r"""Randomly pivoted partial Cholesky of a PSD matrix ``A``.

    Builds ``F`` with ``F Fᵀ ≈ A`` one pivot at a time, touching ``A`` only
    through its diagonal and ``rank`` of its columns (Chen, Epperly, Tropp &
    Webber, 2023). At step ``i`` the pivot ``s`` is drawn with probability
    proportional to the residual diagonal $d_s = [A - FF^\top]_{ss}$, the
    variance not yet explained, and then

    $$
    g = A_{:,s} - F F_{s,:}^\top,\qquad F_{:,i} = g / \sqrt{g_s}.
    $$

    With $k \ge r/\varepsilon + r\log(1/(\varepsilon\eta))$ pivots,
    $\mathbb E\,\operatorname{tr}(A - FF^\top) \le
    (1+\varepsilon)\operatorname{tr}(A - [\![A]\!]_r)$, where
    $\eta = \operatorname{tr}(A - [\![A]\!]_r)/\operatorname{tr}A$. Greedy
    pivoting (``argmax`` of the residual diagonal) has no such guarantee and
    chases outliers. The cost is ``rank`` column evaluations and
    $O(N k^2)$ flops.

    The returned pivots ``S`` make ``F Fᵀ = A[:, S] A[S, S]⁺ A[S, :]``, the
    column Nyström approximation on those columns, so they double as
    landmark (inducing-point) indices. The diagonal and ``column`` can come
    from a kernel evaluated on the fly, so ``A`` is never formed: for 10⁶
    points and ``rank=1000`` this is 1000 kernel columns.

    The residual is guarded like LAPACK ``?pstrf``: once the chosen pivot
    falls below ``N · eps · max|diag A|`` the numerical rank is exhausted,
    and that and every later column of ``F`` is exactly zero, with pivot
    ``-1`` (gh-236, gh-237). Filter with ``pivots[pivots >= 0]``.

    Args:
        diagonal: Diagonal of the PSD matrix ``A``, shape ``(N,)``.
        column: Callable returning column ``s`` of ``A``, shape ``(N,)``, for
            a scalar integer index ``s`` (a traced value inside the loop).
        rank: Number of pivots ``k``.
        pivoting: ``"random"`` samples ``s ∝ max(d, 0)``; ``"greedy"`` takes
            ``argmax(d)``, the classic pivoted Cholesky.
        block_size: Pivots per step. Only ``1`` is implemented; the blocked
            variant (Epperly, Tropp & Webber, 2024) is a follow-up.
        key: PRNG key for ``"random"`` pivoting. ``None`` means
            ``jax.random.PRNGKey(0)``. Ignored by ``"greedy"``.

    Returns:
        ``(F, pivots)``: the factor, shape ``(N, k)``, and the pivot indices,
        shape ``(k,)``, in the order chosen (``-1`` past the numerical rank).

    Raises:
        ValueError: If ``pivoting`` is not ``"random"`` or ``"greedy"``.
        NotImplementedError: If ``block_size != 1``.

    Examples:
        Pick 20 landmarks from 1000 points without forming the kernel matrix.

        >>> import jax.numpy as jnp, jax.random as jr, gaussx
        >>> X = jr.normal(jr.key(0), (1000,))
        >>> def column(j):
        ...     return jnp.exp(-0.5 * (X - X[j]) ** 2)
        >>> F, pivots = gaussx.rp_cholesky(jnp.ones(1000), column, 20, key=jr.key(1))
        >>> F.shape, pivots.shape
        ((1000, 20), (20,))
        >>> Z = X[pivots]  # landmarks for Nyström / Falkon / SVGP
    """
    if pivoting not in ("random", "greedy"):
        raise ValueError(f"pivoting must be 'random' or 'greedy', got {pivoting!r}")
    if block_size != 1:
        raise NotImplementedError("rp_cholesky supports only block_size=1 for now")
    return _pivoted_cholesky(diagonal, column, rank, pivoting, key)


def _pivoted_cholesky(
    diagonal: Float[Array, " N"],
    column: Callable[[Int[Array, ""]], Float[Array, " N"]],
    rank: int,
    pivoting: Literal["random", "greedy"],
    key: jax.Array | None,
    *,
    approximate_diagonal: bool = False,
) -> tuple[Float[Array, "N k"], Int[Array, " k"]]:
    """The `rp_cholesky` loop.

    With ``approximate_diagonal`` the diagonal only *selects* pivots: each
    pivot's value is read from its exact column instead, and the diagonal
    entry is corrected to it, so the factor columns stay exact however rough
    the diagonal (gh-361; e.g. a Hutchinson estimate). With an exact
    diagonal both are the same number up to rounding.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # LAPACK ?pstrf stopping criterion: n * eps * max diagonal entry.
    tol = diagonal.shape[0] * jnp.finfo(diagonal.dtype).eps * jnp.max(jnp.abs(diagonal))

    def body(i, carry):
        F, pivots, diagonal = carry
        residual = diagonal - reduce(F * F, "n k -> n", "sum")
        if pivoting == "greedy":
            s = jnp.argmax(residual)
        else:
            # Entries at or below the guard (including chosen pivots, whose
            # residual is rounding noise) get probability zero.
            usable = residual > tol
            log_weights = jnp.where(
                usable, jnp.log(jnp.where(usable, residual, 1.0)), -jnp.inf
            )
            s = jax.random.categorical(jax.random.fold_in(key, i), log_weights)
        raw = column(s)
        residual_column = raw - F @ F[s, :]
        if approximate_diagonal:
            pivot = residual_column[s]
            diagonal = diagonal.at[s].set(raw[s])
        else:
            pivot = residual[s]
        ok = pivot > tol
        # Double-where keeps the sqrt's gradient finite when guarded.
        denom = jnp.sqrt(jnp.where(ok, pivot, 1.0))
        col = residual_column / denom
        F = F.at[:, i].set(jnp.where(ok, col, 0.0))
        pivots = pivots.at[i].set(jnp.where(ok, s, -1).astype(pivots.dtype))
        return F, pivots, diagonal

    F0 = jnp.zeros((diagonal.shape[0], rank), dtype=diagonal.dtype)
    pivots0 = jnp.full((rank,), -1, dtype=jnp.int32)
    F, pivots, _ = jax.lax.fori_loop(0, rank, body, (F0, pivots0, diagonal))
    return F, pivots
