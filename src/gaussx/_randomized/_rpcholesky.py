"""Randomly pivoted partial Cholesky (RPCholesky) of a PSD matrix A.

A is N × N and PSD, known only through its diagonal and its columns. L_k is the N × k
factor with A ≈ L_k L_kᵀ, d the residual diagonal of A − L_k L_kᵀ, s the chosen pivot
and g the residual column at it. The guard τ is the ``?pstrf`` numerical-rank tolerance.
"""

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
    """Randomly pivoted partial Cholesky of a PSD matrix A.

        A ≈ L_k L_kᵀ,   s ~ max(d, 0),   g = A_:,s − L_k (L_k)_s,:ᵀ,
        (L_k)_:,i = g / √g_s

    Touches A only through its diagonal and k of its columns (Chen, Epperly, Tropp &
    Webber, 2023). With ⟦A⟧_r the best rank-r approximation of A and
    k ≥ r/ε + r log(1/(ε η)) pivots,

        𝔼 tr(A − L_k L_kᵀ) ≤ (1 + ε) tr(A − ⟦A⟧_r),   η = tr(A − ⟦A⟧_r) / tr(A)

    Greedy pivoting (the argmax of d) has no such guarantee and chases outliers. The
    cost is k column evaluations and O(N k²) flops.

    The returned pivots S make L_k L_kᵀ = A_:,S (A_S,S)⁺ A_S,:, the column Nyström
    approximation on those columns, so they double as landmark (inducing-point)
    indices. The diagonal and ``column`` can come from a kernel evaluated on the fly,
    so A is never formed: for 10⁶ points and ``rank=1000`` this is 1000 kernel columns.

    The residual is guarded like LAPACK ``?pstrf``: once the chosen pivot falls below
    τ = N ε max_i |A_ii| the numerical rank is exhausted, and that and every later
    column of L_k is exactly zero, with pivot ``-1`` (gh-236, gh-237). Filter with
    ``pivots[pivots >= 0]``.

    Algorithm:
        1. Compute the residual diagonal d of A − L_k L_kᵀ.
        2. Draw the pivot s with probability proportional to d (or take its argmax).
        3. Read column s of A and subtract what L_k explains, giving g.
        4. Store g / √g_s, or zeros and pivot ``-1`` once the pivot falls below τ.

    Args:
        diagonal: Diagonal of the PSD matrix A, of length N.
        column: Callable returning column s of A, of length N, for a scalar integer
            index s (a traced value inside the loop).
        rank: Number of pivots k.
        pivoting: ``"random"`` samples s ∝ max(d, 0); ``"greedy"`` takes the argmax
            of d, the classic pivoted Cholesky.
        block_size: Pivots per step. Only ``1`` is implemented; the blocked variant
            (Epperly, Tropp & Webber, 2024) is a follow-up.
        key: PRNG key for ``"random"`` pivoting. ``None`` means
            ``jax.random.PRNGKey(0)``. Ignored by ``"greedy"``.

    Returns:
        Tuple ``(factor, pivots)`` holding L_k (N × k) and the pivot indices, of
        length k, in the order chosen (``-1`` past the numerical rank).

    Raises:
        ValueError: If ``pivoting`` is not ``"random"`` or ``"greedy"``.
        NotImplementedError: If ``block_size != 1``.

    Examples:
        Pick 20 landmarks from 1000 points without forming the kernel matrix:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import gaussx
        >>> inputs = jr.normal(jr.key(0), (1000,))
        >>> def column(s):
        ...     return jnp.exp(-0.5 * (inputs - inputs[s]) ** 2)
        >>> factor, pivots = gaussx.rp_cholesky(
        ...     jnp.ones(1000), column, 20, key=jr.key(1)
        ... )
        >>> factor.shape, pivots.shape
        ((1000, 20), (20,))
        >>> inputs[pivots].shape
        (20,)

        A rank-one matrix leaves the surplus column zero, with pivot ``-1``:

        >>> u = jnp.array([1.0, 2.0, 3.0])
        >>> matrix = jnp.outer(u, u)
        >>> factor, pivots = gaussx.rp_cholesky(
        ...     jnp.diag(matrix), lambda s: matrix[:, s], 2, pivoting="greedy"
        ... )
        >>> bool(pivots[1] == -1), bool(jnp.all(factor[:, 1] == 0.0))
        (True, True)

        An unknown ``pivoting`` is rejected:

        >>> gaussx.rp_cholesky(jnp.ones(3), lambda s: u, 2, pivoting="smallest")
        Traceback (most recent call last):
        ...
        ValueError: pivoting must be 'random' or 'greedy', got 'smallest'

    References:
        Chen, Y., Epperly, E. N., Tropp, J. A. & Webber, R. J. (2023).
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

        g = A_:,s − L_k (L_k)_s,:ᵀ,   (L_k)_:,i = g / √g_s

    With ``approximate_diagonal`` the diagonal only *selects* pivots: each pivot's
    value is read from its exact column instead, and the diagonal entry is corrected
    to it, so the factor columns stay exact however rough the diagonal (gh-361; e.g.
    a Hutchinson estimate). With an exact diagonal both are the same number up to
    rounding.

    Args:
        diagonal: Diagonal of A, or an estimate of it, of length N.
        column: Callable returning column s of A.
        rank: Number of pivots k.
        pivoting: ``"greedy"`` takes the argmax of d; anything else samples.
        key: PRNG key for sampling. ``None`` means ``jax.random.PRNGKey(0)``.
        approximate_diagonal: Read each pivot from its column, not the diagonal.

    Returns:
        Tuple ``(factor, pivots)`` holding L_k and the pivot indices.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # N ε
    relative_tolerance = diagonal.shape[0] * jnp.finfo(diagonal.dtype).eps
    # |A_ii|
    magnitudes = jnp.abs(diagonal)
    # τ = N ε max_i |A_ii|, the LAPACK ?pstrf stopping criterion
    tolerance = relative_tolerance * jnp.max(magnitudes)

    def body(step, carry):
        factor, pivots, diagonal = carry
        # L_k ⊙ L_k
        squared = factor * factor
        # ∑_i (L_k)_n,i²
        explained = reduce(squared, "n k -> n", "sum")
        # d = diag(A) − ∑_i (L_k)_n,i²
        residual = diagonal - explained
        if pivoting == "greedy":
            # s = argmax d
            pivot_index = jnp.argmax(residual)
        else:
            # Entries at or below the guard (including chosen pivots, whose residual
            # is rounding noise) get probability zero.
            usable = residual > tolerance
            # Double-where keeps the log's gradient finite when masked.
            safe_residual = jnp.where(usable, residual, 1.0)
            # log d
            safe_log_residual = jnp.log(safe_residual)
            # log d on the usable entries, −∞ elsewhere
            log_weights = jnp.where(usable, safe_log_residual, -jnp.inf)
            # a key per step
            step_key = jax.random.fold_in(key, step)
            # s ~ d, up to normalisation
            pivot_index = jax.random.categorical(step_key, log_weights)
        # A_:,s
        raw_column = column(pivot_index)
        # (L_k)_s,:
        pivot_row = factor[pivot_index, :]
        # L_k (L_k)_s,:ᵀ
        explained_column = factor @ pivot_row
        # g = A_:,s − L_k (L_k)_s,:ᵀ
        residual_column = raw_column - explained_column
        if approximate_diagonal:
            # g_s, read from the exact column
            pivot = residual_column[pivot_index]
            # d_s = A_ss, corrected once its column is known
            diagonal = diagonal.at[pivot_index].set(raw_column[pivot_index])
        else:
            # d_s
            pivot = residual[pivot_index]
        accepted = pivot > tolerance
        # Double-where keeps the sqrt's gradient finite when guarded.
        safe_pivot = jnp.where(accepted, pivot, 1.0)
        # √g_s
        denominator = jnp.sqrt(safe_pivot)
        # g / √g_s
        new_column = residual_column / denominator
        # zeros past the numerical rank
        guarded_column = jnp.where(accepted, new_column, 0.0)
        # (L_k)_:,i = g / √g_s
        factor = factor.at[:, step].set(guarded_column)
        # s, or −1 past the numerical rank
        guarded_index = jnp.where(accepted, pivot_index, -1)
        guarded_index = guarded_index.astype(pivots.dtype)
        pivots = pivots.at[step].set(guarded_index)
        return factor, pivots, diagonal

    initial_factor = jnp.zeros((diagonal.shape[0], rank), dtype=diagonal.dtype)
    initial_pivots = jnp.full((rank,), -1, dtype=jnp.int32)
    initial_carry = (initial_factor, initial_pivots, diagonal)
    factor, pivots, _ = jax.lax.fori_loop(0, rank, body, initial_carry)
    return factor, pivots
