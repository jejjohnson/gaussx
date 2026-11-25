"""Temporal precision builders: iid, RW1, RW2 and AR(1).

Each is a quadratic form (τ / 2) ‖D x‖₂² for a banded difference operator D, so
Λ = τ Dᵀ D is banded and fits a `BlockTriDiag`, whose Cholesky, solve,
log-determinant and selected inverse all cost O(N d³) for N blocks of size d.
Cyclic variants couple the two ends, which a block-tridiagonal band cannot hold;
they are returned as a `SparseOperator`.
"""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from gaussx._einx import rearrange
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._sparse import SparseOperator


_PSD = frozenset({lx.positive_semidefinite_tag})


def _as_float(value: ArrayLike) -> Array:
    """Convert a value to an inexact array, keeping a floating input's dtype.

    Args:
        value: Scalar or array; an integer or boolean input becomes a weakly typed
            float, which takes the dtype of what it meets, as a Python float does.

    Returns:
        The value as an inexact array.
    """
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        # an integer times a Python float is a weakly typed float
        value = value * 1.0
    return value


def iid_precision(n: int, tau: Float[ArrayLike, ""]) -> lx.DiagonalLinearOperator:
    """Precision Λ = τ I of n independent effects with variance 1 / τ.

        x_i ~ 𝒩(0, 1 / τ),   i = 1, …, n

    Args:
        n: Number of effects.
        tau: Precision τ > 0 (may be traced).

    Returns:
        Λ as a `lineax.DiagonalLinearOperator` of size n.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> precision = gaussx.iid_precision(5, tau=2.0)
        >>> precision.in_size()
        5
        >>> bool(jnp.allclose(precision.as_matrix(), 2.0 * jnp.eye(5)))
        True
    """
    tau = _as_float(tau)
    # τ 1, the diagonal of Λ
    diagonal = jnp.full((n,), tau, dtype=tau.dtype)
    # Λ = τ I
    return lx.DiagonalLinearOperator(diagonal)


def rw1_structure(
    n: int,
    *,
    spacing: Float[ArrayLike, " n-1"] | float | None = None,
    cyclic: bool = False,
) -> BlockTriDiag | SparseOperator:
    """Structure matrix R = D₁ᵀ W D₁ of a first-order random walk.

        x_t+1 − x_t ~ 𝒩(0, h_t / τ),   Λ = τ R,   R = D₁ᵀ diag(1 / h) D₁

    R is the weighted path-graph Laplacian with edge weights 1 / h_t (unit
    weights for regular spacing), so it is singular with null space span{1}.

    Args:
        n: Number of nodes.
        spacing: Gaps h_t between consecutive nodes, shape ``(n − 1,)``
            (``(n,)`` with ``cyclic=True``, the last one closing the cycle), or a
            scalar. ``None`` means unit spacing.
        cyclic: Join node n − 1 back to node 0 (a cycle-graph Laplacian, e.g. a
            seasonal effect). The corner entries leave the tridiagonal band, so
            the result is a `SparseOperator`.

    Returns:
        R as a positive-semidefinite `BlockTriDiag` with 1 × 1 blocks, or a
        symmetric `SparseOperator` with ``cyclic=True``. A single node has no
        increment, and its R is zero.

    Raises:
        ValueError: If n < 1.

    Examples:
        Constants are in the null space:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> structure = gaussx.rw1_structure(5)
        >>> bool(jnp.allclose(structure.mv(jnp.ones(5)), 0.0))
        True

        A cycle with irregular gaps is sparse and still sums to zero by row:

        >>> gaps = jnp.array([0.5, 1.0, 2.0, 1.0])
        >>> cycle = gaussx.rw1_structure(4, spacing=gaps, cyclic=True)
        >>> type(cycle).__name__
        'SparseOperator'
        >>> bool(jnp.allclose(cycle.mv(jnp.ones(4)), 0.0))
        True
    """
    if n < 1:
        raise ValueError(f"rw1_structure needs n >= 1, got {n}.")
    num_edges = n if cyclic else n - 1
    if spacing is None:
        dtype = jnp.result_type(float)
        # w_t = 1
        weights = jnp.ones(num_edges, dtype=dtype)
    else:
        gaps = _as_float(spacing)
        # w_t = 1 / h_t
        inverse_gaps = 1.0 / gaps
        weights = jnp.broadcast_to(inverse_gaps, (num_edges,))
    if cyclic:
        if n == 1:  # the closing edge is a self-loop, which adds nothing
            weights = jnp.zeros_like(weights)
        return _cyclic_structure(n, [weights], [1])
    # [R]_t,t = w_t−1 + w_t, the weighted degree
    degree = jnp.zeros(n, dtype=weights.dtype)
    degree = degree.at[1:].add(weights)
    degree = degree.at[:-1].add(weights)
    diagonal_blocks = rearrange(degree, "(n a b) -> n a b", a=1, b=1)
    if n == 1:  # no sub-diagonal, which einx rejects
        no_edges = jnp.zeros((0, 1, 1), dtype=weights.dtype)
        return BlockTriDiag(diagonal_blocks, no_edges, tags=_PSD)
    # [R]_t+1,t = −w_t
    negated_weights = -weights
    sub_diagonal_blocks = rearrange(negated_weights, "(n a b) -> n a b", a=1, b=1)
    return BlockTriDiag(diagonal_blocks, sub_diagonal_blocks, tags=_PSD)


def rw2_structure(n: int, *, cyclic: bool = False) -> BlockTriDiag | SparseOperator:
    """Structure matrix R = D₂ᵀ D₂ of a second-order random walk.

        x_t+1 − 2 x_t + x_t−1 ~ 𝒩(0, 1 / τ),   Λ = τ R,   R = D₂ᵀ D₂

    Λ is pentadiagonal, the discrete cubic smoothing spline. The null space of R
    is span{1, t} (span{1} for ``cyclic=True``).

    The band is stored as a `BlockTriDiag` with 2 × 2 blocks, which needs an even
    size: for **odd** n the operator has n + 1 rows, the last one a decoupled node
    with unit precision. That node does not interact with the others, so strip it
    from results (``x[:n]``, ``diag_inv(R)[:n]``); it adds nothing to log det R
    and log 1 = 0 otherwise (log τ once R is scaled by τ).

    Args:
        n: Number of nodes (at least 3).
        cyclic: Wrap the second differences around (a seasonal RW2). The result
            is then a `SparseOperator` of size exactly n.

    Returns:
        R as a positive-semidefinite `BlockTriDiag` of size 2⌈n / 2⌉, or a
        symmetric `SparseOperator` with ``cyclic=True``.

    Raises:
        ValueError: If n < 3.

    Examples:
        Linear trends are in the null space:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> structure = gaussx.rw2_structure(6)
        >>> bool(jnp.allclose(structure.mv(jnp.arange(6.0)), 0.0))
        True

        An odd n gets one padding row:

        >>> gaussx.rw2_structure(7).in_size()
        8

        Too few nodes for a second difference:

        >>> gaussx.rw2_structure(2)
        Traceback (most recent call last):
        ...
        ValueError: rw2_structure needs n >= 3, got 2.
    """
    if n < 3:
        raise ValueError(f"rw2_structure needs n >= 3, got {n}.")
    dtype = jnp.result_type(float)
    if cyclic:
        ones = jnp.ones(n, dtype=dtype)
        # [R]_t,t = 6
        main_band = 6.0 * ones
        # [R]_t+1,t = −4
        first_band = -4.0 * ones
        # [R]_t+2,t = 1
        return _cyclic_structure(n, [main_band, first_band, ones], [0, 1, 2])
    # The bands of D₂ᵀ D₂ for the open chain, padded to an even size with a decoupled
    # node. Row r of D₂ is (1, −2, 1) on the nodes r, r + 1, r + 2.
    size = n + (n % 2)
    num_rows = n - 2
    main_band = np.zeros(size)
    for shift, square in ((0, 1.0), (1, 4.0), (2, 1.0)):
        main_band[shift : shift + num_rows] += square
    first_band = np.zeros(size)
    for shift in (0, 1):
        first_band[shift : shift + num_rows] -= 2.0
    second_band = np.zeros(size)
    second_band[:num_rows] = 1.0
    if size != n:
        # [R]_n,n = 1, the padding node
        main_band[n] = 1.0
    # 2 × 2 blocks: nodes (2b, 2b + 1) on the diagonal, coupled to (2b + 2, 2b + 3)
    num_blocks = size // 2
    diagonal = np.zeros((num_blocks, 2, 2))
    diagonal[:, 0, 0] = main_band[0::2]
    diagonal[:, 1, 1] = main_band[1::2]
    diagonal[:, 1, 0] = diagonal[:, 0, 1] = first_band[0::2]
    sub_diagonal = np.zeros((num_blocks - 1, 2, 2))
    # [R]_2b+2,2b, [R]_2b+2,2b+1 and [R]_2b+3,2b+1
    sub_diagonal[:, 0, 0] = second_band[0::2][:-1]
    sub_diagonal[:, 0, 1] = first_band[1::2][:-1]
    sub_diagonal[:, 1, 1] = second_band[1::2][:-1]
    return BlockTriDiag(
        jnp.asarray(diagonal, dtype=dtype),
        jnp.asarray(sub_diagonal, dtype=dtype),
        tags=_PSD,
    )


def ar1_precision(
    n: int, rho: Float[ArrayLike, ""], tau: Float[ArrayLike, ""]
) -> BlockTriDiag:
    """Precision of a stationary AR(1) process with marginal precision τ.

        x_t = ρ x_t−1 + η_t,   x_0 ~ 𝒩(0, 1 / τ)
        Λ = (τ / (1 − ρ²)) tridiag(−ρ, 1 + ρ², −ρ),   with 1 in the two corners

    Every marginal variance is 1 / τ and the innovation precision is
    τ / (1 − ρ²).

    Args:
        n: Number of time points (at least 2).
        rho: Lag-one correlation ρ, ``|ρ| < 1`` (may be traced).
        tau: Marginal precision τ > 0 (may be traced).

    Returns:
        Λ as a positive-definite `BlockTriDiag` with 1 × 1 blocks.

    Raises:
        ValueError: If n < 2.

    Examples:
        The marginal variance is 1 / τ:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> precision = gaussx.ar1_precision(50, rho=0.8, tau=10.0)
        >>> bool(jnp.allclose(gaussx.diag_inv(precision), 0.1, atol=1e-3))
        True

        One time point has no lag-one term:

        >>> gaussx.ar1_precision(1, rho=0.8, tau=10.0)
        Traceback (most recent call last):
        ...
        ValueError: ar1_precision needs n >= 2, got 1.
    """
    if n < 2:
        raise ValueError(f"ar1_precision needs n >= 2, got {n}.")
    rho = _as_float(rho)
    tau = _as_float(tau)
    dtype = jnp.result_type(rho, tau)
    # τ / (1 − ρ²)
    scale = tau / (1.0 - rho**2)
    # 1 + ρ²
    interior = 1.0 + rho**2
    # tridiag main band: 1 + ρ², with 1 in the two corners
    main = jnp.full(n, interior, dtype=dtype)
    main = main.at[0].set(1.0)
    main = main.at[-1].set(1.0)
    # tridiag off band: −ρ
    negated_rho = -rho
    off = jnp.full(n - 1, negated_rho, dtype=dtype)
    # [Λ]_t,t
    scaled_main = scale * main
    diagonal_blocks = rearrange(scaled_main, "(n a b) -> n a b", a=1, b=1)
    # [Λ]_t+1,t
    scaled_off = scale * off
    sub_diagonal_blocks = rearrange(scaled_off, "(n a b) -> n a b", a=1, b=1)
    return BlockTriDiag(diagonal_blocks, sub_diagonal_blocks, tags=_PSD)


def _cyclic_structure(n: int, bands: list[Array], offsets: list[int]) -> SparseOperator:
    """Symmetric circulant-pattern operator from lower bands.

    ``bands[k][t]`` is entry ``(t + offsets[k] mod n, t)``. For the RW1 cycle
    (``offsets == [1]``) the single band holds the edge weights and the diagonal
    is derived from them as the weighted degree.

    Args:
        n: Number of nodes.
        bands: The lower bands.
        offsets: Offset of each band below the diagonal.

    Returns:
        A symmetric positive-semidefinite `SparseOperator` of size n.
    """
    index = np.arange(n)
    if offsets == [1]:
        weights = bands[0]
        # w_t−1, the weight of the edge entering node t
        previous_weights = jnp.roll(weights, 1)
        # [R]_t,t = w_t−1 + w_t
        degree = weights + previous_weights
        rows = np.r_[index, (index + 1) % n]
        cols = np.r_[index, index]
        # [R]_t+1,t = −w_t
        negated_weights = -weights
        values = jnp.concatenate([degree, negated_weights])
    else:
        rows = np.concatenate([(index + k) % n for k in offsets])
        cols = np.concatenate([index for _ in offsets])
        values = jnp.concatenate(bands)
    return SparseOperator.from_coo(
        rows, cols, values, (n, n), symmetric=True, tags=_PSD
    )
