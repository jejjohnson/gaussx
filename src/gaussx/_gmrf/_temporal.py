r"""Temporal precision builders: iid, RW1, RW2 and AR(1).

Each is a quadratic form $\tfrac{\tau}{2}\|Dx\|^2$ for a banded difference
operator $D$, so $Q = \tau D^\top D$ is banded and fits a `BlockTriDiag`,
whose Cholesky, solve, log-determinant and selected inverse all cost
$O(N d^3)$. Cyclic variants couple the two ends, which a block-tridiagonal
band cannot hold; they are returned as a `SparseOperator`.
"""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from gaussx._einx import einsum, rearrange
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._sparse import SparseOperator


_PSD = frozenset({lx.positive_semidefinite_tag})


def _as_float(value: ArrayLike) -> Array:
    """``value`` as an inexact array, keeping a floating input's dtype."""
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype(jnp.result_type(float))
    return value


def iid_precision(n: int, tau: Float[ArrayLike, ""]) -> lx.DiagonalLinearOperator:
    r"""Precision ``τ I`` of ``n`` independent effects with variance ``1/τ``.

    Args:
        n: Number of effects.
        tau: Precision ``τ > 0`` (may be traced).

    Returns:
        A `lineax.DiagonalLinearOperator` of size ``n``.

    Examples:
        ```python
        import gaussx

        Q = gaussx.iid_precision(5, tau=2.0)
        Q.as_matrix()  # 2 I
        ```
    """
    tau = _as_float(tau)
    return lx.DiagonalLinearOperator(jnp.full((n,), tau, dtype=tau.dtype))


def rw1_structure(
    n: int,
    *,
    spacing: Float[ArrayLike, " n-1"] | float | None = None,
    cyclic: bool = False,
) -> BlockTriDiag | SparseOperator:
    r"""Structure matrix ``R = D₁ᵀ W D₁`` of a first-order random walk.

    The increments ``x_{i+1} − x_i ~ N(0, h_i/τ)`` give the precision
    ``τ R`` with ``R = D₁ᵀ diag(1/h) D₁``: the weighted path-graph Laplacian
    with edge weights ``1/h_i`` (unit weights for regular spacing). ``R`` is
    singular with null space ``span{1}``.

    Args:
        n: Number of nodes.
        spacing: Gaps ``h_i`` between consecutive nodes, shape ``(n − 1,)``
            (``(n,)`` with ``cyclic=True``, the last one closing the cycle),
            or a scalar. ``None`` means unit spacing.
        cyclic: Join node ``n − 1`` back to node ``0`` (a cycle-graph
            Laplacian, e.g. a seasonal effect). The corner entries leave
            the tridiagonal band, so the result is a `SparseOperator`.

    Returns:
        A positive-semidefinite `BlockTriDiag` with ``1 × 1`` blocks, or a
        symmetric `SparseOperator` with ``cyclic=True``.

    Examples:
        ```python
        import jax.numpy as jnp
        import gaussx

        R = gaussx.rw1_structure(5)
        R.mv(jnp.ones(5))  # zeros: constants are in the null space
        ```
    """
    n_edges = n if cyclic else n - 1
    if spacing is None:
        weights = jnp.ones(n_edges, dtype=jnp.result_type(float))
    else:
        h = _as_float(spacing)
        weights = jnp.broadcast_to(1.0 / h, (n_edges,))
    if cyclic:
        return _cyclic_structure(n, [weights], [1])
    degree = jnp.zeros(n, dtype=weights.dtype).at[1:].add(weights).at[:-1].add(weights)
    return BlockTriDiag(
        rearrange(degree, "(n a b) -> n a b", a=1, b=1),
        rearrange(-weights, "(n a b) -> n a b", a=1, b=1),
        tags=_PSD,
    )


def rw2_structure(n: int, *, cyclic: bool = False) -> BlockTriDiag | SparseOperator:
    r"""Structure matrix ``R = D₂ᵀ D₂`` of a second-order random walk.

    The second differences ``x_{i+1} − 2x_i + x_{i−1} ~ N(0, 1/τ)`` give the
    pentadiagonal precision ``τ R``, the discrete cubic smoothing spline. Its
    null space is ``span{1, t}`` (``span{1}`` for ``cyclic=True``).

    The band is stored as a `BlockTriDiag` with ``2 × 2`` blocks, which needs
    an even size: for **odd** ``n`` the operator has ``n + 1`` rows, the last
    one a decoupled node with unit precision. That node does not interact
    with the others, so strip it from results (``x[:n]``,
    ``diag_inv(R)[:n]``); it adds nothing to ``log|R|`` and ``log 1 = 0``
    otherwise (``log τ`` once ``R`` is scaled by ``τ``).

    Args:
        n: Number of nodes (at least 3).
        cyclic: Wrap the second differences around (a seasonal RW2). The
            result is then a `SparseOperator` of size exactly ``n``.

    Returns:
        A positive-semidefinite `BlockTriDiag` of size ``2⌈n/2⌉``, or a
        symmetric `SparseOperator` with ``cyclic=True``.

    Raises:
        ValueError: If ``n < 3``.

    Examples:
        ```python
        import jax.numpy as jnp
        import gaussx

        R = gaussx.rw2_structure(6)
        t = jnp.arange(6.0)
        R.mv(t)  # zeros: linear trends are in the null space
        ```
    """
    if n < 3:
        raise ValueError(f"rw2_structure needs n >= 3, got {n}.")
    dtype = jnp.result_type(float)
    if cyclic:
        ones = jnp.ones(n, dtype=dtype)
        return _cyclic_structure(n, [6.0 * ones, -4.0 * ones, ones], [0, 1, 2])
    # D₂ᵀD₂ for the open chain, padded to an even size with a decoupled node.
    size = n + (n % 2)
    D = np.zeros((n - 2, size))
    rows = np.arange(n - 2)
    D[rows, rows], D[rows, rows + 1], D[rows, rows + 2] = 1.0, -2.0, 1.0
    D = jnp.asarray(D, dtype=dtype)
    R = einsum(D, D, "k i, k j -> i j")
    if size != n:
        R = R.at[n, n].set(1.0)
    blocks = rearrange(R, "(N a) (M b) -> N M a b", a=2, b=2)
    num = size // 2
    diagonal = blocks[jnp.arange(num), jnp.arange(num)]
    sub = blocks[jnp.arange(1, num), jnp.arange(num - 1)]
    return BlockTriDiag(diagonal, sub, tags=_PSD)


def ar1_precision(
    n: int, rho: Float[ArrayLike, ""], tau: Float[ArrayLike, ""]
) -> BlockTriDiag:
    r"""Precision of a stationary AR(1) process with marginal precision ``τ``.

    ``x_t = rho x_{t−1} + ε_t``, started from its stationary law, has

    $$
    Q = \frac{\tau}{1-\rho^2}\operatorname{tridiag}(-\rho,\ 1+\rho^2,\ -\rho)
    $$

    with ``1`` in the two corners, so every marginal variance is ``1/τ``
    and the innovation precision is ``τ/(1 − rho²)``.

    Args:
        n: Number of time points (at least 2).
        rho: Lag-one correlation, ``|rho| < 1`` (may be traced).
        tau: Marginal precision ``τ > 0`` (may be traced).

    Returns:
        A positive-definite `BlockTriDiag` with ``1 × 1`` blocks.

    Examples:
        ```python
        import jax.numpy as jnp
        import gaussx

        Q = gaussx.ar1_precision(50, rho=0.8, tau=10.0)
        gaussx.diag_inv(Q)  # all 0.1: the marginal variance 1/τ
        ```
    """
    if n < 2:
        raise ValueError(f"ar1_precision needs n >= 2, got {n}.")
    rho = _as_float(rho)
    tau = _as_float(tau)
    dtype = jnp.result_type(rho, tau)
    scale = tau / (1.0 - rho**2)
    main = jnp.full(n, 1.0 + rho**2, dtype=dtype).at[0].set(1.0).at[-1].set(1.0)
    off = jnp.full(n - 1, -rho, dtype=dtype)
    return BlockTriDiag(
        rearrange(scale * main, "(n a b) -> n a b", a=1, b=1),
        rearrange(scale * off, "(n a b) -> n a b", a=1, b=1),
        tags=_PSD,
    )


def _cyclic_structure(n: int, bands: list[Array], offsets: list[int]) -> SparseOperator:
    """Symmetric circulant-pattern operator from lower bands.

    ``bands[k][i]`` is entry ``(i + offsets[k] mod n, i)``. For the RW1 cycle
    the single band holds the edge weights and the diagonal is derived from
    them as the weighted degree.
    """
    idx = np.arange(n)
    if offsets == [1]:  # weighted cycle Laplacian from edge weights
        weights = bands[0]
        degree = weights + jnp.roll(weights, 1)
        rows = np.r_[idx, (idx + 1) % n]
        cols = np.r_[idx, idx]
        values = jnp.concatenate([degree, -weights])
    else:
        rows = np.concatenate([(idx + k) % n for k in offsets])
        cols = np.concatenate([idx for _ in offsets])
        values = jnp.concatenate(bands)
    return SparseOperator.from_coo(
        rows, cols, values, (n, n), symmetric=True, tags=_PSD
    )
