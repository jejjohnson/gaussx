r"""Matérn fields through the SPDE approach, on meshes and on regular grids.

The SPDE $(\kappa^2-\Delta)^{\alpha/2}(\tau x) = \mathcal W$ has Matérn
covariance with smoothness $\nu = \alpha - d/2$ (Lindgren, Rue & Lindström,
2011). Its P1 finite-element discretisation with the lumped mass matrix
$\tilde C$ and stiffness $G$ gives the sparse precision

$$
K = \kappa^2\tilde C + G,\qquad Q_1 = \tau^2 K,\qquad Q_2 = \tau^2K\tilde C^{-1}K,
\qquad Q_\alpha = K\tilde C^{-1}Q_{\alpha-2}\tilde C^{-1}K,
$$

i.e. $Q_\alpha = \tau^2 K(\tilde C^{-1}K)^{\alpha-1}$. On a regular grid with
spacing $h$, $\tilde C = h^d I$ and $G = h^{d-2}(L_1\oplus\cdots\oplus L_d)$,
so $Q_\alpha$ is a function of one Kronecker sum and every operation is
exact through the factor eigenvectors.

**Boundary effects.** Both discretisations impose natural (Neumann)
boundary conditions, which inflate the marginal variance within about one
practical range of the boundary (up to twice the stationary value at an
edge, four times at a corner). Extend the domain by at least one range
beyond the region of interest (a larger raster, or an outer mesh with
coarser triangles) and discard the extension; periodic axes of a grid
(longitude on a global raster) have no boundary.
"""

from __future__ import annotations

import functools as ft
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.scipy.special import gammaln
from jaxtyping import Array, ArrayLike, Float

from gaussx._gmrf._temporal import _as_float
from gaussx._linalg._eigen_factorization import EigenFactorization
from gaussx._operators._sparse import (
    _PLAN_CACHE_SIZE,
    SparseOperator,
    SparsityPattern,
    _canonicalise,
    _positions,
)
from gaussx._operators._spectral_function import SpectralFunction


def spde_precision(
    C_lumped: lx.DiagonalLinearOperator,
    G: SparseOperator,
    kappa: Float[ArrayLike, ""],
    tau: Float[ArrayLike, ""],
    alpha: int,
) -> SparseOperator:
    r"""SPDE precision ``Q_α = τ² K (C̃⁻¹K)^{α−1}`` with ``K = κ²C̃ + G``.

    The pattern of ``Q_α`` (the ``(α−1)``-ring neighbourhood of ``G``'s) and
    the index triples of each sparse product are computed once on the host
    per pattern and cached; only the values depend on ``κ`` and ``τ``, so
    ``jit``, ``grad`` and ``vmap`` over them never redo the symbolic work.
    The result goes to sparse Cholesky through `gaussx.solve`,
    `gaussx.logdet` and `gaussx.diag_inv` as those dispatch for a
    `SparseOperator`.

    Args:
        C_lumped: Lumped (diagonal) mass matrix ``C̃``, from
            `gaussx.fem_matrices`.
        G: Stiffness matrix, from `gaussx.fem_matrices`.
        kappa: Inverse range parameter ``κ > 0`` (may be traced).
        tau: Scale ``τ > 0`` (may be traced).
        alpha: Integer order ``α ≥ 1``; ``nu = α − d/2``.

    Returns:
        A symmetric positive-definite `SparseOperator`.

    Raises:
        ValueError: If ``alpha`` is not a positive integer or the sizes
            disagree.

    Examples:
        ```python
        import numpy as np
        import gaussx

        # Two triangles forming the unit square
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        C, G = gaussx.fem_matrices(vertices, triangles)
        kappa, tau, alpha = gaussx.matern_spde_params(
            range=0.5, sigma=1.0, nu=1.0, d=2
        )
        Q = gaussx.spde_precision(C, G, kappa, tau, alpha)  # (4, 4), alpha = 2
        ```
    """
    if not isinstance(alpha, int | np.integer) or alpha < 1:
        raise ValueError(f"alpha must be a positive integer, got {alpha!r}.")
    if not isinstance(G, SparseOperator):
        raise TypeError(f"G must be a SparseOperator, got {type(G).__name__}.")
    c = lx.diagonal(C_lumped)
    n = G.pattern.shape[0]
    if c.shape != (n,):
        raise ValueError(f"C_lumped must have size {n}, got {c.shape[0]}.")
    kappa = _as_float(kappa)
    tau = _as_float(tau)
    K = G.add_diagonal(kappa**2 * c)
    Q = K
    for _ in range(int(alpha) - 1):
        Q = _sandwich(K, 1.0 / c, Q)
    return SparseOperator(
        tau**2 * Q.values,
        Q.pattern,
        tags=frozenset({lx.positive_semidefinite_tag}),
    )


def spde_precision_grid(
    shape: tuple[int, ...],
    kappa: Float[ArrayLike, ""],
    tau: Float[ArrayLike, ""],
    alpha: int,
    *,
    spacing: float = 1.0,
    periodic: bool | tuple[bool, ...] = False,
) -> SpectralFunction:
    r"""SPDE precision on a regular grid: a function of a Kronecker sum.

    With spacing ``h`` the right-triangle mesh has ``C̃ = h^d I`` and
    ``G = h^{d−2}(L_1 ⊕ … ⊕ L_d)``, the ``(2d+1)``-point Laplacian, so

    $$
    Q_\alpha = \tau^2 h^d\big(\kappa^2 I
        + h^{-2}(L_1\oplus\cdots\oplus L_d)\big)^\alpha,
    $$

    which is ``τ²h²(κ² + λ/h²)^α`` on the eigenvalues ``λ`` of the Kronecker
    sum for a raster (``d = 2``). Each ``L_m`` is the path-graph Laplacian
    (natural boundary) or, on a periodic axis, the cycle-graph Laplacian.
    Its eigendecomposition is computed once on the host, so `gaussx.solve`,
    `gaussx.logdet`, `gaussx.diag_inv` and exact sampling
    (`SpectralFunction.sqrt_matmul` with ``inverse=True``) cost
    ``O(Σ_m n_m³)`` once and ``O(N Σ_m n_m)`` per call, with no mesh and no
    Cholesky. In the interior it equals `spde_precision` on the matching
    right-triangle mesh; at a non-periodic boundary the mesh's lumped mass
    and half-weight boundary edges differ (see the module notes on
    boundary effects and domain extension).

    Args:
        shape: Grid shape ``(n_1, …, n_d)``; vectors are flattened row-major.
        kappa: Inverse range ``κ > 0`` in units of the coordinates (may be
            traced).
        tau: Scale ``τ > 0`` (may be traced).
        alpha: Integer order ``α ≥ 1``.
        spacing: Grid spacing ``h`` (the same on every axis).
        periodic: Wrap all axes (``True``) or some of them, e.g.
            ``(False, True)`` for a global latitude-longitude raster.

    Returns:
        A positive-definite `gaussx.SpectralFunction`.

    Raises:
        ValueError: If ``alpha`` is not a positive integer or ``periodic``
            has the wrong length.

    Examples:
        ```python
        import gaussx

        # Matérn nu = 1 on a 30 x 60 raster, wrapping the second axis
        Q = gaussx.spde_precision_grid(
            (30, 60), kappa=0.3, tau=1.0, alpha=2, periodic=(False, True)
        )
        sd = gaussx.diag_inv(Q) ** 0.5  # exact, through the factor eigenvectors
        ```
    """
    if not isinstance(alpha, int | np.integer) or alpha < 1:
        raise ValueError(f"alpha must be a positive integer, got {alpha!r}.")
    shape = tuple(int(n) for n in shape)
    if isinstance(periodic, bool):
        periodic = (periodic,) * len(shape)
    if len(periodic) != len(shape):
        raise ValueError(
            f"periodic must have one entry per axis ({len(shape)}), got {periodic}."
        )
    kappa = _as_float(kappa)
    tau = _as_float(tau)
    h = jnp.asarray(spacing, dtype=jnp.result_type(kappa, tau))
    factors = [
        _cast(
            EigenFactorization.from_matrix(_laplacian_1d(n, wrap), symmetric=True),
            h.dtype,
        )
        for n, wrap in zip(shape, periodic, strict=True)
    ]
    symbol = _MaternSymbol(kappa, tau, h, int(alpha), len(shape))
    return SpectralFunction.from_eigen_factorizations(
        factors, symbol, tags=frozenset({lx.positive_semidefinite_tag})
    )


def matern_spde_params(
    range: Float[ArrayLike, ""],
    sigma: Float[ArrayLike, ""],
    nu: float,
    d: int,
) -> tuple[Array, Array, int]:
    r"""SPDE parameters ``(κ, τ, α)`` of a Matérn field.

    ``κ = √(8 nu)/rho`` for the practical range ``rho`` (correlation ≈ 0.13 at
    distance ``rho``), ``α = nu + d/2``, and ``τ`` from the marginal variance

    $$
    \sigma^2 = \frac{\Gamma(\nu)}{\Gamma(\alpha)(4\pi)^{d/2}\kappa^{2\nu}\tau^2}.
    $$

    Args:
        range: Practical range ``rho > 0`` (may be traced).
        sigma: Marginal standard deviation ``σ > 0`` (may be traced).
        nu: Smoothness ``nu > 0``, concrete.
        d: Spatial dimension, concrete.

    Returns:
        ``(kappa, tau, alpha)`` with an integer ``alpha``.

    Raises:
        ValueError: If ``nu + d/2`` is not a positive integer (rational
            non-integer ``α`` is not supported).

    Examples:
        ```python
        import gaussx

        kappa, tau, alpha = gaussx.matern_spde_params(
            range=50.0, sigma=2.0, nu=1.0, d=2
        )  # alpha == 2
        ```
    """
    alpha_float = float(nu) + d / 2
    alpha = round(alpha_float)
    if alpha < 1 or not math.isclose(alpha, alpha_float):
        raise ValueError(
            f"alpha = nu + d/2 = {alpha_float} must be a positive integer; "
            "non-integer alpha is not supported."
        )
    rho = _as_float(range)
    sigma = _as_float(sigma)
    kappa = jnp.sqrt(8.0 * nu) / rho
    log_tau2 = (
        gammaln(nu)
        - gammaln(float(alpha))
        - 0.5 * d * math.log(4.0 * math.pi)
        - 2.0 * nu * jnp.log(kappa)
        - 2.0 * jnp.log(sigma)
    )
    return kappa, jnp.exp(0.5 * log_tau2).astype(kappa.dtype), alpha


class _MaternSymbol(eqx.Module):
    """``f(λ) = τ² hᵈ (κ² + λ/h²)^α``, with the parameters as pytree leaves."""

    kappa: Array
    tau: Array
    spacing: Array
    alpha: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __call__(self, lam: Array) -> Array:
        h = self.spacing
        return self.tau**2 * h**self.dim * (self.kappa**2 + lam / h**2) ** self.alpha


def _laplacian_1d(n: int, periodic: bool) -> np.ndarray:
    """Path-graph (or, periodic, cycle-graph) Laplacian with unit weights."""
    heads = np.arange(n if periodic and n > 1 else n - 1)
    tails = (heads + 1) % n
    L = np.zeros((n, n))
    np.add.at(L, (heads, heads), 1.0)
    np.add.at(L, (tails, tails), 1.0)
    np.add.at(L, (heads, tails), -1.0)
    np.add.at(L, (tails, heads), -1.0)
    return L


def _cast(factor: EigenFactorization, dtype: jnp.dtype) -> EigenFactorization:
    return jax.tree.map(lambda leaf: leaf.astype(dtype), factor)


def _sandwich(
    A: SparseOperator, d: Float[Array, " n"], B: SparseOperator
) -> SparseOperator:
    """``A diag(d) B`` for symmetric ``A`` and ``B`` with a symmetric product.

    The product's pattern and the index triples feeding each stored entry
    come from `_sandwich_plan` (host, cached); the values are one
    ``segment_sum``.
    """
    pattern, p, q, k, target = _sandwich_plan(A.pattern, B.pattern)
    contributions = A._full_values()[p] * d[k] * B._full_values()[q]
    values = jax.ops.segment_sum(contributions, target, num_segments=pattern.nnz)
    return SparseOperator(values, pattern)


@ft.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _sandwich_plan(
    a: SparsityPattern, b: SparsityPattern
) -> tuple[SparsityPattern, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pattern of ``A D B`` (lower triangle) and its index triples.

    Entry ``p = (i, k)`` of the full ``A`` meets every entry ``q = (k, j)``
    of the full ``B`` in row ``k``, contributing ``A_p d_k B_q`` to
    ``(i, j)``; only ``i ≥ j`` is kept, since the product is symmetric.
    """
    a_rows, a_cols, _ = a._full
    b_rows, b_cols, _ = b._full
    counts = np.bincount(b_rows, minlength=b.shape[0])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    reps = counts[a_cols]
    p = np.repeat(np.arange(a_rows.shape[0]), reps)
    offset = np.arange(p.shape[0]) - np.repeat(np.cumsum(reps) - reps, reps)
    q = starts[a_cols[p]] + offset
    i, j = a_rows[p], b_cols[q]
    keep = i >= j
    p, q, i, j = p[keep], q[keep], i[keep], j[keep]
    rows, cols, _ = _canonicalise(i, j, a.shape, True)
    pattern = SparsityPattern._from_canonical(rows, cols, a.shape, True)
    return (
        pattern,
        p.astype(np.int32),
        q.astype(np.int32),
        a_cols[p].astype(np.int32),
        _positions(pattern, i, j),
    )
