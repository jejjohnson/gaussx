"""Matérn fields through the SPDE approach, on meshes and on regular grids.

The SPDE (κ² − ∇²)^(α/2) (τ x) = white noise has Matérn covariance with smoothness
ν = α − D_s / 2 (Lindgren, Rue & Lindström, 2011). Its P1 finite-element
discretisation with the lumped mass matrix C̃ and stiffness G gives the sparse
precision

    K = κ² C̃ + G,   Λ_1 = τ² K,   Λ_2 = τ² K C̃⁻¹ K,
    Λ_α = K C̃⁻¹ Λ_α−2 C̃⁻¹ K,

i.e. Λ_α = τ² K (C̃⁻¹ K)^(α−1). On a regular grid with spacing h, C̃ = h^D_s I and
G = h^(D_s−2) (L_1 ⊕ … ⊕ L_D_s), so Λ_α is a function of one Kronecker sum and
every operation is exact through the factor eigenvectors.

**Boundary effects.** Both discretisations impose natural (Neumann) boundary
conditions, which inflate the marginal variance within about one practical range
of the boundary (up to twice the stationary value at an edge, four times at a
corner). Extend the domain by at least one range beyond the region of interest (a
larger raster, or an outer mesh with coarser triangles) and discard the extension;
periodic axes of a grid (longitude on a global raster) have no boundary.
"""

from __future__ import annotations

import functools as ft
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from gaussx._gmrf._temporal import _as_float
from gaussx._linalg._eigen_factorization import EigenDecomposition
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
    """SPDE precision Λ_α = τ² K (C̃⁻¹ K)^(α−1) with K = κ² C̃ + G.

    The pattern of Λ_α (the (α − 1)-ring neighbourhood of the pattern of G) and the
    index triples of each sparse product are computed once on the host per pattern
    and cached; only the values depend on κ and τ, so ``jit``, ``grad`` and ``vmap``
    over them never redo the symbolic work. The result goes to sparse Cholesky
    through `gaussx.solve`, `gaussx.logdet` and `gaussx.diag_inv` as those dispatch
    for a `SparseOperator`.

    Algorithm:
        1. Shift the stiffness: K = κ² C̃ + G.
        2. Multiply α − 1 times, Λ ← K C̃⁻¹ Λ, on the cached product patterns.
        3. Scale the stored values by τ².

    Args:
        C_lumped: Lumped (diagonal) mass matrix C̃, from `gaussx.fem_matrices`.
        G: Stiffness matrix G, from `gaussx.fem_matrices`.
        kappa: Inverse range κ > 0 (may be traced).
        tau: Scale τ > 0 (may be traced).
        alpha: Integer order α ≥ 1; ν = α − D_s / 2.

    Returns:
        Λ_α as a symmetric positive-definite `SparseOperator`.

    Raises:
        ValueError: If ``alpha`` is not a positive integer or the sizes disagree.
        TypeError: If ``G`` is not a `SparseOperator`.

    Examples:
        Two triangles forming the unit square, at Matérn ν = 1:

        >>> import numpy as np
        >>> import gaussx
        >>> vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        >>> triangles = np.array([[0, 1, 2], [0, 2, 3]])
        >>> mass, stiffness = gaussx.fem_matrices(vertices, triangles)
        >>> kappa, tau, alpha = gaussx.matern_spde_params(
        ...     range=0.5, sigma=1.0, nu=1.0, d=2
        ... )
        >>> precision = gaussx.spde_precision(mass, stiffness, kappa, tau, alpha)
        >>> alpha, precision.in_size()
        (2, 4)

        The order must be a positive integer:

        >>> gaussx.spde_precision(mass, stiffness, kappa, tau, 0)
        Traceback (most recent call last):
        ...
        ValueError: alpha must be a positive integer, got 0.
    """
    if not isinstance(alpha, int | np.integer) or alpha < 1:
        raise ValueError(f"alpha must be a positive integer, got {alpha!r}.")
    if not isinstance(G, SparseOperator):
        raise TypeError(f"G must be a SparseOperator, got {type(G).__name__}.")
    mass_diagonal = lx.diagonal(C_lumped)
    n = G.pattern.shape[0]
    if mass_diagonal.shape != (n,):
        raise ValueError(f"C_lumped must have size {n}, got {mass_diagonal.shape[0]}.")
    kappa = _as_float(kappa)
    tau = _as_float(tau)
    # κ² diag(C̃)
    scaled_mass = kappa**2 * mass_diagonal
    # K = κ² C̃ + G
    shifted_stiffness = G.add_diagonal(scaled_mass)
    # Λ_1 / τ² = K
    precision = shifted_stiffness
    for _ in range(int(alpha) - 1):
        # C̃⁻¹
        inverse_mass = 1.0 / mass_diagonal
        # Λ ← K C̃⁻¹ Λ
        precision = _sandwich(shifted_stiffness, inverse_mass, precision)
    # τ² K (C̃⁻¹ K)^(α−1), on the stored values
    values = tau**2 * precision.values
    return SparseOperator(
        values,
        precision.pattern,
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
    """SPDE precision on a regular grid: a function of a Kronecker sum.

        Λ_α = τ² h^D_s (κ² I + h⁻² (L_1 ⊕ … ⊕ L_D_s))^α

    With spacing h the right-triangle mesh has C̃ = h^D_s I and
    G = h^(D_s−2) (L_1 ⊕ … ⊕ L_D_s), the (2 D_s + 1)-point Laplacian. On the
    eigenvalues γ of the Kronecker sum, this is τ² h² (κ² + γ / h²)^α for a raster
    (D_s = 2). Each L_m is the path-graph Laplacian (natural boundary) or, on a
    periodic axis, the cycle-graph Laplacian. Its eigendecomposition is computed
    once on the host, so `gaussx.solve`, `gaussx.logdet`, `gaussx.diag_inv` and exact
    sampling (`SpectralFunction.sqrt_matmul` with ``inverse=True``) cost
    O(∑_m n_m³) once and O(N ∑_m n_m) per call, with no mesh and no Cholesky. In the
    interior it equals `spde_precision` on the matching right-triangle mesh; at a
    non-periodic boundary the lumped mass and half-weight boundary edges of the mesh
    differ (see the module notes on boundary effects and domain extension).

    Algorithm:
        1. Eigendecompose each axis Laplacian L_m on the host, in the dtype of h.
        2. Wrap the parameters in the symbol f(γ) = τ² h^D_s (κ² + γ / h²)^α.
        3. Return f of the Kronecker sum, from the factor eigendecompositions.

    Args:
        shape: Grid shape ``(n_1, …, n_D_s)``; vectors are flattened row-major.
        kappa: Inverse range κ > 0 in units of the coordinates (may be traced).
        tau: Scale τ > 0 (may be traced).
        alpha: Integer order α ≥ 1.
        spacing: Grid spacing h (the same on every axis).
        periodic: Wrap all axes (``True``) or some of them, e.g.
            ``(False, True)`` for a global latitude-longitude raster.

    Returns:
        Λ_α as a positive-definite `gaussx.SpectralFunction`.

    Raises:
        ValueError: If ``alpha`` is not a positive integer or ``periodic`` has the
            wrong length.

    Examples:
        Matérn ν = 1 on a 4 × 6 raster, wrapping the second axis:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> precision = gaussx.spde_precision_grid(
        ...     (4, 6), kappa=0.3, tau=1.0, alpha=2, periodic=(False, True)
        ... )
        >>> precision.in_size()
        24

        Marginal standard deviations are exact, through the factor eigenvectors:

        >>> jnp.sqrt(gaussx.diag_inv(precision)).shape
        (24,)

        ``periodic`` needs one entry per axis:

        >>> gaussx.spde_precision_grid(
        ...     (4, 6), kappa=0.3, tau=1.0, alpha=2, periodic=(True,)
        ... )
        Traceback (most recent call last):
        ...
        ValueError: periodic must have one entry per axis (2), got (True,).
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
    dtype = jnp.result_type(kappa, tau)
    grid_spacing = jnp.asarray(spacing, dtype=dtype)
    factors = []
    for size, wrap in zip(shape, periodic, strict=True):
        # L_m, on the host
        laplacian = _laplacian_1d(size, wrap)
        # L_m = Q_m diag(γ_m) Q_mᵀ
        eigen = EigenDecomposition.from_matrix(laplacian, symmetric=True)
        factors.append(_cast(eigen, grid_spacing.dtype))
    # f(γ) = τ² h^D_s (κ² + γ / h²)^α
    symbol = _MaternSymbol(kappa, tau, grid_spacing, int(alpha), len(shape))
    return SpectralFunction.from_eigen_factorizations(
        factors, symbol, tags=frozenset({lx.positive_semidefinite_tag})
    )


def matern_spde_params(
    range: Float[ArrayLike, ""],
    sigma: Float[ArrayLike, ""],
    nu: float,
    d: int,
) -> tuple[Array, Array, int]:
    """SPDE parameters (κ, τ, α) of a Matérn field.

        κ = √(8 ν) / ρ,   α = ν + D_s / 2
        σ² = Γ(ν) / (Γ(α) (4π)^(D_s/2) κ^(2ν) τ²)

    ρ is the practical range (correlation ≈ 0.13 at distance ρ), and τ solves the
    marginal variance equation.

    Algorithm:
        1. Check that α = ν + D_s / 2 is a positive integer.
        2. Set κ from the practical range.
        3. Solve the marginal variance for log τ², so the gamma functions and powers
           do not overflow.

    Args:
        range: Practical range ρ > 0 (may be traced).
        sigma: Marginal standard deviation σ > 0 (may be traced).
        nu: Smoothness ν > 0, concrete.
        d: Spatial dimension D_s, concrete.

    Returns:
        Tuple ``(kappa, tau, alpha)`` holding κ, τ and an integer α.

    Raises:
        ValueError: If ν + D_s / 2 is not a positive integer (rational non-integer
            α is not supported).

    Examples:
        A raster at ν = 1 has α = 2:

        >>> import gaussx
        >>> kappa, tau, alpha = gaussx.matern_spde_params(
        ...     range=50.0, sigma=2.0, nu=1.0, d=2
        ... )
        >>> alpha, round(float(kappa), 4)
        (2, 0.0566)

        A non-integer order has no sparse precision:

        >>> gaussx.matern_spde_params(
        ...     range=50.0, sigma=2.0, nu=0.5, d=2
        ... )  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: alpha = nu + d/2 = 1.5 must be a positive integer; non-integer
        alpha is not supported.
    """
    alpha_float = float(nu) + d / 2
    alpha = round(alpha_float)
    if alpha < 1 or not math.isclose(alpha, alpha_float):
        raise ValueError(
            f"alpha = nu + d/2 = {alpha_float} must be a positive integer; "
            "non-integer alpha is not supported."
        )
    practical_range = _as_float(range)
    sigma = _as_float(sigma)
    # κ = √(8 ν) / ρ
    root_eight_nu = jnp.sqrt(8.0 * nu)
    kappa = root_eight_nu / practical_range
    # log τ² = log Γ(ν) − log Γ(α) − (D_s / 2) log 4π − 2 ν log κ − 2 log σ
    log_tau2 = (
        jss.gammaln(nu)
        - jss.gammaln(float(alpha))
        - 0.5 * d * math.log(4.0 * math.pi)
        - 2.0 * nu * jnp.log(kappa)
        - 2.0 * jnp.log(sigma)
    )
    # τ = exp(½ log τ²)
    tau = jnp.exp(0.5 * log_tau2)
    return kappa, tau.astype(kappa.dtype), alpha


class _MaternSymbol(eqx.Module):
    """Spectral symbol f(γ) = τ² h^D_s (κ² + γ / h²)^α, with pytree parameters.

    Attributes:
        kappa: Inverse range κ.
        tau: Scale τ.
        spacing: Grid spacing h.
        alpha: Integer order α.
        dim: Spatial dimension D_s.
    """

    kappa: Array
    tau: Array
    spacing: Array
    alpha: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __call__(self, lam: Array) -> Array:
        """Evaluate the symbol on eigenvalues of the Kronecker sum.

        Args:
            lam: Eigenvalues γ.

        Returns:
            f(γ), the eigenvalues of Λ_α.
        """
        h = self.spacing
        # τ² h^D_s
        prefactor = self.tau**2 * h**self.dim
        # κ² + γ / h²
        shifted = self.kappa**2 + lam / h**2
        # τ² h^D_s (κ² + γ / h²)^α
        return prefactor * shifted**self.alpha


def _laplacian_1d(n: int, periodic: bool) -> np.ndarray:
    """Path-graph (or, periodic, cycle-graph) Laplacian L_m with unit weights.

    Args:
        n: Number of nodes on the axis.
        periodic: Join the last node back to the first.

    Returns:
        The dense L_m as a NumPy array.
    """
    heads = np.arange(n if periodic and n > 1 else n - 1)
    tails = (heads + 1) % n
    laplacian = np.zeros((n, n))
    np.add.at(laplacian, (heads, heads), 1.0)
    np.add.at(laplacian, (tails, tails), 1.0)
    np.add.at(laplacian, (heads, tails), -1.0)
    np.add.at(laplacian, (tails, heads), -1.0)
    return laplacian


def _cast(factor: EigenDecomposition, dtype: jnp.dtype) -> EigenDecomposition:
    """Cast every leaf of an eigendecomposition.

    Args:
        factor: The eigendecomposition.
        dtype: The target dtype.

    Returns:
        The same factor, in ``dtype``.
    """
    return jax.tree.map(lambda leaf: leaf.astype(dtype), factor)


def _sandwich(
    A: SparseOperator, d: Float[Array, " n"], B: SparseOperator
) -> SparseOperator:
    """Sparse product A diag(d) B for symmetric A, B with a symmetric product.

        [A diag(d) B]_ij = ∑_k [A]_ik d_k [B]_kj

    The pattern of the product and the index triples feeding each stored entry come
    from `_sandwich_plan` (host, cached); the values are one ``segment_sum``.

    Args:
        A: The left factor.
        d: The diagonal.
        B: The right factor.

    Returns:
        The product, on its lower-triangle pattern.
    """
    pattern, a_entry, b_entry, middle, target = _sandwich_plan(A.pattern, B.pattern)
    # [A]_ik d_k
    left_terms = A._full_values()[a_entry] * d[middle]
    # [A]_ik d_k [B]_kj
    contributions = left_terms * B._full_values()[b_entry]
    # ∑_k [A]_ik d_k [B]_kj
    values = jax.ops.segment_sum(contributions, target, num_segments=pattern.nnz)
    return SparseOperator(values, pattern)


@ft.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _sandwich_plan(
    a: SparsityPattern, b: SparsityPattern
) -> tuple[SparsityPattern, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pattern of A diag(d) B (lower triangle) and its index triples.

    Entry p = (i, k) of the full A meets every entry q = (k, j) of the full B in
    row k, contributing [A]_p d_k [B]_q to (i, j); only i ≥ j is kept, since the
    product is symmetric.

    Args:
        a: Pattern of A.
        b: Pattern of B.

    Returns:
        Tuple ``(pattern, p, q, k, target)``: the pattern of the product, the
        indices p, q and k of each contribution, and the stored position it lands
        on.
    """
    a_rows, a_cols, _ = a._full
    b_rows, b_cols, _ = b._full
    counts = np.bincount(b_rows, minlength=b.shape[0])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    repeats = counts[a_cols]
    a_entry = np.repeat(np.arange(a_rows.shape[0]), repeats)
    offset = np.arange(a_entry.shape[0]) - np.repeat(
        np.cumsum(repeats) - repeats, repeats
    )
    b_entry = starts[a_cols[a_entry]] + offset
    row, col = a_rows[a_entry], b_cols[b_entry]
    keep = row >= col
    a_entry, b_entry, row, col = a_entry[keep], b_entry[keep], row[keep], col[keep]
    rows, cols, _ = _canonicalise(row, col, a.shape, True)
    pattern = SparsityPattern._from_canonical(rows, cols, a.shape, True)
    return (
        pattern,
        a_entry.astype(np.int32),
        b_entry.astype(np.int32),
        a_cols[a_entry].astype(np.int32),
        _positions(pattern, row, col),
    )
