"""Quadrature rules: Gauss-Hermite, sigma and cubature point sets."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from gaussx._einx import rearrange, reduce, repeat
from gaussx._primitives._sqrt import sqrt


# |centre mean weight| at which float32 has lost about half its ~7 digits.
_CENTRE_WEIGHT_WARN = 1e3


def gauss_hermite_points(
    order: int,
    dim: int,
    dtype: DTypeLike | None = None,
) -> tuple[Float[Array, "P D"], Float[Array, " P"]]:
    """Tensor-product Gauss-Hermite points and weights for a standard Gaussian.

        ∫ g(z) exp(−‖z‖²/2) dz ≈ ∑ᵢ wᵢ g(zᵢ),    P = order^dim

    with probabilists' Hermite polynomials. The weights sum to (2π)^(dim/2), so
    𝔼[g(z)] for z ~ 𝒩(0, I) is (2π)^(−dim/2) times this sum.

    Algorithm:
        1. Take the 1-D points and weights.
        2. For dim > 1, take the tensor grid of the points, and the product
           over dimensions of the weights.

    Args:
        order: Number of quadrature points per dimension.
        dim: Dimensionality of the integration domain.
        dtype: Floating dtype of the returned points and weights.
            Defaults to JAX's default float dtype. Unlike the other rules
            here this one has no mean to take a dtype from, so callers
            propagating an input dtype must pass it explicitly —
            `gaussx.GaussHermiteIntegrator` does.

    Returns:
        Tuple ``(points, weights)`` holding zᵢ and wᵢ.

    Examples:
        >>> import jax.numpy as jnp
        >>> points, weights = gauss_hermite_points(3, 2)
        >>> points.shape, weights.shape
        ((9, 2), (9,))

        The weights integrate the normalising constant of exp(−z²/2):

        >>> _, weights = gauss_hermite_points(5, 1)
        >>> bool(jnp.allclose(jnp.sum(weights), jnp.sqrt(2.0 * jnp.pi)))
        True
    """
    # 1-D probabilists' points and weights
    points_1d_np, weights_1d_np = np.polynomial.hermite_e.hermegauss(order)
    points_1d = jnp.array(points_1d_np, dtype=dtype)
    weights_1d = jnp.array(weights_1d_np, dtype=dtype)

    if dim == 1:
        return repeat(points_1d, "point -> point dim", dim=1), weights_1d

    # jnp, not einx: stack raises the recorded "Need at least one array to
    # stack." at dim = 0.
    grids = jnp.meshgrid(*([points_1d] * dim), indexing="ij")
    stacked = jnp.stack(grids)
    points = rearrange(stacked, "D ... -> (...) D")

    weight_grids = jnp.meshgrid(*([weights_1d] * dim), indexing="ij")
    weight_stack = jnp.stack(weight_grids)
    # wᵢ = ∏_d w_d,i
    weights = reduce(weight_stack, "D ... -> (...)", "prod")

    return points, weights


def _warn_cancelling_centre_weight(
    mean: Float[Array, " N"], alpha: float, kappa: float
) -> None:
    """Warn when a float32 rule's centre mean weight cancels catastrophically.

        w_m,0 = λ / (D_x + λ),    λ = α² (D_x + κ) − D_x

    Skipped for traced ``alpha`` / ``kappa`` (no concrete value to test).

    Args:
        mean: Mean μ_x, whose dtype and size D_x decide the check.
        alpha: Spread parameter α.
        kappa: Secondary scaling κ.
    """
    if jnp.finfo(mean.dtype).bits >= 64:
        return
    try:
        concrete_alpha, concrete_kappa = float(alpha), float(kappa)
    except jax.errors.ConcretizationTypeError:  # traced under jit / vmap
        return
    state_size = mean.shape[0]
    # λ = α² (D_x + κ) − D_x
    scaling = concrete_alpha**2 * (state_size + concrete_kappa) - state_size
    if state_size + scaling <= 0:
        return
    # w_m,0 = λ / (D_x + λ)
    centre_weight = scaling / (state_size + scaling)
    if abs(centre_weight) >= _CENTRE_WEIGHT_WARN:
        warnings.warn(
            f"sigma_points/UnscentedIntegrator with alpha={concrete_alpha:g}, "
            f"kappa={concrete_kappa:g} in {jnp.dtype(mean.dtype).name}: the centre "
            f"mean weight is {centre_weight:.1e}, so the moments are recovered by "
            "cancellation and lose about log10|w0| digits. Use alpha=1.0 (gh-310).",
            UserWarning,
            stacklevel=3,
        )


def sigma_points(
    mean: Float[Array, " N"],
    cov: lx.AbstractLinearOperator,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[Float[Array, "P N"], Float[Array, " P"], Float[Array, " P"]]:
    """Scaled unscented sigma points and weights, 2 D_x + 1 of them.

        χ₀ = μ_x,    χ_±d = μ_x ± √(D_x + λ) S_d,    S Sᵀ = Σ_x
        λ = α² (D_x + κ) − D_x
        w_m,0 = λ / (D_x + λ),    w_c,0 = w_m,0 + 1 − α² + β
        w_m,i = w_c,i = 1 / (2 (D_x + λ))

    with S_d the d-th column of S. Uses ``gaussx.sqrt(cov)`` for structured
    square root dispatch.

    Algorithm:
        1. Warn if a float32 centre mean weight would cancel catastrophically.
        2. Take S and scale it by √(D_x + λ).
        3. Place the centre point and the two sets of D_x outer points.
        4. Build the mean and covariance weights in the input dtype.

    Args:
        mean: Mean μ_x.
        cov: Covariance Σ_x.
        alpha: Spread parameter α. Controls how far sigma points are
            from the mean. Default ``1e-3`` (the classic Wan-van der Merwe
            scaled transform). See the warning below; ``alpha=1.0`` is the
            recommended choice, especially in float32.
        beta: Prior distribution parameter β. ``beta=2`` is optimal for
            Gaussians. Default ``2.0``.
        kappa: Secondary scaling parameter κ. Default ``0.0``.

    Warning:
        The centre mean weight w_m,0 = λ / (D_x + λ) ≈ 1 − 1/α² is about
        ``-1e6`` at the default ``alpha=1e-3``. The moments are then recovered
        by cancellation between weights of order ``1e6``, which costs about
        six digits; in float32 a smooth nonlinearity comes out wrong in
        the second digit (gh-310). ``alpha=1.0`` (with ``beta=2``,
        ``kappa=0``) is the symmetric 2 D_x + 1 rule with a zero centre mean
        weight, exact for affine maps, and is what every gaussx-internal
        default (`moment_transform`, the nonlinear Kalman filters) uses.
        A `UserWarning` is emitted when the actual centre mean weight
        (computed from ``alpha``, ``kappa`` and D_x) has magnitude
        ``>= 1e3`` with a float32 (or lower precision) mean; it is skipped
        when ``alpha`` or ``kappa`` is traced.

    Returns:
        Tuple ``(chi, w_m, w_c)`` holding the sigma points χᵢ, the mean
        weights w_m,i and the covariance weights w_c,i.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> points, mean_weights, covariance_weights = sigma_points(
        ...     jnp.zeros(2), covariance, alpha=1.0
        ... )
        >>> points.shape
        (5, 2)
        >>> bool(jnp.allclose(jnp.sum(mean_weights), 1.0))
        True
    """
    _warn_cancelling_centre_weight(mean, alpha, kappa)
    state_size = mean.shape[0]
    # λ = α² (D_x + κ) − D_x
    scaling = alpha**2 * (state_size + kappa) - state_size
    # D_x + λ
    spread = state_size + scaling

    # S Sᵀ = Σ_x
    covariance_root = sqrt(cov)
    root_matrix = covariance_root.as_matrix()
    # √(D_x + λ) S
    scaled_root = jnp.sqrt(spread) * root_matrix

    # jnp, not einx: einx cannot resolve an empty state (D_x = 0), which returns
    # a single empty point today when α²κ > 0. Each row broadcast stays its own
    # operation, so reverse mode accumulates into μ_x in the same order.
    # χ₀ = μ_x
    centre_point = mean[None, :]
    # χ_+d = μ_x + √(D_x + λ) S_d
    plus_points = mean[None, :] + scaled_root.T
    # χ_−d = μ_x − √(D_x + λ) S_d
    minus_points = mean[None, :] - scaled_root.T
    points = jnp.concatenate([centre_point, plus_points, minus_points])

    # Weights are built in the input dtype: Python floats would otherwise
    # land in float64 under x64 and promote the whole downstream transform.
    dtype = mean.dtype

    # w_m,0 = λ / (D_x + λ)
    centre_mean_weight = scaling / spread
    # w_m,i = 1 / (2 (D_x + λ))
    outer_weight = 1.0 / (2.0 * spread)
    mean_weights = jnp.concatenate(
        [
            jnp.array([centre_mean_weight], dtype=dtype),
            jnp.full(2 * state_size, outer_weight, dtype=dtype),
        ]
    )

    # w_c,0 = λ / (D_x + λ) + 1 − α² + β
    centre_covariance_weight = scaling / spread + (1.0 - alpha**2 + beta)
    covariance_weights = jnp.concatenate(
        [
            jnp.array([centre_covariance_weight], dtype=dtype),
            jnp.full(2 * state_size, outer_weight, dtype=dtype),
        ]
    )

    return points, mean_weights, covariance_weights


def cubature_points(
    mean: Float[Array, " N"],
    cov: lx.AbstractLinearOperator,
) -> tuple[Float[Array, "P N"], Float[Array, " P"]]:
    """Spherical-radial cubature points and weights, 2 D_x of them.

        χ_±d = μ_x ± √D_x S_d,    S Sᵀ = Σ_x,    wᵢ = 1 / (2 D_x)

    This is the cubature Kalman filter (CKF) point set. Uses
    ``gaussx.sqrt(cov)`` for structured square root dispatch.

    Algorithm:
        1. Take S and scale it by √D_x.
        2. Place the two sets of D_x points about μ_x.
        3. Build the equal weights in the input dtype.

    Args:
        mean: Mean μ_x.
        cov: Covariance Σ_x.

    Returns:
        Tuple ``(chi, weights)`` holding the points χᵢ and the equal weights
        wᵢ.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(3))
        >>> points, weights = cubature_points(jnp.zeros(3), covariance)
        >>> points.shape, weights.shape
        ((6, 3), (6,))
    """
    state_size = mean.shape[0]

    # S Sᵀ = Σ_x
    covariance_root = sqrt(cov)
    root_matrix = covariance_root.as_matrix()
    # √D_x S
    scaled_root = jnp.sqrt(state_size) * root_matrix

    # jnp, not einx: einx would raise AxisSizeError at D_x = 0, ahead of the
    # ZeroDivisionError from the weights below.
    # χ_+d = μ_x + √D_x S_d
    plus_points = mean[None, :] + scaled_root.T
    # χ_−d = μ_x − √D_x S_d
    minus_points = mean[None, :] - scaled_root.T
    points = jnp.concatenate([plus_points, minus_points])

    # wᵢ = 1 / (2 D_x)
    weights = jnp.full(2 * state_size, 1.0 / (2.0 * state_size), dtype=mean.dtype)

    return points, weights


def fifth_order_cubature_points(
    mean: Float[Array, " N"],
    cov: lx.AbstractLinearOperator,
) -> tuple[Float[Array, "P N"], Float[Array, " P"]]:
    """Fifth-order fully symmetric cubature points and weights.

        χᵢ = μ_x + S zᵢ,    S Sᵀ = Σ_x,    P = 2 D_x² + 1

    The degree-5 rule: exact for polynomials up to total degree 5 in D_x
    variables, with the standard points zᵢ on two shells plus the origin,

        Shell      Points                 Multiplicity     Weight
        origin     0                      1                2 / (D_x + 2)
        axis       ±r₁ e_a                2 D_x            (4 − D_x) / (2 (D_x + 2)²)
        diagonal   ±r₂ (e_a ± e_b), a<b   2 D_x (D_x − 1)  1 / (D_x + 2)²

    with r₁ = √(D_x + 2) and r₂ = √((D_x + 2) / 2).

    This rule sits between `sigma_points` / `cubature_points` (2 D_x + 1
    and 2 D_x points, degree 3) and a tensor-product `gauss_hermite_points`
    rule (order^D_x points), and is the default integrator in BayesNewton.

    Note:
        The axis weight is negative for D_x > 4 (and zero at D_x = 4).
        That is inherent to the rule, not a bug: the weights still sum to
        one and every moment through degree 5 is reproduced exactly, but
        the rule is not a positive quadrature in high dimension.

    Algorithm:
        1. Compute the two radii and the three shell weights.
        2. Lay out the standard points zᵢ on the shells.
        3. Map them into the input space through S.

    Args:
        mean: Mean μ_x.
        cov: Covariance Σ_x.

    Returns:
        Tuple ``(chi, weights)`` holding the points χᵢ and the weights wᵢ,
        which sum to one.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> points, weights = fifth_order_cubature_points(jnp.zeros(2), covariance)
        >>> points.shape
        (9, 2)
        >>> bool(jnp.allclose(jnp.sum(weights), 1.0))
        True

    References:
        McNamee & Stenger (1967).
    """
    state_size = mean.shape[0]

    # Python floats, not numpy scalars: a np.float64 radius is strongly
    # typed and would promote the points it scales back to float64.
    # r₁ = √(D_x + 2)
    axis_radius = float(np.sqrt(state_size + 2.0))
    # r₂ = √((D_x + 2) / 2)
    diagonal_radius = float(np.sqrt((state_size + 2.0) / 2.0))
    # origin weight 2 / (D_x + 2)
    origin_weight = 2.0 / (state_size + 2.0)
    # axis weight (4 − D_x) / (2 (D_x + 2)²)
    axis_weight = (4.0 - state_size) / (2.0 * (state_size + 2.0) ** 2)
    # diagonal weight 1 / (D_x + 2)²
    diagonal_weight = 1.0 / (state_size + 2.0) ** 2

    # Points and weights are built in the input dtype: the defaults would
    # otherwise land in float64 under x64 and promote the whole transform.
    dtype = mean.dtype

    eye = jnp.eye(state_size, dtype=dtype)
    origin = jnp.zeros((1, state_size), dtype=dtype)
    # ±r₁ e_a
    axis_shell = jnp.concatenate([axis_radius * eye, -axis_radius * eye])

    # ±r₂ (e_a ± e_b) for a < b: all four sign combinations
    rows, cols = np.triu_indices(state_size, k=1)
    n_pairs = rows.size
    shells = [origin, axis_shell]
    if n_pairs:
        basis_a = eye[rows]
        basis_b = eye[cols]
        shells.append(
            jnp.concatenate(
                [
                    diagonal_radius * (basis_a + basis_b),
                    diagonal_radius * (basis_a - basis_b),
                    -diagonal_radius * (basis_a + basis_b),
                    -diagonal_radius * (basis_a - basis_b),
                ]
            )
        )
    standard_points = jnp.concatenate(shells)

    weights = jnp.concatenate(
        [
            jnp.array([origin_weight], dtype=dtype),
            jnp.full(2 * state_size, axis_weight, dtype=dtype),
            jnp.full(4 * n_pairs, diagonal_weight, dtype=dtype),
        ]
    )

    # S Sᵀ = Σ_x
    covariance_root = sqrt(cov).as_matrix()
    # χᵢ = μ_x + S zᵢ
    # jnp, not einx: einx cannot resolve an empty state (D_x = 0), which returns
    # a single empty point here today.
    points = mean[None, :] + standard_points @ covariance_root.T

    return points, weights
