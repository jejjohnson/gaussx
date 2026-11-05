"""Covariance localization: Gaspari-Cohn taper and the Hadamard-localized gain."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._inference._ensemble_stats import (
    _check_obs_noise_shape,
    ensemble_cross_covariance,
)
from gaussx._linalg._linalg import solve_rows
from gaussx._linalg._mixed_precision import stable_squared_distances
from gaussx._linalg._symmetrize import symmetrize
from gaussx._strategies._base import AbstractSolverStrategy


def gaspari_cohn(
    r: Float[Array, "*shape"], c: float | Float[Array, ""]
) -> Float[Array, "*shape"]:
    """Gaspari-Cohn (1999) fifth-order compactly supported taper.

    With z = 2 |r| / c, the standard positive-definite, approximately Gaussian
    localization function is the piecewise rational

        ρ(z) = −¼ z⁵ + ½ z⁴ + ⅝ z³ − (5/3) z² + 1                 0 ≤ z ≤ 1
        ρ(z) = (1/12) z⁵ − ½ z⁴ + ⅝ z³ + (5/3) z² − 5 z + 4 − 2 / (3 z)    1 < z < 2
        ρ(z) = 0                                                  z ≥ 2

    so ρ = 1 at r = 0 and ρ = 0 for |r| ≥ c, where c is the compact-support
    radius, **not** a Gaussian length scale. The taper is C³ at the knots z = 1
    and z = 2.

    Algorithm:
        1. Scale the distances to z = 2 |r| / c.
        2. Guard the 1 / z term of the middle branch.
        3. Evaluate both polynomial branches and select by z.

    Args:
        r: Distances r of any shape, e.g. a pairwise distance matrix.
        c: Compact-support radius c; ρ = 0 beyond |r| = c.

    Returns:
        Taper values in [0, 1], with the shape of ``r``.

    Examples:
        One at zero distance and zero at the support radius:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> gaussx.gaspari_cohn(jnp.array([0.0, 2.0, 3.0]), 2.0).tolist()
        [1.0, 0.0, 0.0]

        The gradient stays finite at r = 0:

        >>> import jax
        >>> slope = jax.grad(lambda r: gaussx.gaspari_cohn(r, 2.0))(jnp.array(0.0))
        >>> bool(jnp.isfinite(slope))
        True

    References:
        Gaspari & Cohn (1999).
    """
    # z = 2 |r| / c
    scaled_distance = 2.0 * jnp.abs(r) / c
    # Guard the 1 / z term: at z = 0 the near branch is selected, but JAX still
    # traces the middle branch, so an unguarded 1 / z poisons the gradient.
    safe_distance = jnp.where(scaled_distance > 0.0, scaled_distance, 1.0)

    # ρ = −¼ z⁵ + ½ z⁴ + ⅝ z³ − (5/3) z² + 1
    near = (
        -0.25 * scaled_distance**5
        + 0.5 * scaled_distance**4
        + 0.625 * scaled_distance**3
        - (5.0 / 3.0) * scaled_distance**2
        + 1.0
    )
    # ρ = (1/12) z⁵ − ½ z⁴ + ⅝ z³ + (5/3) z² − 5 z + 4 − 2 / (3 z)
    mid = (
        (1.0 / 12.0) * scaled_distance**5
        - 0.5 * scaled_distance**4
        + 0.625 * scaled_distance**3
        + (5.0 / 3.0) * scaled_distance**2
        - 5.0 * scaled_distance
        + 4.0
        - 2.0 / (3.0 * safe_distance)
    )
    outer = jnp.where(scaled_distance < 2.0, mid, 0.0)
    return jnp.where(scaled_distance <= 1.0, near, outer)


def euclidean_distance(
    coords_a: Float[Array, "Na D"],
    coords_b: Float[Array, "Nb D"],
) -> Float[Array, "Na Nb"]:
    """Pairwise Euclidean distances ‖aₙ − bₘ‖₂.

    A default ``metric`` for `localization_matrix`. It builds on
    `stable_squared_distances` and takes a gradient-safe square root, so zero
    distances (the diagonal of a self-distance matrix) give no ``NaN``
    gradients.

    Args:
        coords_a: First set of points aₙ, one row per point.
        coords_b: Second set of points bₘ, one row per point.

    Returns:
        The matrix of distances ‖aₙ − bₘ‖₂.

    Examples:
        A 3-4-5 triangle:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> origin = jnp.zeros((1, 2))
        >>> point = jnp.array([[3.0, 4.0]])
        >>> bool(jnp.allclose(gaussx.euclidean_distance(origin, point), 5.0))
        True

        One row and one column per point:

        >>> gaussx.euclidean_distance(jnp.zeros((3, 2)), jnp.ones((4, 2))).shape
        (3, 4)
    """
    # ‖aₙ − bₘ‖₂²
    squared_distance = stable_squared_distances(
        coords_a,
        coords_b,
        compute_dtype=coords_a.dtype,
        accumulate_dtype=coords_a.dtype,
    )
    # The square root has an infinite slope at zero, so keep zero out of it.
    safe_squared_distance = jnp.where(squared_distance > 0.0, squared_distance, 1.0)
    # ‖aₙ − bₘ‖₂
    return jnp.where(squared_distance > 0.0, jnp.sqrt(safe_squared_distance), 0.0)


def haversine_distance(
    coords_a: Float[Array, "Na 2"],
    coords_b: Float[Array, "Nb 2"],
    radius: float = 6.371e6,
) -> Float[Array, "Na Nb"]:
    """Pairwise great-circle (haversine) distances on a sphere.

        h = sin²(Δφ / 2) + cos φ_a cos φ_b sin²(Δϑ / 2)
        d = 2 r arcsin(√h)

    A ``metric`` for `localization_matrix` on geophysical grids, with latitude
    φ and longitude ϑ in **radians**.

    Algorithm:
        1. Broadcast the latitudes and longitudes to one row per point of a
           and one column per point of b.
        2. Form the haversine h, clipped to [0, 1] against round-off.
        3. Return the arc length 2 r arcsin(√h).

    Args:
        coords_a: First set of points (φ, ϑ) in radians, one row per point.
        coords_b: Second set of points (φ, ϑ) in radians, one row per point.
        radius: Sphere radius r in the units of the returned distance; the
            default is the Earth mean radius, 6.371e6 m.

    Returns:
        The matrix of great-circle distances d.

    Examples:
        A quarter of the unit circle along the equator:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> origin = jnp.zeros((1, 2))
        >>> east = jnp.array([[0.0, jnp.pi / 2]])
        >>> distance = gaussx.haversine_distance(origin, east, radius=1.0)
        >>> bool(jnp.allclose(distance, jnp.pi / 2))
        True

        As the metric of a localization taper:

        >>> gaussx.localization_matrix(
        ...     origin, east, 1.0, metric=gaussx.haversine_distance
        ... ).shape
        (1, 1)
    """
    # jnp, not einx: N_a = 0 or N_b = 0 returns an empty matrix here today, and
    # einx raises AxisSizeError on a zero-length axis.
    latitude_a = coords_a[:, 0][:, None]
    longitude_a = coords_a[:, 1][:, None]
    latitude_b = coords_b[:, 0][None, :]
    longitude_b = coords_b[:, 1][None, :]
    # Δφ = φ_b − φ_a
    latitude_difference = latitude_b - latitude_a
    # Δϑ = ϑ_b − ϑ_a
    longitude_difference = longitude_b - longitude_a
    # h = sin²(Δφ / 2) + cos φ_a cos φ_b sin²(Δϑ / 2)
    haversine = (
        jnp.sin(latitude_difference / 2.0) ** 2
        + jnp.cos(latitude_a)
        * jnp.cos(latitude_b)
        * jnp.sin(longitude_difference / 2.0) ** 2
    )
    # d = 2 r arcsin(√h)
    return 2.0 * radius * jnp.arcsin(jnp.sqrt(jnp.clip(haversine, 0.0, 1.0)))


def localization_matrix(
    coords_a: Float[Array, "Na D"],
    coords_b: Float[Array, "Nb D"],
    c: float | Float[Array, ""],
    metric: Callable[
        [Float[Array, "Na D"], Float[Array, "Nb D"]], Float[Array, "Na Nb"]
    ] = euclidean_distance,
) -> Float[Array, "Na Nb"]:
    """Pairwise Gaspari-Cohn taper ρ(d(aₙ, bₘ); c).

    Builds the ρ_xy (state-observation) and ρ_yy (observation-observation)
    matrices consumed by `localized_kalman_gain`.

    Args:
        coords_a: First set of points aₙ, one row per point.
        coords_b: Second set of points bₘ, one row per point.
        c: Gaspari-Cohn compact-support radius c.
        metric: Pairwise distance function d returning one row per point of a
            and one column per point of b. Defaults to `euclidean_distance`;
            pass `haversine_distance` for spherical coordinates.

    Returns:
        The localization matrix, with entries in [0, 1].

    Examples:
        Self-localization has a unit diagonal:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> coords = jnp.array([[0.0], [1.0], [5.0]])
        >>> rho = gaussx.localization_matrix(coords, coords, 2.0)
        >>> jnp.diag(rho).tolist()
        [1.0, 1.0, 1.0]

        Points beyond the radius are decoupled:

        >>> float(rho[0, 2])
        0.0
    """
    # d(aₙ, bₘ)
    distance = metric(coords_a, coords_b)
    # ρ(d; c)
    return gaspari_cohn(distance, c)


def _check_localization_shapes(
    state_size: int,
    obs_size: int,
    localization: Float[Array, "N M"] | None,
    obs_localization: Float[Array, "M M"] | None,
    *,
    localization_name: str = "localization",
    obs_localization_name: str = "obs_localization",
) -> None:
    """Check the taper shapes; ``obs_localization`` only counts with a taper.

    Args:
        state_size: State length D_x.
        obs_size: Observation length D_y.
        localization: State-observation taper ρ_xy, or ``None``.
        obs_localization: Observation-observation taper ρ_yy, or ``None``.
        localization_name: Name used for the state-observation taper in errors.
        obs_localization_name: Name used for the observation taper in errors.

    Raises:
        ValueError: If a taper given alongside ``localization`` has the wrong
            shape.
    """
    if localization is None:
        return
    # Broadcast-compatible but wrong shapes are the danger: a (D_x, 1) taper
    # repeats one observation's taper across all D_y, and a (1, 1)
    # obs_localization rescales the whole observation covariance. Both give a
    # plausible, wrong gain rather than an error.
    if localization.shape != (state_size, obs_size):
        raise ValueError(
            f"{localization_name} must have shape ({state_size}, {obs_size}) to "
            f"match particles and obs_particles, got {localization.shape}."
        )
    if obs_localization is not None and obs_localization.shape != (obs_size, obs_size):
        raise ValueError(
            f"{obs_localization_name} must have shape ({obs_size}, {obs_size}) to "
            f"match obs_particles, got {obs_localization.shape}."
        )


def localized_kalman_gain(
    particles: Float[Array, "J N"],
    obs_particles: Float[Array, "J M"],
    obs_noise: lx.AbstractLinearOperator,
    rho_xy: Float[Array, "N M"],
    rho_yy: Float[Array, "M M"],
    *,
    solver: AbstractSolverStrategy | None = None,
    bessel: bool = True,
) -> Float[Array, "N M"]:
    """Ensemble Kalman gain with Hadamard (Schur-product) localization.

        G = (ρ_xy ⊙ Σ̂_xy) (ρ_yy ⊙ Σ̂_h + Σ_ε)⁻¹

    where Σ̂_xy is the state-observation sample cross-covariance and Σ̂_h the
    sample covariance of the observation-space ensemble. Tapering removes
    spurious long-range sample correlations; because Gaspari-Cohn is positive
    definite, the Schur product theorem keeps ρ_yy ⊙ Σ̂_h positive semi-definite,
    so the observation covariance stays invertible.

    This is the localized counterpart of `ensemble_kalman_gain`. The Hadamard
    product destroys the low-rank structure, so Σ_y is materialised and the
    solve costs O(D_x D_y + D_y³). The unlocalized gain is the c → ∞ limit,
    ρ_xy = ρ_yy = 1.

    Algorithm:
        1. Form the sample covariances Σ̂_xy and Σ̂_h.
        2. Taper them by ρ_xy and ρ_yy.
        3. Add Σ_ε and symmetrise to get Σ_y.
        4. Solve for the gain G row by row.

    Args:
        particles: Prior ensemble x⁽ᵉ⁾ in state space.
        obs_particles: Its image h⁽ᵉ⁾ = h(x⁽ᵉ⁾) in observation space.
        obs_noise: Observation-error covariance Σ_ε.
        rho_xy: State-observation taper ρ_xy.
        rho_yy: Observation-observation taper ρ_yy.
        solver: Solver strategy for the dense Σ_y; ``None`` uses structural
            dispatch.
        bessel: Use the 1 / (|ℰ| − 1) divisor (the EnKF convention, default).

    Returns:
        The dense localized Kalman gain G.

    Raises:
        ValueError: If the two ensembles have different sizes, or if
            ``obs_noise`` or the tapers do not match the state and observation
            dimensions, or if ``bessel`` is set with |ℰ| < 2.

    Examples:
        Unit tapers recover the unlocalized gain:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> particles = jr.normal(jr.key(0), (8, 3))
        >>> obs_particles = particles[:, :2]
        >>> obs_noise = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> gain = gaussx.localized_kalman_gain(
        ...     particles, obs_particles, obs_noise, jnp.ones((3, 2)), jnp.ones((2, 2))
        ... )
        >>> plain = gaussx.ensemble_kalman_gain(particles, obs_particles, obs_noise)
        >>> bool(jnp.allclose(gain, plain, atol=1e-4))
        True

        A zero taper removes that coupling from the gain:

        >>> rho_xy = jnp.ones((3, 2)).at[2, :].set(0.0)
        >>> gain = gaussx.localized_kalman_gain(
        ...     particles, obs_particles, obs_noise, rho_xy, jnp.ones((2, 2))
        ... )
        >>> bool(jnp.allclose(gain[2], 0.0))
        True
    """
    if particles.shape[0] != obs_particles.shape[0]:
        raise ValueError(
            "particles and obs_particles must share the same ensemble size, "
            f"got J={particles.shape[0]} and J={obs_particles.shape[0]}."
        )
    state_size, obs_size = particles.shape[1], obs_particles.shape[1]
    _check_obs_noise_shape(obs_size, obs_noise)
    _check_localization_shapes(
        state_size,
        obs_size,
        rho_xy,
        rho_yy,
        localization_name="rho_xy",
        obs_localization_name="rho_yy",
    )
    # Σ̂_xy
    cross_covariance = ensemble_cross_covariance(
        particles, obs_particles, bessel=bessel
    )
    # Σ̂_h
    obs_ensemble_covariance = ensemble_cross_covariance(
        obs_particles, obs_particles, bessel=bessel
    )

    # ρ_xy ⊙ Σ̂_xy
    localized_cross = rho_xy * cross_covariance
    # Σ_y = ρ_yy ⊙ Σ̂_h + Σ_ε
    obs_covariance = rho_yy * obs_ensemble_covariance + obs_noise.as_matrix()
    obs_covariance = symmetrize(obs_covariance)
    obs_covariance_op = lx.MatrixLinearOperator(
        obs_covariance, lx.positive_semidefinite_tag
    )
    # G = (ρ_xy ⊙ Σ̂_xy) Σ_y⁻¹, row by row since Σ_y is symmetric
    return solve_rows(obs_covariance_op, localized_cross, solver=solver)
