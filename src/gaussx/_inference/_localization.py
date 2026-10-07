"""Covariance localization: Gaspari-Cohn taper and the Hadamard-localized gain."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._inference._ensemble_stats import ensemble_cross_covariance
from gaussx._linalg._linalg import solve_rows
from gaussx._linalg._mixed_precision import stable_squared_distances
from gaussx._linalg._symmetrize import symmetrize
from gaussx._strategies._base import AbstractSolverStrategy


def gaspari_cohn(
    r: Float[Array, "*shape"], c: float | Float[Array, ""]
) -> Float[Array, "*shape"]:
    r"""Gaspari-Cohn (1999) fifth-order compactly-supported taper.

    The standard positive-definite, approximately-Gaussian localization
    function. With ``z = 2 |r| / c`` it is the piecewise-rational

    $$
    \begin{aligned}
    \rho = \begin{cases}
      -\tfrac14 z^5 + \tfrac12 z^4 + \tfrac58 z^3 - \tfrac53 z^2 + 1
        & 0 \le z \le 1 \\
      \tfrac1{12} z^5 - \tfrac12 z^4 + \tfrac58 z^3 + \tfrac53 z^2
        - 5 z + 4 - \tfrac{2}{3 z}
        & 1 < z \le 2 \\
      0 & z > 2.
    \end{cases}
    \end{aligned}
    $$

    so ``rho(0) = 1`` and ``rho = 0`` for ``|r| >= c`` (``c`` is the
    compact-support radius, **not** a Gaussian length scale). The taper is
    only $C^1$ at the knots ``z = 1, 2``.

    Differentiability: the ``2 / (3 z)`` term in the middle branch is guarded
    with a safe denominator so reverse-mode gradients are finite at ``r = 0``
    (which would otherwise produce ``NaN`` via the standard ``where`` pitfall).

    Args:
        r: Distances (any shape), e.g. a pairwise distance matrix.
        c: Compact-support radius; ``rho = 0`` beyond ``|r| = c``.

    Returns:
        Taper values in ``[0, 1]``, same shape as ``r``.
    """
    z = 2.0 * jnp.abs(r) / c
    # Guard the 1 / z term: at z = 0 the near branch is selected, but JAX still
    # traces the middle branch, so an unguarded 1 / z poisons the gradient.
    z_safe = jnp.where(z > 0.0, z, 1.0)

    near = -0.25 * z**5 + 0.5 * z**4 + 0.625 * z**3 - (5.0 / 3.0) * z**2 + 1.0
    mid = (
        (1.0 / 12.0) * z**5
        - 0.5 * z**4
        + 0.625 * z**3
        + (5.0 / 3.0) * z**2
        - 5.0 * z
        + 4.0
        - 2.0 / (3.0 * z_safe)
    )
    return jnp.where(z <= 1.0, near, jnp.where(z < 2.0, mid, 0.0))


def euclidean_distance(
    coords_a: Float[Array, "Na D"],
    coords_b: Float[Array, "Nb D"],
) -> Float[Array, "Na Nb"]:
    """Pairwise Euclidean distances ``||a_i - b_j||``.

    A default ``metric`` for `localization_matrix`. Builds on
    `stable_squared_distances` and takes a gradient-safe square root so
    zero distances (e.g. the diagonal of a self-distance matrix) do not produce
    ``NaN`` gradients.

    Args:
        coords_a: First set of points, shape ``(Na, D)``.
        coords_b: Second set of points, shape ``(Nb, D)``.

    Returns:
        Distance matrix of shape ``(Na, Nb)``.
    """
    sq = stable_squared_distances(
        coords_a,
        coords_b,
        compute_dtype=coords_a.dtype,
        accumulate_dtype=coords_a.dtype,
    )
    sq_safe = jnp.where(sq > 0.0, sq, 1.0)
    return jnp.where(sq > 0.0, jnp.sqrt(sq_safe), 0.0)


def haversine_distance(
    coords_a: Float[Array, "Na 2"],
    coords_b: Float[Array, "Nb 2"],
    radius: float = 6.371e6,
) -> Float[Array, "Na Nb"]:
    """Pairwise great-circle (haversine) distances on a sphere.

    A ``metric`` for `localization_matrix` on geophysical grids.
    Coordinates are ``(latitude, longitude)`` in **radians**.

    Args:
        coords_a: First set of points ``(lat, lon)`` in radians, shape ``(Na, 2)``.
        coords_b: Second set of points ``(lat, lon)`` in radians, shape ``(Nb, 2)``.
        radius: Sphere radius in the units of the returned distance (default the
            Earth mean radius, ``6.371e6`` m).

    Returns:
        Great-circle distance matrix of shape ``(Na, Nb)``.
    """
    lat_a = coords_a[:, 0][:, None]
    lon_a = coords_a[:, 1][:, None]
    lat_b = coords_b[:, 0][None, :]
    lon_b = coords_b[:, 1][None, :]
    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    h = (
        jnp.sin(dlat / 2.0) ** 2
        + jnp.cos(lat_a) * jnp.cos(lat_b) * jnp.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius * jnp.arcsin(jnp.sqrt(jnp.clip(h, 0.0, 1.0)))


def localization_matrix(
    coords_a: Float[Array, "Na D"],
    coords_b: Float[Array, "Nb D"],
    c: float | Float[Array, ""],
    metric: Callable[
        [Float[Array, "Na D"], Float[Array, "Nb D"]], Float[Array, "Na Nb"]
    ] = euclidean_distance,
) -> Float[Array, "Na Nb"]:
    """Pairwise Gaspari-Cohn taper ``rho(dist(a_i, b_j); c)``.

    Use this to build the ``rho_xy`` (state-obs) and ``rho_yy`` (obs-obs)
    localization matrices consumed by `localized_kalman_gain`.

    Args:
        coords_a: First set of points, shape ``(Na, D)``.
        coords_b: Second set of points, shape ``(Nb, D)``.
        c: Gaspari-Cohn compact-support radius.
        metric: Pairwise distance function returning an ``(Na, Nb)`` matrix.
            Defaults to `euclidean_distance`; pass
            `haversine_distance` for spherical coordinates.

    Returns:
        Localization matrix of shape ``(Na, Nb)`` with entries in ``[0, 1]``.
    """
    return gaspari_cohn(metric(coords_a, coords_b), c)


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
    r"""Ensemble Kalman gain with Hadamard (Schur-product) localization.

    Computes

    $$
    K = (\rho_{xy} \circ P_{xy})\,(\rho_{yy} \circ P_{yy} + R)^{-1},
    $$

    where ``P_xy`` is the state-observation cross-covariance and ``P_yy`` the
    observation-space ensemble covariance. Tapering kills spurious long-range
    sample correlations; because Gaspari-Cohn is positive-definite, the Schur
    product theorem keeps ``rho_yy . P_yy`` PSD, so the innovation covariance
    stays invertible.

    This is the localized counterpart of `ensemble_kalman_gain`. Unlike
    that routine, the Hadamard product destroys the low-rank structure, so the
    innovation covariance is materialized densely and the solve is
    ``O(N M + M^3)``. Recover the unlocalized gain as the ``c -> inf`` limit
    (``rho_xy = rho_yy = 1``).

    Args:
        particles: Prior ensemble in state space, shape ``(J, N)``.
        obs_particles: Prior ensemble in observation space, shape ``(J, M)``.
        obs_noise: Observation error covariance operator ``R``, shape ``(M, M)``.
        rho_xy: State-observation localization matrix, shape ``(N, M)``.
        rho_yy: Observation-observation localization matrix, shape ``(M, M)``.
        solver: Optional solver strategy for the dense innovation solve.
        bessel: Use the ``1 / (J - 1)`` divisor (EnKF convention, default).

    Returns:
        Dense localized Kalman gain of shape ``(N, M)``.
    """
    if particles.shape[0] != obs_particles.shape[0]:
        raise ValueError(
            "particles and obs_particles must share the same ensemble size, "
            f"got J={particles.shape[0]} and J={obs_particles.shape[0]}."
        )
    cross_cov = ensemble_cross_covariance(particles, obs_particles, bessel=bessel)
    obs_cov = ensemble_cross_covariance(obs_particles, obs_particles, bessel=bessel)

    localized_cross = rho_xy * cross_cov
    innovation = rho_yy * obs_cov + obs_noise.as_matrix()
    innovation = symmetrize(innovation)
    innovation_op = lx.MatrixLinearOperator(innovation, lx.positive_semidefinite_tag)
    return solve_rows(innovation_op, localized_cross, solver=solver)
