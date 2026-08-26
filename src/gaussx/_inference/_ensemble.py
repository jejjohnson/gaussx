"""Ensemble covariance and cross-covariance recipes."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from gaussx._linalg._linalg import solve_rows
from gaussx._linalg._mixed_precision import stable_squared_distances
from gaussx._linalg._symmetrize import symmetrize
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._lazy_algebra import ScaledOperator
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._operators._sum_kronecker import SumOfKroneckers
from gaussx._primitives._cholesky import DenseFallbackWarning, cholesky
from gaussx._primitives._sqrt import dense_symmetric_sqrt
from gaussx._strategies._base import AbstractSolverStrategy


def ensemble_covariance(
    particles: Float[Array, "J N"],
    *,
    bessel: bool = False,
) -> LowRankUpdate:
    r"""Empirical covariance from an ensemble as a low-rank operator.

    Returns ``C = c X'^T X'`` with ``c = 1 / J`` when ``bessel=False``
    (default, maximum likelihood) and ``c = 1 / (J - 1)`` when
    ``bessel=True`` (unbiased / ensemble Kalman filter convention).
    The result is a ``LowRankUpdate`` of rank ``<= J-1`` rather than
    materializing the full ``(N, N)`` matrix.  Efficient when
    ``J << N``.

    Args:
        particles: Ensemble of shape ``(J, N)``.
        bessel: If True, apply the ``1 / (J - 1)`` Bessel correction
            used throughout the ensemble Kalman filter literature. This
            lower-level helper defaults to False for backwards compatibility;
            `ensemble_kalman_gain` defaults to True for the EnKF
            convention.

    Returns:
        A ``LowRankUpdate`` operator representing the empirical
        covariance, with a zero base and ``J``-column low-rank factor.
    """
    J, N = particles.shape
    _check_ensemble_size(J, bessel)
    mean = jnp.mean(particles, axis=0)
    deviations = particles - mean[None, :]  # (J, N)

    divisor = J - 1 if bessel else J
    U = deviations.T / jnp.sqrt(divisor)  # (N, J)

    base = lx.DiagonalLinearOperator(jnp.zeros(N, dtype=particles.dtype))
    return LowRankUpdate(base, U)


def ensemble_cross_covariance(
    particles_theta: Float[Array, "J N"],
    particles_G: Float[Array, "J M"],
    *,
    bessel: bool = False,
) -> Float[Array, "N M"]:
    r"""Cross-covariance between two ensemble sets.

    Computes ``C^{theta,G} = c sum_j (theta_j - bar)(G_j - bar)^T``
    with ``c = 1 / J`` by default or ``c = 1 / (J - 1)`` when
    ``bessel=True``.

    Args:
        particles_theta: First ensemble, shape ``(J, N)``.
        particles_G: Second ensemble, shape ``(J, M)``.
        bessel: If True, apply the ``1 / (J - 1)`` Bessel correction
            used by ensemble Kalman filter recipes. This lower-level helper
            defaults to False for backwards compatibility; `ensemble_kalman_gain`
            defaults to True for the EnKF convention.

    Returns:
        Cross-covariance array of shape ``(N, M)``.
    """
    J = particles_theta.shape[0]
    _check_ensemble_size(J, bessel)
    dev_theta = particles_theta - jnp.mean(particles_theta, axis=0, keepdims=True)
    dev_G = particles_G - jnp.mean(particles_G, axis=0, keepdims=True)
    divisor = J - 1 if bessel else J
    return (dev_theta.T @ dev_G) / divisor


def _check_ensemble_size(J: int, bessel: bool) -> None:
    if J < 1:
        raise ValueError(f"Ensemble must have at least one particle, got J={J}.")
    if bessel and J < 2:
        raise ValueError(
            "Bessel correction requires J >= 2 particles (divisor is J - 1); "
            f"got J={J}. Pass bessel=False for a maximum-likelihood divisor."
        )


def ensemble_kalman_gain(
    particles: Float[Array, "J N"],
    obs_particles: Float[Array, "J M"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
    bessel: bool = True,
) -> Float[Array, "N M"]:
    r"""Kalman gain from an ensemble and its image in observation space.

    Computes ``K = C^{xH} (C^{HH} + R)^{-1}``, where ``C^{xH}`` is the
    state-observation cross-covariance and ``C^{HH}`` is the
    observation-space ensemble covariance. The innovation covariance
    ``S = C^{HH} + R`` is assembled as a ``LowRankUpdate`` so
    ``solve_rows`` can use structural dispatch via the Woodbury identity.

    Args:
        particles: Prior ensemble in state space, shape ``(J, N)``.
        obs_particles: Prior ensemble in observation space, shape ``(J, M)``.
        obs_noise: Observation error covariance operator, shape ``(M, M)``.
        solver: Optional solver strategy. ``None`` uses structural dispatch.
        bessel: Defaults to True, unlike the lower-level covariance helpers,
            because this recipe follows the unbiased EnKF convention. Use
            False for maximum-likelihood recipes with a ``1 / J`` divisor.

    Returns:
        Dense Kalman gain of shape ``(N, M)``.
    """
    if particles.shape[0] != obs_particles.shape[0]:
        raise ValueError(
            "particles and obs_particles must share the same ensemble size, "
            f"got J={particles.shape[0]} and J={obs_particles.shape[0]}."
        )
    cross_cov = ensemble_cross_covariance(
        particles,
        obs_particles,
        bessel=bessel,
    )
    innovation_cov = ensemble_covariance(obs_particles, bessel=bessel)
    innovation_cov = LowRankUpdate(obs_noise, innovation_cov.U)
    return solve_rows(innovation_cov, cross_cov, solver=solver)


# ---------------------------------------------------------------------------
# Covariance localization (Gaspari-Cohn taper + Hadamard-localized gain)
# ---------------------------------------------------------------------------


def gaspari_cohn(r: Float[Array, "*shape"], c: float) -> Float[Array, "*shape"]:
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
    c: float,
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


# ---------------------------------------------------------------------------
# Ensemble analysis (stochastic / perturbed-observation EnKF update)
# ---------------------------------------------------------------------------


def _noise_factor(
    obs_noise: lx.AbstractLinearOperator,
    *,
    allow_dense: bool = False,
) -> lx.AbstractLinearOperator:
    """A factor ``L`` with ``L L^T = R``, valid for singular ``R``.

    Perturbations are drawn as ``eps_j = L n_j``, so ``L`` has three jobs, and
    neither `gaussx.cholesky` nor `gaussx.sqrt` does all three on its own:

    1. **Exact.** ``enkf_analysis`` promises ``eps ~ N(0, R)``. An approximate
       factor gives the perturbations the wrong covariance and biases the
       analysis silently, so a truncated Lanczos square root is not an
       acceptable default however cheap it is.
    2. **Defined for a singular ``R``.** That is a documented, supported case
       -- ``dense_innovation=True`` exists for it -- and perturbations are
       drawn before that flag is ever consulted, so a positive-*definite*-only
       factor poisons the analysis with ``NaN`` regardless of what the caller
       asked for.
    3. **Structure-preserving.** ``M`` is routinely large enough that
       materialising ``R`` is the thing the structured operator existed to
       avoid, and a dense fallback trades a ``NaN`` for an ``OOM``.

    `gaussx.cholesky` fails (2) at every dense leaf -- whether that leaf is the
    whole operator or one block of a `gaussx.BlockDiag` -- because it bottoms
    out in `jax.numpy.linalg.cholesky`, which returns ``NaN`` for a positive
    *semi*-definite matrix such as ``diag(1, 1, 0)``. Only the diagonal case
    escapes, its "Cholesky" being an elementwise ``sqrt`` that takes a zero in
    its stride. `gaussx.sqrt` fixes (2) but breaks (1) for
    `gaussx.SumOfKroneckers` and (3) for `gaussx.BlockTriDiag`.

    So the dispatch is by operator, taking whichever is better per structure:

    - Diagonal and identity: `gaussx.cholesky`, already exact and PSD-safe.
    - `gaussx.BlockDiag` / `gaussx.Kronecker`: recurse. Both factor into
      independent leaves -- ``(L_1 (x) L_2)(L_1 (x) L_2)^T = A_1 (x) A_2`` --
      so structure survives *and* each leaf gets the PSD-safe treatment.
    - `gaussx.BlockTriDiag`: `gaussx.cholesky`, whose banded recurrence is
      ``O(N d^3)`` against ``O((N d)^3)`` dense. It has no PSD-safe blockwise
      analogue, so it is the one structure where (2) and (3) genuinely
      conflict -- hence ``allow_dense``, below.
    - Anything else, including every dense leaf: the symmetric square root.

    That last branch is where `gaussx.KroneckerSum` and
    `gaussx.SumOfKroneckers` land. Both densify -- exactly as `gaussx.cholesky`
    did -- rather than take their matrix-free `gaussx.sqrt` routes, which are
    respectively not traceable (a Python ``bool`` on a data-dependent
    definiteness check) and not exact.

    The result is the *symmetric* square root at dense leaves rather than a
    triangular factor. That is a different factorisation of the same ``R``:
    both satisfy ``L L^T = R``, so the draws are distributionally identical,
    but a given key gives a different (equally valid) realisation.

    Args:
        obs_noise: Observation error covariance, shape ``(M, M)``.
        allow_dense: Whether the caller's gain path materialises ``R`` anyway.
            When it does, requirement (3) is already spent and cannot justify
            declining (2), so `gaussx.BlockTriDiag` takes the PSD-safe dense
            factor too and a singular ``R`` stops being fatal. Only
            `enkf_analysis`'s Woodbury route -- no localization, structured
            innovation -- keeps ``R`` unmaterialised, so only there is the
            banded factor worth a ``NaN``.

    Returns:
        An operator ``L`` such that ``L L^T = obs_noise``.
    """
    if isinstance(obs_noise, lx.TaggedLinearOperator):
        return _noise_factor(obs_noise.operator, allow_dense=allow_dense)
    if isinstance(obs_noise, lx.IdentityLinearOperator | lx.DiagonalLinearOperator):
        return cholesky(obs_noise)
    if isinstance(obs_noise, BlockTriDiag) and not allow_dense:
        return cholesky(obs_noise)
    if isinstance(obs_noise, BlockDiag):
        return BlockDiag(
            *(_noise_factor(op, allow_dense=allow_dense) for op in obs_noise.operators)
        )
    if isinstance(obs_noise, Kronecker):
        return Kronecker(
            *(_noise_factor(op, allow_dense=allow_dense) for op in obs_noise.operators)
        )
    # Only worth saying when the caller could act on it: if the gain path
    # materialises R regardless, drawing the perturbations elsewhere saves
    # nothing.
    if isinstance(obs_noise, SumOfKroneckers) and not allow_dense:
        warnings.warn(
            "enkf_analysis materialises a SumOfKroneckers obs_noise to draw "
            "exact perturbations. For a matrix-free alternative, sample "
            "eps ~ N(0, R) yourself -- sumkronecker_sample or "
            "sqrt(obs_noise, lanczos_order=...) -- and pass them as "
            "perturbed_obs=observation + eps. Both are approximate, so that "
            "is an opt-in, not the default.",
            DenseFallbackWarning,
            stacklevel=2,
        )
    return lx.MatrixLinearOperator(dense_symmetric_sqrt(obs_noise.as_matrix()))


def _check_analysis_shapes(
    particles: Float[Array, "J N"],
    obs_particles: Float[Array, "J M"],
    observation: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    bessel: bool,
) -> tuple[int, int, int]:
    """Shape agreement for an ensemble analysis step. Returns ``(J, N, M)``."""
    n_ens, n_state = particles.shape
    _check_ensemble_size(n_ens, bessel)
    if obs_particles.shape[0] != n_ens:
        raise ValueError(
            "particles and obs_particles must share the same ensemble size, "
            f"got J={n_ens} and J={obs_particles.shape[0]}."
        )
    n_obs = obs_particles.shape[1]
    if observation.shape != (n_obs,):
        raise ValueError(
            f"observation must have shape ({n_obs},) to match obs_particles, "
            f"got {observation.shape}."
        )
    # Without this an operator of the wrong size broadcasts against the (M, M)
    # empirical covariance instead of raising -- a (1, 1) R against M = 3 adds
    # the scalar to every entry and yields a plausible but wrong gain.
    if (obs_noise.in_size(), obs_noise.out_size()) != (n_obs, n_obs):
        raise ValueError(
            f"obs_noise must be ({n_obs}, {n_obs}) to match obs_particles, got "
            f"({obs_noise.out_size()}, {obs_noise.in_size()})."
        )
    return n_ens, n_state, n_obs


def _check_localization_shapes(
    n_state: int,
    n_obs: int,
    localization: Float[Array, "N M"] | None,
    obs_localization: Float[Array, "M M"] | None,
) -> None:
    """Taper shapes. ``obs_localization`` is only consulted alongside a taper."""
    if localization is None:
        return
    # Broadcast-compatible but wrong shapes are the danger: an (N, 1) taper
    # repeats one observation's taper across all M, and a (1, 1)
    # obs_localization rescales the whole observation covariance. Both give
    # a plausible, wrong gain rather than an error.
    if localization.shape != (n_state, n_obs):
        raise ValueError(
            f"localization must have shape ({n_state}, {n_obs}) to match "
            f"particles and obs_particles, got {localization.shape}."
        )
    if obs_localization is not None and obs_localization.shape != (n_obs, n_obs):
        raise ValueError(
            f"obs_localization must have shape ({n_obs}, {n_obs}) to match "
            f"obs_particles, got {obs_localization.shape}."
        )


def _analysis_gain(
    particles: Float[Array, "J N"],
    obs_particles: Float[Array, "J M"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    localization: Float[Array, "N M"] | None,
    obs_localization: Float[Array, "M M"] | None,
    solver: AbstractSolverStrategy | None,
    use_dense: bool,
    bessel: bool,
) -> Float[Array, "N M"]:
    """``K = C^{xH} (C^{HH} + R)^{-1}`` by whichever route ``use_dense`` picks."""
    if localization is None and not use_dense:
        # Fewer members than observations: the Woodbury capacitance is (J, J)
        # and cheap, so let `ensemble_kalman_gain` keep the low-rank structure.
        return ensemble_kalman_gain(
            particles, obs_particles, obs_noise, solver=solver, bessel=bessel
        )  # (N, M)
    if localization is None:
        # A `LowRankUpdate` innovation would send `solve_rows` through Woodbury
        # and form a (J, J) capacitance matrix -- 320 GB at J = 200_000 -- so
        # assemble the (M, M) innovation densely instead. Same solve, same
        # answer.
        cross_cov = ensemble_cross_covariance(particles, obs_particles, bessel=bessel)
        obs_cov = ensemble_cross_covariance(obs_particles, obs_particles, bessel=bessel)
        innovation = symmetrize(obs_cov + obs_noise.as_matrix())  # (M, M)
        innovation_op = lx.MatrixLinearOperator(
            innovation, lx.positive_semidefinite_tag
        )
        return solve_rows(innovation_op, cross_cov, solver=solver)  # (N, M)
    rho_yy = (
        jnp.ones((obs_particles.shape[1],) * 2, dtype=particles.dtype)
        if obs_localization is None
        else obs_localization
    )
    return localized_kalman_gain(
        particles,
        obs_particles,
        obs_noise,
        localization,
        rho_yy,
        solver=solver,
        bessel=bessel,
    )  # (N, M)


def enkf_analysis(
    particles: Float[Array, "J N"],
    obs_particles: Float[Array, "J M"],
    observation: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    key: PRNGKeyArray | None = None,
    perturbed_obs: Float[Array, "J M"] | None = None,
    localization: Float[Array, "N M"] | None = None,
    obs_localization: Float[Array, "M M"] | None = None,
    solver: AbstractSolverStrategy | None = None,
    dense_innovation: bool | None = None,
    bessel: bool = True,
) -> Float[Array, "J N"]:
    r"""Stochastic (perturbed-observation) ensemble Kalman analysis step.

    Updates a prior ensemble $X^f$ toward an observation $y$:

    $$
    X^a_j = X^f_j + K\,(y + \varepsilon_j - \mathcal{H}(X^f_j)),
    \qquad \varepsilon_j \sim N(0, R),
    $$

    with $K$ from `ensemble_kalman_gain` (or `localized_kalman_gain` when
    ``localization`` is given). The observation operator enters only through
    ``obs_particles`` -- the image $\mathcal{H}(X^f)$ of the prior ensemble in
    observation space -- so nonlinear operators need no special handling.

    The perturbation $\varepsilon_j$ is what keeps the analysis spread correct.
    The deterministic update $X^a_j = X^f_j + K(y - \mathcal{H}(X^f_j))$ drives
    the ensemble covariance to $(I - KH)P(I - KH)^\top$ instead of $(I - KH)P$,
    i.e. under-dispersive. There is deliberately no ``perturb=False`` flag: the
    deterministic alternative is a different filter (the square-root / ETKF
    family, see `etkf_transform`), not an option on this one.

    Two ways to supply the observation perturbations:

    - ``key`` -- draw $\varepsilon_j \sim N(0, R)$ internally, via a Cholesky
      factor of ``obs_noise``.
    - ``perturbed_obs`` -- pass a pre-built perturbed-observation ensemble
      $y + \varepsilon_j$. Preferred when the same noise realisation must be
      reused across filters, and when the perturbations come from a nonlinear
      observation model rather than an additive $R$.

    Exactly one of ``key`` / ``perturbed_obs`` must be given.

    Known limitation. The update is a Gaussian one, applied in whatever
    coordinates the caller supplies. For a non-Gaussian prior it is biased, and
    the bias does **not** shrink with ensemble size -- it is an error of
    coordinates, not of sampling. On the lognormal / logit-normal prior of
    Chipilski (2025), whose exact posterior mean is ``[0.548062, 0.353937]``,
    the physical-space update plateaus several percent off that value and stays
    there as $J$ grows by two orders of magnitude.

    The fix is to conjugate the update with a bijection $\Gamma$ that
    Gaussianises the prior -- call this function on $\Gamma^{-1}(X^f)$ and map
    the result back through $\Gamma$: the ensemble Kalman filter's Gaussian
    assumption is a statement about coordinates, not about the algorithm. Pass
    the same ``perturbed_obs`` through both routes to compare them on one noise
    realisation.

    That conjugated update is *exact Bayes* only under conditions worth stating
    precisely, because it is easy to over-claim. It needs the population limit
    -- with a finite ensemble the gain is empirical and the perturbations are
    Monte Carlo, so the result is an estimate either way -- and it needs the
    observation model to be **affine with additive Gaussian noise** in the same
    latent coordinates that Gaussianise the prior. A merely "Gaussian
    likelihood" is not enough: $y = \zeta^2 + \varepsilon$ has Gaussian noise
    and a non-Gaussian posterior that no Kalman update reproduces. Outside
    those conditions conjugation is an approximation with no guaranteed
    ordering against the physical-space update -- usually much better, but a
    badly matched $\Gamma$ can make the latent joint less Gaussian and do
    worse.

    Args:
        particles: Prior ensemble in state space, shape ``(J, N)``.
        obs_particles: Prior ensemble in observation space, shape ``(J, M)``.
        observation: The observation, shape ``(M,)``.
        obs_noise: Observation error covariance $R$, shape ``(M, M)``.
        key: PRNG key for internally drawn perturbations. Mutually exclusive
            with ``perturbed_obs``.
        perturbed_obs: Pre-built perturbed observation ensemble, shape
            ``(J, M)``. Mutually exclusive with ``key``.
        localization: Optional state-observation taper $\rho_{xy}$, shape
            ``(N, M)``, e.g. from `localization_matrix`. When given, the gain
            comes from `localized_kalman_gain` instead of
            `ensemble_kalman_gain`.
        obs_localization: Optional observation-observation taper $\rho_{yy}$,
            shape ``(M, M)``. Only consulted when ``localization`` is given;
            defaults to all-ones, i.e. no tapering of the innovation
            covariance.
        solver: Optional solver strategy for the innovation solve. ``None``
            uses structural dispatch. A matrix-free strategy (e.g. `CGSolver`)
            wants ``dense_innovation=False`` so it is handed the structured
            operator instead of a materialised one.
        dense_innovation: Whether to form the ``(M, M)`` innovation covariance
            densely. ``None`` (default) chooses by shape, as described in the
            note below. ``False`` keeps the structured `LowRankUpdate` no
            matter the shapes -- what a matrix-free solver wants. ``True``
            forces the dense assembly, which is the way out when ``obs_noise``
            is only positive *semi*-definite, since the structured route
            solves against ``obs_noise`` itself.
        bessel: Use the $1/(J-1)$ divisor. Defaults to ``True``, matching
            `ensemble_kalman_gain`.

    Returns:
        Analysis ensemble, shape ``(J, N)``.

    Note:
        How the innovation covariance $C^{HH} + R$ is assembled defaults to a
        choice made from the static shapes, because the two regimes have wildly
        different costs. With $J < M$ the gain comes from
        `ensemble_kalman_gain`, which keeps the ensemble term low-rank and
        inverts a $(J, J)$ Woodbury capacitance -- the right choice for the
        geoscience regime of a few dozen members against many observations.
        With $J \ge M$ that capacitance is the larger of the two (320 GB at
        $J = 200{,}000$), so the $(M, M)$ innovation is formed densely instead.
        Both routes solve the same system and agree to round-off.

        Shapes are the wrong criterion in two cases, which is why
        ``dense_innovation`` exists to override it:

        - **A matrix-free solver.** With $J \ge M$ and $M$ still large, the
          dense assembly allocates an $(M, M)$ array before the solver is ever
          called -- around 40 GB at $M = 100{,}000$ in float32 -- even though
          an iterative strategy could work through matvecs on the structured
          operator. Pass ``dense_innovation=False``.
        - **Singular observation noise.** The Woodbury route solves against
          $R$ itself, so a positive *semi*-definite $R$ divides by zero and
          returns infinities or ``NaN`` even when $C^{HH} + R$ is perfectly
          invertible -- e.g. $R = \mathrm{diag}(1, 1, 0)$ with ensemble
          anomalies spanning the third observation direction. The $J < M$ path
          therefore requires $R$ to be positive **definite**; with a singular
          $R$, pass ``dense_innovation=True`` to solve the full innovation
          instead. This is not checked: PSD-ness of an arbitrary operator is
          not something this function can establish cheaply, and certainly not
          under ``jit``.

    Raises:
        ValueError: If neither or both of ``key`` / ``perturbed_obs`` are
            given, if the ensemble sizes disagree, or if the observation-space
            shapes disagree.

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> from gaussx import enkf_analysis
        >>> key, subkey = jr.split(jr.key(0))
        >>> prior = jr.normal(subkey, (500, 3))           # (J, N)
        >>> H = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> obs_prior = prior @ H.T                       # (J, M)
        >>> R = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> posterior = enkf_analysis(
        ...     prior, obs_prior, jnp.array([1.0, -1.0]), R, key=key
        ... )
        >>> posterior.shape
        (500, 3)
    """
    if (key is None) == (perturbed_obs is None):
        raise ValueError(
            "Pass exactly one of 'key' (draw perturbations from obs_noise) or "
            "'perturbed_obs' (supply them directly)."
        )

    n_ens, n_state, n_obs = _check_analysis_shapes(
        particles, obs_particles, observation, obs_noise, bessel
    )

    # Whether to form the (M, M) innovation densely. `None` picks by shape; an
    # explicit value overrides that, which is what a matrix-free solver needs.
    # Decided here rather than at the gain, because the factor below needs it.
    use_dense = n_ens >= n_obs if dense_innovation is None else dense_innovation

    # Branch on `key` rather than on `perturbed_obs`: the check above makes the
    # two equivalent, but this way each branch narrows the argument it uses.
    if key is not None:
        # eps_j = L n_j with R = L L^T. The factor dispatches on structure, so
        # a DiagonalLinearOperator stays diagonal and its matvec stays O(M) --
        # materialising R here would allocate a dense (M, M) and cost O(M^3),
        # which for the low-rank branch below (J < M, M possibly enormous) would
        # OOM before the gain is ever formed.
        #
        # Unless the gain is about to allocate that (M, M) anyway: every route
        # but Woodbury calls `obs_noise.as_matrix()`, and once the dense cost
        # is being paid regardless there is nothing left to protect by holding
        # on to a positive-definite-only factor.
        factor = _noise_factor(
            obs_noise, allow_dense=use_dense or localization is not None
        )
        noise = jr.normal(key, (n_ens, n_obs), dtype=particles.dtype)  # (J, M)
        perturbed = observation[None, :] + jax.vmap(factor.mv)(noise)  # (J, M)
    else:
        perturbed = perturbed_obs
        if perturbed is None or perturbed.shape != (n_ens, n_obs):
            raise ValueError(
                f"perturbed_obs must have shape ({n_ens}, {n_obs}) to match "
                f"obs_particles, got "
                f"{None if perturbed is None else perturbed.shape}."
            )

    _check_localization_shapes(n_state, n_obs, localization, obs_localization)

    gain = _analysis_gain(
        particles,
        obs_particles,
        obs_noise,
        localization=localization,
        obs_localization=obs_localization,
        solver=solver,
        use_dense=use_dense,
        bessel=bessel,
    )  # (N, M)

    innovation = perturbed - obs_particles  # (J, M)
    return particles + innovation @ gain.T  # (J, M) @ (M, N) -> (J, N)


# ---------------------------------------------------------------------------
# Ensemble inflation (multiplicative, RTPP, RTPS)
# ---------------------------------------------------------------------------


def inflate_multiplicative(
    ensemble: Float[Array, "J N"],
    factor: float,
) -> Float[Array, "J N"]:
    r"""Multiplicative ensemble inflation about the mean.

    Restores ensemble spread lost to sampling error / model collapse by scaling
    perturbations: ``x_j <- x_bar + factor (x_j - x_bar)``.

    Args:
        ensemble: Ensemble of shape ``(J, N)``.
        factor: Inflation factor ``>= 1`` (e.g. ``1.02``-``1.10``).

    Returns:
        Inflated ensemble, shape ``(J, N)``. The mean is unchanged.
    """
    mean = jnp.mean(ensemble, axis=0, keepdims=True)
    return mean + factor * (ensemble - mean)


def inflate_rtpp(
    posterior: Float[Array, "J N"],
    prior: Float[Array, "J N"],
    alpha: float,
) -> Float[Array, "J N"]:
    r"""Relaxation to prior perturbations (RTPP; Zhang et al. 2004).

    Relaxes posterior perturbations toward the prior perturbations while keeping
    the posterior mean: ``x'^a <- (1 - alpha) x'^a + alpha x'^f``, where the
    perturbations are taken about each ensemble's own mean.

    Args:
        posterior: Analysis ensemble, shape ``(J, N)``.
        prior: Forecast ensemble, shape ``(J, N)``.
        alpha: Relaxation weight in ``[0, 1]``.

    Returns:
        Relaxed analysis ensemble, shape ``(J, N)``. The posterior mean is
        preserved.
    """
    post_mean = jnp.mean(posterior, axis=0, keepdims=True)
    post_pert = posterior - post_mean
    prior_pert = prior - jnp.mean(prior, axis=0, keepdims=True)
    return post_mean + (1.0 - alpha) * post_pert + alpha * prior_pert


def inflate_rtps(
    posterior: Float[Array, "J N"],
    prior: Float[Array, "J N"],
    beta: float,
    eps: float = 1e-12,
) -> Float[Array, "J N"]:
    r"""Relaxation to prior spread (RTPS; Whitaker & Hamill 2012).

    Scales each posterior perturbation, per coordinate, so the analysis spread
    relaxes back toward the prior spread:
    ``x'^a <- x'^a [ (1 - beta) + beta sigma^f / sigma^a ]``, with ``sigma`` the
    per-coordinate ensemble standard deviation.

    Args:
        posterior: Analysis ensemble, shape ``(J, N)``.
        prior: Forecast ensemble, shape ``(J, N)``.
        beta: Relaxation weight in ``[0, 1]``.
        eps: Floor on the posterior std to avoid division by zero.

    Returns:
        Spread-restored analysis ensemble, shape ``(J, N)``. The posterior mean
        is preserved.
    """
    post_mean = jnp.mean(posterior, axis=0, keepdims=True)
    post_pert = posterior - post_mean
    sigma_post = jnp.std(posterior, axis=0)
    sigma_prior = jnp.std(prior, axis=0)
    scale = (1.0 - beta) + beta * sigma_prior / (sigma_post + eps)
    return post_mean + post_pert * scale[None, :]


# ---------------------------------------------------------------------------
# Ensemble transform (ETKF) -- deterministic square-root analysis
# ---------------------------------------------------------------------------


def etkf_transform(
    obs_particles: Float[Array, "J M"],
    y: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    inflation: float = 1.0,
) -> tuple[Float[Array, " J"], Float[Array, "J J"]]:
    r"""Ensemble Transform Kalman Filter (ETKF) analysis weights.

    Deterministic (perturbed-obs-free) ensemble square-root analysis in the
    ``J``-dimensional ensemble space (Bishop et al. 2001; Hunt et al. 2007).
    With raw observation perturbations ``Y = H X'^f`` (columns are members) and
    ``d = y - H x_bar^f``,

    $$
    \tilde{A}^{-1} = \tfrac{J-1}{\lambda} I + Y^T R^{-1} Y, \qquad
    \bar{w} = \tilde{A}\, Y^T R^{-1} d, \qquad
    W = \big((J-1)\,\tilde{A}\big)^{1/2},
    $$

    where ``lambda`` is the (multiplicative) ``inflation`` and ``W`` is the
    **symmetric** square root. The analysis ensemble is reconstructed as

    $$
    \bar{x}^a = \bar{x}^f + X'^f \bar{w}, \qquad X'^a = X'^f\, W.
    $$

    The symmetric (eigendecomposition) square root -- not a Cholesky factor --
    is required: because the observation perturbations are zero-mean, ``1`` is
    an eigenvector of ``W`` with eigenvalue ``1``, which makes the transform
    exactly mean-preserving (``sum_j X'^a_j = 0``).

    Args:
        obs_particles: Forecast ensemble in observation space, shape ``(J, M)``.
        y: Observation vector, shape ``(M,)``.
        obs_noise: Observation error covariance operator ``R``, shape ``(M, M)``.
        inflation: Multiplicative covariance inflation ``lambda >= 1``, applied
            to the prior term ``(J - 1) / lambda``.

    Returns:
        ``(w_mean, transform)`` where ``w_mean`` has shape ``(J,)`` and
        ``transform`` has shape ``(J, J)``. Apply to forecast state
        perturbations ``Xp`` (shape ``(J, N)``) as
        ``x_bar^a = x_bar^f + w_mean @ Xp`` and ``X'^a = transform @ Xp``.
    """
    n_ens = obs_particles.shape[0]
    obs_mean = jnp.mean(obs_particles, axis=0)
    obs_pert = obs_particles - obs_mean[None, :]  # (J, M), zero-mean rows

    r_matrix = obs_noise.as_matrix()
    # R^{-1} applied to the (M, .) right-hand sides.
    rinv_pert = jnp.linalg.solve(r_matrix, obs_pert.T)  # (M, J)
    rinv_d = jnp.linalg.solve(r_matrix, y - obs_mean)  # (M,)

    precision = (n_ens - 1) / inflation * jnp.eye(n_ens) + obs_pert @ rinv_pert
    precision = symmetrize(precision)
    analysis_cov = jnp.linalg.inv(precision)  # tilde A, (J, J)

    w_mean = analysis_cov @ (obs_pert @ rinv_d)  # (J,)
    transform = _symmetric_sqrt((n_ens - 1) * analysis_cov)
    return w_mean, transform


def _symmetric_sqrt(matrix: Float[Array, "J J"]) -> Float[Array, "J J"]:
    """Symmetric (eigendecomposition) square root of an SPD matrix.

    Thin alias for `gaussx._primitives._sqrt.dense_symmetric_sqrt`, which
    carries a Sylvester-equation JVP so that the derivative stays finite at
    repeated eigenvalues. `etkf_transform` square-roots an analysis covariance
    that always has them -- ``Y R^-1 Y^T`` has rank at most ``min(M, J - 1)``,
    so the prior eigenvalue survives with multiplicity ``J - M`` whenever
    ``M < J`` -- and while the naive derivative happens to stay finite on the
    tangents that arise there, it is one input away from not doing so.
    """
    return dense_symmetric_sqrt(matrix)


# ---------------------------------------------------------------------------
# Ensemble Kalman inversion (EKI)
# ---------------------------------------------------------------------------


def eki_step(
    particles: Float[Array, "J N"],
    obs_particles: Float[Array, "J M"],
    observation: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    dt: float | Float[Array, ""] = 1.0,
    step: lx.AbstractLinearOperator | None = None,
    key: PRNGKeyArray | None = None,
    perturbed_obs: Float[Array, "J M"] | None = None,
    deterministic: bool = False,
    localization: Float[Array, "N M"] | None = None,
    obs_localization: Float[Array, "M M"] | None = None,
    solver: AbstractSolverStrategy | None = None,
    dense_innovation: bool | None = None,
    bessel: bool = True,
) -> Float[Array, "J N"]:
    r"""One ensemble Kalman inversion (EKI) update.

    A single tempered Kalman update of an ensemble against a fixed
    observation (Iglesias, Law & Stuart 2013). The gain is
    $K = C^{uG}(C^{GG} + R/\Delta t)^{-1}$, and the stochastic
    (perturbed-observation) update is

    $$
    u_j \leftarrow u_j + \Lambda\,K\,
        \big(y + \varepsilon_j/\sqrt{\Delta t} - \mathcal{G}(u_j)\big),
    \qquad \varepsilon_j \sim N(0, R).
    $$

    This is `enkf_analysis` with two knobs added, and reduces to it exactly at
    ``dt=1``, ``step=None``. Everything the forward model does enters through
    ``obs_particles``, so this function is pure array-in / array-out: there is
    no iteration, no stopping rule, and no $\mathcal{G}$. The driver that
    supplies the schedule lives outside this package.

    **Tempering (``dt``).** One iteration replaces $R$ by $R/\Delta t$, i.e.
    a likelihood raised to the power $\Delta t$. Over a schedule with
    $\sum_n \Delta t_n = 1$ the composition is *exactly* one Bayesian update
    in the linear-Gaussian population limit -- the precisions add,
    $C_N^{-1} = C_0^{-1} + \sum_n \Delta t_n\, A^\top R^{-1} A$ -- which is
    what makes an EKI schedule a tempering path rather than a heuristic. The
    sum condition is load-bearing: at $\sum \Delta t_n \neq 1$ the result is
    the posterior of a different problem, over- or under-weighting the data.
    ``dt`` may be a traced scalar, so an adaptive schedule (see
    `discrepancy_step_size`) stays inside ``jit``.

    **State-side step (``step``).** In the gradient-flow view
    $\dot{u} = -C^{uu}\nabla\Phi(u)$, ``step`` is the operator $\Lambda$ in
    the Euler step $u \leftarrow u + \Lambda C^{uG} S^{-1}(y - \mathcal{G}(u))$.
    It is applied by ``step.mv`` to each member's *increment*, so a
    `gaussx.BlockDiag` of scaled identities gives a different rate per state
    block -- parameters, latents, initial conditions -- without densifying an
    $(N, N)$ matrix. $\Lambda$ changes the trajectory, not the fixed point:
    where $K(y - \bar{\mathcal{G}}) = 0$ the increment is zero for every
    $\Lambda$, invertible or not.

    **Deterministic variant.** ``deterministic=True`` replaces the perturbed
    observations with an ETKF square-root transform (`etkf_transform` at
    $R/\Delta t$), applied to the increment so that $\Lambda$ still acts on a
    difference:

    $$
    \bar{u} \leftarrow \bar{u} + \Lambda K (y - \bar{\mathcal{G}}),
    \qquad
    U'^a = U'^f + \Lambda\,(W U'^f - U'^f).
    $$

    At $\Lambda = I$ the anomaly update collapses to $U'^a = W U'^f$, i.e.
    plain `etkf_transform`. This is the variant to use for the exactness
    property above: the stochastic one is exact only in expectation, so a
    finite ensemble carries Monte Carlo error on top of the tempering.

    Args:
        particles: Prior ensemble in state space, shape ``(J, N)``.
        obs_particles: Its image $\mathcal{G}(u_j)$ in observation space,
            shape ``(J, M)``.
        observation: The observation $y$, shape ``(M,)``.
        obs_noise: Observation error covariance $R$, shape ``(M, M)``. Scaled
            to $R/\Delta t$ as a lazy `gaussx.ScaledOperator`, never
            materialized here, so a structured $R$ keeps its structured solve.
        dt: Observation-side tempering step $\Delta t > 0$. Positivity is not
            checked -- it may be traced.
        step: State-side operator $\Lambda$, shape ``(N, N)``. ``None`` is the
            identity, and skips the matvec rather than building one.
        key: PRNG key for internally drawn perturbations
            $\varepsilon_j \sim N(0, R)$, which are then scaled by
            $1/\sqrt{\Delta t}$ to match $R/\Delta t$. Mutually exclusive with
            ``perturbed_obs``; both must be ``None`` when ``deterministic``.
        perturbed_obs: Pre-built perturbed observation ensemble, shape
            ``(J, M)``, used **as given** -- the caller owns the
            $1/\sqrt{\Delta t}$ scaling. Mutually exclusive with ``key``.
        deterministic: Use the ETKF square-root transform instead of perturbed
            observations.
        localization: Optional state-observation taper $\rho_{xy}$, shape
            ``(N, M)``. Stochastic variant only.
        obs_localization: Optional observation-observation taper $\rho_{yy}$,
            shape ``(M, M)``. Only consulted when ``localization`` is given.
        solver: Optional solver strategy for the innovation solve. ``None``
            uses structural dispatch.
        dense_innovation: Whether to form the ``(M, M)`` innovation densely.
            ``None`` chooses by shape. Same contract as `enkf_analysis`,
            including that a positive *semi*-definite $R$ needs ``True``.
        bessel: Use the $1/(J-1)$ divisor. Must stay ``True`` when
            ``deterministic``, since `etkf_transform` is $1/(J-1)$ throughout
            and a mismatched gain would move the mean and the anomalies by
            inconsistent amounts.

    Returns:
        The updated ensemble, shape ``(J, N)``.

    Raises:
        ValueError: If the shapes disagree; if ``key`` / ``perturbed_obs`` are
            not given exactly once in the stochastic variant, or given at all
            in the deterministic one; or if ``deterministic`` is combined with
            ``localization`` or with ``bessel=False``.

    Note:
        ``deterministic=True`` rejects ``localization`` rather than ignoring
        it. Schur-product localization has no square-root analogue: tapering
        the gain but not the transform would localize the mean update and
        leave the anomalies unlocalized, an inconsistent analysis that looks
        like a working one. (The LETKF localizes by domain decomposition
        instead, which is a different construction, not this argument.)

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> from gaussx import eki_step
        >>> key, subkey = jr.split(jr.key(0))
        >>> u = jr.normal(subkey, (200, 3))                 # (J, N)
        >>> A = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> G = u @ A.T                                     # (J, M)
        >>> R = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> eki_step(u, G, jnp.array([1.0, -1.0]), R, dt=0.5, key=key).shape
        (200, 3)
    """
    n_ens, n_state, n_obs = _check_analysis_shapes(
        particles, obs_particles, observation, obs_noise, bessel
    )
    if deterministic:
        if key is not None or perturbed_obs is not None:
            raise ValueError(
                "The deterministic variant draws no perturbations; pass "
                "neither 'key' nor 'perturbed_obs' with deterministic=True."
            )
        if localization is not None:
            raise ValueError(
                "deterministic=True does not support 'localization': tapering "
                "the gain without tapering the ETKF transform would localize "
                "the mean update and not the anomalies. Use the stochastic "
                "variant, or localize by domain decomposition (LETKF)."
            )
        if not bessel:
            raise ValueError(
                "deterministic=True requires bessel=True, because "
                "etkf_transform uses the 1 / (J - 1) divisor throughout and a "
                "1 / J gain would update the mean and the anomalies by "
                "inconsistent amounts."
            )
    elif (key is None) == (perturbed_obs is None):
        raise ValueError(
            "Pass exactly one of 'key' (draw perturbations from obs_noise) or "
            "'perturbed_obs' (supply them directly)."
        )
    if step is not None and (step.in_size(), step.out_size()) != (n_state, n_state):
        raise ValueError(
            f"step must be ({n_state}, {n_state}) to match particles, got "
            f"({step.out_size()}, {step.in_size()})."
        )
    _check_localization_shapes(n_state, n_obs, localization, obs_localization)

    dt = jnp.asarray(dt)
    if dt.ndim != 0:
        raise ValueError(f"dt must be a scalar, got shape {dt.shape}.")
    # R / dt as a lazy scale. `solve` unwraps a MulLinearOperator by dividing
    # the inner solve, so a Diagonal / BlockDiag / Kronecker R keeps its own
    # solve rather than falling back to a dense (M, M). At dt = 1 the scalar is
    # exactly 1.0 and every route is bit-for-bit `enkf_analysis`.
    tempered_noise = ScaledOperator(obs_noise, 1.0 / dt)

    use_dense = n_ens >= n_obs if dense_innovation is None else dense_innovation
    gain = _analysis_gain(
        particles,
        obs_particles,
        tempered_noise,
        localization=localization,
        obs_localization=obs_localization,
        solver=solver,
        use_dense=use_dense,
        bessel=bessel,
    )  # (N, M)

    if deterministic:
        obs_mean = jnp.mean(obs_particles, axis=0)  # (M,)
        anomalies = particles - jnp.mean(particles, axis=0, keepdims=True)  # (J, N)
        _, transform = etkf_transform(obs_particles, observation, tempered_noise)
        mean_increment = gain @ (observation - obs_mean)  # (N,)
        # The increment, not the transformed anomalies: Lambda acts on
        # differences, so Lambda = I leaves `transform @ anomalies` exactly.
        anomaly_increment = transform @ anomalies - anomalies  # (J, N)
        if step is not None:
            mean_increment = step.mv(mean_increment)
            anomaly_increment = jax.vmap(step.mv)(anomaly_increment)
        return particles + mean_increment[None, :] + anomaly_increment

    if key is not None:
        # eps_j ~ N(0, R) drawn against the *unscaled* R, then scaled by
        # 1/sqrt(dt) to give N(0, R/dt). Drawing against R/dt instead would
        # hand `_noise_factor` a MulLinearOperator it has no structured branch
        # for, and densify an (M, M) for nothing.
        factor = _noise_factor(
            obs_noise, allow_dense=use_dense or localization is not None
        )
        noise = jr.normal(key, (n_ens, n_obs), dtype=particles.dtype)  # (J, M)
        perturbation = jax.vmap(factor.mv)(noise) / jnp.sqrt(dt)  # (J, M)
        perturbed = observation[None, :] + perturbation  # (J, M)
    else:
        perturbed = perturbed_obs
        if perturbed is None or perturbed.shape != (n_ens, n_obs):
            raise ValueError(
                f"perturbed_obs must have shape ({n_ens}, {n_obs}) to match "
                f"obs_particles, got "
                f"{None if perturbed is None else perturbed.shape}."
            )

    increment = (perturbed - obs_particles) @ gain.T  # (J, M) @ (M, N) -> (J, N)
    if step is not None:
        increment = jax.vmap(step.mv)(increment)
    return particles + increment


def tikhonov_augment(
    particles: Float[Array, "J N"],
    obs_particles: Float[Array, "J M"],
    observation: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    prior_mean: Float[Array, " N"],
    prior_cov: lx.AbstractLinearOperator,
) -> tuple[
    Float[Array, "J M+N"],
    Float[Array, " M+N"],
    lx.AbstractLinearOperator,
]:
    r"""Observation augmentation for Tikhonov-regularised EKI (TEKI).

    Puts the prior $N(m_0, C_0)$ into an EKI step by treating the state as its
    own observation (Chada, Stuart & Tong 2020):

    $$
    y_{\text{aug}} = \begin{bmatrix} y \\ m_0 \end{bmatrix},
    \qquad
    \mathcal{G}_{\text{aug}}(u) = \begin{bmatrix} \mathcal{G}(u) \\ u
        \end{bmatrix},
    \qquad
    R_{\text{aug}} = \operatorname{blockdiag}(R, C_0),
    $$

    so the augmented least-squares functional is the regularised one,
    $\tfrac12\|y - \mathcal{G}(u)\|_R^2 + \tfrac12\|u - m_0\|_{C_0}^2$.
    Unregularised EKI collapses onto the data-misfit minimiser and, for an
    ill-posed problem, keeps going; the prior term is what stops it.

    A helper rather than a flag on `eki_step`: the triple goes straight into
    `eki_step` (and into `discrepancy_step_size`, whose $M$ is then $M + N$),
    nothing inside the step knows about priors, and $C_0$ stays an operator, so
    a `gaussx.Kronecker` prior keeps its structured solve inside the
    `gaussx.BlockDiag`.

    Args:
        particles: Ensemble in state space, shape ``(J, N)``.
        obs_particles: Its image $\mathcal{G}(u_j)$, shape ``(J, M)``.
        observation: The observation $y$, shape ``(M,)``.
        obs_noise: Observation error covariance $R$, shape ``(M, M)``.
        prior_mean: Prior mean $m_0$, shape ``(N,)``.
        prior_cov: Prior covariance $C_0$, shape ``(N, N)``.

    Returns:
        ``(obs_particles_aug, observation_aug, obs_noise_aug)`` with shapes
        ``(J, M + N)``, ``(M + N,)`` and ``(M + N, M + N)``. Pass them to
        `eki_step` in place of ``obs_particles``, ``observation`` and
        ``obs_noise``; ``particles`` is unchanged.

    Raises:
        ValueError: If any of the shapes disagree.

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> from gaussx import eki_step, tikhonov_augment
        >>> key, subkey = jr.split(jr.key(0))
        >>> u = jr.normal(subkey, (200, 3))
        >>> A = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> R = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> C0 = lx.DiagonalLinearOperator(jnp.ones(3))
        >>> G_aug, y_aug, R_aug = tikhonov_augment(
        ...     u, u @ A.T, jnp.array([1.0, -1.0]), R, jnp.zeros(3), C0
        ... )
        >>> eki_step(u, G_aug, y_aug, R_aug, key=key).shape
        (200, 3)
    """
    n_ens, n_state = particles.shape
    n_obs = obs_particles.shape[1]
    if obs_particles.shape[0] != n_ens:
        raise ValueError(
            "particles and obs_particles must share the same ensemble size, "
            f"got J={n_ens} and J={obs_particles.shape[0]}."
        )
    if observation.shape != (n_obs,):
        raise ValueError(
            f"observation must have shape ({n_obs},) to match obs_particles, "
            f"got {observation.shape}."
        )
    if (obs_noise.in_size(), obs_noise.out_size()) != (n_obs, n_obs):
        raise ValueError(
            f"obs_noise must be ({n_obs}, {n_obs}) to match obs_particles, got "
            f"({obs_noise.out_size()}, {obs_noise.in_size()})."
        )
    if prior_mean.shape != (n_state,):
        raise ValueError(
            f"prior_mean must have shape ({n_state},) to match particles, got "
            f"{prior_mean.shape}."
        )
    if (prior_cov.in_size(), prior_cov.out_size()) != (n_state, n_state):
        raise ValueError(
            f"prior_cov must be ({n_state}, {n_state}) to match particles, got "
            f"({prior_cov.out_size()}, {prior_cov.in_size()})."
        )

    obs_particles_aug = jnp.concatenate([obs_particles, particles], axis=1)
    observation_aug = jnp.concatenate([observation, prior_mean])
    return obs_particles_aug, observation_aug, BlockDiag(obs_noise, prior_cov)


def discrepancy_step_size(
    obs_particles: Float[Array, "J M"],
    observation: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    remaining: Float[Array, ""],
    bessel: bool = True,
) -> Float[Array, ""]:
    r"""Adaptive EKI tempering step: the data misfit controller.

    Iglesias & Yang (2021), eq. (14) -- the selection rule of their EKI-DMC
    (Algorithm 3), which needs no tuning parameter. With the per-particle
    least-squares functional

    $$
    \Phi_j = \tfrac12 \big\| R^{-1/2}(y - \mathcal{G}(u_j)) \big\|^2
           = \tfrac12 (y - \mathcal{G}(u_j))^\top R^{-1}
             (y - \mathcal{G}(u_j)),
    $$

    and $\bar\Phi$, $\sigma^2_\Phi$ its empirical mean and variance across the
    ensemble,

    $$
    \Delta t = \min\!\left(
        \max\!\left(\frac{M}{2\bar\Phi},\;
                    \sqrt{\frac{M}{2\sigma^2_\Phi}}\right),\;
        \text{remaining}\right).
    $$

    The two candidates are the paper's statistical discrepancy principle
    applied to the tempered sub-problem $y = \mathcal{G}(u) + \sqrt{\alpha}\eta$
    with $\alpha = 1/\Delta t$: since $\|R^{-1/2}(y - \mathcal{G}(u))\|^2$ is
    $\chi^2_M$ under the correct model, it has mean $M$ (their C1, accuracy,
    giving $M/2\bar\Phi$) and variance $2M$ (their C2, uncertainty, giving
    $\sqrt{M/2\sigma^2_\Phi}$). The **max** is deliberate and enforces *at
    least one* of the two, not both: their Remark 2 notes that a wide prior
    makes $\bar\Phi \ll \sigma_\Phi$, so C1 binds; a narrow prior centred far
    from the truth flips it, and C2 then licenses the larger step. The outer
    min is the tempering budget $1 - t_n$, which is also the stopping rule --
    the driver halts on the iteration where it binds.

    Note:
        gh-230 specified both terms in the misfit of the ensemble *mean*,
        $\|R^{-1/2}(y - \bar{\mathcal{G}})\|^2$. This follows the paper
        instead: eq. (13) defines $\Phi_n$ as the set of per-particle
        functionals and eq. (14) takes their mean and variance. The
        distinction is not cosmetic -- the second term's $\sigma^2_\Phi$ is
        identically zero for any single vector, so a mean-misfit reading would
        make C2 infinite and the ``max`` vacuous.

    Only ``obs_noise`` solves are used; $R^{-1/2}$ is never formed.

    Args:
        obs_particles: Ensemble in observation space $\mathcal{G}(u_j)$, shape
            ``(J, M)``. Under `tikhonov_augment` pass the augmented ensemble,
            so that $M$ counts the augmented observations.
        observation: The observation $y$, shape ``(M,)``.
        obs_noise: Observation error covariance $R$, shape ``(M, M)``.
        remaining: Tempering budget left, $1 - \sum_{k<n}\Delta t_k$. May be
            traced.
        bessel: Use the $1/(J-1)$ divisor for $\sigma^2_\Phi$. Defaults to
            ``True``, matching the rest of this module.

    Returns:
        The step $\Delta t$, a scalar. Never exceeds ``remaining``, and is
        positive whenever ``remaining`` is: a degenerate ensemble
        ($\sigma^2_\Phi = 0$) sends the second candidate to $+\infty$, so the
        ``max`` saturates and ``remaining`` is returned.

    Raises:
        ValueError: If the shapes disagree, or if ``bessel`` is set with
            ``J < 2``.

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> from gaussx import discrepancy_step_size
        >>> G = jr.normal(jr.key(0), (50, 4))
        >>> R = lx.DiagonalLinearOperator(jnp.ones(4))
        >>> dt = discrepancy_step_size(
        ...     G, jnp.zeros(4), R, remaining=jnp.asarray(1.0)
        ... )
        >>> bool(0.0 < dt <= 1.0)
        True
    """
    n_ens, n_obs = obs_particles.shape
    _check_ensemble_size(n_ens, bessel)
    if observation.shape != (n_obs,):
        raise ValueError(
            f"observation must have shape ({n_obs},) to match obs_particles, "
            f"got {observation.shape}."
        )
    if (obs_noise.in_size(), obs_noise.out_size()) != (n_obs, n_obs):
        raise ValueError(
            f"obs_noise must be ({n_obs}, {n_obs}) to match obs_particles, got "
            f"({obs_noise.out_size()}, {obs_noise.in_size()})."
        )

    residuals = observation[None, :] - obs_particles  # (J, M)
    weighted = solve_rows(obs_noise, residuals)  # (J, M), R^{-1} r_j
    misfit = 0.5 * jnp.sum(residuals * weighted, axis=-1)  # (J,), Phi_j

    mean_misfit = jnp.mean(misfit)
    var_misfit = jnp.var(misfit, ddof=1 if bessel else 0)
    accuracy = n_obs / (2.0 * mean_misfit)  # C1
    uncertainty = jnp.sqrt(n_obs / (2.0 * var_misfit))  # C2
    return jnp.minimum(jnp.maximum(accuracy, uncertainty), remaining)
