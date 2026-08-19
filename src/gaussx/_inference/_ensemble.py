"""Ensemble covariance and cross-covariance recipes."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from gaussx._linalg._linalg import solve_rows
from gaussx._linalg._mixed_precision import stable_squared_distances
from gaussx._linalg._symmetrize import symmetrize
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._primitives._cholesky import cholesky
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
            uses structural dispatch.
        bessel: Use the $1/(J-1)$ divisor. Defaults to ``True``, matching
            `ensemble_kalman_gain`.

    Returns:
        Analysis ensemble, shape ``(J, N)``.

    Note:
        How the innovation covariance $C^{HH} + R$ is assembled is chosen from
        the static shapes, because the two regimes have wildly different costs.
        With $J < M$ the gain comes from `ensemble_kalman_gain`, which keeps
        the ensemble term low-rank and inverts a $(J, J)$ Woodbury capacitance
        -- the right choice for the geoscience regime of a few dozen members
        against many observations. With $J \ge M$ that capacitance is the
        larger of the two (320 GB at $J = 200{,}000$), so the $(M, M)$
        innovation is formed densely instead. Both routes solve the same
        system and agree to round-off.

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

    n_ens = particles.shape[0]
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

    # Branch on `key` rather than on `perturbed_obs`: the check above makes the
    # two equivalent, but this way each branch narrows the argument it uses.
    if key is not None:
        # eps_j = L n_j with R = L L^T. `cholesky` dispatches on structure, so a
        # DiagonalLinearOperator stays diagonal and its matvec stays O(M) --
        # materialising R here would allocate a dense (M, M) and cost O(M^3),
        # which for the low-rank branch below (J < M, M possibly enormous) would
        # OOM before the gain is ever formed.
        chol = cholesky(obs_noise)
        noise = jr.normal(key, (n_ens, n_obs), dtype=particles.dtype)  # (J, M)
        perturbed = observation[None, :] + jax.vmap(chol.mv)(noise)  # (J, M)
    else:
        perturbed = perturbed_obs
        if perturbed is None or perturbed.shape != (n_ens, n_obs):
            raise ValueError(
                f"perturbed_obs must have shape ({n_ens}, {n_obs}) to match "
                f"obs_particles, got "
                f"{None if perturbed is None else perturbed.shape}."
            )

    if localization is None and n_ens < n_obs:
        # Fewer members than observations: the Woodbury capacitance is (J, J)
        # and cheap, so let `ensemble_kalman_gain` keep the low-rank structure.
        gain = ensemble_kalman_gain(
            particles, obs_particles, obs_noise, solver=solver, bessel=bessel
        )  # (N, M)
    elif localization is None:
        # J >= M, the usual regime for this function. A `LowRankUpdate`
        # innovation would send `solve_rows` through Woodbury and form a
        # (J, J) capacitance matrix -- 320 GB at J = 200_000 -- so assemble
        # the (M, M) innovation densely instead. Same solve, same answer.
        cross_cov = ensemble_cross_covariance(particles, obs_particles, bessel=bessel)
        obs_cov = ensemble_cross_covariance(obs_particles, obs_particles, bessel=bessel)
        innovation = symmetrize(obs_cov + obs_noise.as_matrix())  # (M, M)
        innovation_op = lx.MatrixLinearOperator(
            innovation, lx.positive_semidefinite_tag
        )
        gain = solve_rows(innovation_op, cross_cov, solver=solver)  # (N, M)
    else:
        rho_yy = (
            jnp.ones((n_obs, n_obs), dtype=particles.dtype)
            if obs_localization is None
            else obs_localization
        )
        gain = localized_kalman_gain(
            particles,
            obs_particles,
            obs_noise,
            localization,
            rho_yy,
            solver=solver,
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
    """Symmetric (eigendecomposition) square root of an SPD matrix."""
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    sqrt_eigvals = jnp.sqrt(jnp.maximum(eigvals, 0.0))
    return (eigvecs * sqrt_eigvals[None, :]) @ eigvecs.T
