"""Stochastic (perturbed-observation) ensemble Kalman analysis and its gain."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from gaussx._inference._ensemble_stats import (
    _check_ensemble_size,
    ensemble_covariance,
    ensemble_cross_covariance,
)
from gaussx._inference._localization import localized_kalman_gain
from gaussx._linalg._linalg import solve_rows
from gaussx._linalg._symmetrize import symmetrize
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._operators._sum_kronecker import SumOfKroneckers
from gaussx._primitives._cholesky import cholesky, warn_dense_fallback
from gaussx._primitives._sqrt import dense_symmetric_sqrt
from gaussx._strategies._base import AbstractSolverStrategy


def ensemble_kalman_gain(
    particles: Float[Array, "J N"],
    obs_particles: Float[Array, "J M"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
    bessel: bool = True,
    dense_innovation: bool | None = None,
) -> Float[Array, "N M"]:
    r"""Kalman gain from an ensemble and its image in observation space.

    Computes ``K = C^{xH} (C^{HH} + R)^{-1}``, where ``C^{xH}`` is the
    state-observation cross-covariance and ``C^{HH}`` is the
    observation-space ensemble covariance. The innovation covariance
    ``S = C^{HH} + R`` is assembled by one of two routes, which solve the
    same system and agree to round-off:

    - **Woodbury** (``dense_innovation=False``): ``S`` is a `LowRankUpdate`
      of ``R``, and ``solve_rows`` inverts a ``(J, J)`` capacitance through
      ``R``'s own structured solve. Costs ``O(J^3 + J^2 M)`` and never
      materialises ``R`` -- the route for a few dozen members against many
      observations. It solves against ``R`` itself, so ``R`` must be
      positive **definite**.
    - **Dense** (``dense_innovation=True``): ``S`` is formed as an ``(M, M)``
      matrix, calling ``obs_noise.as_matrix()``, and solved directly. Costs
      ``O(J M^2 + M^3)`` -- the route for many members against few
      observations, where the capacitance would be the larger matrix
      (3.2 GB in float64 at ``J = 20 000``). It also handles a positive
      *semi*-definite ``R``.

    ``dense_innovation=None`` (the default) picks dense when ``J >= M`` and
    Woodbury otherwise, the same rule as `enkf_analysis` and `eki_step`.

    Args:
        particles: Prior ensemble in state space, shape ``(J, N)``.
        obs_particles: Prior ensemble in observation space, shape ``(J, M)``.
        obs_noise: Observation error covariance operator, shape ``(M, M)``.
        solver: Optional solver strategy. ``None`` uses structural dispatch.
            A matrix-free strategy (e.g. `CGSolver`) wants
            ``dense_innovation=False`` so it is handed the structured operator.
        bessel: Defaults to True, unlike the lower-level covariance helpers,
            because this recipe follows the unbiased EnKF convention. Use
            False for maximum-likelihood recipes with a ``1 / J`` divisor.
        dense_innovation: Whether to form the ``(M, M)`` innovation densely.
            ``None`` chooses by shape (dense when ``J >= M``); ``True`` /
            ``False`` force the dense / Woodbury route.

    Returns:
        Dense Kalman gain of shape ``(N, M)``.
    """
    n_ens, n_obs = obs_particles.shape
    if particles.shape[0] != n_ens:
        raise ValueError(
            "particles and obs_particles must share the same ensemble size, "
            f"got J={particles.shape[0]} and J={n_ens}."
        )
    use_dense = n_ens >= n_obs if dense_innovation is None else dense_innovation
    cross_cov = ensemble_cross_covariance(
        particles,
        obs_particles,
        bessel=bessel,
    )
    if use_dense:
        # A `LowRankUpdate` innovation would send `solve_rows` through Woodbury
        # and form a (J, J) capacitance matrix -- 320 GB at J = 200_000 -- so
        # assemble the (M, M) innovation densely instead. Same solve, same
        # answer.
        obs_cov = ensemble_cross_covariance(obs_particles, obs_particles, bessel=bessel)
        innovation = symmetrize(obs_cov + obs_noise.as_matrix())  # (M, M)
        innovation_op = lx.MatrixLinearOperator(
            innovation, lx.positive_semidefinite_tag
        )
        return solve_rows(innovation_op, cross_cov, solver=solver)  # (N, M)
    # Fewer members than observations: the Woodbury capacitance is (J, J) and
    # cheap, so keep the ensemble term low-rank.
    innovation_cov = ensemble_covariance(obs_particles, bessel=bessel)
    innovation_cov = LowRankUpdate(obs_noise, innovation_cov.U)
    return solve_rows(innovation_cov, cross_cov, solver=solver)


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
        warn_dense_fallback(
            "enkf_analysis materialises a SumOfKroneckers obs_noise to draw "
            "exact perturbations. For a matrix-free alternative, sample "
            "eps ~ N(0, R) yourself -- sumkronecker_sample or "
            "sqrt(obs_noise, lanczos_order=...) -- and pass them as "
            "perturbed_obs=observation + eps. Both are approximate, so that "
            "is an opt-in, not the default."
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
    n_obs = _check_observation_shapes(obs_particles, observation, obs_noise)
    return n_ens, n_state, n_obs


def _check_observation_shapes(
    obs_particles: Float[Array, "J M"],
    observation: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    observation_name: str = "observation",
) -> int:
    """Observation-space shape agreement. Returns ``M``."""
    if obs_particles.ndim != 2:
        raise ValueError(
            f"obs_particles must have shape (J, M), got {obs_particles.shape}."
        )
    n_obs = obs_particles.shape[1]
    if observation.shape != (n_obs,):
        raise ValueError(
            f"{observation_name} must have shape ({n_obs},) to match "
            f"obs_particles, got {observation.shape}."
        )
    # Without this an operator of the wrong size broadcasts against the (M, M)
    # empirical covariance instead of raising -- a (1, 1) R against M = 3 adds
    # the scalar to every entry and yields a plausible but wrong gain.
    if (obs_noise.in_size(), obs_noise.out_size()) != (n_obs, n_obs):
        raise ValueError(
            f"obs_noise must be ({n_obs}, {n_obs}) to match obs_particles, got "
            f"({obs_noise.out_size()}, {obs_noise.in_size()})."
        )
    return n_obs


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
    if localization is None:
        return ensemble_kalman_gain(
            particles,
            obs_particles,
            obs_noise,
            solver=solver,
            bessel=bessel,
            dense_innovation=use_dense,
        )  # (N, M)
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

    - ``key`` -- draw $\varepsilon_j \sim N(0, R)$ internally, via an exact
      PSD factor $L L^\top = R$ of ``obs_noise`` that keeps its structure (a
      symmetric square root at dense leaves, so a singular $R$ is fine).
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
