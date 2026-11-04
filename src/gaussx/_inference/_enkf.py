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
    """Kalman gain from an ensemble and its image in observation space.

        G = Σ̂_xy Σ_y⁻¹,   Σ_y = Σ̂_h + Σ_ε

    where Σ̂_xy is the state-observation sample cross-covariance and Σ̂_h the
    sample covariance of the observation-space ensemble h⁽ᵉ⁾ = h(x⁽ᵉ⁾). Σ_y is
    assembled by one of two routes, which solve the same system and agree to
    round-off:

    - **Woodbury** (``dense_innovation=False``): Σ_y is a `LowRankUpdate` of
      Σ_ε, and `solve_rows` inverts an (|ℰ|, |ℰ|) capacitance through Σ_ε's own
      structured solve. It costs O(|ℰ|³ + |ℰ|² D_y) and never materialises Σ_ε,
      so it is the route for a few dozen members against many observations. It
      solves against Σ_ε itself, so Σ_ε must be positive **definite**.
    - **Dense** (``dense_innovation=True``): Σ_y is formed as a (D_y, D_y)
      matrix, calling ``obs_noise.as_matrix()``, and solved directly. It costs
      O(|ℰ| D_y² + D_y³), so it is the route for many members against few
      observations, where the capacitance would be the larger matrix (3.2 GB
      in float64 at |ℰ| = 20 000). It also handles a positive *semi*-definite
      Σ_ε.

    ``dense_innovation=None`` (the default) picks dense when |ℰ| ≥ D_y and
    Woodbury otherwise, the same rule as `enkf_analysis` and `eki_step`.

    Algorithm:
        1. Choose the route for Σ_y from the shapes or ``dense_innovation``.
        2. Form the cross-covariance Σ̂_xy.
        3. Assemble Σ_y densely, or as a low-rank update of Σ_ε.
        4. Solve for the gain G = Σ̂_xy Σ_y⁻¹ row by row.

    Args:
        particles: Prior ensemble x⁽ᵉ⁾ in state space.
        obs_particles: Its image h⁽ᵉ⁾ = h(x⁽ᵉ⁾) in observation space.
        obs_noise: Observation-error covariance Σ_ε.
        solver: Solver strategy for Σ_y; ``None`` uses structural dispatch. A
            matrix-free strategy (e.g. `CGSolver`) wants
            ``dense_innovation=False`` so it is handed the structured operator.
        bessel: Defaults to ``True``, unlike the lower-level covariance
            helpers, because this recipe follows the unbiased EnKF convention.
            Use ``False`` for maximum-likelihood recipes with a 1 / |ℰ| divisor.
        dense_innovation: Whether to form the (D_y, D_y) Σ_y densely. ``None``
            chooses by shape (dense when |ℰ| ≥ D_y); ``True`` and ``False``
            force the dense and the Woodbury route.

    Returns:
        The dense Kalman gain G.

    Raises:
        ValueError: If the two ensembles have different sizes, or if ``bessel``
            is set with |ℰ| < 2.

    Examples:
        Eight members of a three-dimensional state with two observations:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> particles = jr.normal(jr.key(0), (8, 3))
        >>> obs_particles = particles[:, :2]
        >>> obs_noise = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> gain = gaussx.ensemble_kalman_gain(particles, obs_particles, obs_noise)
        >>> gain.shape
        (3, 2)

        The two routes agree to round-off:

        >>> woodbury = gaussx.ensemble_kalman_gain(
        ...     particles, obs_particles, obs_noise, dense_innovation=False
        ... )
        >>> bool(jnp.allclose(gain, woodbury, atol=1e-4))
        True
    """
    ensemble_size, obs_size = obs_particles.shape
    if particles.shape[0] != ensemble_size:
        raise ValueError(
            "particles and obs_particles must share the same ensemble size, "
            f"got J={particles.shape[0]} and J={ensemble_size}."
        )
    use_dense = (
        ensemble_size >= obs_size if dense_innovation is None else dense_innovation
    )
    # Σ̂_xy
    cross_covariance = ensemble_cross_covariance(
        particles,
        obs_particles,
        bessel=bessel,
    )
    if use_dense:
        # A `LowRankUpdate` Σ_y would send `solve_rows` through Woodbury and
        # form an (|ℰ|, |ℰ|) capacitance matrix, 320 GB at |ℰ| = 200 000, so
        # assemble the (D_y, D_y) Σ_y densely instead. Same solve, same answer.
        # Σ̂_h
        obs_ensemble_covariance = ensemble_cross_covariance(
            obs_particles, obs_particles, bessel=bessel
        )
        # Σ_y = Σ̂_h + Σ_ε
        obs_covariance = obs_ensemble_covariance + obs_noise.as_matrix()
        obs_covariance = symmetrize(obs_covariance)
        obs_covariance_op = lx.MatrixLinearOperator(
            obs_covariance, lx.positive_semidefinite_tag
        )
        # G = Σ̂_xy Σ_y⁻¹
        return solve_rows(obs_covariance_op, cross_covariance, solver=solver)
    # Fewer members than observations: the Woodbury capacitance is (|ℰ|, |ℰ|)
    # and cheap, so keep the ensemble term low-rank.
    # Σ̂_h = U Uᵀ
    obs_ensemble_covariance_op = ensemble_covariance(obs_particles, bessel=bessel)
    # Σ_y = Σ_ε + U Uᵀ
    obs_covariance_op = LowRankUpdate(obs_noise, obs_ensemble_covariance_op.U)
    # G = Σ̂_xy Σ_y⁻¹
    return solve_rows(obs_covariance_op, cross_covariance, solver=solver)


def _noise_factor(
    obs_noise: lx.AbstractLinearOperator,
    *,
    allow_dense: bool = False,
) -> lx.AbstractLinearOperator:
    """A factor L with L Lᵀ = Σ_ε, valid for a singular Σ_ε.

    Perturbations are drawn as ε⁽ᵉ⁾ = L n⁽ᵉ⁾ with n⁽ᵉ⁾ ~ 𝒩(0, I), so L has three
    jobs, and neither `gaussx.cholesky` nor `gaussx.sqrt` does all three alone:

    1. **Exact.** `enkf_analysis` promises ε⁽ᵉ⁾ ~ 𝒩(0, Σ_ε). An approximate
       factor gives the perturbations the wrong covariance and biases the
       analysis silently, so a truncated Lanczos square root is not an
       acceptable default however cheap it is.
    2. **Defined for a singular Σ_ε.** That is a documented, supported case
       (``dense_innovation=True`` exists for it), and perturbations are drawn
       before that flag is ever consulted, so a positive-*definite*-only factor
       poisons the analysis with ``NaN`` regardless of what the caller asked
       for.
    3. **Structure-preserving.** D_y is routinely large enough that
       materialising Σ_ε is the thing the structured operator existed to avoid,
       and a dense fallback trades a ``NaN`` for an ``OOM``.

    `gaussx.cholesky` fails (2) at every dense leaf, whether that leaf is the
    whole operator or one block of a `gaussx.BlockDiag`, because it bottoms out
    in `jax.numpy.linalg.cholesky`, which returns ``NaN`` for a positive
    *semi*-definite matrix such as diag(1, 1, 0). Only the diagonal case
    escapes, its "Cholesky" being an elementwise square root that takes a zero
    in its stride. `gaussx.sqrt` fixes (2) but breaks (1) for
    `gaussx.SumOfKroneckers` and (3) for `gaussx.BlockTriDiag`.

    So the dispatch is by operator, taking whichever is better per structure:

    - Diagonal and identity: `gaussx.cholesky`, already exact and PSD-safe.
    - `gaussx.BlockDiag` / `gaussx.Kronecker`: recurse. Both factor into
      independent leaves, (L₁ ⊗ L₂)(L₁ ⊗ L₂)ᵀ = A₁ ⊗ A₂, so structure survives
      *and* each leaf gets the PSD-safe treatment.
    - `gaussx.BlockTriDiag`: `gaussx.cholesky`, whose banded recurrence is
      O(N d³) against O((N d)³) dense. It has no PSD-safe blockwise analogue,
      so it is the one structure where (2) and (3) genuinely conflict, hence
      ``allow_dense`` below.
    - Anything else, including every dense leaf: the symmetric square root.

    That last branch is where `gaussx.KroneckerSum` and
    `gaussx.SumOfKroneckers` land. Both densify, exactly as `gaussx.cholesky`
    did, rather than take their matrix-free `gaussx.sqrt` routes, which are
    respectively not traceable (a Python ``bool`` on a data-dependent
    definiteness check) and not exact.

    The result is the *symmetric* square root at dense leaves rather than a
    triangular factor. That is a different factorisation of the same Σ_ε: both
    satisfy L Lᵀ = Σ_ε, so the draws are distributionally identical, but a
    given key gives a different (equally valid) realisation.

    Args:
        obs_noise: Observation-error covariance Σ_ε.
        allow_dense: Whether the caller's gain path materialises Σ_ε anyway.
            When it does, requirement (3) is already spent and cannot justify
            declining (2), so `gaussx.BlockTriDiag` takes the PSD-safe dense
            factor too and a singular Σ_ε stops being fatal. Only
            `enkf_analysis`'s Woodbury route (no localization, structured
            Σ_y) keeps Σ_ε unmaterialised, so only there is the banded factor
            worth a ``NaN``.

    Returns:
        An operator L such that L Lᵀ = Σ_ε.
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
    """Check shape agreement for an ensemble analysis step.

    Args:
        particles: Prior ensemble x⁽ᵉ⁾ in state space.
        obs_particles: Its image h⁽ᵉ⁾ = h(x⁽ᵉ⁾) in observation space.
        observation: The observation y.
        obs_noise: Observation-error covariance Σ_ε.
        bessel: Whether the caller divides by |ℰ| − 1 rather than |ℰ|.

    Returns:
        Tuple ``(ensemble_size, state_size, obs_size)`` holding |ℰ|, D_x and
        D_y.

    Raises:
        ValueError: If the ensemble is too small for the divisor, or if any of
            the shapes disagree.
    """
    ensemble_size, state_size = particles.shape
    _check_ensemble_size(ensemble_size, bessel)
    if obs_particles.shape[0] != ensemble_size:
        raise ValueError(
            "particles and obs_particles must share the same ensemble size, "
            f"got J={ensemble_size} and J={obs_particles.shape[0]}."
        )
    obs_size = _check_observation_shapes(obs_particles, observation, obs_noise)
    return ensemble_size, state_size, obs_size


def _check_observation_shapes(
    obs_particles: Float[Array, "J M"],
    observation: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    observation_name: str = "observation",
) -> int:
    """Check shape agreement in observation space.

    Args:
        obs_particles: Observation-space ensemble h⁽ᵉ⁾.
        observation: The observation y.
        obs_noise: Observation-error covariance Σ_ε.
        observation_name: Name of ``observation`` in the error message.

    Returns:
        The observation length D_y.

    Raises:
        ValueError: If ``obs_particles`` is not a matrix, or if ``observation``
            or ``obs_noise`` do not match D_y.
    """
    if obs_particles.ndim != 2:
        raise ValueError(
            f"obs_particles must have shape (J, M), got {obs_particles.shape}."
        )
    obs_size = obs_particles.shape[1]
    if observation.shape != (obs_size,):
        raise ValueError(
            f"{observation_name} must have shape ({obs_size},) to match "
            f"obs_particles, got {observation.shape}."
        )
    # Without this an operator of the wrong size broadcasts against the
    # (D_y, D_y) sample covariance instead of raising: a (1, 1) Σ_ε against
    # D_y = 3 adds the scalar to every entry and yields a plausible, wrong gain.
    if (obs_noise.in_size(), obs_noise.out_size()) != (obs_size, obs_size):
        raise ValueError(
            f"obs_noise must be ({obs_size}, {obs_size}) to match obs_particles, "
            f"got ({obs_noise.out_size()}, {obs_noise.in_size()})."
        )
    return obs_size


def _check_localization_shapes(
    n_state: int,
    n_obs: int,
    localization: Float[Array, "N M"] | None,
    obs_localization: Float[Array, "M M"] | None,
) -> None:
    """Check the taper shapes; ``obs_localization`` only counts with a taper.

    Args:
        n_state: State length D_x.
        n_obs: Observation length D_y.
        localization: State-observation taper ρ_xy, or ``None``.
        obs_localization: Observation-observation taper ρ_yy, or ``None``.

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
    """Gain G = Σ̂_xy Σ_y⁻¹ by whichever route ``use_dense`` picks.

    Args:
        particles: Prior ensemble x⁽ᵉ⁾ in state space.
        obs_particles: Its image h⁽ᵉ⁾ = h(x⁽ᵉ⁾) in observation space.
        obs_noise: Observation-error covariance Σ_ε.
        localization: State-observation taper ρ_xy, or ``None`` for no
            localization.
        obs_localization: Observation-observation taper ρ_yy; ``None`` with a
            ``localization`` means all ones.
        solver: Solver strategy for Σ_y; ``None`` uses structural dispatch.
        use_dense: Whether to form Σ_y densely when there is no localization.
        bessel: Whether to use the 1 / (|ℰ| − 1) divisor.

    Returns:
        The dense Kalman gain G.
    """
    if localization is None:
        return ensemble_kalman_gain(
            particles,
            obs_particles,
            obs_noise,
            solver=solver,
            bessel=bessel,
            dense_innovation=use_dense,
        )
    obs_taper = (
        jnp.ones((obs_particles.shape[1],) * 2, dtype=particles.dtype)
        if obs_localization is None
        else obs_localization
    )
    return localized_kalman_gain(
        particles,
        obs_particles,
        obs_noise,
        localization,
        obs_taper,
        solver=solver,
        bessel=bessel,
    )


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
    """Stochastic (perturbed-observation) ensemble Kalman analysis step.

        x⁽ᵉ⁾ ← x⁽ᵉ⁾ + G (y + ε⁽ᵉ⁾ − h(x⁽ᵉ⁾)),   ε⁽ᵉ⁾ ~ 𝒩(0, Σ_ε)

    with G from `ensemble_kalman_gain`, or from `localized_kalman_gain` when
    ``localization`` is given. The observation operator enters only through
    ``obs_particles``, the image h⁽ᵉ⁾ = h(x⁽ᵉ⁾) of the prior ensemble in
    observation space, so nonlinear operators need no special handling.

    The perturbation ε⁽ᵉ⁾ is what keeps the analysis spread correct. The update
    x⁽ᵉ⁾ ← x⁽ᵉ⁾ + G (y − h(x⁽ᵉ⁾)) drives the ensemble covariance to
    (I − G H) Σ_x (I − G H)ᵀ instead of (I − G H) Σ_x, which is under-dispersive.
    There is deliberately no ``perturb=False`` flag: the deterministic
    alternative is a different filter (the square-root or ETKF family, see
    `etkf_transform`), not an option on this one.

    Two ways to supply the observation perturbations:

    - ``key`` draws ε⁽ᵉ⁾ ~ 𝒩(0, Σ_ε) internally, via an exact PSD factor
      L Lᵀ = Σ_ε of ``obs_noise`` that keeps its structure (a symmetric square
      root at dense leaves, so a singular Σ_ε is fine).
    - ``perturbed_obs`` passes a pre-built perturbed-observation ensemble
      y + ε⁽ᵉ⁾. Prefer it when the same noise realisation must be reused across
      filters, and when the perturbations come from a nonlinear observation
      model rather than an additive Σ_ε.

    Exactly one of ``key`` and ``perturbed_obs`` must be given.

    Known limitation. The update is a Gaussian one, applied in whatever
    coordinates the caller supplies. For a non-Gaussian prior it is biased, and
    the bias does **not** shrink with ensemble size: it is an error of
    coordinates, not of sampling. On the lognormal / logit-normal prior of
    Chipilski (2025), whose exact posterior mean is [0.548062, 0.353937], the
    physical-space update plateaus several percent off that value and stays
    there as |ℰ| grows by two orders of magnitude.

    The fix is to conjugate the update with a bijection Γ that Gaussianises the
    prior: call this function on Γ⁻¹(x⁽ᵉ⁾) and map the result back through Γ.
    The ensemble Kalman filter's Gaussian assumption is a statement about
    coordinates, not about the algorithm. Pass the same ``perturbed_obs``
    through both routes to compare them on one noise realisation.

    That conjugated update is *exact Bayes* only under conditions worth stating
    precisely, because it is easy to over-claim. It needs the population limit
    (with a finite ensemble the gain is empirical and the perturbations are
    Monte Carlo, so the result is an estimate either way), and it needs the
    observation model to be **affine with additive Gaussian noise** in the same
    latent coordinates that Gaussianise the prior. A merely "Gaussian
    likelihood" is not enough: y = z² + ε has Gaussian noise and a non-Gaussian
    posterior that no Kalman update reproduces. Outside those conditions
    conjugation is an approximation with no guaranteed ordering against the
    physical-space update: usually much better, but a badly matched Γ can make
    the latent joint less Gaussian and do worse.

    Algorithm:
        1. Check the arguments and pick the route for Σ_y from the shapes or
           ``dense_innovation``.
        2. Draw y + ε⁽ᵉ⁾ from ``key``, or take ``perturbed_obs`` as given.
        3. Form the gain G, tapered when ``localization`` is given.
        4. Add G (y + ε⁽ᵉ⁾ − h⁽ᵉ⁾) to each member.

    Args:
        particles: Prior ensemble x⁽ᵉ⁾ in state space.
        obs_particles: Prior ensemble h⁽ᵉ⁾ = h(x⁽ᵉ⁾) in observation space.
        observation: The observation y.
        obs_noise: Observation-error covariance Σ_ε.
        key: PRNG key for internally drawn perturbations. Mutually exclusive
            with ``perturbed_obs``.
        perturbed_obs: Pre-built perturbed-observation ensemble y + ε⁽ᵉ⁾.
            Mutually exclusive with ``key``.
        localization: Optional state-observation taper ρ_xy, e.g. from
            `localization_matrix`. When given, the gain comes from
            `localized_kalman_gain` instead of `ensemble_kalman_gain`.
        obs_localization: Optional observation-observation taper ρ_yy. Only
            consulted when ``localization`` is given; defaults to all ones,
            i.e. no tapering of Σ̂_h.
        solver: Solver strategy for Σ_y; ``None`` uses structural dispatch. A
            matrix-free strategy (e.g. `CGSolver`) wants
            ``dense_innovation=False`` so it is handed the structured operator
            instead of a materialised one.
        dense_innovation: Whether to form the (D_y, D_y) Σ_y densely. ``None``
            (default) chooses by shape, as described in the note below.
            ``False`` keeps the structured `LowRankUpdate` no matter the shapes,
            which is what a matrix-free solver wants. ``True`` forces the dense
            assembly, which is the way out when ``obs_noise`` is only positive
            *semi*-definite, since the structured route solves against
            ``obs_noise`` itself.
        bessel: Use the 1 / (|ℰ| − 1) divisor. Defaults to ``True``, matching
            `ensemble_kalman_gain`.

    Returns:
        The analysis ensemble.

    Note:
        How Σ_y = Σ̂_h + Σ_ε is assembled defaults to a choice made from the
        static shapes, because the two regimes have wildly different costs.
        With |ℰ| < D_y the gain comes from `ensemble_kalman_gain`, which keeps
        the ensemble term low-rank and inverts an (|ℰ|, |ℰ|) Woodbury
        capacitance, the right choice for the geoscience regime of a few dozen
        members against many observations. With |ℰ| ≥ D_y that capacitance is
        the larger of the two (320 GB at |ℰ| = 200 000), so the (D_y, D_y) Σ_y
        is formed densely instead. Both routes solve the same system and agree
        to round-off.

        Shapes are the wrong criterion in two cases, which is why
        ``dense_innovation`` exists to override them:

        - **A matrix-free solver.** With |ℰ| ≥ D_y and D_y still large, the
          dense assembly allocates a (D_y, D_y) array before the solver is ever
          called, around 40 GB at D_y = 100 000 in float32, even though an
          iterative strategy could work through matvecs on the structured
          operator. Pass ``dense_innovation=False``.
        - **Singular observation noise.** The Woodbury route solves against
          Σ_ε itself, so a positive *semi*-definite Σ_ε divides by zero and
          returns infinities or ``NaN`` even when Σ̂_h + Σ_ε is perfectly
          invertible, e.g. Σ_ε = diag(1, 1, 0) with ensemble anomalies spanning
          the third observation direction. The |ℰ| < D_y path therefore requires
          Σ_ε to be positive **definite**; with a singular Σ_ε, pass
          ``dense_innovation=True`` to solve the full Σ_y instead. This is not
          checked: positive semi-definiteness of an arbitrary operator is not
          something this function can establish cheaply, and certainly not
          under ``jit``.

    Raises:
        ValueError: If neither or both of ``key`` and ``perturbed_obs`` are
            given, if the ensemble sizes disagree, or if the observation-space
            or taper shapes disagree.

    Examples:
        A drawn perturbation, against two observations of a linear model:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> key, subkey = jr.split(jr.key(0))
        >>> prior = jr.normal(subkey, (500, 3))
        >>> obs_operator = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> obs_prior = prior @ obs_operator.T
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> observation = jnp.array([1.0, -1.0])
        >>> posterior = gaussx.enkf_analysis(
        ...     prior, obs_prior, observation, obs_noise, key=key
        ... )
        >>> posterior.shape
        (500, 3)

        Passing the observations unperturbed still updates every member:

        >>> repeated = jnp.broadcast_to(observation, obs_prior.shape)
        >>> gaussx.enkf_analysis(
        ...     prior, obs_prior, observation, obs_noise, perturbed_obs=repeated
        ... ).shape
        (500, 3)

        ``key`` and ``perturbed_obs`` are mutually exclusive:

        >>> gaussx.enkf_analysis(
        ...     prior, obs_prior, observation, obs_noise
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: Pass exactly one of 'key' or 'perturbed_obs'

    References:
        Chipilski (2025).
    """
    if (key is None) == (perturbed_obs is None):
        raise ValueError(
            "Pass exactly one of 'key' (draw perturbations from obs_noise) or "
            "'perturbed_obs' (supply them directly)."
        )

    ensemble_size, state_size, obs_size = _check_analysis_shapes(
        particles, obs_particles, observation, obs_noise, bessel
    )

    # Whether to form the (D_y, D_y) Σ_y densely. `None` picks by shape; an
    # explicit value overrides that, which is what a matrix-free solver needs.
    # Decided here rather than at the gain, because the factor below needs it.
    use_dense = (
        ensemble_size >= obs_size if dense_innovation is None else dense_innovation
    )

    # Branch on `key` rather than on `perturbed_obs`: the check above makes the
    # two equivalent, but this way each branch narrows the argument it uses.
    if key is not None:
        # The factor dispatches on structure, so a DiagonalLinearOperator stays
        # diagonal and its matvec stays O(D_y). Materialising Σ_ε here would
        # allocate a dense (D_y, D_y) and cost O(D_y³), which for the low-rank
        # branch (|ℰ| < D_y, D_y possibly enormous) would OOM before the gain is
        # ever formed.
        #
        # Unless the gain is about to allocate that (D_y, D_y) anyway: every
        # route but Woodbury calls `obs_noise.as_matrix()`, and once the dense
        # cost is being paid regardless there is nothing left to protect by
        # holding on to a positive-definite-only factor.
        # Σ_ε = L Lᵀ
        noise_factor = _noise_factor(
            obs_noise, allow_dense=use_dense or localization is not None
        )
        # n⁽ᵉ⁾ ~ 𝒩(0, I)
        noise = jr.normal(key, (ensemble_size, obs_size), dtype=particles.dtype)
        # jnp, not einx: an empty D_y returns the prior ensemble here today, and
        # einx raises AxisSizeError on a zero-length axis.
        # ε⁽ᵉ⁾ = L n⁽ᵉ⁾
        perturbation = jax.vmap(noise_factor.mv)(noise)
        # y + ε⁽ᵉ⁾
        perturbed = observation[None, :] + perturbation
    else:
        perturbed = perturbed_obs
        if perturbed is None or perturbed.shape != (ensemble_size, obs_size):
            raise ValueError(
                f"perturbed_obs must have shape ({ensemble_size}, {obs_size}) to "
                f"match obs_particles, got "
                f"{None if perturbed is None else perturbed.shape}."
            )

    _check_localization_shapes(state_size, obs_size, localization, obs_localization)

    # G = Σ̂_xy Σ_y⁻¹
    gain = _analysis_gain(
        particles,
        obs_particles,
        obs_noise,
        localization=localization,
        obs_localization=obs_localization,
        solver=solver,
        use_dense=use_dense,
        bessel=bessel,
    )

    # r⁽ᵉ⁾ = y + ε⁽ᵉ⁾ − h⁽ᵉ⁾
    residuals = perturbed - obs_particles
    # jnp, not einx: an empty D_x or D_y returns an empty update here today, and
    # einx raises AxisSizeError on a zero-length axis.
    # G r⁽ᵉ⁾, as rows
    increment = residuals @ gain.T
    # x⁽ᵉ⁾ ← x⁽ᵉ⁾ + G r⁽ᵉ⁾
    return particles + increment
