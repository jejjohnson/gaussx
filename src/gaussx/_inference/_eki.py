"""Ensemble Kalman inversion (EKI): tempered step, TEKI prior, adaptive step."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from gaussx._einx import reduce
from gaussx._inference._enkf import (
    _analysis_gain,
    _check_analysis_shapes,
    _noise_factor,
)
from gaussx._inference._ensemble_stats import (
    _check_ensemble_size,
    _check_obs_noise_shape,
)
from gaussx._inference._etkf import _etkf_increments
from gaussx._inference._localization import _check_localization_shapes
from gaussx._linalg._linalg import solve_rows
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._lazy_algebra import ScaledOperator
from gaussx._strategies._base import AbstractSolverStrategy


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
    """One ensemble Kalman inversion (EKI) update.

        G = Σ̂_xy (Σ̂_h + Σ_ε / Δt)⁻¹
        x⁽ᵉ⁾ ← x⁽ᵉ⁾ + α G (y + ε⁽ᵉ⁾ / √Δt − h(x⁽ᵉ⁾)),   ε⁽ᵉ⁾ ~ 𝒩(0, Σ_ε)

    A single tempered Kalman update of an ensemble against a fixed observation.
    This is `enkf_analysis` with two knobs added, and reduces to it exactly at
    ``dt=1``, ``step=None``. Everything the forward model does enters through
    ``obs_particles``, so this function is pure array-in, array-out: there is no
    iteration, no stopping rule and no h. The driver that supplies the schedule
    lives outside this package.

    **Tempering (``dt``).** One iteration replaces Σ_ε by Σ_ε / Δt, i.e. a
    likelihood raised to the power Δt. Over a schedule with ∑ᵢ Δt⁽ⁱ⁾ = 1 the
    composition is *exactly* one Bayesian update in the linear-Gaussian
    population limit, because the precisions add,
    Λ_x|y = Λ_x + ∑ᵢ Δt⁽ⁱ⁾ Hᵀ Σ_ε⁻¹ H, which is what makes an EKI schedule a
    tempering path rather than a heuristic. The sum condition is load-bearing:
    at ∑ᵢ Δt⁽ⁱ⁾ ≠ 1 the result is the posterior of a different problem, over- or
    under-weighting the data. ``dt`` may be a traced scalar, so an adaptive
    schedule (see `discrepancy_step_size`) stays inside ``jit``.

    **State-side step (``step``).** In the gradient-flow view
    ẋ = −Σ̂_x ∇_x 𝒥(x), ``step`` is the operator α in the Euler step
    x ← x + α Σ̂_xy Σ_y⁻¹ (y − h(x)). It is applied by ``step.mv`` to each
    member's *increment*, so a `gaussx.BlockDiag` of scaled identities gives a
    different rate per state block (parameters, latents, initial conditions)
    without densifying a (D_x, D_x) matrix. α changes the trajectory, not the
    fixed point: where G (y − h̄) = 0 the increment is zero for every α,
    invertible or not.

    **Deterministic variant.** ``deterministic=True`` replaces the perturbed
    observations with an ETKF square-root transform (`etkf_transform` at
    Σ_ε / Δt), applied to the increment so that α still acts on a difference:

        x̄ ← x̄ + α G (y − h̄)
        X̃ ← X̃ + α (W X̃ − X̃), one row per member

    At α = I the anomaly update collapses to X̃ ← W X̃, i.e. plain
    `etkf_transform`. This is the variant to use for the exactness property
    above: the stochastic one is exact only in expectation, so a finite ensemble
    carries Monte Carlo error on top of the tempering.

    Algorithm:
        1. Check the arguments and scale Σ_ε to the lazy Σ_ε / Δt.
        2. Deterministic variant: get the mean and anomaly increments from the
           ETKF update behind `etkf_analysis`, apply α, and add both.
        3. Stochastic variant: form the gain G against Σ_ε / Δt.
        4. Draw y + ε⁽ᵉ⁾ / √Δt from ``key``, or take ``perturbed_obs`` as given.
        5. Add α G (y + ε⁽ᵉ⁾ / √Δt − h⁽ᵉ⁾) to each member.

    Args:
        particles: Prior ensemble x⁽ᵉ⁾ in state space.
        obs_particles: Its image h⁽ᵉ⁾ = h(x⁽ᵉ⁾) in observation space.
        observation: The observation y.
        obs_noise: Observation-error covariance Σ_ε. Scaled to Σ_ε / Δt as a lazy
            `gaussx.ScaledOperator`, so a structured Σ_ε keeps its own solve in
            both variants.
        dt: Observation-side tempering step Δt. It may be traced, so a step that
            is not positive is not an error: it returns ``particles`` unchanged,
            with a zero derivative.
        step: State-side operator α. ``None`` is the identity, and skips the
            matvec rather than building one.
        key: PRNG key for internally drawn perturbations ε⁽ᵉ⁾ ~ 𝒩(0, Σ_ε), which
            are then scaled by 1 / √Δt to match Σ_ε / Δt. Mutually exclusive
            with ``perturbed_obs``; both must be ``None`` when ``deterministic``.
        perturbed_obs: Pre-built perturbed-observation ensemble, used **as
            given**: the caller owns the 1 / √Δt scaling. Mutually exclusive
            with ``key``.
        deterministic: Use the ETKF square-root transform instead of perturbed
            observations.
        localization: Optional state-observation taper ρ_xy. Stochastic variant
            only.
        obs_localization: Optional observation-observation taper ρ_yy. Only
            consulted when ``localization`` is given.
        solver: Solver strategy for Σ_y; ``None`` uses structural dispatch.
        dense_innovation: Whether to form the (D_y, D_y) Σ_y densely. ``None``
            chooses by shape. Same contract as `enkf_analysis`, including that a
            positive *semi*-definite Σ_ε needs ``True``. Stochastic variant only:
            the deterministic one forms no gain, because `etkf_transform`
            supplies the mean update as well, and it solves against Σ_ε itself.
        bessel: Use the 1 / (|ℰ| − 1) divisor. Must stay ``True`` when
            ``deterministic``, since `etkf_transform` is 1 / (|ℰ| − 1) throughout
            and a mismatched gain would move the mean and the anomalies by
            inconsistent amounts.

    Returns:
        The updated ensemble.

    Raises:
        ValueError: If the shapes disagree; if ``key`` and ``perturbed_obs`` are
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

    Examples:
        A tempered stochastic step against a linear model:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> key, subkey = jr.split(jr.key(0))
        >>> particles = jr.normal(subkey, (200, 3))
        >>> obs_operator = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> obs_particles = particles @ obs_operator.T
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> observation = jnp.array([1.0, -1.0])
        >>> gaussx.eki_step(
        ...     particles, obs_particles, observation, obs_noise, dt=0.5, key=key
        ... ).shape
        (200, 3)

        The deterministic variant needs no key:

        >>> gaussx.eki_step(
        ...     particles, obs_particles, observation, obs_noise, deterministic=True
        ... ).shape
        (200, 3)

        It does not support localization:

        >>> gaussx.eki_step(
        ...     particles,
        ...     obs_particles,
        ...     observation,
        ...     obs_noise,
        ...     deterministic=True,
        ...     localization=jnp.ones((3, 2)),
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: deterministic=True does not support 'localization'

    References:
        Iglesias, Law & Stuart (2013).
    """
    ensemble_size, state_size, obs_size = _check_analysis_shapes(
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
    if step is not None and (step.in_size(), step.out_size()) != (
        state_size,
        state_size,
    ):
        raise ValueError(
            f"step must be ({state_size}, {state_size}) to match particles, got "
            f"({step.out_size()}, {step.in_size()})."
        )
    _check_localization_shapes(state_size, obs_size, localization, obs_localization)

    dt = jnp.asarray(dt)
    if dt.ndim != 0:
        raise ValueError(f"dt must be a scalar, got shape {dt.shape}.")
    # Σ_ε / Δt as a lazy scale. `solve` unwraps a MulLinearOperator by dividing
    # the inner solve, so a Diagonal, BlockDiag or Kronecker Σ_ε keeps its own
    # solve rather than falling back to a dense (D_y, D_y). At Δt = 1 the scalar
    # is exactly 1.0 and every route is bit-for-bit `enkf_analysis`.
    # A step that is not positive moves nothing. Its 1 / Δt and 1 / √Δt are kept out
    # of the update, where ∞ × 0 would be NaN in the value and in the gradient.
    positive_step = dt > 0.0
    safe_dt = jnp.where(positive_step, dt, 1.0)
    tempered_noise = ScaledOperator(obs_noise, 1.0 / safe_dt)

    if deterministic:
        # By the push-through identity G (y − h̄) = X̃ᵀ w̄, so the ETKF weights
        # give the mean update too, and no separate solve for the gain is needed.
        # X̃ᵀ w̄ and W X̃ at Σ_ε / Δt
        mean_increment, transformed_anomalies = _etkf_increments(
            particles, obs_particles, observation, tempered_noise, 1.0, solver, True
        )
        # x̄ = (1 / |ℰ|) ∑ₑ x⁽ᵉ⁾
        particle_mean = reduce(particles, "member state -> state", "mean")
        # x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄
        anomalies = einx.subtract(
            "member state, state -> member state", particles, particle_mean
        )
        # The increment, not the transformed anomalies: α acts on differences,
        # so α = I leaves W X̃ exactly.
        # W X̃ − X̃
        anomaly_increment = transformed_anomalies - anomalies
        if step is not None:
            # α X̃ᵀ w̄
            mean_increment = step.mv(mean_increment)
            # α (W X̃ − X̃), one row per member
            anomaly_increment = jax.vmap(step.mv)(anomaly_increment)
        # x⁽ᵉ⁾ + α X̃ᵀ w̄
        shifted = einx.add(
            "member state, state -> member state", particles, mean_increment
        )
        # x⁽ᵉ⁾ ← x⁽ᵉ⁾ + α X̃ᵀ w̄ + α (W x̃⁽ᵉ⁾ − x̃⁽ᵉ⁾)
        return jnp.where(positive_step, shifted + anomaly_increment, particles)

    use_dense = (
        ensemble_size >= obs_size if dense_innovation is None else dense_innovation
    )
    # G = Σ̂_xy (Σ̂_h + Σ_ε / Δt)⁻¹
    gain = _analysis_gain(
        particles,
        obs_particles,
        tempered_noise,
        localization=localization,
        obs_localization=obs_localization,
        solver=solver,
        use_dense=use_dense,
        bessel=bessel,
    )

    if key is not None:
        # ε⁽ᵉ⁾ ~ 𝒩(0, Σ_ε) drawn against the *unscaled* Σ_ε, then scaled by
        # 1 / √Δt to give 𝒩(0, Σ_ε / Δt). Drawing against Σ_ε / Δt instead would
        # hand `_noise_factor` a MulLinearOperator it has no structured branch
        # for, and densify a (D_y, D_y) for nothing.
        # Σ_ε = L Lᵀ
        noise_factor = _noise_factor(
            obs_noise, allow_dense=use_dense or localization is not None
        )
        # n⁽ᵉ⁾ ~ 𝒩(0, I)
        noise = jr.normal(key, (ensemble_size, obs_size), dtype=particles.dtype)
        # ε⁽ᵉ⁾ = L n⁽ᵉ⁾
        drawn_noise = jax.vmap(noise_factor.mv)(noise)
        # ε⁽ᵉ⁾ / √Δt
        perturbation = drawn_noise / jnp.sqrt(safe_dt)
        # jnp, not einx: an empty D_y returns the prior ensemble here today, and
        # einx raises AxisSizeError on a zero-length axis.
        # y + ε⁽ᵉ⁾ / √Δt
        perturbed = observation[None, :] + perturbation
    else:
        perturbed = perturbed_obs
        if perturbed is None or perturbed.shape != (ensemble_size, obs_size):
            raise ValueError(
                f"perturbed_obs must have shape ({ensemble_size}, {obs_size}) to "
                f"match obs_particles, got "
                f"{None if perturbed is None else perturbed.shape}."
            )

    # r⁽ᵉ⁾ = y + ε⁽ᵉ⁾ / √Δt − h⁽ᵉ⁾
    residuals = perturbed - obs_particles
    # jnp, not einx: an empty D_x or D_y returns an empty update here today, and
    # einx raises AxisSizeError on a zero-length axis.
    # G r⁽ᵉ⁾, as rows
    increment = residuals @ gain.T
    if step is not None:
        # α G r⁽ᵉ⁾, one row per member
        increment = jax.vmap(step.mv)(increment)
    # x⁽ᵉ⁾ ← x⁽ᵉ⁾ + α G r⁽ᵉ⁾
    return jnp.where(positive_step, particles + increment, particles)


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
    """Observation augmentation for Tikhonov-regularised EKI (TEKI).

        y_aug = [y, μ_x],   h_aug(x) = [h(x), x],   Σ_ε,aug = blockdiag(Σ_ε, Σ_x)

    Puts the prior 𝒩(μ_x, Σ_x) into an EKI step by treating the state as its own
    observation, so the augmented least-squares cost is the regularised one,

        𝒥(x) = ½ ‖y − h(x)‖²_Σ_ε⁻¹ + ½ ‖x − μ_x‖²_Σ_x⁻¹

    Unregularised EKI collapses onto the data-misfit minimiser and, for an
    ill-posed problem, keeps going; the prior term is what stops it.

    A helper rather than a flag on `eki_step`: the triple goes straight into
    `eki_step` (and into `discrepancy_step_size`, whose D_y is then D_y + D_x),
    nothing inside the step knows about priors, and Σ_x stays an operator, so a
    `gaussx.Kronecker` prior keeps its structured solve inside the
    `gaussx.BlockDiag`.

    Args:
        particles: Ensemble x⁽ᵉ⁾ in state space.
        obs_particles: Its image h⁽ᵉ⁾ = h(x⁽ᵉ⁾).
        observation: The observation y.
        obs_noise: Observation-error covariance Σ_ε.
        prior_mean: Prior mean μ_x.
        prior_cov: Prior covariance Σ_x.

    Returns:
        Tuple ``(obs_particles_aug, observation_aug, obs_noise_aug)`` holding
        h_aug(x⁽ᵉ⁾), y_aug and Σ_ε,aug, of observation length D_y + D_x. Pass them
        to `eki_step` in place of ``obs_particles``, ``observation`` and
        ``obs_noise``; ``particles`` is unchanged.

    Raises:
        ValueError: If any of the shapes disagree.

    Examples:
        The augmented observation stacks the state after the data:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> key, subkey = jr.split(jr.key(0))
        >>> particles = jr.normal(subkey, (200, 3))
        >>> obs_operator = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> prior_cov = lx.DiagonalLinearOperator(jnp.ones(3))
        >>> obs_aug, y_aug, noise_aug = gaussx.tikhonov_augment(
        ...     particles,
        ...     particles @ obs_operator.T,
        ...     jnp.array([1.0, -1.0]),
        ...     obs_noise,
        ...     jnp.zeros(3),
        ...     prior_cov,
        ... )
        >>> obs_aug.shape, y_aug.shape
        ((200, 5), (5,))

        The triple goes straight into `eki_step`:

        >>> gaussx.eki_step(particles, obs_aug, y_aug, noise_aug, key=key).shape
        (200, 3)

    References:
        Chada, Stuart & Tong (2020).
    """
    ensemble_size, state_size = particles.shape
    obs_size = obs_particles.shape[1]
    if obs_particles.shape[0] != ensemble_size:
        raise ValueError(
            "particles and obs_particles must share the same ensemble size, "
            f"got J={ensemble_size} and J={obs_particles.shape[0]}."
        )
    if observation.shape != (obs_size,):
        raise ValueError(
            f"observation must have shape ({obs_size},) to match obs_particles, "
            f"got {observation.shape}."
        )
    _check_obs_noise_shape(obs_size, obs_noise)
    if prior_mean.shape != (state_size,):
        raise ValueError(
            f"prior_mean must have shape ({state_size},) to match particles, got "
            f"{prior_mean.shape}."
        )
    if (prior_cov.in_size(), prior_cov.out_size()) != (state_size, state_size):
        raise ValueError(
            f"prior_cov must be ({state_size}, {state_size}) to match particles, "
            f"got ({prior_cov.out_size()}, {prior_cov.in_size()})."
        )

    # jnp, not einx: |ℰ| = 0 returns an empty ensemble here today, and einx
    # raises AxisSizeError on a zero-length axis.
    # h_aug(x⁽ᵉ⁾) = [h⁽ᵉ⁾, x⁽ᵉ⁾]
    obs_particles_aug = jnp.concatenate([obs_particles, particles], axis=1)
    # y_aug = [y, μ_x]
    observation_aug = jnp.concatenate([observation, prior_mean])
    # Σ_ε,aug = blockdiag(Σ_ε, Σ_x)
    obs_noise_aug = BlockDiag(obs_noise, prior_cov)
    return obs_particles_aug, observation_aug, obs_noise_aug


def discrepancy_step_size(
    obs_particles: Float[Array, "J M"],
    observation: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    remaining: Float[Array, ""],
    bessel: bool = True,
) -> Float[Array, ""]:
    """Adaptive EKI tempering step: the data misfit controller.

        𝒥⁽ᵉ⁾ = ½ (y − h⁽ᵉ⁾)ᵀ Σ_ε⁻¹ (y − h⁽ᵉ⁾)
        Δt = min(max(D_y / (2 𝒥̄), √(D_y / (2 σ²_𝒥))), remaining)

    Equation (14) of Iglesias & Yang (2021), the selection rule of their EKI-DMC
    (Algorithm 3), which needs no tuning parameter. 𝒥̄ and σ²_𝒥 are the sample
    mean and variance of the per-member cost 𝒥⁽ᵉ⁾ across the ensemble.

    The two candidates are the paper's statistical discrepancy principle applied
    to the tempered sub-problem y = h(x) + ε / √Δt. Since
    (y − h(x))ᵀ Σ_ε⁻¹ (y − h(x)) is χ² with D_y degrees of freedom under the
    correct model, it has mean D_y (their C1, accuracy, giving D_y / (2 𝒥̄)) and
    variance 2 D_y (their C2, uncertainty, giving √(D_y / (2 σ²_𝒥))). The **max**
    is deliberate and enforces *at least one* of the two, not both: their
    Remark 2 notes that a wide prior makes 𝒥̄ ≪ σ_𝒥, so C1 binds; a narrow prior
    centred far from the truth flips it, and C2 then licenses the larger step.
    The outer min is the tempering budget, which is also the stopping rule: the
    driver halts on the iteration where it binds.

    Only ``obs_noise`` solves are used; no square root of Σ_ε is ever formed.

    Algorithm:
        1. Form the residuals r⁽ᵉ⁾ = y − h⁽ᵉ⁾ and solve Σ_ε⁻¹ r⁽ᵉ⁾ row by row.
        2. Form the cost 𝒥⁽ᵉ⁾ = ½ r⁽ᵉ⁾ᵀ Σ_ε⁻¹ r⁽ᵉ⁾ of each member.
        3. Take its sample mean 𝒥̄ and variance σ²_𝒥.
        4. Return the larger candidate, capped by ``remaining``.

    Args:
        obs_particles: Ensemble in observation space h⁽ᵉ⁾ = h(x⁽ᵉ⁾). Under
            `tikhonov_augment` pass the augmented ensemble, so that D_y counts
            the augmented observations.
        observation: The observation y.
        obs_noise: Observation-error covariance Σ_ε.
        remaining: Tempering budget left, 1 − ∑ₖ Δt⁽ᵏ⁾ over the steps k already
            taken. May be traced.
        bessel: Use the 1 / (|ℰ| − 1) divisor for σ²_𝒥. Defaults to ``True``,
            matching the rest of this module.

    Returns:
        The step Δt, a scalar. It never exceeds ``remaining``, and is positive
        whenever ``remaining`` is: a degenerate ensemble (σ²_𝒥 = 0) sends the
        second candidate to +∞, so the max saturates and ``remaining`` is
        returned. It is 0 when ``remaining`` is not positive.

    Raises:
        ValueError: If the shapes disagree, or if ``bessel`` is set with
            |ℰ| < 2.

    Note:
        gh-230 specified both terms in the misfit of the ensemble *mean*,
        (y − h̄)ᵀ Σ_ε⁻¹ (y − h̄). This follows the paper instead: equation (13)
        defines the costs per member and equation (14) takes their mean and
        variance. The distinction is not cosmetic: σ²_𝒥 is identically zero for
        any single vector, so a mean-misfit reading would make C2 infinite and
        the max vacuous.

    Examples:
        A step inside the budget:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> obs_particles = jr.normal(jr.key(0), (50, 4))
        >>> obs_noise = lx.DiagonalLinearOperator(jnp.ones(4))
        >>> observation = jnp.zeros(4)
        >>> dt = gaussx.discrepancy_step_size(
        ...     obs_particles, observation, obs_noise, remaining=jnp.asarray(1.0)
        ... )
        >>> bool(0.0 < dt <= 1.0)
        True

        A small budget caps the step:

        >>> capped = gaussx.discrepancy_step_size(
        ...     obs_particles, observation, obs_noise, remaining=jnp.asarray(1e-3)
        ... )
        >>> bool(jnp.allclose(capped, 1e-3))
        True

    References:
        Iglesias & Yang (2021).
    """
    ensemble_size, obs_size = obs_particles.shape
    _check_ensemble_size(ensemble_size, bessel)
    if observation.shape != (obs_size,):
        raise ValueError(
            f"observation must have shape ({obs_size},) to match obs_particles, "
            f"got {observation.shape}."
        )
    _check_obs_noise_shape(obs_size, obs_noise)

    # jnp, not einx: an empty D_y returns a step here today, and einx raises
    # AxisSizeError on a zero-length axis.
    # r⁽ᵉ⁾ = y − h⁽ᵉ⁾
    residuals = observation[None, :] - obs_particles
    # Σ_ε⁻¹ r⁽ᵉ⁾
    whitened_residuals = solve_rows(obs_noise, residuals)
    # 𝒥⁽ᵉ⁾ = ½ r⁽ᵉ⁾ᵀ Σ_ε⁻¹ r⁽ᵉ⁾
    cost = 0.5 * jnp.sum(residuals * whitened_residuals, axis=-1)

    # 𝒥̄
    mean_cost = jnp.mean(cost)
    # σ²_𝒥
    variance_cost = jnp.var(cost, ddof=1 if bessel else 0)
    # C1: D_y / (2 𝒥̄)
    accuracy = obs_size / (2.0 * mean_cost)
    # C2: √(D_y / (2 σ²_𝒥))
    uncertainty = jnp.sqrt(obs_size / (2.0 * variance_cost))
    # Δt = min(max(C1, C2), max(remaining, 0)), since round-off can take the budget
    # just below zero in a fixed-length driver
    return jnp.minimum(jnp.maximum(accuracy, uncertainty), jnp.maximum(remaining, 0.0))
