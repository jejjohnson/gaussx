"""Ensemble Kalman inversion (EKI): tempered step, TEKI prior, adaptive step."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from gaussx._einx import einsum
from gaussx._inference._enkf import (
    _analysis_gain,
    _check_analysis_shapes,
    _check_localization_shapes,
    _noise_factor,
)
from gaussx._inference._ensemble_stats import _check_ensemble_size
from gaussx._inference._etkf import etkf_transform
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
            to $R/\Delta t$ as a lazy `gaussx.ScaledOperator`, so a
            structured $R$ keeps its own solve in both variants.
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
            Stochastic variant only: the deterministic one forms no gain --
            `etkf_transform` supplies the mean update as well -- and solves
            against $R$ itself.
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

    if deterministic:
        # By the push-through identity K (y - G_bar) = U'^T w_mean, so the
        # ETKF weights give the mean update too, and no separate innovation
        # solve for the gain is needed.
        anomalies = particles - jnp.mean(particles, axis=0, keepdims=True)  # (J, N)
        w_mean, transform = etkf_transform(
            obs_particles, observation, tempered_noise, solver=solver
        )
        mean_increment = einsum(w_mean, anomalies, "J, J N -> N")  # (N,)
        # The increment, not the transformed anomalies: Lambda acts on
        # differences, so Lambda = I leaves `transform @ anomalies` exactly.
        anomaly_increment = transform @ anomalies - anomalies  # (J, N)
        if step is not None:
            mean_increment = step.mv(mean_increment)
            anomaly_increment = jax.vmap(step.mv)(anomaly_increment)
        return particles + mean_increment[None, :] + anomaly_increment

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
