"""Exact discretisation of linear time-invariant SDEs.

The SDE dx(t) = F_c x(t) dt + L_c db(t) has drift matrix F_c, diffusion
(noise-effect) matrix L_c, white-noise spectral density Q_c, diffusion
covariance B_c = L_c Q_c L_cᵀ and, when it exists, stationary covariance
Σ_x,∞. Over a step Δt_t it discretises exactly to

    x_t = F_t x_t−1 + η_t,    η_t ~ 𝒩(0, Σ_η,t),    F_t = exp(F_c Δt_t),

either from Σ_x,∞ (`process_noise_covariance`) or by matrix-fraction
decomposition (`discretize_mfd`), which needs no Σ_x,∞.
"""

from __future__ import annotations

from typing import overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange, reduce
from gaussx._linalg._linalg import cov_transform
from gaussx._linalg._symmetrize import symmetrize


@overload
def process_noise_covariance(
    A: Float[Array, "N N"],
    Pinf: Float[Array, "N N"],
) -> Float[Array, "N N"]: ...


@overload
def process_noise_covariance(
    A: lx.AbstractLinearOperator,
    Pinf: Float[Array, "N N"] | lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator: ...


@overload
def process_noise_covariance(
    A: Float[Array, "N N"] | lx.AbstractLinearOperator,
    Pinf: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator: ...


def process_noise_covariance(
    A: Float[Array, "N N"] | lx.AbstractLinearOperator,
    Pinf: Float[Array, "N N"] | lx.AbstractLinearOperator,
) -> Float[Array, "N N"] | lx.AbstractLinearOperator:
    """Model-noise covariance Σ_η,t of a step, from the stationary covariance.

        Σ_η,t = Σ_x,∞ − F_t Σ_x,∞ F_tᵀ

    This is the standard steady-state trick: it follows from stationarity
    of the discretised recursion, Σ_x,∞ = F_t Σ_x,∞ F_tᵀ + Σ_η,t, and
    avoids matrix-fraction integration (Särkkä & Solin 2019, §6.4). It is
    the forward direction of the discrete Lyapunov equation that
    `gaussx.discrete_lyapunov_solve` inverts: this maps Σ_x,∞ to Σ_η,t,
    that one maps Σ_η,t to Σ_x,∞ for the same F_t.

    The congruence F_t Σ_x,∞ F_tᵀ is delegated to `gaussx.cov_transform`,
    so structure is exploited rather than re-derived: a diagonal Σ_x,∞
    skips its dense materialisation, and matched `gaussx.Kronecker` /
    `gaussx.BlockDiag` operands stay factorised through `gaussx.sandwich`.
    Passing **operators** returns a lazy operator: nothing is materialised,
    and the structural class of the result follows `gaussx.sandwich`.
    Passing **arrays** returns an array.

    The result is not symmetrised. Callers that need a covariance clean of
    floating-point asymmetry should wrap it in `gaussx.symmetrize`, as
    `gaussx.SDEKernel.discretise` does.

    Args:
        A: Transition F_t, an array or an operator.
        Pinf: Stationary covariance Σ_x,∞, an array or an operator.

    Returns:
        Model-noise covariance Σ_η,t: an operator when either argument is
        an operator, otherwise an array.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> transition = jnp.array([[0.9, 0.1], [0.0, 0.8]])
        >>> stationary = jnp.array([[1.0, 0.2], [0.2, 2.0]])
        >>> noise = gaussx.process_noise_covariance(transition, stationary)
        >>> expected = stationary - transition @ stationary @ transition.T
        >>> bool(jnp.allclose(noise, expected))
        True

        With operators the result stays lazy:

        >>> import lineax as lx
        >>> noise_op = gaussx.process_noise_covariance(
        ...     lx.MatrixLinearOperator(transition),
        ...     lx.MatrixLinearOperator(stationary, lx.symmetric_tag),
        ... )
        >>> isinstance(noise_op, lx.AbstractLinearOperator)
        True
        >>> bool(jnp.allclose(noise_op.as_matrix(), expected))
        True
    """
    stationary_covariance_op = (
        Pinf
        if isinstance(Pinf, lx.AbstractLinearOperator)
        else lx.MatrixLinearOperator(Pinf, lx.symmetric_tag)
    )
    # One implementation of the congruence, for both paths. Wrapping a dense
    # Σ_x,∞ costs nothing at runtime: it is pytree bookkeeping at trace
    # time, and the dense branch of ``cov_transform`` evaluates the identical
    # ``A @ Pinf @ A.T`` expression.
    # F_t Σ_x,∞ F_tᵀ
    propagated = cov_transform(A, stationary_covariance_op)

    if isinstance(A, lx.AbstractLinearOperator) or isinstance(
        Pinf, lx.AbstractLinearOperator
    ):
        # Σ_η,t = Σ_x,∞ − F_t Σ_x,∞ F_tᵀ, lazily
        return stationary_covariance_op - propagated
    # Σ_η,t = Σ_x,∞ − F_t Σ_x,∞ F_tᵀ
    return Pinf - propagated.as_matrix()


# The augmented block below contains exp(−F_cᵀ Δt), which *grows* for a
# stable drift, so a stiff F_c or a long step overflows the float range even
# when the process covariance itself is small and finite. The remedy is
# exact rather than approximate: discretise over Δt / 2^m, then compose
# the result back up m times via
#
#     F(2h) = F(h) F(h),    Σ_η(2h) = F(h) Σ_η(h) F(h)ᵀ + Σ_η(h)
#
# Only as many doublings as the drift actually needs are applied, since
# each one costs a little precision. _MFD_MAX_SQUARINGS bounds the static
# unrolling; _MFD_EXPONENT_BUDGET is the per-step ‖F_c‖ Δt kept below the
# float32 overflow point of ~88 with margin to spare.
_MFD_MAX_SQUARINGS = 12
_MFD_EXPONENT_BUDGET = 30.0
# Largest diffusion *times substep* left unscaled. Van Loan's generator
# carries B_c in its off-diagonal block and is exponentiated as M Δt, so
# expm starts returning NaN well below the float range -- measured at 1e6 for
# an otherwise trivial problem.
_MFD_DIFFUSION_BUDGET = 1e3


def _van_loan(
    drift: Float[Array, "d d"],
    diffusion_covariance: Float[Array, "d d"],
    dt: Float[Array, ""],
) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
    """One unscaled Van Loan discretisation; see `discretize_mfd`.

        M = [[F_c, B_c], [0, −F_cᵀ]],    exp(M Δt) = [[F_t, C], [0, F_t⁻ᵀ]]
        Σ_η = C F_tᵀ

    Args:
        drift: Drift matrix F_c.
        diffusion_covariance: Diffusion covariance B_c = L_c Q_c L_cᵀ.
        dt: Step Δt.

    Returns:
        Tuple ``(transition, model_noise_covariance)`` holding F_t and the
        symmetrised Σ_η.
    """
    state_size = drift.shape[0]
    dtype = jnp.result_type(drift, diffusion_covariance)

    # Van Loan's augmented generator. The key property is that it is block
    # upper triangular, so its exponential is too, and the three blocks of
    # exp(M Δt) each carry a piece of the answer: F_t = exp(F_c Δt),
    # D = exp(−F_cᵀ Δt) and, the whole point,
    # C = (∫_0^Δt exp(F_c s) B_c exp(F_cᵀ s) ds) D. The integral we want
    # is therefore recovered as Σ_η = C D⁻¹, without ever discretising the
    # integral or solving a Lyapunov equation.
    # −F_cᵀ
    drift_transpose = rearrange(drift, "row col -> col row")
    # M = [[F_c, B_c], [0, −F_cᵀ]]
    generator = jnp.block(
        [
            [drift, diffusion_covariance],
            [jnp.zeros((state_size, state_size), dtype=dtype), -drift_transpose],
        ]
    )
    # exp(M Δt)
    block = jsl.expm(generator * dt)

    # F_t = exp(F_c Δt). This is also the top-left block; computing it
    # separately costs one small extra expm and keeps the transition
    # matrix, the more load-bearing of the two outputs, independent of the
    # conditioning of the 2D_x-square augmented problem.
    transition = jsl.expm(drift * dt)

    # C, the top-right block: the integral post-multiplied by D.
    cross = block[:state_size, state_size:]

    # Σ_η = C D⁻¹. No inverse or solve is formed: D = exp(−F_cᵀ Δt), so
    # D⁻¹ = exp(F_cᵀ Δt) = F_tᵀ exactly. Measured at 0.0 to 2.2e-16 against
    # an explicit inv(D) across stable, unstable and undamped-oscillatory
    # drifts.
    #
    # The symmetrise is not cosmetic: C F_tᵀ is a product of two unrelated
    # exponentials and is asymmetric at the 1e-16 level, which downstream
    # Cholesky factorisations will reject.
    # Σ_η = C F_tᵀ
    model_noise_covariance = einsum(
        cross, transition, "row inner, col inner -> row col"
    )
    return transition, symmetrize(model_noise_covariance)


def _double_step(
    operands: tuple[Float[Array, "d d"], Float[Array, "d d"]],
) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
    """Compose a step with itself: the discretisation over twice the step.

        F(2h) = F(h) F(h),    Σ_η(2h) = F(h) Σ_η(h) F(h)ᵀ + Σ_η(h)

    Args:
        operands: Tuple of the transition F(h) and model-noise covariance
            Σ_η(h) over a step h.

    Returns:
        Tuple of F(2h) and the symmetrised Σ_η(2h).
    """
    transition, model_noise_covariance = operands
    # F(2h) = F(h) F(h)
    doubled_transition = einsum(
        transition, transition, "next mid, mid prev -> next prev"
    )
    # F(h) Σ_η(h)
    transition_noise = einsum(
        transition, model_noise_covariance, "next prev, prev col -> next col"
    )
    # F(h) Σ_η(h) F(h)ᵀ
    propagated = einsum(
        transition_noise, transition, "next col, other col -> next other"
    )
    # Σ_η(2h) = F(h) Σ_η(h) F(h)ᵀ + Σ_η(h)
    doubled_noise = symmetrize(propagated + model_noise_covariance)
    return doubled_transition, doubled_noise


def discretize_mfd(
    F: Float[Array, "d d"],
    Q_c: Float[Array, "d d"],
    dt: Float[Array, ""],
) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
    """Discretise a linear SDE by matrix-fraction decomposition.

        F_t = exp(F_c Δt),    Σ_η,t = ∫_0^Δt exp(F_c s) B_c exp(F_cᵀ s) ds

    for the discrete-time model x_t = F_t x_t−1 + η_t, η_t ~ 𝒩(0, Σ_η,t).
    Both come from one 2D_x × 2D_x matrix exponential (Van Loan 1978). For
    the augmented generator

        M = [[F_c, B_c], [0, −F_cᵀ]],    exp(M Δt) = [[F_t, C], [0, D]]

    the integral is Σ_η,t = C D⁻¹, and D = exp(−F_cᵀ Δt) is always
    invertible with D⁻¹ = F_tᵀ, so no inverse or solve is formed here.

    Unlike the stationary route in `gaussx.SDEKernel.discretise`, this
    needs no Σ_x,∞ and is well defined for **every** F_c, including a
    learned F_c whose eigenvalues sum to zero in pairs. The stationary
    route recovers Σ_x,∞ from the Lyapunov equation
    F_c Σ_x,∞ + Σ_x,∞ F_cᵀ + B_c = 0, which has a unique solution only
    when no two eigenvalues of F_c sum to zero. An undamped oscillatory
    mode has the eigenvalue pair ±iω, which always does, so the Lyapunov
    route degenerates.

    Algorithm:
        1. Check Δt ≥ 0, and pick the number m of step halvings that keeps
           ‖F_c‖₁ Δt / 2^m within budget.
        2. Scale B_c down by a power of two when B_c Δt / 2^m would
           overflow the exponential.
        3. Discretise over the substep Δt / 2^m with one Van Loan
           exponential.
        4. Double back up m times, F(2h) = F(h)², Σ_η(2h) =
           F(h) Σ_η(h) F(h)ᵀ + Σ_η(h), and undo the diffusion scaling.

    Note:
        The obstruction is degeneracy of that Sylvester system, not
        instability. The identity Σ_η,t = Σ_x,∞ − F_t Σ_x,∞ F_tᵀ holds for
        unstable F_c, and even when the Lyapunov solution is not PSD and so
        is not a valid covariance. Constraining F_c to be Hurwitz would not
        fix the degeneracy, and would forbid exactly the oscillatory modes
        this function exists to support.

    Args:
        F: Drift matrix F_c.
        Q_c: Diffusion covariance B_c = L_c Q_c L_cᵀ (not the spectral
            density Q_c itself, unless L_c = I).
        dt: Time step Δt. Must be non-negative; checked with
            `equinox.error_if`, so under ``jit`` the error fires at
            evaluation rather than trace time.

    Returns:
        Tuple ``(A, Q)`` holding the transition F_t and the model-noise
        covariance Σ_η,t, which is symmetrised since C F_tᵀ is not
        symmetric to floating point.

    Raises:
        equinox.EquinoxRuntimeError: If Δt < 0, or if ‖F_c‖ Δt needs more
            than 12 step halvings.

    Examples:
        A Matérn-1/2 SDE, F_c = −1 and B_c = 2, matches its stationary
        closed form Σ_η,t = 1 − exp(−2Δt):

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> transition, noise = gaussx.discretize_mfd(
        ...     jnp.array([[-1.0]]), jnp.array([[2.0]]), jnp.asarray(0.5)
        ... )
        >>> bool(jnp.allclose(transition, jnp.exp(-0.5)))
        True
        >>> bool(jnp.allclose(noise, 1.0 - jnp.exp(-1.0)))
        True

        An undamped oscillator, which has no unique Lyapunov solution:

        >>> drift = jnp.array([[0.0, 1.0], [-4.0, 0.0]])
        >>> diffusion = jnp.array([[0.0, 0.0], [0.0, 1.0]])
        >>> step = jnp.asarray(0.3)
        >>> transition, noise = gaussx.discretize_mfd(drift, diffusion, step)
        >>> transition.shape, bool(jnp.all(jnp.linalg.eigvalsh(noise) > 0.0))
        ((2, 2), True)
    """
    # A negative step would silently run the exponential backwards and
    # return a Σ_η that is not a covariance. error_if defers the check to
    # evaluation time so this stays traceable under jit.
    dt = eqx.error_if(dt, dt < 0, "discretize_mfd requires dt >= 0.")

    # How many doublings are needed to keep the augmented exponential in
    # range. The 1-norm bounds the spectral radius, so this is
    # conservative. Traced, but only ever used as a *predicate* below --
    # never as a trip count -- so the whole function stays reverse-mode
    # differentiable, unlike a lax.while_loop.
    # column sums of |F_c|
    column_abs_sums = reduce(jnp.abs(F), "row col -> col", "sum")
    # ‖F_c‖₁ Δt
    drift_scale = column_abs_sums.max() * dt
    # ⌈log₂(‖F_c‖₁ Δt / budget)⌉, floored at 0
    required = jnp.ceil(jnp.log2(jnp.maximum(drift_scale / _MFD_EXPONENT_BUDGET, 1.0)))
    # m
    n_squarings = jnp.clip(required, 0.0, _MFD_MAX_SQUARINGS)

    # Beyond the cap the scaled step is still too stiff and the augmented
    # exponential would overflow anyway, so say so rather than returning a
    # silent NaN. In float32 this starts to bite around ‖F_c‖ Δt ~ 1e6.
    dt = eqx.error_if(
        dt,
        required > _MFD_MAX_SQUARINGS,
        f"discretize_mfd: ||F|| * dt is too large to discretise in one step "
        f"(more than {_MFD_MAX_SQUARINGS} doublings would be needed). Split "
        f"the interval into shorter steps, or rescale time.",
    )

    # Σ_η is *linear* in B_c, so the diffusion's magnitude can be divided out
    # of the exponential and multiplied back afterwards. Without this the
    # augmented generator inherits ‖B_c‖ in its off-diagonal block and
    # overflows for a large diffusion even when the drift is benign and the
    # answer trivial -- F_c = 0 with B_c = 1e6 already returns NaN. Scaling
    # the *step* would not help there and would wrongly reject it: the
    # growth is linear in B_c, not exponential.
    # Scale by a power of two, and only when the magnitude actually
    # threatens the exponential. A power of two is exact in binary floating
    # point, so it introduces no rounding of its own, and leaving small
    # diffusions alone matters: normalising unconditionally by the largest
    # entry pushes smaller modes down by the same factor, and one that
    # underflows is lost outright rather than merely rounded.
    #
    # A diffusion whose dynamic range exceeds the dtype's own cannot
    # survive any uniform scaling -- in float32, diag(1e30, 1e-20) has no
    # scaling that keeps both representable. That is a limit of the format,
    # not of the scheme.
    #
    # The budget applies to what the exponential actually sees, which is
    # B_c scaled by the *substep*: `_van_loan` exponentiates M Δt, so a
    # modest diffusion over a long step is just as dangerous as a large one
    # over a short step. B_c = 1e3 with Δt = 1e5 presents the same 1e8
    # off-diagonal magnitude that NaNs at B_c = 1e8, Δt = 1.
    # h = Δt / 2^m
    substep = dt / 2.0**n_squarings
    # max |B_c| h
    effective_diffusion = jnp.abs(Q_c).max() * substep
    # s = 2^⌈log₂(max |B_c| h / budget)⌉ when over budget, else 1
    diffusion_scale = jnp.where(
        effective_diffusion > _MFD_DIFFUSION_BUDGET,
        2.0 ** jnp.ceil(jnp.log2(effective_diffusion / _MFD_DIFFUSION_BUDGET)),
        1.0,
    )

    # F(h), Σ_η(h) / s
    transition, model_noise_covariance = _van_loan(F, Q_c / diffusion_scale, substep)

    # Compose back up. The unroll is static, but each doubling is placed
    # behind a lax.cond rather than a select so an inactive step is not
    # evaluated at all -- a well-scaled drift, which is the common case,
    # pays nothing here. lax.cond is used over a while_loop because it
    # stays reverse-mode differentiable.
    #
    # Under vmap (as in discretize_mfd_sequence) a cond with a batched
    # predicate lowers back to a select, so batched callers do pay for the
    # inactive branches; that is a JAX limitation, not an oversight.
    for i in range(_MFD_MAX_SQUARINGS):
        # F(2h), Σ_η(2h) while i < m
        transition, model_noise_covariance = jax.lax.cond(
            i < n_squarings,
            _double_step,
            lambda operands: operands,
            (transition, model_noise_covariance),
        )

    # Σ_η,t = s (Σ_η,t / s), undoing the diffusion scaling
    return transition, model_noise_covariance * diffusion_scale


def discretize_mfd_sequence(
    F: Float[Array, "d d"],
    Q_c: Float[Array, "d d"],
    dt: Float[Array, " N"],
) -> tuple[Float[Array, "N d d"], Float[Array, "N d d"]]:
    """Vectorised `gaussx.discretize_mfd` over a stack of steps Δt_1:T.

        F_t = exp(F_c Δt_t),    Σ_η,t,    t = 1, …, T

    Args:
        F: Drift matrix F_c.
        Q_c: Diffusion covariance B_c = L_c Q_c L_cᵀ.
        dt: Time steps Δt_1:T. All must be non-negative.

    Returns:
        Tuple ``(A_seq, Q_seq)`` stacking F_t and Σ_η,t over time.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> transitions, noises = gaussx.discretize_mfd_sequence(
        ...     jnp.array([[-1.0]]), jnp.array([[2.0]]), jnp.array([0.1, 0.5, 1.0])
        ... )
        >>> transitions.shape, noises.shape
        ((3, 1, 1), (3, 1, 1))
    """
    return jax.vmap(lambda step: discretize_mfd(F, Q_c, step))(dt)
