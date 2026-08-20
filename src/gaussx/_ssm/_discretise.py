"""Exact discretisation of stationary linear SDEs."""

from __future__ import annotations

from typing import overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lineax as lx
from jaxtyping import Array, Float

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
    r"""Compute process noise from stationary covariance.

    Computes:

        Q = Pinf - A @ Pinf @ A^T

    For a discrete-time state-space model with stationary covariance
    ``Pinf`` and transition matrix ``A``. This is the standard
    "steady-state trick": it follows from stationarity of the discretised
    recursion ``Pinf = A Pinf A^T + Q``, and avoids matrix-fraction
    integration (Särkkä & Solin 2019, §6.4).

    It is the forward direction of the discrete Lyapunov equation that
    `gaussx.discrete_lyapunov_solve` inverts: this maps
    ``Pinf -> Q``, that one maps ``Q -> Pinf`` for the same ``A``.

    The ``A Pinf A^T`` term is delegated to
    `gaussx.cov_transform`, so structure is exploited rather than
    re-derived: a diagonal ``Pinf`` skips its ``(N, N)`` materialization, and
    matched `gaussx.Kronecker` / `gaussx.BlockDiag`
    operands stay factorised through `gaussx.sandwich`.

    Passing **operators** returns a lazy operator — nothing is materialized,
    and the structural class of the result follows `gaussx.sandwich`.
    Passing **arrays** returns an array, as before.

    The result is not symmetrised. Callers that need a covariance clean of
    floating-point asymmetry should wrap it in `gaussx.symmetrize`,
    as `gaussx.SDEKernel.discretise` does.

    Args:
        A: State transition matrix, shape ``(N, N)`` — array or operator.
        Pinf: Stationary covariance, shape ``(N, N)`` — array or operator.

    Returns:
        Process noise covariance Q, shape ``(N, N)``. An operator when
        either argument is an operator, otherwise an array.
    """
    Pinf_op = (
        Pinf
        if isinstance(Pinf, lx.AbstractLinearOperator)
        else lx.MatrixLinearOperator(Pinf, lx.symmetric_tag)
    )
    # One implementation of the congruence, for both paths. Wrapping a dense
    # ``Pinf`` costs nothing at runtime — it is pytree bookkeeping at trace
    # time, and the dense branch of ``cov_transform`` evaluates the identical
    # ``A @ Pinf @ A.T`` expression.
    congruence = cov_transform(A, Pinf_op)

    if isinstance(A, lx.AbstractLinearOperator) or isinstance(
        Pinf, lx.AbstractLinearOperator
    ):
        return Pinf_op - congruence
    return Pinf - congruence.as_matrix()


# The augmented block below contains exp(-F^T dt), which *grows* for a
# stable drift, so a stiff F or a long step overflows the float range even
# when the process covariance itself is small and finite. The remedy is
# exact rather than approximate: discretise over dt / 2**k, then compose
# the result back up k times via
#
#     A(2h) = A(h) A(h),    Q(2h) = A(h) Q(h) A(h)^T + Q(h)
#
# Only as many doublings as the drift actually needs are applied, since
# each one costs a little precision. _MFD_MAX_SQUARINGS bounds the static
# unrolling; _MFD_EXPONENT_BUDGET is the per-step ||F|| dt kept below the
# float32 overflow point of ~88 with margin to spare.
_MFD_MAX_SQUARINGS = 12
_MFD_EXPONENT_BUDGET = 30.0


def _van_loan(
    F: Float[Array, "d d"],
    Q_c: Float[Array, "d d"],
    dt: Float[Array, ""],
) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
    """One unscaled Van Loan discretisation; see `discretise_mfd`."""
    d = F.shape[0]
    dtype = jnp.result_type(F, Q_c)

    # Van Loan's augmented generator. The key property is that it is block
    # upper triangular, so its exponential is too, and the three blocks of
    # exp(Phi dt) each carry a piece of the answer:
    #
    #     Phi = [  F   Q_c ]          exp(Phi dt) = [ A   C ]
    #           [  0  -F^T ]                        [ 0   D ]
    #
    # with A = exp(F dt), D = exp(-F^T dt), and -- the whole point --
    # C = (integral_0^dt exp(F s) Q_c exp(F^T s) ds) . D. The integral we
    # want is therefore recovered as Q = C D^-1, without ever discretising
    # the integral or solving a Lyapunov equation.
    generator = jnp.block(
        [
            [F, Q_c],
            [jnp.zeros((d, d), dtype=dtype), -F.T],
        ]
    )
    block = jsl.expm(generator * dt)

    # A = exp(F dt). This is also block[:d, :d]; computing it separately
    # costs one small extra expm and keeps the transition matrix -- the
    # more load-bearing of the two outputs -- independent of the
    # conditioning of the 2d-square augmented problem.
    A = jsl.expm(F * dt)

    # C, the top-right block: the integral post-multiplied by D.
    cross = block[:d, d:]

    # Q = C D^-1. No inverse or solve is formed: D = exp(-F^T dt), so
    # D^-1 = exp(F^T dt) = (exp(F dt))^T = A^T exactly. Measured at 0.0 to
    # 2.2e-16 against an explicit inv(D) across stable, unstable and
    # undamped-oscillatory drifts.
    #
    # The symmetrise is not cosmetic: C @ A^T is a product of two
    # unrelated exponentials and is asymmetric at the 1e-16 level, which
    # downstream Cholesky factorisations will reject.
    return A, symmetrize(cross @ A.T)


def discretise_mfd(
    F: Float[Array, "d d"],
    Q_c: Float[Array, "d d"],
    dt: Float[Array, ""],
) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
    r"""Discretise a linear SDE by matrix-fraction decomposition.

    Returns $(A, Q)$ for the discrete-time model $x_k = A x_{k-1} + q_k$
    with $q_k \sim N(0, Q)$, where

    $$
    A = e^{F\,\Delta t}, \qquad
    Q = \int_0^{\Delta t} e^{Fs}\, Q_c\, e^{F^\top s}\, ds .
    $$

    Both come from one $2d \times 2d$ matrix exponential (Van Loan 1978).
    For the augmented generator

    $$
    \Phi = \begin{bmatrix} F & Q_c \\ 0 & -F^\top \end{bmatrix},
    \qquad
    e^{\Phi \Delta t}
        = \begin{bmatrix} A & C \\ 0 & D \end{bmatrix},
    $$

    the integral is $Q = C D^{-1}$, and $D = e^{-F^\top \Delta t}$ is
    always invertible with $D^{-1} = A^\top$ — so no inverse or solve is
    formed here.

    Unlike the stationary route in `gaussx.SDEKernel.discretise`, this
    needs no $P_\infty$ and is well defined for **every** $F$, including a
    learned $F$ whose eigenvalues sum to zero in pairs. The stationary
    route recovers $P_\infty$ from the Lyapunov equation
    $F P + P F^\top + Q_c = 0$, which has a unique solution only when
    $\lambda_i(F) + \lambda_j(F) \neq 0$ for all $i, j$. An undamped
    oscillatory mode has $\lambda = \pm i\omega$, so $\lambda + \bar\lambda
    = 0$ always and the Lyapunov route degenerates.

    Note:
        The obstruction is degeneracy of that Sylvester system, not
        instability. The identity $Q = P_\infty - A P_\infty A^\top$ holds
        for unstable $F$, and even when the Lyapunov solution is not PSD
        and so is not a valid covariance. Constraining $F$ to be Hurwitz
        would not fix the degeneracy, and would forbid exactly the
        oscillatory modes this function exists to support.

    Args:
        F: Continuous-time drift matrix, shape ``(d, d)``.
        Q_c: Continuous-time diffusion covariance $L Q_c L^\top$, shape
            ``(d, d)``.
        dt: Time step. Must be non-negative; checked with
            `equinox.error_if`, so under ``jit`` the error fires at
            evaluation rather than trace time.

    Returns:
        Tuple ``(A, Q)``, both shape ``(d, d)``. ``Q`` is symmetrised —
        ``C @ A.T`` is not symmetric to floating point.
    """
    # A negative step would silently run the exponential backwards and
    # return a Q that is not a covariance. error_if defers the check to
    # evaluation time so this stays traceable under jit.
    dt = eqx.error_if(dt, dt < 0, "discretise_mfd requires dt >= 0.")

    # How many doublings are needed to keep the augmented exponential in
    # range. The 1-norm bounds the spectral radius, so this is
    # conservative. Traced, but only ever used as a *predicate* below --
    # never as a trip count -- so the whole function stays reverse-mode
    # differentiable, unlike a lax.while_loop.
    drift_scale = jnp.abs(F).sum(axis=0).max() * dt
    required = jnp.ceil(jnp.log2(jnp.maximum(drift_scale / _MFD_EXPONENT_BUDGET, 1.0)))
    n_squarings = jnp.clip(required, 0.0, _MFD_MAX_SQUARINGS)

    # Beyond the cap the scaled step is still too stiff and the augmented
    # exponential would overflow anyway, so say so rather than returning a
    # silent NaN. In float32 this starts to bite around ||F|| dt ~ 1e6.
    dt = eqx.error_if(
        dt,
        required > _MFD_MAX_SQUARINGS,
        f"discretise_mfd: ||F|| * dt is too large to discretise in one step "
        f"(more than {_MFD_MAX_SQUARINGS} doublings would be needed). Split "
        f"the interval into shorter steps, or rescale time.",
    )

    # Q is *linear* in Q_c, so the diffusion's magnitude can be divided out
    # of the exponential and multiplied back afterwards. Without this the
    # augmented generator inherits ||Q_c|| in its off-diagonal block and
    # overflows for a large diffusion even when the drift is benign and the
    # answer trivial -- F = 0 with Q_c = 1e6 already returns NaN. Scaling
    # the *step* would not help there and would wrongly reject it: the
    # growth is linear in Q_c, not exponential.
    diffusion_scale = jnp.maximum(jnp.abs(Q_c).max(), jnp.finfo(Q_c.dtype).tiny)

    A, Q = _van_loan(F, Q_c / diffusion_scale, dt / 2.0**n_squarings)

    # Compose back up. The unroll is static, but each doubling is placed
    # behind a lax.cond rather than a select so an inactive step is not
    # evaluated at all -- a well-scaled drift, which is the common case,
    # pays nothing here. lax.cond is used over a while_loop because it
    # stays reverse-mode differentiable.
    #
    # Under vmap (as in discretise_mfd_sequence) a cond with a batched
    # predicate lowers back to a select, so batched callers do pay for the
    # inactive branches; that is a JAX limitation, not an oversight.
    def _double(operands):
        A_i, Q_i = operands
        return A_i @ A_i, symmetrize(A_i @ Q_i @ A_i.T + Q_i)

    for i in range(_MFD_MAX_SQUARINGS):
        A, Q = jax.lax.cond(i < n_squarings, _double, lambda operands: operands, (A, Q))

    return A, Q * diffusion_scale


def discretise_mfd_sequence(
    F: Float[Array, "d d"],
    Q_c: Float[Array, "d d"],
    dt: Float[Array, " N"],
) -> tuple[Float[Array, "N d d"], Float[Array, "N d d"]]:
    """Vectorised `gaussx.discretise_mfd` over a vector of time steps.

    Args:
        F: Continuous-time drift matrix, shape ``(d, d)``.
        Q_c: Continuous-time diffusion covariance, shape ``(d, d)``.
        dt: Time steps, shape ``(N,)``. All must be non-negative.

    Returns:
        Tuple ``(A_seq, Q_seq)``, both shape ``(N, d, d)``.
    """
    return jax.vmap(lambda step: discretise_mfd(F, Q_c, step))(dt)
