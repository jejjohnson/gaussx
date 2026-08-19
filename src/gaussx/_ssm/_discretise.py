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
