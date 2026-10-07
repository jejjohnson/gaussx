"""Discrete Algebraic Riccati Equation (DARE) solver."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from jaxtyping import Array, Bool, Float, PyTree, Scalar

from gaussx._deprecation import warn_deprecated
from gaussx._einx import einsum, rearrange
from gaussx._linalg._linalg import solve_matrix
from gaussx._linalg._symmetrize import symmetrize
from gaussx._ssm._utils import (
    _as_operator,
    _innovation_covariance,
    _left_matmul,
    _materialise,
)
from gaussx._strategies._base import AbstractSolverStrategy


class DAREResult(eqx.Module):
    """Result of DARE solver.

    Attributes:
        P_inf: Steady-state covariance, shape ``(D, D)``.
        K_inf: Steady-state Kalman gain, shape ``(D, M)``.
        converged: Scalar boolean indicating convergence.
    """

    P_inf: Float[Array, "D D"]
    K_inf: Float[Array, "D M"]
    converged: Bool[Array, ""]


class _DoublingState(eqx.Module):
    P_pred: Float[Array, "D D"]
    converged: Bool[Array, ""]


class _DoublingSolver(optx.AbstractFixedPointSolver):
    """The structure-preserving doubling algorithm as an optimistix solver.

    The whole doubling iteration runs in ``init`` (it never evaluates the
    Riccati map ``fn``), and ``terminate`` reports the solve finished
    before the first step, so the iterate loop is empty and ``postprocess``
    returns the doubling solution. Wrapping it this way lets
    `optimistix.fixed_point` attach its implicit adjoint to a solve whose
    forward pass is the doubling algorithm unchanged. ``args`` is
    ``(A, H, Q, R, G)`` with ``G = Hᵀ R⁻¹ H``.

    ``rtol`` is the doubling tolerance on the iterate's relative change;
    ``atol`` and ``norm`` complete optimistix's solver interface and are
    unused.
    """

    rtol: float
    max_iter: int = 100
    atol: float = 0.0
    norm: Callable[[PyTree], Scalar] = optx.max_norm

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        del fn, y, options, f_struct, aux_struct, tags
        A, _, Q, _, G = args
        eye = jnp.eye(A.shape[0], dtype=A.dtype)

        def _cond(state):
            *_, i, converged = state
            return (i < self.max_iter) & (~converged)

        def _body(state):
            A_k, G_k, H_k, i, _ = state
            W = eye + G_k @ H_k
            Winv_A = jnp.linalg.solve(W, A_k)
            Winv_G = jnp.linalg.solve(W, G_k)
            A_next = A_k @ Winv_A
            G_next = symmetrize(G_k + A_k @ Winv_G @ rearrange(A_k, "i j -> j i"))
            H_next = symmetrize(H_k + rearrange(A_k, "i j -> j i") @ H_k @ Winv_A)
            change = jnp.max(jnp.abs(H_next - H_k))
            converged = change <= self.rtol * jnp.max(jnp.abs(H_next))
            return A_next, G_next, H_next, i + 1, converged

        # SDA on the control-form DARE with A_c = Aᵀ, B_c = Hᵀ: the iterate
        # ``H_k`` converges to the predicted steady-state covariance P⁻.
        init_state = (rearrange(A, "i j -> j i"), G, Q, 0, jnp.array(False))
        *_, P_pred, _, converged = jax.lax.while_loop(_cond, _body, init_state)
        return _DoublingState(P_pred=P_pred, converged=converged)

    def step(self, fn, y, args, options, state, tags):
        del options, tags
        # Never reached (``terminate`` is immediately true); a Picard step
        # keeps the solver well defined if called directly.
        new_y, aux = fn(y, args)
        return new_y, state, aux

    def terminate(self, fn, y, args, options, state, tags):
        del fn, y, args, options, state, tags
        return jnp.array(True), optx.RESULTS.successful

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        del fn, y, args, options, tags, result
        return state.P_pred, aux, {}


def _filter_update(P_pred, H_o, R_i, woodbury, solver):
    """The Kalman gain and filtered covariance from a predicted covariance."""
    S_op = _innovation_covariance(H_o, P_pred, R_i, woodbury=woodbury)
    HP_pred = _left_matmul(H_o, P_pred)
    K = rearrange(solve_matrix(S_op, HP_pred, solver=solver), "m d -> d m")
    return K, symmetrize(P_pred - K @ HP_pred)


def dare(
    A: Float[Array, "D D"] | lx.AbstractLinearOperator,
    H: Float[Array, "M D"] | lx.AbstractLinearOperator,
    Q: Float[Array, "D D"] | lx.AbstractLinearOperator,
    R: Float[Array, "M M"] | lx.AbstractLinearOperator,
    *,
    P_init: Float[Array, "D D"] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    solver: AbstractSolverStrategy | None = None,
    woodbury_innovation: bool = False,
) -> DAREResult:
    r"""Steady-state Kalman filter covariance and gain (the filtering DARE).

    Solves for the fixed point of the Kalman predict-update recursion,

        Predict:  P⁻ = A P Aᵀ + Q
        Update:   S = H P⁻ Hᵀ + R
                  K = P⁻ Hᵀ S⁻¹
                  P = (I - KH) P⁻

    with the structure-preserving doubling algorithm (SDA; Chu, Fan & Lin,
    2005) applied to the predicted-covariance Riccati equation
    ``P⁻ = A P⁻ Aᵀ − A P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹ H P⁻ Aᵀ + Q``. SDA converges
    quadratically -- after ``k`` doublings the error is
    ``O(rho^(2^k))`` for contraction rate ``rho`` -- so slowly mixing
    dynamics (``A`` near the unit circle) need a dozen or so doublings
    where iterating the recursion itself needs thousands (gh-294). Each
    doubling costs a few ``D × D`` solves.

    Convergence is declared when the doubling iterate changes by at most
    ``tol`` relative to its largest entry; by quadratic convergence the
    returned solution is then accurate to about ``tol²``. Check
    ``converged`` when calling ``dare`` directly;
    `infinite_horizon_filter` and `infinite_horizon_smoother` raise on a
    non-converged result.

    Requires ``R`` invertible and the usual stabilisability /
    detectability conditions for a unique stabilising solution.

    **Gradients** take the implicit path. The doubling algorithm is wrapped
    as an `optimistix` fixed-point solver for the Riccati map

    $$
    \mathcal{R}(P^-) = A\big(P^- - P^- H^\top (H P^- H^\top + R)^{-1}
        H P^-\big)A^\top + Q,
    $$

    and run through `optimistix.fixed_point` with
    `optimistix.ImplicitAdjoint`. The forward value is exactly the doubling
    solution; derivatives with respect to $\theta = (A, H, Q, R)$ follow from
    the implicit function theorem on $g(P^-, \theta) =
    \mathcal{R}(P^-; \theta) - P^-$,

    $$
    \frac{\partial P^-_\star}{\partial \theta}
        = -\big(\partial_{P} g\big)^{-1}\, \partial_\theta g,
    $$

    one linear solve at the fixed point instead of differentiating the
    iterations (Blondel et al., 2022). Without it ``jax.grad`` could not go
    through `dare` at all: reverse mode does not support the doubling
    ``while_loop``. The solve materialises the $D^2 \times D^2$ Jacobian of
    $g$ and factors it densely (optimistix's default), so the backward pass
    costs $O(D^6)$: cheap for the state dimensions of SDE kernels, heavy
    beyond $D \approx 50$. The derivative is only meaningful when
    ``converged`` is ``True``.

    Pseudocode:

        P⁻ = fixed_point(Riccati map, solver=SDA)        (implicit adjoint)
        S = H P⁻ Hᵀ + R;  K = P⁻ Hᵀ S⁻¹;  P = P⁻ − K H P⁻

    Args:
        A: Transition matrix or operator, shape ``(D, D)``.
        H: Observation matrix or operator, shape ``(M, D)``.
        Q: Process noise covariance or operator, shape ``(D, D)``.
        R: Observation noise covariance or operator, shape ``(M, M)``.
        P_init: Deprecated and ignored: the doubling algorithm needs no
            initial guess.
        max_iter: Maximum number of doubling steps.
        tol: Convergence tolerance on the relative change of the iterate.
        solver: Optional solver strategy for structured linear algebra
            (``R`` solves and the final gain). When ``None``, falls back to
            structural dispatch.
        woodbury_innovation: When ``True``, build ``S = H P⁻ Hᵀ + R``
            as a `gaussx.LowRankUpdate` so structured ``R`` uses
            Woodbury solves for the final gain.

    Returns:
        A `DAREResult` containing the steady-state *filtered* covariance,
        Kalman gain, and convergence flag.

    References:
        Chu, E. K.-W., Fan, H.-Y. & Lin, W.-W. (2005). A structure-preserving
        doubling algorithm for continuous-time algebraic Riccati equations.
        *Linear Algebra and its Applications* 396, 55-80.

        Blondel, M., Berthet, Q., Cuturi, M., Frostig, R., Hoyer, S.,
        Llinares-López, F., Pedregosa, F. & Vert, J.-P. (2022). Efficient and
        modular implicit differentiation. *NeurIPS 35*.
    """
    if P_init is not None:
        warn_deprecated(
            "dare(P_init=...) is deprecated and ignored: the doubling "
            "algorithm needs no initial guess. It will be removed in gaussx 0.7.0."
        )
    A_op = _as_operator(A)
    H_op = _as_operator(H)
    A_dense = A_op.as_matrix()
    H_dense = H_op.as_matrix()
    Q_dense = _materialise(Q)
    # Keep ``R`` lazy when the Woodbury innovation path will consume the
    # operator directly — avoids an O(M²) allocation for large structured
    # noise (e.g. ``DiagonalLinearOperator`` with large ``M``).
    R_for_innovation = (
        R
        if woodbury_innovation and isinstance(R, lx.AbstractLinearOperator)
        else _materialise(R)
    )

    def _riccati(P_pred, args):
        A_d, H_o, Q_d, R_i, _ = args
        _, P_filt = _filter_update(P_pred, H_o, R_i, woodbury_innovation, solver)
        return symmetrize(A_d @ P_filt @ rearrange(A_d, "i j -> j i") + Q_d)

    # G = Hᵀ R⁻¹ H seeds the doubling iteration. It rides along in ``args``
    # for the solver; the Riccati map ignores it, so the implicit adjoint
    # gives it a zero cotangent and H, R are differentiated through the map.
    dtype = jnp.result_type(A_dense, H_dense, Q_dense)
    Rinv_H = solve_matrix(_as_operator(R), H_dense, solver=solver)
    G_0 = symmetrize(einsum(H_dense, Rinv_H, "m i, m j -> i j"))
    Q_dense = Q_dense.astype(dtype)
    sol = optx.fixed_point(
        _riccati,
        _DoublingSolver(rtol=tol, max_iter=max_iter),
        jnp.zeros_like(Q_dense),
        args=(A_dense.astype(dtype), H_op, Q_dense, R_for_innovation, G_0),
        max_steps=None,
        adjoint=optx.ImplicitAdjoint(),
        throw=False,
    )
    P_pred, converged = sol.value, sol.state.converged

    # Filtered covariance and gain from the steady-state prediction.
    K_inf, P_inf = _filter_update(
        P_pred, H_op, R_for_innovation, woodbury_innovation, solver
    )

    return DAREResult(P_inf=P_inf, K_inf=K_inf, converged=converged)
