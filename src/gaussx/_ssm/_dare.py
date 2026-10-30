"""Steady state Σ_x,∞, G_∞ of the Kalman filter: the filtering DARE solver."""

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
    """Output of `dare`: the steady-state covariance Σ_x,∞ and gain G_∞.

    Attributes:
        P_inf: Steady-state filtered covariance Σ_x,∞.
        K_inf: Steady-state Kalman gain G_∞.
        converged: Whether the doubling iteration met ``tol`` within
            ``max_iter`` steps.
    """

    P_inf: Float[Array, "D D"]
    K_inf: Float[Array, "D M"]
    converged: Bool[Array, ""]


class _DoublingState(eqx.Module):
    """State of `_DoublingSolver`: the converged Σ_pred,∞ and its flag."""

    predicted_covariance: Float[Array, "D D"]
    converged: Bool[Array, ""]


class _DoublingSolver(optx.AbstractFixedPointSolver):
    """The structure-preserving doubling algorithm as an optimistix solver.

        W^(j) = I + Λ_obs^(j) Σ_pred^(j)
        F^(j+1) = F^(j) W^(j)⁻¹ F^(j)
        Λ_obs^(j+1) = Λ_obs^(j) + F^(j) W^(j)⁻¹ Λ_obs^(j) F^(j)ᵀ
        Σ_pred^(j+1) = Σ_pred^(j) + F^(j)ᵀ Σ_pred^(j) W^(j)⁻¹ F^(j)

    started from F^(0) = Fᵀ, Λ_obs^(0) = Hᵀ Σ_ε⁻¹ H and Σ_pred^(0) = Σ_η;
    j counts doublings, and Σ_pred^(j) converges to the steady-state
    predicted covariance Σ_pred,∞. These are the iterates A_k, G_k, H_k of
    Chu, Fan & Lin (2005) on the control-form DARE with A_c = Fᵀ and
    B_c = Hᵀ.

    The whole doubling iteration runs in ``init`` (it never evaluates the
    Riccati map ``fn``), and ``terminate`` reports the solve finished before
    the first step, so the iterate loop is empty and ``postprocess`` returns
    the doubling solution. Wrapping it this way lets
    `optimistix.fixed_point` attach its implicit adjoint to a solve whose
    forward pass is the doubling algorithm unchanged. ``args`` is
    ``(F, H, Σ_η, Σ_ε, Λ_obs^(0))``.

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
        transition, _, process_noise, _, obs_information = args
        identity = jnp.eye(transition.shape[0], dtype=transition.dtype)

        def _cond(state):
            *_, iteration, converged = state
            return (iteration < self.max_iter) & (~converged)

        def _body(state):
            doubled_transition, information, covariance, iteration, _ = state
            # Λ_obs^(j) Σ_pred^(j)
            information_covariance = einsum(
                information, covariance, "row mid, mid col -> row col"
            )
            # W^(j) = I + Λ_obs^(j) Σ_pred^(j)
            coupling = identity + information_covariance
            # W^(j)⁻¹ F^(j)
            coupling_solve_transition = jnp.linalg.solve(coupling, doubled_transition)
            # W^(j)⁻¹ Λ_obs^(j)
            coupling_solve_information = jnp.linalg.solve(coupling, information)
            # F^(j+1) = F^(j) W^(j)⁻¹ F^(j)
            next_transition = einsum(
                doubled_transition,
                coupling_solve_transition,
                "row mid, mid col -> row col",
            )
            # F^(j) W^(j)⁻¹ Λ_obs^(j)
            transition_information = einsum(
                doubled_transition,
                coupling_solve_information,
                "row mid, mid col -> row col",
            )
            # F^(j) W^(j)⁻¹ Λ_obs^(j) F^(j)ᵀ
            information_update = einsum(
                transition_information,
                doubled_transition,
                "row mid, col mid -> row col",
            )
            # Λ_obs^(j+1) = Λ_obs^(j) + F^(j) W^(j)⁻¹ Λ_obs^(j) F^(j)ᵀ
            next_information = symmetrize(information + information_update)
            # F^(j)ᵀ Σ_pred^(j)
            transition_covariance = einsum(
                doubled_transition, covariance, "mid row, mid col -> row col"
            )
            # F^(j)ᵀ Σ_pred^(j) W^(j)⁻¹ F^(j)
            covariance_update = einsum(
                transition_covariance,
                coupling_solve_transition,
                "row mid, mid col -> row col",
            )
            # Σ_pred^(j+1) = Σ_pred^(j) + F^(j)ᵀ Σ_pred^(j) W^(j)⁻¹ F^(j)
            next_covariance = symmetrize(covariance + covariance_update)
            # max |Σ_pred^(j+1) − Σ_pred^(j)| ≤ rtol max |Σ_pred^(j+1)|
            change = jnp.max(jnp.abs(next_covariance - covariance))
            converged = change <= self.rtol * jnp.max(jnp.abs(next_covariance))
            return (
                next_transition,
                next_information,
                next_covariance,
                iteration + 1,
                converged,
            )

        # F^(0) = Fᵀ, the control form
        transition_transpose = rearrange(transition, "row col -> col row")
        init_state = (
            transition_transpose,
            obs_information,
            process_noise,
            0,
            jnp.array(False),
        )
        *_, predicted_covariance, _, converged = jax.lax.while_loop(
            _cond, _body, init_state
        )
        return _DoublingState(
            predicted_covariance=predicted_covariance, converged=converged
        )

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
        return state.predicted_covariance, aux, {}


def _filter_update(
    predicted_covariance: Float[Array, "D D"],
    obs_operator: lx.AbstractLinearOperator,
    obs_noise_covariance: Float[Array, "M M"] | lx.AbstractLinearOperator,
    woodbury: bool,
    solver: AbstractSolverStrategy | None,
) -> tuple[Float[Array, "D M"], Float[Array, "D D"]]:
    """Kalman gain and filtered covariance from a predicted covariance.

        Σ_y = H Σ_pred Hᵀ + Σ_ε,   G = Σ_pred Hᵀ Σ_y⁻¹
        Σ_x = Σ_pred − G H Σ_pred

    Args:
        predicted_covariance: Predicted covariance Σ_pred.
        obs_operator: Observation operator H.
        obs_noise_covariance: Observation-error covariance Σ_ε.
        woodbury: When ``True``, Σ_y is a `gaussx.LowRankUpdate` on Σ_ε.
        solver: Solver strategy for Σ_y; ``None`` uses structural dispatch.

    Returns:
        Tuple ``(gain, filtered_covariance)`` holding G and Σ_x.
    """
    # Σ_y = H Σ_pred Hᵀ + Σ_ε
    obs_covariance = _innovation_covariance(
        obs_operator, predicted_covariance, obs_noise_covariance, woodbury=woodbury
    )
    # H Σ_pred
    obs_projection = _left_matmul(obs_operator, predicted_covariance)
    # Gᵀ = Σ_y⁻¹ H Σ_pred
    gain_transpose = solve_matrix(obs_covariance, obs_projection, solver=solver)
    # G = Σ_pred Hᵀ Σ_y⁻¹
    gain = rearrange(gain_transpose, "obs state -> state obs")
    # G H Σ_pred
    correction = einsum(gain, obs_projection, "state obs, obs col -> state col")
    # Σ_x = Σ_pred − G H Σ_pred
    filtered_covariance = symmetrize(predicted_covariance - correction)
    return gain, filtered_covariance


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
    """Steady-state Kalman filter covariance and gain (the filtering DARE).

        Σ_pred,∞ = F Σ_x,∞ Fᵀ + Σ_η
        Σ_y,∞ = H Σ_pred,∞ Hᵀ + Σ_ε,   G_∞ = Σ_pred,∞ Hᵀ Σ_y,∞⁻¹
        Σ_x,∞ = Σ_pred,∞ − G_∞ H Σ_pred,∞

    for the time-invariant model x_t = F x_t−1 + η_t, y_t = H x_t + ε_t
    with η_t ~ 𝒩(0, Σ_η) and ε_t ~ 𝒩(0, Σ_ε): the fixed point of the
    Kalman predict-update recursion. The predicted covariance solves the
    Riccati equation Σ_pred,∞ = Ric(Σ_pred,∞), with the Riccati map

        Ric(Σ_pred) = F (Σ_pred − Σ_pred Hᵀ (H Σ_pred Hᵀ + Σ_ε)⁻¹ H Σ_pred) Fᵀ
                    + Σ_η

    which is solved with the structure-preserving doubling algorithm (SDA;
    Chu, Fan & Lin, 2005). SDA converges quadratically: after j doublings
    the error shrinks like the 2^j-th power of the closed-loop contraction
    rate, so slowly mixing
    dynamics (F near the unit circle) need a dozen or so doublings where
    iterating the recursion itself needs thousands (gh-294). Each doubling
    costs a few D_x × D_x solves.

    Convergence is declared when the doubling iterate changes by at most
    ``tol`` relative to its largest entry; by quadratic convergence the
    returned solution is then accurate to about ``tol²``. Check
    ``converged`` when calling ``dare`` directly;
    `infinite_horizon_filter` and `infinite_horizon_smoother` raise on a
    non-converged result. A unique stabilising solution needs Σ_ε
    invertible and (F, H) stabilisable and detectable.

    Gradients take the implicit path. The doubling algorithm is wrapped as
    an `optimistix` fixed-point solver for Ric and run through
    `optimistix.fixed_point` with `optimistix.ImplicitAdjoint`. The forward
    value is exactly the doubling solution; derivatives with respect to
    θ = (F, H, Σ_η, Σ_ε) follow from the implicit function theorem on
    g(Σ_pred, θ) = Ric(Σ_pred; θ) − Σ_pred,

        ∂Σ_pred,∞/∂θ = −(∂g/∂Σ_pred)⁻¹ ∂g/∂θ,

    one linear solve at the fixed point instead of differentiating the
    iterations (Blondel et al., 2022). Without it ``jax.grad`` could not go
    through `dare` at all: reverse mode does not support the doubling
    ``while_loop``. The solve materialises the D_x² × D_x² Jacobian of g
    and factors it densely (optimistix's default), so the backward pass
    costs O(D_x⁶): cheap for the state sizes of SDE kernels, heavy beyond
    D_x ≈ 50. The derivative is only meaningful when ``converged`` is
    ``True``.

    Algorithm:
        1. Seed the doubling with Λ_obs^(0) = Hᵀ Σ_ε⁻¹ H.
        2. Solve Σ_pred,∞ = Ric(Σ_pred,∞) by SDA, inside
           `optimistix.fixed_point` with the implicit adjoint.
        3. Form Σ_y,∞, solve for G_∞ and update to Σ_x,∞.

    Args:
        A: Linear dynamics F, as an array or an operator.
        H: Observation operator H.
        Q: Model-error covariance Σ_η.
        R: Observation-error covariance Σ_ε.
        P_init: Deprecated and ignored: the doubling algorithm needs no
            initial guess.
        max_iter: Maximum number of doubling steps.
        tol: Convergence tolerance on the relative change of the iterate.
        solver: Solver strategy for Σ_ε and Σ_y,∞ (the seed and the final
            gain); ``None`` uses structural dispatch.
        woodbury_innovation: When ``True``, build Σ_y = H Σ_pred Hᵀ + Σ_ε
            as a `gaussx.LowRankUpdate`, so a structured Σ_ε gets Woodbury
            solves.

    Returns:
        A `DAREResult` with the steady-state filtered covariance Σ_x,∞
        (``P_inf``), the gain G_∞ (``K_inf``) and the convergence flag.

    Examples:
        A damped random walk observed through its first component:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> transition = 0.9 * jnp.eye(2)
        >>> obs_operator = jnp.array([[1.0, 0.0]])
        >>> process_noise = 0.1 * jnp.eye(2)
        >>> obs_noise = jnp.array([[0.5]])
        >>> result = gaussx.dare(transition, obs_operator, process_noise, obs_noise)
        >>> result.P_inf.shape, result.K_inf.shape, bool(result.converged)
        ((2, 2), (2, 1), True)

        G_∞ is the Kalman gain of the steady predicted covariance:

        >>> predicted = 0.81 * result.P_inf + process_noise
        >>> gain = gaussx.kalman_gain(
        ...     lx.MatrixLinearOperator(predicted),
        ...     lx.MatrixLinearOperator(obs_operator),
        ...     lx.MatrixLinearOperator(obs_noise),
        ... )
        >>> bool(jnp.allclose(gain, result.K_inf, atol=1e-4))
        True

        The Woodbury path with a diagonal Σ_ε gives the same fixed point:

        >>> woodbury = gaussx.dare(
        ...     transition,
        ...     obs_operator,
        ...     process_noise,
        ...     lx.DiagonalLinearOperator(jnp.array([0.5])),
        ...     woodbury_innovation=True,
        ... )
        >>> bool(jnp.allclose(woodbury.P_inf, result.P_inf, atol=1e-4))
        True

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
    transition_op = _as_operator(A)
    obs_operator_op = _as_operator(H)
    transition_dense = transition_op.as_matrix()
    obs_operator_dense = obs_operator_op.as_matrix()
    process_noise_dense = _materialise(Q)
    # Keep Σ_ε lazy when the Woodbury path consumes the operator directly:
    # it avoids an O(D_y²) allocation for large structured noise (e.g. a
    # ``DiagonalLinearOperator`` with large D_y).
    obs_noise_innovation = (
        R
        if woodbury_innovation and isinstance(R, lx.AbstractLinearOperator)
        else _materialise(R)
    )

    def _riccati(predicted_covariance, args):
        transition_d, obs_operator_o, process_noise_d, obs_noise_i, _ = args
        _, filtered_covariance = _filter_update(
            predicted_covariance,
            obs_operator_o,
            obs_noise_i,
            woodbury_innovation,
            solver,
        )
        # F Σ_x
        transition_covariance = einsum(
            transition_d, filtered_covariance, "next prev, prev col -> next col"
        )
        # F Σ_x Fᵀ
        propagated = einsum(
            transition_covariance, transition_d, "next col, other col -> next other"
        )
        # Ric(Σ_pred) = F Σ_x Fᵀ + Σ_η
        return symmetrize(propagated + process_noise_d)

    # Λ_obs^(0) = Hᵀ Σ_ε⁻¹ H seeds the doubling iteration. It rides along in
    # ``args`` for the solver; the Riccati map ignores it, so the implicit
    # adjoint gives it a zero cotangent and H, Σ_ε are differentiated
    # through the map.
    dtype = jnp.result_type(transition_dense, obs_operator_dense, process_noise_dense)
    # Σ_ε⁻¹ H
    whitened_obs_operator = solve_matrix(
        _as_operator(R), obs_operator_dense, solver=solver
    )
    # Λ_obs^(0) = Hᵀ Σ_ε⁻¹ H
    obs_information = symmetrize(
        einsum(
            obs_operator_dense,
            whitened_obs_operator,
            "obs row, obs col -> row col",
        )
    )
    process_noise_dense = process_noise_dense.astype(dtype)
    solution = optx.fixed_point(
        _riccati,
        _DoublingSolver(rtol=tol, max_iter=max_iter),
        jnp.zeros_like(process_noise_dense),
        args=(
            transition_dense.astype(dtype),
            obs_operator_op,
            process_noise_dense,
            obs_noise_innovation,
            obs_information,
        ),
        max_steps=None,
        adjoint=optx.ImplicitAdjoint(),
        throw=False,
    )
    steady_predicted_covariance = solution.value
    converged = solution.state.converged

    # G_∞ and Σ_x,∞ from the steady-state prediction Σ_pred,∞
    steady_gain, steady_covariance = _filter_update(
        steady_predicted_covariance,
        obs_operator_op,
        obs_noise_innovation,
        woodbury_innovation,
        solver,
    )

    return DAREResult(P_inf=steady_covariance, K_inf=steady_gain, converged=converged)
