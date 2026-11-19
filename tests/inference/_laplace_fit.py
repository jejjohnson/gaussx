"""A caller-owned Laplace loop on gaussx's public Newton step, for the tests.

gaussx ships one Newton step (`gaussx.laplace_model`, `gaussx.LaplaceModel.newton`)
and the result at a mode (`gaussx.laplace_result`); the loop that iterates them,
``argminax.solve.laplace_mode``, lives downstream of gaussx (gh-816). The tests
need a converged, implicitly differentiable mode to check those parts against
dense references, so `laplace_fit` is that loop: the damped Newton iteration to
‖Δx‖_∞ ≤ tol (1 + ‖x‖_∞) with gradients stopped, then implicit tangents
∂_θ x̂ = Λ_x|y⁻¹ ∂_θ(−∇_x 𝒥) through `jax.lax.custom_root`. It is the body of the
former ``gaussx.laplace_mode``, removed after gaussx 0.7.0.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

import gaussx as gx
from gaussx._inference._laplace import _factor


def _stop_gradient(tree: Any) -> Any:
    """Stop gradients through the inexact arrays of a pytree.

    Args:
        tree: Any pytree.

    Returns:
        The same pytree, with its inexact array leaves wrapped.
    """
    dynamic, static = eqx.partition(tree, eqx.is_inexact_array)
    return eqx.combine(jax.lax.stop_gradient(dynamic), static)


@eqx.filter_jit
def _newton_loop(
    model: gx.LaplaceModel,
    init_state: Float[Array, " N"],
    max_iter: int,
    tol: float,
    damping: float,
) -> tuple[Array, Array, Array]:
    """Damped Newton iteration to ‖Δx‖_∞ ≤ tol (1 + ‖x‖_∞).

        x⁽ʲ⁺¹⁾ = x⁽ʲ⁾ + α (x̃ − x⁽ʲ⁾)

    with x̃ the Newton target of `gaussx.LaplaceModel.newton`.

    Algorithm:
        1. Floor ``tol`` at 64 machine epsilons of the dtype.
        2. Step until the sup-norm of the step passes the test, or ``max_iter``.

    Args:
        model: The model, with gradients stopped by the caller.
        init_state: Starting point x⁽⁰⁾.
        max_iter: Maximum Newton steps.
        tol: Relative tolerance on the step's sup-norm.
        damping: Step size α.

    Returns:
        Tuple ``(state, n_iter, converged)``.
    """
    eps = float(jnp.finfo(init_state.dtype).eps)
    tol = max(tol, 64.0 * eps)

    def threshold(state: Array) -> Array:
        # jnp, not einx: einx cannot resolve an empty state (D_x = 0).
        # tol (1 + ‖x‖_∞)
        return tol * (1.0 + jnp.max(jnp.abs(state)))

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        state, iteration, step_norm = carry
        return (iteration < max_iter) & (step_norm > threshold(state))

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        state, iteration, _ = carry
        # x̃ − x⁽ʲ⁾
        step = model.newton(state) - state
        # x⁽ʲ⁺¹⁾ = x⁽ʲ⁾ + α (x̃ − x⁽ʲ⁾)
        next_state = state + damping * step
        # jnp, not einx: einx cannot resolve an empty state (D_x = 0).
        # ‖Δx‖_∞
        step_norm = jnp.max(jnp.abs(next_state - state))
        return next_state, iteration + 1, step_norm

    init = (
        init_state,
        jnp.asarray(0),
        jnp.asarray(jnp.inf, dtype=init_state.dtype),
    )
    state, n_iter, step_norm = jax.lax.while_loop(cond, body, init)
    return state, n_iter, step_norm <= threshold(state)


def _implicit_mode(
    model: gx.LaplaceModel, mode_estimate: Float[Array, " N"]
) -> Float[Array, " N"]:
    """Attach implicit-function tangents to a converged mode.

        ∂_θ x̂ = Λ_x|y⁻¹ ∂_θ(−∇_x 𝒥)

    The root function is −∇_x 𝒥(x), or with the hard constraint
    −P ∇_x 𝒥(P x) − Π_V x (P = I − Π_V), whose roots are the constrained modes
    and whose Jacobian −(P Λ_x|y P + Π_V) is symmetric.

    Algorithm:
        1. Define the root function and factorise Λ_x|y once at the mode, with
           gradients stopped.
        2. Solve tangents with that factor, kriged for the constraint, wrapped in
           `jax.lax.custom_linear_solve` so that higher derivatives see Λ_x|y's
           dependence on θ through the matvec.

    Args:
        model: The model, carrying θ.
        mode_estimate: The converged mode from `_newton_loop`.

    Returns:
        The same point, now implicitly differentiable in θ.
    """
    constrained = model.null_space is not None

    def root(state: Array) -> Array:
        if not constrained:
            return model.grad(state)
        # P x = x − Π_V x
        inner = state - model.null_component(state)
        # −∇_x 𝒥(P x)
        ascent = model.grad(inner)
        # −P ∇_x 𝒥(P x) − Π_V x
        return ascent - model.null_component(ascent) - model.null_component(state)

    fixed = _stop_gradient(model)
    frozen_mode = jax.lax.stop_gradient(mode_estimate)
    factor = _factor(fixed.hessian(frozen_mode)[1])

    def neg_inverse(_: Any, b: Array) -> Array:
        # Π_V b
        null = fixed.null_component(b)
        # krige(Λ_x|y⁻¹ (b − Π_V b))
        inner = fixed.krige(factor, factor.solve(b - null))[0]
        return -(inner + null)

    def tangent_solve(matvec: Any, b: Array) -> Array:
        return jax.lax.custom_linear_solve(matvec, b, neg_inverse, symmetric=True)

    return jax.lax.custom_root(
        root, jax.lax.stop_gradient(mode_estimate), lambda _, x0: x0, tangent_solve
    )


def laplace_fit(
    prior: gx.GaussianMRF | gx.IntrinsicGMRF,
    likelihood: Any,
    *,
    projector: Any = None,
    offset: Any = None,
    init: Any = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    damping: float = 1.0,
) -> gx.LaplaceResult:
    """Mode, Hessian and log-marginal of the Laplace approximation.

    Args:
        prior: The latent field, a `GaussianMRF` or `IntrinsicGMRF`.
        likelihood: A site-factorising likelihood holding y.
        projector: The projector A; ``None`` is the identity.
        offset: The offset o, a scalar or of length M.
        init: Starting point x⁽⁰⁾; zeros by default.
        max_iter: Maximum Newton steps.
        tol: Relative tolerance on the step's sup-norm.
        damping: Step size α in (0, 1].

    Returns:
        The `LaplaceResult`.
    """
    model = gx.laplace_model(prior, likelihood, projector=projector, offset=offset)
    state_size = prior.event_shape[0]
    dtype = prior.loc.dtype
    if init is None:
        init_state = jnp.zeros(state_size, dtype=dtype)
    else:
        init_state = jnp.asarray(init, dtype=dtype)
    # x⁽⁰⁾ ← x⁽⁰⁾ − Π_V x⁽⁰⁾
    init_state = init_state - model.null_component(init_state)
    frozen_model = _stop_gradient(model)
    frozen_init = jax.lax.stop_gradient(init_state)
    mode_estimate, n_iter, converged = _newton_loop(
        frozen_model, frozen_init, max_iter, tol, damping
    )
    mode = _implicit_mode(model, mode_estimate)
    return gx.laplace_result(model, prior, mode, n_iter, converged)
