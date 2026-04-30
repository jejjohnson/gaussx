"""Chandrupatla root finder implemented for optimistix."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array


Scalar = Array


class _ChandrupatlaState(eqx.Module):
    lower: Scalar
    upper: Scalar
    c: Scalar
    f_lower: Scalar
    f_upper: Scalar
    f_c: Scalar
    t: Scalar
    error: Scalar
    iterations: Array
    at_fixed_point: Array


class Chandrupatla(optx.AbstractRootFinder[Scalar, Scalar, Any, _ChandrupatlaState]):
    """Hybrid inverse-quadratic / bisection solver.

    This follows the acceptance test from Chandrupatla (1997), falling back to
    bisection when the inverse-quadratic proposal is unsafe.
    """

    rtol: float
    atol: float
    norm: ClassVar[Callable[[Any], Scalar]] = optx.rms_norm

    def init(
        self,
        fn: Callable[[Scalar, Any], tuple[Scalar, Any]],
        y: Scalar,
        args: Any,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: Any,
        tags: frozenset[object],
    ) -> _ChandrupatlaState:
        del aux_struct, tags
        lower = jnp.asarray(options["lower"], f_struct.dtype)
        upper = jnp.asarray(options["upper"], f_struct.dtype)
        y = jnp.asarray(y, f_struct.dtype)

        lower, upper, y = jnp.broadcast_arrays(lower, upper, y)

        if not isinstance(f_struct, jax.ShapeDtypeStruct) or f_struct.shape != ():
            raise ValueError(
                "Chandrupatla can only be used to find roots of scalar functions."
            )

        f_lower, _ = fn(lower, args)
        f_upper, _ = fn(upper, args)

        same_sign = ((f_lower > 0) & (f_upper > 0)) | ((f_lower < 0) & (f_upper < 0))
        lower = eqx.error_if(
            lower,
            jnp.any(same_sign),
            msg="The root is not contained in [lower, upper].",
        )

        return _ChandrupatlaState(
            lower=lower,
            upper=upper,
            c=lower,
            f_lower=f_lower,
            f_upper=f_upper,
            f_c=f_lower,
            t=jnp.full_like(lower, 0.5),
            error=jnp.full_like(f_lower, jnp.inf),
            iterations=jnp.zeros_like(f_lower, dtype=jnp.int32),
            at_fixed_point=jnp.zeros_like(f_lower, dtype=bool),
        )

    def step(
        self,
        fn: Callable[[Scalar, Any], tuple[Scalar, Any]],
        y: Scalar,
        args: Any,
        options: dict[str, Any],
        state: _ChandrupatlaState,
        tags: frozenset[object],
    ) -> tuple[Scalar, _ChandrupatlaState, Any]:
        del options, tags, y
        x_new = (1.0 - state.t) * state.lower + state.t * state.upper
        f_new, aux = fn(x_new, args)

        same_sign = jnp.sign(f_new) == jnp.sign(state.f_lower)
        new_lower = x_new
        new_upper = jnp.where(same_sign, state.upper, state.lower)
        new_c = jnp.where(same_sign, state.lower, state.upper)
        new_f_lower = f_new
        new_f_upper = jnp.where(same_sign, state.f_upper, state.f_lower)
        new_f_c = jnp.where(same_sign, state.f_lower, state.f_upper)

        denom_xi = jnp.where(
            new_c == new_upper,
            jnp.finfo(new_lower.dtype).tiny,
            new_c - new_upper,
        )
        xi = jnp.clip((new_lower - new_upper) / denom_xi, 0.0, 1.0)

        denom_phi = jnp.where(
            new_f_c == new_f_upper,
            jnp.finfo(new_f_lower.dtype).tiny,
            new_f_c - new_f_upper,
        )
        phi = (new_f_lower - new_f_upper) / denom_phi

        denom_candidate = jnp.where(
            new_upper == new_lower,
            jnp.finfo(new_lower.dtype).tiny,
            new_upper - new_lower,
        )
        candidate = new_f_lower / (new_f_upper - new_f_lower) * new_f_c / (
            new_f_upper - new_f_c
        ) + (new_c - new_lower) / denom_candidate * new_f_lower / (
            new_f_c - new_f_lower
        ) * new_f_upper / (new_f_c - new_f_upper)

        xi_sqrt = jnp.sqrt(xi)
        accept_iqi = (1.0 - jnp.sqrt(1.0 - xi) < phi) & (xi_sqrt > phi)
        t_proposed = jnp.where(accept_iqi, candidate, 0.5)

        width = jnp.abs(new_upper - new_c)
        scale = self.atol + self.rtol * jnp.abs(0.5 * (new_upper + new_c))
        interval_tolerance = scale / jnp.maximum(width, jnp.finfo(new_upper.dtype).tiny)
        t_next = jnp.clip(t_proposed, interval_tolerance, 1.0 - interval_tolerance)

        best_error = jnp.where(
            jnp.abs(new_f_lower) < jnp.abs(new_f_upper), new_f_lower, new_f_upper
        )
        at_fixed_point = (x_new == state.lower) & (state.t == 0.5)

        new_state = _ChandrupatlaState(
            lower=new_lower,
            upper=new_upper,
            c=new_c,
            f_lower=new_f_lower,
            f_upper=new_f_upper,
            f_c=new_f_c,
            t=t_next,
            error=best_error,
            iterations=state.iterations + 1,
            at_fixed_point=at_fixed_point,
        )
        next_y = (1.0 - t_next) * new_lower + t_next * new_upper
        return next_y, new_state, aux

    def terminate(
        self,
        fn: Callable[[Scalar, Any], tuple[Scalar, Any]],
        y: Scalar,
        args: Any,
        options: dict[str, Any],
        state: _ChandrupatlaState,
        tags: frozenset[object],
    ) -> tuple[Array, optx.RESULTS]:
        del fn, y, args, options, tags
        width = jnp.abs(state.upper - state.c)
        scale = self.atol + self.rtol * jnp.abs(0.5 * (state.upper + state.c))
        interval_small = width < scale
        residual_small = jnp.abs(state.error) < self.atol
        return (
            interval_small | residual_small | state.at_fixed_point,
            optx.RESULTS.successful,
        )

    def postprocess(
        self,
        fn: Callable[[Scalar, Any], tuple[Scalar, Any]],
        y: Scalar,
        aux: Any,
        args: Any,
        options: dict[str, Any],
        state: _ChandrupatlaState,
        tags: frozenset[object],
        result: optx.RESULTS,
    ) -> tuple[Scalar, Any, dict[str, Any]]:
        del fn, args, options, state, tags, result
        return y, aux, {}
