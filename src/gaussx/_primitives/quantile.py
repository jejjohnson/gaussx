"""Quantile inversion primitives built on optimistix."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optimistix as optx
from jaxtyping import Array, Float

from gaussx._primitives._chandrupatla import Chandrupatla


def mixture_quantile(
    cdf_fn: Callable[[Float[Array, ...]], Float[Array, ...]],
    q: Float[Array, ...],
    lower: Float[Array, ...],
    upper: Float[Array, ...],
    *,
    solver: optx.AbstractRootFinder | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_steps: int = 64,
    throw: bool = False,
) -> Float[Array, ...]:
    """Invert a scalar CDF at one or more quantile levels.

    Args:
        cdf_fn: Callable returning the CDF evaluated at ``x``.
        q: Quantile levels in ``[0, 1]``.
        lower: Lower bracket values with ``cdf_fn(lower) <= q``.
        upper: Upper bracket values with ``cdf_fn(upper) >= q``.
        solver: Optional optimistix root finder. Defaults to ``Chandrupatla``.
        rtol: Relative tolerance for the root finder.
        atol: Absolute tolerance for the root finder.
        max_steps: Maximum iterations for the solver.
        throw: If ``True``, raise on solver failure.

    Returns:
        Quantile values broadcast to ``q`` and bracket shapes.
    """
    solver = solver or Chandrupatla(rtol=rtol, atol=atol)  # ty: ignore[missing-argument]
    q = jnp.asarray(q)
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)

    lower_cdf = cdf_fn(lower)
    upper_cdf = cdf_fn(upper)
    bracket_ok = (lower_cdf <= q) & (upper_cdf >= q)
    message = "mixture_quantile expects q to be bracketed by cdf(lower) and cdf(upper)."
    lower = eqx.error_if(lower, jnp.any(~bracket_ok), msg=message)

    def solve_one(target: Float[Array, ...]) -> Float[Array, ...]:
        def fn(x: Float[Array, ...], _: None) -> Float[Array, ...]:
            return cdf_fn(x) - target

        midpoint = 0.5 * (lower + upper)
        solution = optx.root_find(
            fn,
            solver,
            midpoint,
            options={"lower": lower, "upper": upper},
            max_steps=max_steps,
            has_aux=False,
            throw=throw,
        )
        return solution.value

    if q.ndim == 0:
        return solve_one(q)

    quantile_first = jnp.moveaxis(q, -1, 0)
    mapped = jax.vmap(solve_one)(quantile_first)
    return jnp.moveaxis(mapped, 0, -1)


def mixture_quantile_gaussian_approx(
    means: Float[Array, "... ensemble"],
    stds: Float[Array, "... ensemble"],
    q: Float[Array, ...],
) -> Float[Array, ...]:
    """Gaussian approximation quantiles for an ensemble."""

    means = jnp.asarray(means)
    stds = jnp.asarray(stds)
    q = jnp.asarray(q)

    mean = jnp.mean(means, axis=-1)
    second_moment = jnp.mean(stds**2 + means**2, axis=-1)
    variance = second_moment - mean**2
    std = jnp.sqrt(jnp.maximum(variance, 0.0))
    return mean[..., None] + std[..., None] * jsp.special.ndtri(q)
