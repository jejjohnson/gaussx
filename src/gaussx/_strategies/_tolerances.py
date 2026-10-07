"""Dtype-aware default tolerances for the iterative strategies (gh-327).

The iterative strategies leave their tolerances as ``None`` by default and
resolve them here, at solve time, from the operator's dtype. Float64 keeps
the historical constants. In float32, a CG relative residual of ``1e-5`` is
out of reach once the condition number passes about ``1e3``. CG then runs out
of steps, or breaks down, and returns an iterate worse than zero. On a
κ = 1e4 system float32 CG converges at ``1e-3`` (432 steps) but not at
``sqrt(eps) ≈ 3.5e-4``, so ``1e-3`` is the float32 default.

Absolute tolerances are not relaxed to a constant: that would declare the
zero iterate converged for any right-hand side smaller than it (e.g.
``‖b‖ = 1e-4`` with ``atol = 1e-3``). But lineax's CG checks every entry,
``|r_i| <= atol + rtol |b_i|``, so an entry with ``b_i ≈ 0`` must reach
``atol`` itself, and a fixed ``1e-5`` sits at float32's rounding floor for a
right-hand side of order one: whether CG converges then depends on the CPU's
instruction set (gh-639). `resolve_atol` therefore scales a low-precision
default with the right-hand side, ``max(atol₆₄, sqrt(eps) ‖b‖_∞)``. It stays
``atol₆₄`` for a small ``b``, and it can never accept the zero iterate,
because the largest entry of ``b`` exceeds ``sqrt(eps) ‖b‖_∞``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jax.typing import DTypeLike


# Default tolerance by the component width of the dtype, in bytes. Float64
# uses each strategy's own historical default.
_LOW_PRECISION_TOLERANCE = {4: 1e-3, 2: 1e-2}


def operator_dtype(
    operator: lx.AbstractLinearOperator, *arrays: jax.Array
) -> DTypeLike:
    """The lowest-precision dtype a solve with *operator* runs in.

    Looks at the input and the output structure (they may differ, e.g. a
    rectangular mixed-precision operator) and any *arrays* (the right-hand
    side), and returns the narrowest float among them, since the default
    tolerance must be reachable in all of them.
    """
    leaves = [
        *jax.tree.leaves(operator.in_structure()),
        *jax.tree.leaves(operator.out_structure()),
        *arrays,
    ]
    dtypes = [
        jnp.dtype(leaf.dtype)
        for leaf in leaves
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    if not dtypes:
        return jnp.result_type(float)
    return min(dtypes, key=lambda d: jnp.finfo(d).bits)


def resolve_tolerance(
    value: float | None, dtype: DTypeLike, float64_default: float
) -> float:
    """*value* if set, else the default tolerance for *dtype*.

    Args:
        value: A tolerance the user set, or ``None`` for the default.
        dtype: The dtype the solve runs in.
        float64_default: The strategy's float64 default. Lower precisions
            use ``1e-3`` (float32) or ``1e-2`` (16-bit floats) when that is
            looser.

    Returns:
        The tolerance to use.
    """
    if value is not None:
        return value
    width = jnp.finfo(dtype).bits // 8
    if width >= 8:
        return float64_default
    return max(float64_default, _LOW_PRECISION_TOLERANCE.get(width, 1e-2))


def resolve_atol(
    value: float | None,
    dtype: DTypeLike,
    float64_default: float,
    vector: jax.Array,
) -> float | jax.Array:
    """*value* if set, else the default absolute tolerance for *dtype*.

    Args:
        value: An absolute tolerance the user set, or ``None`` for the default.
        dtype: The dtype the solve runs in.
        float64_default: The strategy's float64 default, also the floor in
            lower precisions.
        vector: The right-hand side ``b``. Below float64 the default is
            ``max(float64_default, sqrt(eps) * max|b_i|)`` (gh-639).

    Returns:
        The absolute tolerance to use: a float, or a scalar array when it
        depends on *vector*.
    """
    if value is not None:
        return value
    if jnp.finfo(dtype).bits >= 64:
        return float64_default
    scale = max(jnp.max(jnp.abs(leaf)) for leaf in jax.tree.leaves(vector))
    floor = jnp.sqrt(jnp.finfo(dtype).eps).astype(dtype)
    return jnp.maximum(jnp.asarray(float64_default, dtype), floor * scale.astype(dtype))
