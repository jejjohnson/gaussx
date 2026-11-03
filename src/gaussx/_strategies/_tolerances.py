"""Dtype-aware default tolerances for the iterative strategies (gh-327).

An iterative solve of A x = b stops at the first iterate x_j whose residual
r_j = b − A x_j is within tolerance. lineax's CG checks every entry,

    |r_j,n| ≤ tol_abs + tol_rel |b_n|,   n = 1, …, N,

with the absolute and relative tolerances tol_abs (``atol``) and tol_rel
(``rtol``). The iterative strategies leave both as ``None`` by default and
resolve them here, at solve time, from the operator's dtype. Float64 keeps
the historical constants. In float32, tol_rel = 1e-5 is out of reach once
the condition number κ passes about 1e3: CG then runs out of steps, or
breaks down, and returns an iterate worse than zero. On a κ = 1e4 system
float32 CG converges at 1e-3 (432 steps) but not at √eps ≈ 3.5e-4, with eps the
machine epsilon, so 1e-3 is the float32 default.

Absolute tolerances are not relaxed to a constant: that would declare the
zero iterate converged for any right-hand side smaller than it (e.g.
‖b‖ = 1e-4 with tol_abs = 1e-3). But an entry with b_n ≈ 0 must reach
tol_abs itself, and a fixed 1e-5 sits at float32's rounding floor for a
right-hand side of order one: whether CG converges then depends on the CPU's
instruction set (gh-639). So below float64, with the default ``atol``, the
CG strategies solve the rescaled system A y = b / s with s = maxₙ |b_n| and
tol_abs = √eps, and return x = s y (`rhs_scaling`). The effective absolute
tolerance is √eps maxₙ |b_n|: relative to b, so the zero iterate is never
accepted for a nonzero b however small. The solver's own tolerances stay
constants, so no tangent reaches the lineax solver and its implicit
derivative solves use the same fixed tol_abs. s carries no gradient (it only
changes the iteration count); a zero or non-finite b uses s = 1, the
unscaled solve.
"""

from __future__ import annotations

import math

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
    """The lowest-precision dtype a solve with ``operator`` runs in.

    Looks at the input and the output structure (they may differ, e.g. a
    rectangular mixed-precision operator) and at ``arrays`` (the right-hand
    side), and returns the narrowest float among them, since the default
    tolerance must be reachable in all of them.

    Args:
        operator: Linear operator A.
        *arrays: Further arrays the solve touches, such as b.

    Returns:
        The narrowest inexact dtype, or the default float dtype when there
        is none.

    Examples:
        A float32 right-hand side narrows a float32 operator's solve:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from gaussx._strategies._tolerances import operator_dtype
        >>> A = lx.DiagonalLinearOperator(jnp.ones(2, dtype=jnp.float32))
        >>> str(operator_dtype(A, jnp.ones(2, dtype=jnp.float16)))
        'float16'
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
    """``value`` if set, else the default tolerance for ``dtype``.

        tol = value                                  if set
        tol = tol_64                                 in float64
        tol = max(tol_64, 1e-3)                      in float32
        tol = max(tol_64, 1e-2)                      in narrower floats

    Args:
        value: A tolerance the user set, or ``None`` for the default.
        dtype: The dtype the solve runs in.
        float64_default: The strategy's float64 default tol_64. Lower
            precisions use 1e-3 (float32) or 1e-2 (narrower floats) when
            that is looser.

    Returns:
        The tolerance to use.

    Examples:
        >>> import jax.numpy as jnp
        >>> from gaussx._strategies._tolerances import resolve_tolerance
        >>> resolve_tolerance(None, jnp.float32, 1e-5)
        0.001
        >>> resolve_tolerance(None, jnp.float16, 1e-5)
        0.01
        >>> resolve_tolerance(1e-7, jnp.float32, 1e-5)
        1e-07
    """
    if value is not None:
        return value
    # component width of the dtype, in bytes
    width = jnp.finfo(dtype).bits // 8
    if width >= 8:
        return float64_default
    low_precision_tolerance = _LOW_PRECISION_TOLERANCE.get(width, 1e-2)
    # tol = max(tol_64, tol_low)
    return max(float64_default, low_precision_tolerance)


def rhs_scaling(
    value: float | None,
    dtype: DTypeLike,
    float64_default: float,
    vector: jax.Array,
) -> tuple[float, jax.Array | None]:
    """The absolute tolerance, and the right-hand-side scale to solve with.

        s = maxₙ |b_n|,   A y = b / s,   x = s y,   tol_abs = √eps

    Args:
        value: An absolute tolerance tol_abs the user set, or ``None`` for
            the default.
        dtype: The dtype the solve runs in, with machine epsilon eps.
        float64_default: The strategy's float64 default tol_abs.
        vector: The right-hand side b.

    Returns:
        ``(atol, None)`` when ``value`` is set, the solve runs in float64 or
        b is empty: solve A x = b with that tol_abs. Otherwise
        ``(√eps, s)`` with s = maxₙ |b_n| (gradient stopped; 1 for a zero
        or non-finite b): solve A y = b / s and return x = s y (gh-639).

    Examples:
        A float32 right-hand side is scaled by its largest entry:

        >>> import jax.numpy as jnp
        >>> from gaussx._strategies._tolerances import rhs_scaling
        >>> b = jnp.array([0.5, -2.0], dtype=jnp.float32)
        >>> atol, scale = rhs_scaling(None, jnp.float32, 1e-5, b)
        >>> round(atol, 6), float(scale)
        (0.000345, 2.0)

        A user-set tolerance, or a zero b, leaves the system as it is:

        >>> rhs_scaling(1e-6, jnp.float32, 1e-5, b)
        (1e-06, None)
        >>> _, scale = rhs_scaling(None, jnp.float32, 1e-5, jnp.zeros(2))
        >>> float(scale)
        1.0
    """
    if value is not None:
        return value, None
    leaves = jax.tree.leaves(vector)
    if jnp.finfo(dtype).bits >= 64 or any(leaf.size == 0 for leaf in leaves):
        return float64_default, None
    # maxₙ |b_n| per leaf
    leaf_maxima = [jnp.max(jnp.abs(leaf)) for leaf in leaves]
    # s = maxₙ |b_n|
    scale = max(leaf_maxima)
    # s only changes the iteration count, so it carries no gradient
    scale = jax.lax.stop_gradient(scale)
    is_usable = jnp.isfinite(scale) & (scale > 0)
    # s = 1 for a zero or non-finite b
    scale = jnp.where(is_usable, scale, jnp.ones_like(scale))
    machine_epsilon = float(jnp.finfo(dtype).eps)
    # tol_abs = √eps
    scaled_atol = math.sqrt(machine_epsilon)
    return scaled_atol, scale
