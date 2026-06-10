"""Symmetrization helper shared across the package.

Leaf module (no gaussx imports) so primitives, recipes, and solvers can
all use it without import cycles.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def symmetrize(mat: Float[Array, "... n n"]) -> Float[Array, "... n n"]:
    """Symmetrize the trailing two axes: ``0.5 * (X + X^T)``.

    Eliminates residual floating-point asymmetry after covariance
    updates. Works on batched stacks ``(..., n, n)``.

    Args:
        mat: Square matrix or batched stack of square matrices.

    Returns:
        The symmetric part of ``mat``.
    """
    return 0.5 * (mat + jnp.swapaxes(mat, -1, -2))
