"""Symmetric part ½ (A + Aᵀ) of a square matrix or a stack of them.

A leaf module (no gaussx imports), so primitives, recipes and solvers can all
use it without import cycles. A is a generic square matrix.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def symmetrize(mat: Float[Array, "... n n"]) -> Float[Array, "... n n"]:
    """Symmetric part of the trailing two axes, removing round-off asymmetry.

        sym(A) = ½ (A + Aᵀ)

    Used after covariance updates, whose round-off leaves A slightly
    asymmetric. Leading axes are batch axes.

    Args:
        mat: Square matrix A, or a stack of them over leading batch axes.

    Returns:
        The symmetric part ½ (A + Aᵀ), with the shape and dtype of ``mat``.

    Examples:
        A single matrix:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> matrix = jnp.array([[1.0, 2.0], [0.0, 3.0]])
        >>> symmetric = gaussx.symmetrize(matrix)
        >>> bool(jnp.allclose(symmetric, jnp.array([[1.0, 1.0], [1.0, 3.0]])))
        True

        A stack of matrices is symmetrised one by one:

        >>> stack = jnp.stack([matrix, 2.0 * matrix])
        >>> symmetric_stack = gaussx.symmetrize(stack)
        >>> symmetric_stack.shape
        (2, 2, 2)
        >>> bool(jnp.allclose(symmetric_stack[1], 2.0 * symmetric))
        True
    """
    # Aᵀ on the trailing two axes. Not einx: its "..." cannot solve for a
    # zero-length batch axis, which the empty-time-axis filters pass here.
    mat_transpose = jnp.swapaxes(mat, -1, -2)
    # ½ (A + Aᵀ)
    return 0.5 * (mat + mat_transpose)
