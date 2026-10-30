"""Woodbury solve of a low-rank update (A + U D Uᵀ) x = b from its parts.

A is the N × N base operator, U the N × k low-rank factor, D = diag(d) the
k × k diagonal scaling and C = D⁻¹ + Uᵀ A⁻¹ U the k × k capacitance matrix.
"""

from __future__ import annotations

import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._primitives._solve import solve


def woodbury_solve(
    base: lx.AbstractLinearOperator,
    U: Float[Array, "N k"],
    D: Float[Array, " k"],
    b: Float[Array, " N"],
) -> Float[Array, " N"]:
    """Solve (A + U D Uᵀ) x = b by the Woodbury identity, without the operator.

        (A + U D Uᵀ)⁻¹ b = A⁻¹ b − A⁻¹ U C⁻¹ Uᵀ A⁻¹ b,   C = D⁻¹ + Uᵀ A⁻¹ U

    A convenience for callers who hold the parts but not a
    `gaussx.LowRankUpdate`; it builds one and hands it to `gaussx.solve`,
    whose structured rule applies the identity.

    Args:
        base: Base operator A, N × N.
        U: Low-rank factor U, N × k.
        D: Diagonal d of the k × k scaling D = diag(d).
        b: Right-hand side b.

    Returns:
        The solution x = (A + U D Uᵀ)⁻¹ b.

    Examples:
        A diagonal base with a rank-one update matches the dense solve:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> base = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0, 3.0]))
        >>> U = jnp.array([[1.0], [0.5], [0.0]])
        >>> D = jnp.array([2.0])
        >>> b = jnp.array([1.0, 0.0, 1.0])
        >>> x = gaussx.woodbury_solve(base, U, D, b)
        >>> dense = jnp.diag(jnp.array([1.0, 2.0, 3.0])) + 2.0 * jnp.outer(U, U)
        >>> bool(jnp.allclose(x, jnp.linalg.solve(dense, b), atol=1e-5))
        True
    """
    # A + U D Uᵀ, as a structured operator
    low_rank_update = LowRankUpdate(base, U, D)
    # x = (A + U D Uᵀ)⁻¹ b, through the Woodbury rule of `solve`
    return solve(low_rank_update, b)
