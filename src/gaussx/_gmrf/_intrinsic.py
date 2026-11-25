"""The pseudo-inverse of a singular structure matrix, through grounding.

With P the nodes that ground R (`gaussx._operators._grounding`), G the inverse of
the grounded block padded with zeros on P, V the null-space basis and
P⊥ = I − V (Vᵀ V)⁻¹ Vᵀ, the pseudo-inverse is

    R⁺ = P⊥ G P⊥,

since G is a symmetric generalised inverse of R and P⊥ = R⁺ R = R R⁺. A draw
g ~ 𝒩(0, G) is a draw from the grounded precision with its P entries set to zero,
so P⊥ g ~ 𝒩(0, R⁺).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._linalg._diag_inv import diag_inv
from gaussx._operators._grounding import ground, ground_nodes, node_mask
from gaussx._operators._sparse import SparseOperator
from gaussx._primitives._solve import solve
from gaussx._sparse._factor import sparse_cholesky


def null_dual(null_space: Float[Array, "N c"]) -> Float[Array, "N c"]:
    """Dual basis V (Vᵀ V)⁻¹ of the null space.

        P⊥ x = x − V (Vᵀ V)⁻¹ Vᵀ x

    Args:
        null_space: The basis V.

    Returns:
        V (Vᵀ V)⁻¹, shape ``(N, c)``.
    """
    # Vᵀ V
    null_gram = einsum(null_space, null_space, "state row, state col -> row col")
    # Vᵀ
    null_rows = rearrange(null_space, "state null -> null state")
    # (Vᵀ V)⁻¹ Vᵀ, a small dense solve
    solved = jnp.linalg.solve(null_gram, null_rows)
    # V (Vᵀ V)⁻¹
    return rearrange(solved, "null state -> state null")


def pinv_diagonal(
    structure: lx.AbstractLinearOperator, null_space: Float[Array, "N c"]
) -> Float[Array, " N"]:
    """Diagonal of the pseudo-inverse R⁺ of a singular structure.

        diag(R⁺) = diag(P⊥ G P⊥)
                 = diag(G) − 2 diag(D (G V)ᵀ) + diag(D Vᵀ G V Dᵀ),
        D = V (Vᵀ V)⁻¹

    Args:
        structure: The structure R; anything but a (tagged) `SparseOperator` or
            `BlockTriDiag` is densified.
        null_space: Basis V of ker R.

    Returns:
        The diagonal of R⁺.
    """
    nodes = ground_nodes(null_space)
    free = ~node_mask(nodes, structure.in_size())
    grounded = ground(structure, nodes)
    grounded_solve, grounded_diag_inv = _solve_and_diag_inv(grounded)

    def inverse(column: Float[Array, " N"]) -> Float[Array, " N"]:
        # G v: the grounded solve, with v and the result zero on P
        return jnp.where(free, grounded_solve(jnp.where(free, column, 0.0)), 0.0)

    # diag(G)
    inverse_diagonal = jnp.where(free, grounded_diag_inv(), 0.0)
    dual = null_dual(null_space)
    # G V
    solved_null_space = jax.vmap(inverse, in_axes=1, out_axes=1)(null_space)
    # Vᵀ G V
    null_covariance = einsum(null_space, solved_null_space, "n a, n b -> a b")
    # diag(D (G V)ᵀ)
    cross = einsum(dual, solved_null_space, "n a, n a -> n")
    # diag(D Vᵀ G V Dᵀ)
    null_part = einsum(dual, null_covariance, dual, "n a, a b, n b -> n")
    # diag(R⁺)
    return inverse_diagonal - 2.0 * cross + null_part


def grounded_structure(
    structure: lx.AbstractLinearOperator, null_space: Float[Array, "N c"]
) -> tuple[lx.AbstractLinearOperator, Array]:
    """The grounded structure, and the mask of its free nodes.

    Args:
        structure: The structure R.
        null_space: Basis V of ker R.

    Returns:
        Tuple ``(grounded, free)``: the positive-definite grounded R, and ``True``
        at the nodes that were not fixed.
    """
    nodes = ground_nodes(null_space)
    return ground(structure, nodes), ~node_mask(nodes, structure.in_size())


def _solve_and_diag_inv(
    operator: lx.AbstractLinearOperator,
) -> tuple[Callable[[Array], Array], Callable[[], Array]]:
    """The solve and inverse diagonal of an operator, from one sparse factor.

    Args:
        operator: The positive-definite operator.

    Returns:
        The maps b ↦ A⁻¹ b and () ↦ diag(A⁻¹).
    """
    if isinstance(operator, SparseOperator):
        factor = sparse_cholesky(operator)
        return factor.solve, factor.diag_inv
    return (lambda b: solve(operator, b)), (lambda: diag_inv(operator))
