"""The pseudo-inverse of a singular structure matrix, through grounding.

With P the nodes that ground R (`gaussx._operators._grounding`), G the inverse of
the grounded block padded with zeros on P, V the null-space basis and
P⊥ = I − Q Qᵀ with V = Q R, the pseudo-inverse is

    R⁺ = P⊥ G P⊥,

since G is a symmetric generalised inverse of R and P⊥ = R⁺ R = R R⁺. A draw
g ~ 𝒩(0, G) is a draw from the grounded precision with its P entries set to zero,
so P⊥ g ~ 𝒩(0, R⁺).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._linalg._diag_inv import diag_inv
from gaussx._operators._grounding import ground, ground_nodes, node_mask
from gaussx._operators._sparse import SparseOperator
from gaussx._primitives._solve import solve
from gaussx._sparse._factor import sparse_cholesky


def null_dual(null_space: Float[Array, "N c"]) -> Float[Array, "N c"]:
    """Dual basis V (Vᵀ V)⁻¹ of the null space, through V = Q R.

        V (Vᵀ V)⁻¹ = Q R⁻ᵀ

    The QR factorisation keeps the condition number of V, where a solve with Vᵀ V
    squares it.

    Args:
        null_space: The basis V.

    Returns:
        V (Vᵀ V)⁻¹, shape ``(N, c)``.
    """
    # V = Q R
    orthonormal, upper = jnp.linalg.qr(null_space)
    orthonormal_rows = rearrange(orthonormal, "state null -> null state")
    # R⁻¹ Qᵀ
    solved = jsl.solve_triangular(upper, orthonormal_rows, lower=False)
    # Q R⁻ᵀ
    return rearrange(solved, "null state -> state null")


def pinv_diagonal(
    structure: lx.AbstractLinearOperator, null_space: Float[Array, "N c"]
) -> Float[Array, " N"]:
    """Diagonal of the pseudo-inverse R⁺ of a singular structure.

        diag(R⁺) = diag(P⊥ G P⊥)
                 = diag(G) − 2 diag(Q (G Q)ᵀ) + diag(Q Qᵀ G Q Qᵀ),   V = Q R

    Args:
        structure: The structure R; anything but a (tagged or scaled)
            `SparseOperator` or `BlockTriDiag` is densified.
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
    # an orthonormal basis Q of the null space
    basis, _ = jnp.linalg.qr(null_space)
    # G Q
    solved_basis = jax.vmap(inverse, in_axes=1, out_axes=1)(basis)
    # Qᵀ G Q
    null_covariance = einsum(basis, solved_basis, "n a, n b -> a b")
    # diag(Q (G Q)ᵀ)
    cross = einsum(basis, solved_basis, "n a, n a -> n")
    # diag(Q Qᵀ G Q Qᵀ)
    null_part = einsum(basis, null_covariance, basis, "n a, a b, n b -> n")
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
