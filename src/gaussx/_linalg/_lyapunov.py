"""Discrete Lyapunov equation X − A X Aᵀ = B, solved in the eigenbasis of A.

A is a square N × N matrix, B the N × N right-hand side and X the symmetric
solution; A = V diag(γ) V⁻¹ is the eigendecomposition of A, with eigenvalues
γ_i and eigenvectors in the columns of V. The N² × N² Kronecker matrix
I − A ⊗ A of the vectorised equation is never formed.
"""

from __future__ import annotations

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._linalg._symmetrize import symmetrize


def discrete_lyapunov_solve(
    G: Float[Array, "N N"],
    Q: Float[Array, "N N"],
) -> Float[Array, "N N"]:
    """Solve the discrete Lyapunov equation X − A X Aᵀ = B by diagonalising A.

        X − A X Aᵀ = B,   A = V diag(γ) V⁻¹
        X̃_ij = B̃_ij / (1 − γ_i γ_j),   B̃ = V⁻¹ B V⁻ᵀ,   X = V X̃ Vᵀ

    The cost is O(N³) (one general eigendecomposition and a few matrix
    products) against O(N⁶) for the vectorised solve with I − A ⊗ A, and
    the memory drops from O(N⁴) to O(N²).

    The solution is unique if and only if γ_i γ_j ≠ 1 for every pair of
    eigenvalues of A; a stable A (spectral radius below 1) guarantees it.

    !!! warning

        This assumes A is diagonalisable. For a defective A (a Jordan
        block with eigenvalue magnitude below 1, say) V is singular and
        the result is ``NaN`` or ``inf``. The equation still has a unique
        solution then, but finding it needs a Schur decomposition
        (Bartels-Stewart), which JAX does not expose; fall back to the
        vectorised (I − A ⊗ A) solve for such matrices.

    Algorithm:
        1. Eigendecompose A = V diag(γ) V⁻¹ (complex in general).
        2. Rotate the right-hand side, B̃ = V⁻¹ B V⁻ᵀ, with two solves
           against V instead of forming V⁻¹.
        3. Divide entrywise, X̃_ij = B̃_ij / (1 − γ_i γ_j).
        4. Rotate back, X = V X̃ Vᵀ, keep the real part and symmetrise.

    Args:
        G: Square matrix A, stable for a unique solution.
        Q: Right-hand side B, symmetric positive semi-definite in the
            Kalman-filter use (the model-error covariance Σ_η).

    Returns:
        The symmetric solution X of X − A X Aᵀ = B.

    Examples:
        A stable diagonal A has the closed form X_ij = B_ij / (1 − γ_i γ_j):

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> A = jnp.diag(jnp.array([0.5, 0.2]))
        >>> B = jnp.eye(2)
        >>> X = gaussx.discrete_lyapunov_solve(A, B)
        >>> bool(jnp.allclose(jnp.diag(X), jnp.array([1 / 0.75, 1 / 0.96])))
        True

        A general stable A satisfies the equation:

        >>> A = jnp.array([[0.5, 0.3], [-0.2, 0.4]])
        >>> X = gaussx.discrete_lyapunov_solve(A, B)
        >>> bool(jnp.allclose(X - A @ X @ A.T, B, atol=1e-5))
        True
    """
    # A = V diag(γ) V⁻¹
    eigenvalues, eigenvectors = jnp.linalg.eig(G)
    rhs_complex = Q.astype(eigenvectors.dtype)
    # Solves against V rather than an explicit V⁻¹: more stable, and cheaper
    # when V is well conditioned.
    # V⁻¹ B
    rotated_rows = jnp.linalg.solve(eigenvectors, rhs_complex)
    # (V⁻¹ B)ᵀ
    rotated_rows_transpose = rearrange(rotated_rows, "row col -> col row")
    # B̃ᵀ = V⁻¹ (V⁻¹ B)ᵀ
    rotated_transpose = jnp.linalg.solve(eigenvectors, rotated_rows_transpose)
    # B̃ = V⁻¹ B V⁻ᵀ
    rotated_rhs = rearrange(rotated_transpose, "row col -> col row")
    # γ_i γ_j
    eigenvalue_products = einx.multiply("row, col -> row col", eigenvalues, eigenvalues)
    # 1 − γ_i γ_j
    denominator = 1.0 - eigenvalue_products
    # X̃_ij = B̃_ij / (1 − γ_i γ_j)
    rotated_solution = rotated_rhs / denominator
    # V X̃
    half_rotated = einsum(eigenvectors, rotated_solution, "row mid, mid col -> row col")
    # X = V X̃ Vᵀ
    solution_complex = einsum(half_rotated, eigenvectors, "row mid, col mid -> row col")
    solution = jnp.real(solution_complex)
    # Remove round-off asymmetry, as the conditional and infinite-horizon
    # smoother covariances do.
    return symmetrize(solution)
