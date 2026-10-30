"""Precomputed eigendecompositions and shifted Kronecker-sum solves.

These are the building blocks of the matrix-diagonalisation method for
separable operators (tensor-product spectral discretisations of ∇² − c, for
example): factor each one-dimensional operator once, and every solve for any
shift is then a handful of dense rotations and a pointwise division.

A is an n × n matrix with eigendecomposition A = V diag(γ) V⁻¹, eigenvalues
γ_i and right eigenvectors in the columns of V; c is a scalar shift. In a
Kronecker sum, A_d acts on axis d of a tensor X, d = 1, …, D, and ×_d is the
mode-d product.

Unlike the structured solve of `gaussx.KroneckerSum`, which needs symmetric
factors (it uses Vᵀ as the inverse rotation), `EigenDecomposition` stores
V⁻¹ explicitly and so handles non-symmetric but diagonalisable factors with a
real spectrum, such as Chebyshev collocation second-derivative matrices.
"""

from __future__ import annotations

from collections.abc import Sequence

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float

from gaussx._einx import einsum, repeat


_IMAG_TOLERANCE_FACTOR = 1e3


class EigenDecomposition(eqx.Module):
    """Precomputed eigendecomposition A = V diag(γ) V⁻¹ of a square matrix.

        A = V diag(γ) V⁻¹,   (A − c I)⁻¹ b = V diag(1 / (γ − c)) V⁻¹ b

    Built once (on the host, outside ``jit``) and then reused for any number
    of shifted solves (A − c I)⁻¹ b, each costing two dense matrix-vector
    products, O(n²), for any shift c. The shift may be traced, so solves are
    ``jit``- and ``grad``-compatible in c.

    Only real spectra are supported: the factorisation is kept in real
    arithmetic, and `from_matrix` rejects matrices whose eigenvalues have
    non-negligible imaginary parts.

    Attributes:
        eigenvalues: Eigenvalues γ.
        eigenvectors: Right eigenvectors V, one per column.
        eigenvectors_inv: The inverse V⁻¹ (Vᵀ for a symmetric A).

    Examples:
        Factor once, then solve for a shift:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> A = jnp.array([[2.0, 1.0], [0.0, 3.0]])
        >>> factorization = gaussx.EigenDecomposition.from_matrix(A)
        >>> b = jnp.array([1.0, 1.0])
        >>> x = factorization.solve_shifted(b, shift=1.0)
        >>> bool(jnp.allclose((A - jnp.eye(2)) @ x, b, atol=1e-5))
        True
        >>> bool(jnp.allclose(factorization.as_matrix(), A, atol=1e-5))
        True
    """

    eigenvalues: Float[Array, " n"]
    eigenvectors: Float[Array, "n n"]
    eigenvectors_inv: Float[Array, "n n"]

    @classmethod
    def from_matrix(
        cls,
        matrix: ArrayLike | lx.AbstractLinearOperator,
        *,
        symmetric: bool | None = None,
    ) -> EigenDecomposition:
        """Eigendecompose a concrete square matrix A on the host.

            A = V diag(γ) V⁻¹

        Uses NumPy (``eigh`` for a symmetric input, ``eig`` otherwise), so
        it works for non-symmetric matrices on every JAX backend. Because of
        that, ``matrix`` must be concrete: call this outside ``jit`` (when
        constructing a solver, say), not on traced values.

        Args:
            matrix: Square matrix A, or a lineax operator (materialised with
                ``as_matrix()``).
            symmetric: Use the symmetric path. Defaults to
                ``lx.is_symmetric(matrix)`` for operators and ``False`` for
                arrays.

        Returns:
            The factorisation.

        Raises:
            ValueError: If ``matrix`` is not square, is a tracer, or has a
                spectrum with non-negligible imaginary parts.

        Examples:
            A rotation has a complex spectrum and is rejected:

            >>> import jax.numpy as jnp
            >>> import gaussx
            >>> rotation = jnp.array([[0.0, -1.0], [1.0, 0.0]])
            >>> gaussx.EigenDecomposition.from_matrix(
            ...     rotation
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: EigenDecomposition supports real spectra only; ...
        """
        if isinstance(matrix, lx.AbstractLinearOperator):
            if symmetric is None:
                symmetric = lx.is_symmetric(matrix)
            matrix = matrix.as_matrix()
        try:
            mat = np.asarray(matrix)
        except jax.errors.TracerArrayConversionError as err:
            raise ValueError(
                "EigenDecomposition.from_matrix needs a concrete matrix "
                "(it factorizes on the host with NumPy); call it outside jit."
            ) from err
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"matrix must be square, got shape {mat.shape}.")

        if symmetric:
            # A = V diag(γ) Vᵀ
            eigenvalues, eigenvectors = np.linalg.eigh(mat)
            # V⁻¹ = Vᵀ for orthonormal V (host NumPy, so no einx)
            eigenvectors_inv = eigenvectors.T
        else:
            # A = V diag(γ) V⁻¹
            eigenvalues, eigenvectors = np.linalg.eig(mat)
            scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
            tolerance = _IMAG_TOLERANCE_FACTOR * np.finfo(mat.dtype).eps * scale
            if np.max(np.abs(eigenvalues.imag)) > tolerance:
                raise ValueError(
                    "EigenDecomposition supports real spectra only; the largest "
                    f"imaginary part is {np.max(np.abs(eigenvalues.imag)):.3e}."
                )
            eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
            eigenvectors_inv = np.linalg.inv(eigenvectors)
        return cls(
            jnp.asarray(eigenvalues),
            jnp.asarray(eigenvectors),
            jnp.asarray(eigenvectors_inv),
        )

    def as_matrix(self) -> Float[Array, "n n"]:
        """Reassemble A = V diag(γ) V⁻¹.

        Returns:
            The dense matrix A.
        """
        # V diag(γ)
        scaled_eigenvectors = einx.multiply(
            "row col, col -> row col", self.eigenvectors, self.eigenvalues
        )
        # A = V diag(γ) V⁻¹
        return einsum(
            scaled_eigenvectors,
            self.eigenvectors_inv,
            "row mid, mid col -> row col",
        )

    def solve_shifted(
        self,
        rhs: Float[Array, " n *batch"],
        shift: float | Float[Array, ""] = 0.0,
        *,
        drop: Bool[Array, " n"] | None = None,
    ) -> Float[Array, " n *batch"]:
        """Solve (A − c I) x = b along the leading axis of b.

            x = V diag(1 / (γ − c)) V⁻¹ b

        Args:
            rhs: Right-hand side b, the leading axis of length n, optionally
                followed by batch axes.
            shift: Scalar shift c; may be traced.
            drop: Optional boolean mask over the eigenmodes. A masked mode
                gets a zero coefficient instead of 1 / (γ_i − c): a
                restricted pseudo-inverse, to project out a known null space
                such as the constant mode of a pure-Neumann Laplacian.

        Returns:
            The solution x, shaped like ``rhs``.

        Examples:
            Dropping the null mode of a singular matrix:

            >>> import jax.numpy as jnp
            >>> import gaussx
            >>> A = jnp.array([[1.0, -1.0], [-1.0, 1.0]])
            >>> factorization = gaussx.EigenDecomposition.from_matrix(
            ...     A, symmetric=True
            ... )
            >>> null_mode = jnp.abs(factorization.eigenvalues) < 1e-6
            >>> b = jnp.array([1.0, -1.0])
            >>> x = factorization.solve_shifted(b, drop=null_mode)
            >>> bool(jnp.allclose(A @ x, b, atol=1e-5))
            True
        """
        return kronecker_sum_solve((self,), rhs, shift, drop=drop)


def _apply_along_axis(matrix: Float[Array, "m n"], tensor: Array, axis: int) -> Array:
    """Mode product: contract the columns of a matrix with one axis of a tensor.

        (T ×_d M)[…, i, …] = ∑_j M_ij T[…, j, …]

    Args:
        matrix: The matrix M.
        tensor: The tensor T.
        axis: The axis d of T to contract.

    Returns:
        The product T ×_d M, with axis d of length m.
    """
    names = [f"a{index}" for index in range(tensor.ndim)]
    lhs = " ".join([*names[:axis], "col", *names[axis + 1 :]])
    out = " ".join([*names[:axis], "row", *names[axis + 1 :]])
    # T ×_d M
    return einsum(matrix, tensor, f"row col, {lhs} -> {out}")


def kronecker_sum_solve(
    factors: Sequence[EigenDecomposition],
    rhs: Float[Array, "*dims"],
    shift: float | Float[Array, ""] = 0.0,
    *,
    drop: Bool[Array, "*dims"] | None = None,
) -> Float[Array, "*dims"]:
    """Solve a shifted Kronecker-sum system in tensor (unvectorised) form.

        ∑_d X ×_d A_d − c X = B
        (A_1 ⊕ A_2 ⊕ … ⊕ A_D − c I) vec(X) = vec(B)

    A_d acts on axis d of the unknown tensor X, whose D leading axes have
    lengths n_1, …, n_D and may be followed by batch axes. In two dimensions
    this is the Sylvester equation A_1 X + X A_2ᵀ − c X = B. The cost is
    O(N ∑_d n_d) for N = ∏_d n_d unknowns, the N × N Kronecker matrix is
    never formed, and the factors need not be symmetric.

    Algorithm:
        1. Rotate into the eigenbasis, X̂ = B ×_1 V_1⁻¹ ×_2 V_2⁻¹ … ×_D V_D⁻¹.
        2. Divide, X̂[i_1, …, i_D] /= γ¹_i_1 + … + γᴰ_i_D − c.
        3. Rotate back, X = X̂ ×_1 V_1 ×_2 V_2 … ×_D V_D.

    Args:
        factors: One `EigenDecomposition` of A_d per leading axis of
            ``rhs``.
        rhs: Right-hand side B, the D leading axes matching the factor
            sizes, optionally followed by batch axes.
        shift: Scalar shift c; may be traced.
        drop: Optional boolean mask over the D leading axes; masked
            eigen-combinations get a zero coefficient (a restricted
            pseudo-inverse), as in `EigenDecomposition.solve_shifted`.

    Returns:
        The solution X, shaped like ``rhs``.

    Raises:
        ValueError: If the factor sizes do not match the leading axes of
            ``rhs``.

    Examples:
        A 2 × 3 grid with two symmetric factors matches the dense
        Kronecker-sum solve:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> A_1 = jnp.array([[2.0, 0.5], [0.5, 2.0]])
        >>> A_2 = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        >>> factors = [
        ...     gaussx.EigenDecomposition.from_matrix(A_1, symmetric=True),
        ...     gaussx.EigenDecomposition.from_matrix(A_2, symmetric=True),
        ... ]
        >>> B = jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        >>> X = gaussx.kronecker_sum_solve(factors, B, shift=0.5)
        >>> X.shape
        (2, 3)
        >>> dense = (
        ...     jnp.kron(A_1, jnp.eye(3)) + jnp.kron(jnp.eye(2), A_2)
        ...     - 0.5 * jnp.eye(6)
        ... )
        >>> expected = jnp.linalg.solve(dense, jnp.ravel(B))
        >>> bool(jnp.allclose(jnp.ravel(X), expected, atol=1e-5))
        True
    """
    num_factors = len(factors)
    sizes = tuple(factor.eigenvalues.shape[0] for factor in factors)
    if rhs.shape[:num_factors] != sizes:
        raise ValueError(
            f"rhs leading shape {rhs.shape[:num_factors]} does not match factor "
            f"sizes {sizes}."
        )

    # --- rotate into the eigenbasis ---
    rotated = rhs
    for axis, factor in enumerate(factors):
        # X̂ ← X̂ ×_d V_d⁻¹
        rotated = _apply_along_axis(factor.eigenvectors_inv, rotated, axis)

    # --- divide by the shifted eigenvalue sums ---
    grid_names = [f"a{axis}" for axis in range(num_factors)]
    grid_pattern = " ".join(grid_names)
    grid_sizes = dict(zip(grid_names, sizes, strict=True))
    # γ¹_i_1 + … + γᴰ_i_D on the eigen grid, summed factor by factor
    eigenvalue_sum = jnp.zeros((), dtype=factors[0].eigenvalues.dtype)
    for axis, factor in enumerate(factors):
        other_sizes = {
            name: size for name, size in grid_sizes.items() if name != grid_names[axis]
        }
        # γᵈ broadcast over the other eigen axes
        broadcast_eigenvalues = repeat(
            factor.eigenvalues,
            f"{grid_names[axis]} -> {grid_pattern}",
            **other_sizes,
        )
        eigenvalue_sum = eigenvalue_sum + broadcast_eigenvalues
    # γ¹_i_1 + … + γᴰ_i_D − c
    denominator = eigenvalue_sum - shift
    if drop is None:
        # 1 / (γ¹_i_1 + … + γᴰ_i_D − c)
        coefficient = 1.0 / denominator
    else:
        # The inner where keeps the masked division finite, so its gradient
        # is too.
        safe_denominator = jnp.where(drop, 1.0, denominator)
        coefficient = jnp.where(drop, 0.0, 1.0 / safe_denominator)
    batch_pattern = " ".join(f"b{axis}" for axis in range(rotated.ndim - num_factors))
    # X̂[i_1, …, i_D] /= γ¹_i_1 + … + γᴰ_i_D − c, over the batch axes too
    rotated = einx.multiply(
        f"{grid_pattern} {batch_pattern}, {grid_pattern} -> {grid_pattern} "
        f"{batch_pattern}",
        rotated,
        coefficient,
    )

    # --- rotate back ---
    solution = rotated
    for axis, factor in enumerate(factors):
        # X ← X ×_d V_d
        solution = _apply_along_axis(factor.eigenvectors, solution, axis)
    return solution
