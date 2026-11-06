"""`SparseCholeskyFactor` and `sparse_cholesky`, for P A Pᵀ = L Lᵀ."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._operators._sparse import SparseOperator
from gaussx._sparse import _vjp
from gaussx._sparse._cholmod import cholmod_cholesky
from gaussx._sparse._numeric import lower_values, numeric_cholesky, solve_upper
from gaussx._sparse._symbolic import SymbolicCholesky, symbolic_cholesky
from gaussx._sparse._takahashi import takahashi


class SparseCholeskyFactor(eqx.Module):
    """Sparse Cholesky factor on a static symbolic pattern.

        P A Pᵀ = L Lᵀ

    Built by `gaussx.sparse_cholesky` (or `gaussx.cholesky` on a
    `SparseOperator`). ``values`` are the entries of L on the CSC pattern
    of ``symbolic``; ``matrix_values`` are those of the factored matrix (its
    lower triangle, permuted, on the same pattern), the input that
    ``logdet`` and ``solve`` differentiate through their custom VJPs:

    - ``logdet``: d log det A = tr(A⁻¹ dA), so the cotangent is A⁻¹ on
      pattern(A), which Takahashi evaluates in one sweep;
    - ``solve``: b̄ = A⁻¹ x̄ and Ā = −b̄ xᵀ, symmetrised on the pattern.

    Gradients reach the operator's stored values per stored value: with
    ``symmetric=True`` storage an off-diagonal value sets A_ij and A_ji, so its
    cotangent is doubled (2 [A⁻¹]_ij); general storage factors ½(A + Aᵀ) and each
    stored value gets [A⁻¹]_ij. The other methods are differentiated by JAX
    through the factorisation.

    Attributes:
        values: L on its CSC pattern, of length nnz(L).
        matrix_values: Lower triangle of P A Pᵀ on L's pattern.
        symbolic: The static symbolic analysis.

    Examples:
        The precision of a path graph 0 − 1 − 2 − 3 plus a unit shift:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> size = 4
        >>> A = gaussx.SparseOperator.from_coo(
        ...     np.r_[np.arange(size), np.arange(1, size)],
        ...     np.r_[np.arange(size), np.arange(size - 1)],
        ...     jnp.r_[jnp.array([2.0, 3.0, 3.0, 2.0]), -jnp.ones(size - 1)],
        ...     (size, size),
        ...     symmetric=True,
        ... )
        >>> factor = gaussx.sparse_cholesky(A)
        >>> dense = A.as_matrix()
        >>> dense_logdet = jnp.linalg.slogdet(dense)[1]
        >>> bool(jnp.allclose(factor.logdet(), dense_logdet, atol=1e-4))
        True
        >>> b = jnp.ones(size)
        >>> dense_solution = jnp.linalg.solve(dense, b)
        >>> bool(jnp.allclose(factor.solve(b), dense_solution, atol=1e-4))
        True
        >>> dense_variances = jnp.diag(jnp.linalg.inv(dense))
        >>> bool(jnp.allclose(factor.diag_inv(), dense_variances, atol=1e-4))
        True
    """

    values: Float[Array, " nnz_L"]
    matrix_values: Float[Array, " nnz_L"]
    symbolic: SymbolicCholesky = eqx.field(static=True)

    def _L(self) -> Array:
        """The factor's values as ``logdet`` and ``solve`` see them.

        Returns:
            L on its CSC pattern, gradients stopped on the CHOLMOD backend.
        """
        # logdet and solve route their first-order cotangents to
        # ``matrix_values`` and give L none. L stays differentiable on
        # the JAX backend so that reverse-over-reverse (a Hessian) sees how
        # the Takahashi / adjoint cotangents in their backward passes move
        # with the values. CHOLMOD's callback has no derivative.
        if self.symbolic.backend == "cholmod":
            return jax.lax.stop_gradient(self.values)
        return self.values

    def solve(self, b: Float[Array, " n"]) -> Float[Array, " n"]:
        """Solve A x = b through the factor.

            x = Pᵀ L⁻ᵀ L⁻¹ P b

        Args:
            b: Right-hand side b, of length N.

        Returns:
            The solution x, of length N.

        Examples:
            A diagonal A:

            >>> import jax.numpy as jnp
            >>> import numpy as np
            >>> import gaussx
            >>> A = gaussx.SparseOperator.from_coo(
            ...     np.arange(2), np.arange(2), jnp.array([2.0, 4.0]), (2, 2)
            ... )
            >>> solution = gaussx.sparse_cholesky(A).solve(jnp.array([2.0, 2.0]))
            >>> bool(jnp.allclose(solution, jnp.array([1.0, 0.5])))
            True
        """
        sym = self.symbolic
        # y = P b
        permuted_rhs = b[jnp.asarray(sym.perm)]
        # z = L⁻ᵀ L⁻¹ y
        permuted_solution = _vjp.solve(sym, self.matrix_values, self._L(), permuted_rhs)
        # x = Pᵀ z
        return permuted_solution[jnp.asarray(sym.iperm)]

    def logdet(self) -> Float[Array, ""]:
        """Log-determinant of A from the factor's diagonal.

            log det A = 2 ∑ₙ log L_nn

        Returns:
            The log-determinant (NaN if A is not positive definite).

        Examples:
            A diagonal A, and one that is not positive definite:

            >>> import jax.numpy as jnp
            >>> import numpy as np
            >>> import gaussx
            >>> A = gaussx.SparseOperator.from_coo(
            ...     np.arange(2), np.arange(2), jnp.array([1.0, 1.0]), (2, 2)
            ... )
            >>> bool(jnp.allclose(gaussx.sparse_cholesky(A).logdet(), 0.0))
            True
            >>> negated = gaussx.SparseOperator(-A.values, A.pattern)
            >>> bool(jnp.isnan(gaussx.sparse_cholesky(negated).logdet()))
            True
        """
        return _vjp.logdet(self.symbolic, self.matrix_values, self._L())

    def solve_lower_transpose(self, z: Float[Array, " n"]) -> Float[Array, " n"]:
        """Solve through Lᵀ and undo the permutation: a sampler for 𝒩(0, A⁻¹).

            x = Pᵀ L⁻ᵀ z,   z ~ 𝒩(0, I)  ⟹  x ~ 𝒩(0, A⁻¹)

        Args:
            z: The vector z, of length N.

        Returns:
            The vector x, of length N.

        Examples:
            A diagonal A:

            >>> import jax.numpy as jnp
            >>> import numpy as np
            >>> import gaussx
            >>> A = gaussx.SparseOperator.from_coo(
            ...     np.arange(2), np.arange(2), jnp.array([4.0, 16.0]), (2, 2)
            ... )
            >>> factor = gaussx.sparse_cholesky(A)
            >>> x = factor.solve_lower_transpose(jnp.array([2.0, 2.0]))
            >>> bool(jnp.allclose(x, jnp.array([1.0, 0.5])))
            True
        """
        sym = self.symbolic
        # L⁻ᵀ z
        permuted_solution = solve_upper(sym, self.values, z)
        # x = Pᵀ L⁻ᵀ z
        return permuted_solution[jnp.asarray(sym.iperm)]

    def selected_inverse(self) -> SparseOperator:
        """Entries of A⁻¹ on the pattern of L + Lᵀ, in the original order.

            [A⁻¹]_ij for (i, j) ∈ pattern(Pᵀ (L + Lᵀ) P)

        The pattern contains pattern(A). Like the block selected inverse, the
        result holds entries of A⁻¹; it is not an operator equal to A⁻¹ (whose
        other entries are not zero).

        Returns:
            A symmetric `SparseOperator` holding the selected entries.

        Examples:
            A diagonal A:

            >>> import jax.numpy as jnp
            >>> import numpy as np
            >>> import gaussx
            >>> A = gaussx.SparseOperator.from_coo(
            ...     np.arange(2), np.arange(2), jnp.array([2.0, 4.0]), (2, 2)
            ... )
            >>> selected = gaussx.sparse_cholesky(A).selected_inverse()
            >>> bool(jnp.allclose(selected.values, jnp.array([0.5, 0.25])))
            True
        """
        pattern, index = self.symbolic.inverse_plan
        # Z = (P A Pᵀ)⁻¹ on L's pattern
        inverse_entries = takahashi(self.symbolic, self.values)
        return SparseOperator(inverse_entries[jnp.asarray(index)], pattern)

    def diag_inv(self) -> Float[Array, " n"]:
        """The marginal variances, by one Takahashi sweep.

            diag(A⁻¹)

        Returns:
            The diagonal of A⁻¹, of length N, in the original order.

        Examples:
            A diagonal A:

            >>> import jax.numpy as jnp
            >>> import numpy as np
            >>> import gaussx
            >>> A = gaussx.SparseOperator.from_coo(
            ...     np.arange(2), np.arange(2), jnp.array([2.0, 4.0]), (2, 2)
            ... )
            >>> variances = gaussx.sparse_cholesky(A).diag_inv()
            >>> bool(jnp.allclose(variances, jnp.array([0.5, 0.25])))
            True
        """
        sym = self.symbolic
        # Z = (P A Pᵀ)⁻¹ on L's pattern
        inverse_entries = takahashi(sym, self.values)
        # Each column of L starts on its diagonal, in the permuted order.
        return inverse_entries[jnp.asarray(sym.colptr[sym.iperm])]


def sparse_cholesky(
    op: SparseOperator, symbolic: SymbolicCholesky | None = None
) -> SparseCholeskyFactor:
    """Sparse Cholesky factorisation of a symmetric positive-definite operator.

        P A Pᵀ = L Lᵀ

    The symbolic analysis (ordering, elimination tree, pattern of L) is
    host-side and cached per pattern; pass one from `gaussx.symbolic_cholesky`
    to choose the ordering or backend, or to reuse it explicitly. The numeric
    phase is traced: it ``jit``s, ``vmap``s over ``op.values`` and is
    differentiable. With the JAX backend it is a left-looking ``lax.scan``
    over columns, sequential in N and O(∑ⱼ |struct(L_:,j)|²) work.

    Algorithm:
        1. Analyse ``op.pattern``, unless ``symbolic`` is given.
        2. Scatter the stored values onto the lower triangle of P A Pᵀ.
        3. Factor it on the pattern of L with the analysis's backend.

    Args:
        op: Symmetric positive-definite operator A. A general (non-symmetric)
            pattern is factored as ½(A + Aᵀ).
        symbolic: Symbolic analysis of ``op.pattern``. Defaults to
            ``symbolic_cholesky(op.pattern)`` (RCM ordering, JAX backend).

    Returns:
        The factor.

    Raises:
        TypeError: If ``op`` is not a `SparseOperator`.
        ValueError: If ``symbolic`` was computed for a different pattern.

    Examples:
        A 1-D random-walk precision τ R + I: analyse once, factor for many τ.

        >>> import equinox as eqx
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> size = 6
        >>> R = gaussx.SparseOperator.from_coo(
        ...     np.r_[np.arange(size), np.arange(1, size)],
        ...     np.r_[np.arange(size), np.arange(size - 1)],
        ...     jnp.r_[jnp.array([1.0, 2.0, 2.0, 2.0, 2.0, 1.0]), -jnp.ones(size - 1)],
        ...     (size, size),
        ...     symmetric=True,
        ... )
        >>> symbolic = gaussx.symbolic_cholesky(R.pattern)
        >>> def logdet(log_tau):
        ...     scaled = jnp.exp(log_tau) * R.values
        ...     tau_R = eqx.tree_at(lambda op: op.values, R, scaled)
        ...     shifted = tau_R.add_diagonal(jnp.ones(size))
        ...     return gaussx.sparse_cholesky(shifted, symbolic).logdet()
        >>> jax.vmap(logdet)(jnp.linspace(-1.0, 1.0, 4)).shape
        (4,)

        The gradient runs through one Takahashi sweep:

        >>> jax.grad(logdet)(0.0).shape
        ()

        Anything but a `SparseOperator` is rejected:

        >>> gaussx.sparse_cholesky(jnp.eye(2))
        Traceback (most recent call last):
        ...
        TypeError: sparse_cholesky needs a SparseOperator, got ArrayImpl.

        So is an analysis of another pattern:

        >>> other = gaussx.symbolic_cholesky(
        ...     gaussx.SparsityPattern(np.array([0]), np.array([0]), (2, 2))
        ... )
        >>> gaussx.sparse_cholesky(R, other)  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: symbolic was computed for a different sparsity pattern:
        SparsityPattern(shape=(2, 2), nnz=2, symmetric=False) vs
        SparsityPattern(shape=(6, 6), nnz=11, symmetric=True).
    """
    if not isinstance(op, SparseOperator):
        raise TypeError(
            f"sparse_cholesky needs a SparseOperator, got {type(op).__name__}."
        )
    if symbolic is None:
        symbolic = symbolic_cholesky(op.pattern)
    elif symbolic.pattern != op.pattern:
        raise ValueError(
            "symbolic was computed for a different sparsity pattern: "
            f"{symbolic.pattern} vs {op.pattern}."
        )
    # lower(P A Pᵀ) on L's pattern
    matrix_values = lower_values(symbolic, op.values)
    # P A Pᵀ = L Lᵀ
    if symbolic.backend == "cholmod":
        factor_values = cholmod_cholesky(symbolic, matrix_values)
    else:
        factor_values = numeric_cholesky(symbolic, matrix_values)
    return SparseCholeskyFactor(factor_values, matrix_values, symbolic)
