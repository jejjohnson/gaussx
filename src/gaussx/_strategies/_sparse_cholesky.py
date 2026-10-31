"""Sparse Cholesky solver strategy for `SparseOperator` precisions.

A is an N × N symmetric positive-definite sparse operator (a GMRF precision,
say) and P the permutation of a fill-reducing ordering, so P A Pᵀ = L Lᵀ
with a sparse lower-triangular L. A solve is two triangular solves, the
log-determinant is read off the diagonal of L and diag(A⁻¹) comes from one
Takahashi sweep over L.
"""

from __future__ import annotations

import equinox as eqx
import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._sparse import SparseOperator
from gaussx._sparse._factor import SparseCholeskyFactor, sparse_cholesky
from gaussx._sparse._symbolic import Backend, Ordering, symbolic_cholesky
from gaussx._strategies._base import AbstractSolverStrategy


class SparseCholeskySolver(AbstractSolverStrategy):
    """Exact ``solve`` / ``logdet`` / ``diag_inv`` through a sparse Cholesky factor.

        P A Pᵀ = L Lᵀ,   x = A⁻¹ b,   log det A = 2 ∑ₙ log L_nn

    For a symmetric positive-definite `SparseOperator` (a GMRF precision, a
    Laplace Hessian Q + Jᵀ W J). The symbolic analysis is cached per
    sparsity pattern, so only the numeric factorisation runs per call; its
    gradients are exact (Takahashi for ``logdet``, an adjoint solve for
    ``solve``). Pass it explicitly: `AutoSolver` keeps choosing CG for large
    PSD operators, and nothing picks this strategy by size.

    Attributes:
        ordering: Fill-reducing ordering, ``"rcm"``, ``"natural"`` or
            ``"amd"`` (CHOLMOD). See `gaussx.symbolic_cholesky`.
        backend: ``"jax"`` or ``"cholmod"`` (opt-in, needs scikit-sparse).

    Examples:
        A path-graph Laplacian plus a unit shift:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import numpy as np
        >>> import gaussx
        >>> size = 5
        >>> Q = gaussx.SparseOperator.from_coo(
        ...     np.r_[np.arange(size), np.arange(1, size)],
        ...     np.r_[np.arange(size), np.arange(size - 1)],
        ...     jnp.r_[jnp.array([2.0, 3.0, 3.0, 3.0, 2.0]), -jnp.ones(size - 1)],
        ...     (size, size),
        ...     symmetric=True,
        ...     tags=lx.positive_semidefinite_tag,
        ... )
        >>> solver = gaussx.SparseCholeskySolver()
        >>> x = solver.solve(Q, jnp.ones(size))
        >>> bool(jnp.allclose(Q.as_matrix() @ x, jnp.ones(size), atol=1e-5))
        True
        >>> dense_logdet = jnp.linalg.slogdet(Q.as_matrix())[1]
        >>> bool(jnp.allclose(solver.logdet(Q), dense_logdet, atol=1e-4))
        True

        Marginal variances from one Takahashi sweep:

        >>> variances = gaussx.diag_inv(Q, solver=solver)
        >>> dense_variances = jnp.diag(jnp.linalg.inv(Q.as_matrix()))
        >>> bool(jnp.allclose(variances, dense_variances, atol=1e-5))
        True
    """

    ordering: Ordering = eqx.field(static=True, default="rcm")
    backend: Backend = eqx.field(static=True, default="jax")

    def factor(self, operator: lx.AbstractLinearOperator) -> SparseCholeskyFactor:
        """The sparse Cholesky factor of ``operator``.

            P A Pᵀ = L Lᵀ

        Args:
            operator: Sparse operator A, a `SparseOperator` (or a
                lineax-tagged one).

        Returns:
            The factor, on the cached symbolic analysis of its pattern.

        Raises:
            TypeError: If ``operator`` is not a `SparseOperator`.
        """
        if isinstance(operator, lx.TaggedLinearOperator):
            operator = operator.operator
        if not isinstance(operator, SparseOperator):
            raise TypeError(
                "SparseCholeskySolver needs a SparseOperator, got "
                f"{type(operator).__name__}."
            )
        symbolic = symbolic_cholesky(
            operator.pattern, ordering=self.ordering, backend=self.backend
        )
        return sparse_cholesky(operator, symbolic)

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve A x = b with two sparse triangular solves.

            x = Pᵀ L⁻ᵀ L⁻¹ P b

        Args:
            operator: Symmetric positive-definite `SparseOperator` A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            Solution x, of length N.

        Raises:
            TypeError: If ``operator`` is not a `SparseOperator`.
        """
        return self.factor(operator).solve(vector)

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Exact log-determinant from the diagonal of the Cholesky factor.

            log det A = 2 ∑ₙ log L_nn

        Args:
            operator: Symmetric positive-definite `SparseOperator` A, N × N.
            key: Unused; present for protocol compatibility.

        Returns:
            Scalar log det A.

        Raises:
            TypeError: If ``operator`` is not a `SparseOperator`.
        """
        return self.factor(operator).logdet()

    def diag_inv(self, operator: lx.AbstractLinearOperator) -> Float[Array, " n"]:
        """Exact diag(A⁻¹), the marginal variances, by one Takahashi sweep.

            diag(A⁻¹)_n = [A⁻¹]_nn,   n = 1, …, N

        Args:
            operator: Symmetric positive-definite `SparseOperator` A, N × N.

        Returns:
            The N diagonal entries of A⁻¹.

        Raises:
            TypeError: If ``operator`` is not a `SparseOperator`.
        """
        return self.factor(operator).diag_inv()
