"""Sparse Cholesky solver strategy for `SparseOperator` precisions."""

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

    For a symmetric positive-definite `SparseOperator` (a GMRF precision, a
    Laplace Hessian ``Q + AᵀWA``). The symbolic analysis is cached per
    sparsity pattern, so only the numeric factorisation runs per call; its
    gradients are exact (Takahashi for ``logdet``, an adjoint solve for
    ``solve``). Pass it explicitly: `AutoSolver` keeps choosing CG for large
    PSD operators, and nothing picks this strategy by size.

    Attributes:
        ordering: Fill-reducing ordering, ``"rcm"``, ``"natural"`` or
            ``"amd"`` (CHOLMOD). See `gaussx.symbolic_cholesky`.
        backend: ``"jax"`` or ``"cholmod"`` (opt-in, needs scikit-sparse).

    Examples:
        ```python
        import jax.numpy as jnp
        import lineax as lx
        import numpy as np
        import gaussx

        # Path-graph Laplacian plus a unit shift
        n = 5
        Q = gaussx.SparseOperator.from_coo(
            np.r_[np.arange(n), np.arange(1, n)],
            np.r_[np.arange(n), np.arange(n - 1)],
            jnp.r_[jnp.array([2.0, 3.0, 3.0, 3.0, 2.0]), -jnp.ones(n - 1)],
            (n, n),
            symmetric=True,
            tags=lx.positive_semidefinite_tag,
        )
        solver = gaussx.SparseCholeskySolver()
        x = solver.solve(Q, jnp.ones(n))
        ld = solver.logdet(Q)
        variances = gaussx.diag_inv(Q, solver=solver)  # Takahashi
        ```
    """

    ordering: Ordering = eqx.field(static=True, default="rcm")
    backend: Backend = eqx.field(static=True, default="jax")

    def factor(self, operator: lx.AbstractLinearOperator) -> SparseCholeskyFactor:
        """The sparse Cholesky factor of ``operator``.

        Args:
            operator: A `SparseOperator` (or a lineax-tagged one).

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
        """Solve ``A x = b`` with two sparse triangular solves.

        Args:
            operator: A symmetric positive-definite `SparseOperator`.
            vector: Right-hand side ``b``, shape ``(n,)``.

        Returns:
            Solution ``x``, shape ``(n,)``.
        """
        return self.factor(operator).solve(vector)

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Exact ``log|A| = 2 Σ log L_jj``.

        Args:
            operator: A symmetric positive-definite `SparseOperator`.
            key: Unused; present for protocol compatibility.

        Returns:
            Scalar log-determinant.
        """
        return self.factor(operator).logdet()

    def diag_inv(self, operator: lx.AbstractLinearOperator) -> Float[Array, " n"]:
        """Exact ``diag(A⁻¹)`` (marginal variances) by one Takahashi sweep.

        Args:
            operator: A symmetric positive-definite `SparseOperator`.

        Returns:
            Shape ``(n,)``.
        """
        return self.factor(operator).diag_inv()
