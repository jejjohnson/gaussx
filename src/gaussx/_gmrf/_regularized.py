"""Factorisation of a precision that is singular outside a constrained subspace.

A field with a hard constraint Vᵀ x = 0 may have a precision Λ that is positive
definite on {Vᵀ x = 0} only: the posterior of two intrinsic parts read by one
predictor, or of an intrinsic part that no observation reads. Nothing about the
constrained field changes when Λ is replaced by

    Λ' = Λ + V B Vᵀ,   B a positive diagonal,

since the added term is zero on the subspace, and Λ' is positive definite, so the
kriging formulas apply to it. Λ' is dense, but it is never formed. With P nodes that
determine ker Λ (x ∈ ker Λ and x_P = 0 give x = 0), E their unit columns and K a
positive diagonal, M = Λ + E K Eᵀ is positive definite on Λ's sparsity pattern, and

    Λ' = M + U C Uᵀ,   U = [E V],   C = diag(−K, B),

so one factorisation of M, its solves W = M⁻¹ U and the small capacitance matrix
T = C⁻¹ + Uᵀ W give Λ'⁻¹ = M⁻¹ − W T⁻¹ Wᵀ and det Λ' = det M · det T · det C.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Int

from gaussx._einx import einsum, rearrange, reduce
from gaussx._linalg._diag_inv import diag_inv
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._grounding import add_diagonal
from gaussx._operators._sparse import SparseOperator
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._diag import diag
from gaussx._primitives._logdet import logdet
from gaussx._primitives._solve import solve
from gaussx._sparse._factor import SparseCholeskyFactor
from gaussx._strategies._sparse_cholesky import SparseCholeskySolver


# ---------------------------------------------------------------------------
# Factorisations of a positive-definite precision
# ---------------------------------------------------------------------------


class CholeskyFactor(eqx.Module):
    """Λ = L Lᵀ with a structured lower factor (block, diagonal or dense)."""

    operator: lx.AbstractLinearOperator
    lower: lx.AbstractLinearOperator

    def solve(self, b: Float[Array, " N"]) -> Float[Array, " N"]:
        # Lᵀ, formed first: the order of the trace fixes reverse-mode accumulation
        upper = self.lower.T
        # L⁻¹ b
        forward = solve(self.lower, b)
        # Λ⁻¹ b = L⁻ᵀ L⁻¹ b
        return solve(upper, forward)

    def logdet(self) -> Float[Array, ""]:
        # diag(L)
        diagonal = diag(self.lower)
        # log diag(L)
        log_diagonal = jnp.log(diagonal)
        # jnp, not einx: einx cannot resolve an empty factor (D_x = 0).
        # log det Λ = 2 ∑_d log [L]_dd
        return 2.0 * jnp.sum(log_diagonal)

    def diag_inv(self) -> Float[Array, " N"]:
        # diag(Λ⁻¹), by the structured selected inverse
        return diag_inv(self.operator)


class OperatorFactor(eqx.Module):
    """No factor: `gaussx.solve` and `gaussx.logdet` dispatch on Λ itself."""

    operator: lx.AbstractLinearOperator

    def solve(self, b: Float[Array, " N"]) -> Float[Array, " N"]:
        # Λ⁻¹ b
        return solve(self.operator, b)

    def logdet(self) -> Float[Array, ""]:
        # log det Λ
        return logdet(self.operator)

    def diag_inv(self) -> Float[Array, " N"]:
        # diag(Λ⁻¹)
        return diag_inv(self.operator)


_CHOLESKY = (BlockTriDiag, lx.MatrixLinearOperator, lx.DiagonalLinearOperator)


def precision_factor(precision: lx.AbstractLinearOperator) -> Any:
    """Factorise a positive-definite Λ by the cheapest route its class allows.

    Args:
        precision: The operator Λ.

    Returns:
        An object with ``solve(b)``, ``logdet()`` and ``diag_inv()``: a
        `SparseCholeskyFactor`, a `CholeskyFactor`, or an `OperatorFactor` that
        dispatches on Λ.
    """
    if isinstance(precision, SparseOperator):
        # The symbolic analysis is cached per pattern: every Newton step and
        # every θ reuses it, and only the numeric factorisation runs.
        return SparseCholeskySolver().factor(precision)
    if isinstance(precision, _CHOLESKY):
        # Λ = L Lᵀ
        return CholeskyFactor(precision, cholesky(precision))
    return OperatorFactor(precision)


# ---------------------------------------------------------------------------
# A precision singular outside {Vᵀ x = 0}
# ---------------------------------------------------------------------------


class RegularizedFactor(eqx.Module):
    """Λ' = Λ + V B Vᵀ as M + U C Uᵀ, for a Λ singular outside {Vᵀ x = 0}.

        Λ'⁻¹ = M⁻¹ − W T⁻¹ Wᵀ,   W = M⁻¹ U,   T = C⁻¹ + Uᵀ W

    ``solve``, ``logdet``, ``diag_inv`` and ``selected_inverse`` are those of Λ'.
    Kriged onto Vᵀ x = 0 they give the constrained field of Λ, whatever the
    positive K and B; before that they depend on both.

    Attributes:
        grounded: M = Λ + E K Eᵀ.
        base: Its factorisation.
        border: Uᵀ, one row per border column: the D_p nodes, then the D_c
            constraints.
        solved_border: Wᵀ.
        capacitance: T, symmetric and indefinite.
        node_precision: diag(K).
        constraint_precision: diag(B).
        log_det_border: log |det C|.
    """

    grounded: lx.AbstractLinearOperator
    base: Any
    border: Float[Array, "k N"]
    solved_border: Float[Array, "k N"]
    capacitance: Float[Array, "k k"]
    node_precision: Float[Array, " p"]
    constraint_precision: Float[Array, " c"]
    log_det_border: Float[Array, ""]

    @property
    def symbolic(self) -> Any:
        """The symbolic analysis of M, for a sparse Λ."""
        return self.base.symbolic

    def _correction(self) -> Float[Array, "k N"]:
        """T⁻¹ Wᵀ."""
        return jnp.linalg.solve(self.capacitance, self.solved_border)

    def solve(self, b: Float[Array, " N"]) -> Float[Array, " N"]:
        """Λ'⁻¹ b = z − W T⁻¹ Uᵀ z, with z = M⁻¹ b."""
        # M⁻¹ b
        solved = self.base.solve(b)
        # Uᵀ z
        projection = einsum(self.border, solved, "border state, state -> border")
        # T⁻¹ Uᵀ z
        coeffs = jnp.linalg.solve(self.capacitance, projection)
        # z − W T⁻¹ Uᵀ z
        return solved - einsum(
            self.solved_border, coeffs, "border state, border -> state"
        )

    def solve_constraints(self) -> Float[Array, "c N"]:
        """Vᵀ Λ'⁻¹, one row per constraint, with no solve beyond the border's."""
        num_nodes = self.node_precision.shape[0]
        # Vᵀ M⁻¹
        solved = self.solved_border[num_nodes:]
        # Uᵀ M⁻¹ V
        projection = einsum(
            self.border, solved, "border state, con state -> border con"
        )
        # T⁻¹ Uᵀ M⁻¹ V
        coeffs = jnp.linalg.solve(self.capacitance, projection)
        # Vᵀ M⁻¹ − (Uᵀ M⁻¹ V)ᵀ T⁻¹ Wᵀ
        return solved - einsum(
            coeffs, self.solved_border, "border con, border state -> con state"
        )

    def logdet(self) -> Float[Array, ""]:
        """log det Λ' = log det M + log |det T| + log |det C|."""
        _, log_det_capacitance = jnp.linalg.slogdet(self.capacitance)
        return self.base.logdet() + log_det_capacitance + self.log_det_border

    def diag_inv(self) -> Float[Array, " N"]:
        """diag(Λ'⁻¹) = diag(M⁻¹) − diag(W T⁻¹ Wᵀ)."""
        products = self.solved_border * self._correction()
        return self.base.diag_inv() - reduce(products, "border state -> state", "sum")

    def selected_inverse(self) -> SparseOperator:
        """Λ'⁻¹ on the pattern of M's selected inverse, for a sparse Λ."""
        selected = self.base.selected_inverse()
        rows, cols = selected.pattern.rows, selected.pattern.cols
        # [W T⁻¹ Wᵀ]_ab, one term per stored entry
        correction = einsum(
            self.solved_border[:, rows],
            self._correction()[:, cols],
            "border entry, border entry -> entry",
        )
        return SparseOperator(
            selected.values - correction, selected.pattern, tags=selected.tags
        )

    def draw_correction(
        self, solved_constraints: Float[Array, "c N"], kriging_rows: Float[Array, "c N"]
    ) -> Float[Array, "p N"]:
        """Rows Fᵀ with C = C_M + F Fᵀ, what a kriged draw from M⁻¹ lacks.

            C_M = M⁻¹ − M⁻¹ A_cᵀ S_M⁻¹ A_c M⁻¹
            C = C_M + C_M E (K⁻¹ − Eᵀ C_M E)⁻¹ Eᵀ C_M

        with C the constrained covariance of Λ and S_M = A_c M⁻¹ A_cᵀ. The matrix in
        brackets is positive definite because Λ is, on the constrained subspace.

        Args:
            solved_constraints: A_c M⁻¹, one row per constraint.
            kriging_rows: S_M⁻¹ A_c M⁻¹.

        Returns:
            Fᵀ, one row per node of P.
        """
        num_nodes = self.node_precision.shape[0]
        nodes_border = self.border[:num_nodes]
        # Eᵀ M⁻¹
        solved_nodes = self.solved_border[:num_nodes]
        # A_c M⁻¹ E, one row per node
        crossed = einsum(
            nodes_border, solved_constraints, "node state, con state -> node con"
        )
        # Eᵀ C_M
        constrained = solved_nodes - einsum(
            crossed, kriging_rows, "node con, con state -> node state"
        )
        # Eᵀ C_M E
        node_covariance = einsum(
            constrained, nodes_border, "row state, col state -> row col"
        )
        # K⁻¹ − Eᵀ C_M E = L Lᵀ
        gap = jnp.diag(1.0 / self.node_precision) - node_covariance
        symmetric_gap = 0.5 * (gap + rearrange(gap, "row col -> col row"))
        lower = jnp.linalg.cholesky(symmetric_gap)
        # L⁻¹ Eᵀ C_M
        return jax.scipy.linalg.solve_triangular(lower, constrained, lower=True)


def is_groundable(precision: lx.AbstractLinearOperator) -> bool:
    """Whether Λ takes a diagonal at a few nodes and keeps its factorisation."""
    return isinstance(precision, SparseOperator | lx.MatrixLinearOperator) or (
        isinstance(precision, BlockTriDiag) and precision.symmetric
    )


def regularized_factor(
    precision: lx.AbstractLinearOperator,
    constraint_basis: Float[Array, "N c"],
    nodes: Int[Array, " p"],
    solver: SparseCholeskySolver | None = None,
) -> RegularizedFactor:
    """Factorise Λ' = Λ + V B Vᵀ for a Λ that is singular outside {Vᵀ x = 0}.

    K and B follow the diagonal of Λ, which keeps M as well conditioned as the data
    allow; the constrained field does not depend on them, so no gradient flows
    through them.

    Args:
        precision: Λ, a `SparseOperator`, a symmetric `BlockTriDiag` or a dense
            operator (`is_groundable`).
        constraint_basis: The constraint basis V, of full column rank.
        nodes: Nodes P on which every vector of ker Λ is determined.
        solver: The sparse Cholesky strategy; the default one if ``None``.

    Returns:
        The `RegularizedFactor`.
    """
    diagonal = jax.lax.stop_gradient(diag(precision))
    typical = jnp.mean(diagonal)
    # K, the diagonal of Λ at P; its mean at a node Λ does not touch
    node_precision = jnp.where(diagonal[nodes] > 0.0, diagonal[nodes], typical)
    frozen_basis = jax.lax.stop_gradient(constraint_basis)
    squared_basis = frozen_basis**2
    # ∑_i [V]²_ic [Λ]_ii and ∑_i [V]²_ic
    weighted = einsum(squared_basis, diagonal, "state con, state -> con")
    square_norm = reduce(squared_basis, "state con -> con", "sum")
    # B, so that V B Vᵀ has the size of Λ on each constraint
    constraint_precision = (
        jnp.where(weighted > 0.0, weighted, typical * square_norm) / square_norm**2
    )
    size = precision.in_size()
    ground_diagonal = jnp.zeros(size, diagonal.dtype).at[nodes].set(node_precision)
    # M = Λ + E K Eᵀ
    grounded = add_diagonal(precision, ground_diagonal)
    if isinstance(grounded, SparseOperator) and solver is not None:
        base = solver.factor(grounded)
    else:
        base = precision_factor(grounded)
    node_rows = jnp.zeros((nodes.shape[0], size), diagonal.dtype)
    node_rows = node_rows.at[jnp.arange(nodes.shape[0]), nodes].set(1.0)
    constraint_rows = rearrange(constraint_basis, "state con -> con state")
    # Uᵀ = [E V]ᵀ
    border = jnp.concatenate([node_rows, constraint_rows])
    # Wᵀ = (M⁻¹ U)ᵀ
    solved_border = jax.vmap(base.solve)(border)
    # Uᵀ M⁻¹ U
    border_covariance = einsum(border, solved_border, "row state, col state -> row col")
    # diag(C⁻¹) = (−1 / K, 1 / B)
    inverse_diagonal = jnp.concatenate(
        [-1.0 / node_precision, 1.0 / constraint_precision]
    )
    capacitance = jnp.diag(inverse_diagonal) + border_covariance
    # log |det C|
    log_det_border = -jnp.sum(jnp.log(jnp.abs(inverse_diagonal)))
    return RegularizedFactor(
        grounded,
        base,
        border,
        solved_border,
        capacitance,
        node_precision,
        constraint_precision,
        log_det_border,
    )


def has_selected_inverse(factor: Any) -> bool:
    """Whether a factor has the sparse selected inverse of its precision."""
    return isinstance(factor, SparseCholeskyFactor) or (
        isinstance(factor, RegularizedFactor)
        and isinstance(factor.base, SparseCholeskyFactor)
    )
