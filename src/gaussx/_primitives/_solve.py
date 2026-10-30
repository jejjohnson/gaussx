"""Structured solve A x = b, dispatched on the operator type of A.

A is a square N × N operator, b the right-hand side and x the solution. A
Kronecker product is A = A_1 ⊗ … ⊗ A_D with square factors A_d,
d = 1, …, D, and a Kronecker sum is A_1 ⊕ A_2 = A_1 ⊗ I + I ⊗ A_2. A
symmetric factor has the eigendecomposition A_d = Q_d diag(γ_d) Q_dᵀ, with
orthonormal eigenvectors Q_d and eigenvalues γ_d; a diagonalised operator is
A = W⁻¹ diag(γ) W for a forward transform W (an FFT, say). A low-rank update
is A = A₀ + U D Vᵀ, with base A₀, N × k factors U and V and D = diag(d), and
C = I + D Vᵀ A₀⁻¹ U is its k × k capacitance scaled by D. L is a lower
Cholesky factor, A = L Lᵀ.
"""

from __future__ import annotations

import functools as ft
from typing import TYPE_CHECKING

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import (
    BlockTriDiag,
    LowerBlockTriDiag,
    UpperBlockTriDiag,
)
from gaussx._operators._diagonalised import DiagonalizedOperator, as_diagonalized
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import (
    KroneckerSum,
    KroneckerSumSqrt,
    _eigh_factor,
)
from gaussx._operators._low_rank_update import (
    LowRankUpdate,
    orthonormal_scaled_identity,
)
from gaussx._operators._masked import MaskedOperator
from gaussx._operators._sparse import SparseOperator
from gaussx._operators._spectral_function import SpectralFunction
from gaussx._operators._sum_kronecker import (
    SumOfKroneckers,
    _sum_of_kroneckers_solve,
)
from gaussx._primitives._cholesky import cholesky, warn_dense_fallback
from gaussx._primitives._inv import InverseOperator


if TYPE_CHECKING:
    from gaussx._strategies._base import AbstractSolveStrategy


def solve(
    operator: lx.AbstractLinearOperator,
    vector: Float[Array, " n"],
    *,
    solver: lx.AbstractLinearSolver | AbstractSolveStrategy | None = None,
) -> Float[Array, " n"]:
    """Solve A x = b with structural dispatch on the operator type.

        x = A⁻¹ b

    Operators with no structural rule take a dense fallback. With the
    default ``solver=None`` that fallback has numpy semantics: a singular or
    numerically singular A gives a non-finite solution (``nan`` or ``inf``),
    in the value and in its gradients, rather than an error, as
    `gaussx.logdet` already has. A log-density built on it is then
    non-finite too, which a sampler or optimiser can reject; an exception
    would abort the whole run. An explicit ``solver`` keeps lineax's
    behaviour instead, which raises on a singular system or an iterative
    solver that does not converge.

    Structured rules keep their own semantics for singular operators: a
    singular `gaussx.DiagonalizedOperator`, for example, returns the
    minimum-norm pseudo-inverse solution.

    Algorithm:
        1. Hand the whole solve to a gaussx strategy when one is given.
        2. Otherwise take the first structural rule that matches A: a
           diagonal divides, a Kronecker product solves per factor, a
           low-rank update applies the Woodbury identity, a symmetric
           Kronecker sum rotates into its factors' eigenbases, a
           block-tridiagonal operator runs a banded Cholesky, and scalar,
           tagged and composed wrappers recurse into their parts.
        3. Fall back to a dense direct solve (or the given lineax solver).

    Args:
        operator: The square operator A.
        vector: The right-hand side b.
        solver: A lineax solver, which overrides the dense fallback and is
            threaded into the structural rules (e.g. used per Kronecker
            factor); or a gaussx strategy such as `gaussx.CGSolver`, which
            then owns the whole solve (gh-376). ``None`` uses a dense direct
            solve chosen from the operator's tags (Cholesky for positive
            semi-definite, triangular, else LU).

    Returns:
        The solution x. On the default dense fallback, non-finite when A is
        singular.

    Raises:
        TypeError: If *solver* is neither a lineax solver nor a gaussx
            strategy.

    Examples:
        A Kronecker product of diagonals is solved one factor at a time:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A_1 = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> A_2 = lx.DiagonalLinearOperator(jnp.array([4.0, 5.0]))
        >>> A = gaussx.Kronecker(A_1, A_2)  # diag(4, 5, 8, 10)
        >>> x = gaussx.solve(A, jnp.ones(4))
        >>> [round(float(v), 4) for v in x]
        [0.25, 0.2, 0.125, 0.1]

        A dense positive-definite matrix takes the Cholesky fallback:

        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> b = jnp.array([1.0, 2.0])
        >>> x = gaussx.solve(A, b)
        >>> bool(jnp.allclose(matrix @ x, b, atol=1e-5))
        True

        A singular matrix gives a non-finite solution instead of raising:

        >>> singular = lx.MatrixLinearOperator(jnp.zeros((2, 2)))
        >>> bool(jnp.all(jnp.isfinite(gaussx.solve(singular, b))))
        False
    """
    if solver is not None and not isinstance(solver, lx.AbstractLinearSolver):
        # lazy import, cycle: _strategies._base -> _strategies -> _strategies._dense ->
        #   _primitives._solve
        from gaussx._strategies._base import AbstractSolveStrategy

        if isinstance(solver, AbstractSolveStrategy):
            return solver.solve(operator, vector)
        raise TypeError(
            "solver must be a lineax.AbstractLinearSolver (e.g. lineax.CG(...)) "
            "or a gaussx.AbstractSolveStrategy (e.g. gaussx.CGSolver()); got "
            f"{type(solver).__name__}."
        )
    if isinstance(operator, lx.IdentityLinearOperator):
        # x = I⁻¹ b = b
        return vector
    if isinstance(operator, InverseOperator):
        # (A⁻¹)⁻¹ b = A b: one matvec, no factorisation (gh-349).
        return operator.original.mv(vector)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _solve_diagonal(operator, vector)
    if isinstance(operator, DiagonalizedOperator):
        return _solve_diagonalised(operator, vector)
    if isinstance(operator, MaskedOperator) and operator.capacitance is not None:
        return _solve_masked(operator, vector)
    if isinstance(operator, BlockDiag):
        return _solve_block_diag(operator, vector, solver)
    if isinstance(operator, Kronecker):
        return _solve_kronecker(operator, vector, solver)
    if isinstance(operator, LowRankUpdate):
        if operator.rank == 0:
            # x = A₀⁻¹ b when k = 0
            return solve(operator.base, vector, solver=solver)
        shift = orthonormal_scaled_identity(operator)
        if shift is not None:
            return _solve_low_rank_orthonormal(operator, shift, vector)
        return _solve_low_rank(operator, vector, solver)
    if isinstance(operator, SumOfKroneckers):
        return _solve_sum_of_kroneckers(operator, vector, solver)
    if isinstance(operator, KroneckerSum):
        return _solve_kronecker_sum(operator, vector, solver)
    if isinstance(operator, KroneckerSumSqrt):
        return operator.solve(vector)
    if isinstance(operator, SpectralFunction):
        return operator.solve(vector)
    if isinstance(operator, BlockTriDiag) and operator.symmetric:
        # Non-symmetric diagonal blocks take the dense LU fallback (gh-344).
        return _solve_block_tridiag(operator, vector)
    if isinstance(operator, SparseOperator):
        return _solve_sparse(operator, vector, solver)
    if isinstance(operator, LowerBlockTriDiag):
        return _solve_lower_block_tridiag(operator, vector)
    if isinstance(operator, UpperBlockTriDiag):
        return _solve_upper_block_tridiag(operator, vector)
    if isinstance(operator, lx.TaggedLinearOperator):
        return _solve_tagged(operator, vector, solver)
    if isinstance(operator, lx.MulLinearOperator):
        # (c A)⁻¹ b = A⁻¹ b / c
        return solve(operator.operator, vector, solver=solver) / operator.scalar
    if isinstance(operator, lx.DivLinearOperator):
        # (A / c)⁻¹ b = c A⁻¹ b
        return solve(operator.operator, vector, solver=solver) * operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        # (−A)⁻¹ b = −A⁻¹ b
        return -solve(operator.operator, vector, solver=solver)
    if isinstance(operator, lx.ComposedLinearOperator):
        # y = A⁻¹ b, for the composition A B
        intermediate = solve(operator.operator1, vector, solver=solver)
        # x = (A B)⁻¹ b = B⁻¹ y
        return solve(operator.operator2, intermediate, solver=solver)
    if isinstance(operator, lx.AddLinearOperator):
        # ``sum_operator`` builds native lineax sums rather than a
        # `SumOfKroneckers`, so the same reduction has to be reachable here.
        solution = _sum_of_kroneckers_solve(operator, vector)
        if solution is not None:
            return solution
    return _solve_fallback(operator, vector, solver)


@eqx.filter_custom_jvp
def _solve_masked(
    operator: MaskedOperator,
    vector: Float[Array, " m"],
) -> Float[Array, " m"]:
    """Solve the masked system A₀[m, m] x = b by the capacitance method.

        x = y[m],   y with A₀ y = b on the rows kept by the constraints

    m is the set of masked-in indices. The base solve is built here from the
    ``base`` leaf rather than stored (see `MaskedOperator`).

    Algorithm:
        1. Scatter b into the full index space, zero elsewhere.
        2. Solve on the full grid with the operator's precomputed
           capacitance solver, which pins the coupled masked-out entries
           of y to zero.
        3. Gather the masked-in entries y[m].

    Args:
        operator: Masked operator A₀[m, m] with a precomputed capacitance.
        vector: Right-hand side b on the masked-in indices.

    Returns:
        The solution x on the masked-in indices.
    """
    assert operator.capacitance is not None
    full_size = operator.base.in_size()
    masked_in_indices = jnp.where(operator.col_mask, size=operator.in_size())[0]
    full_zeros = jnp.zeros(
        full_size, dtype=jnp.result_type(vector, operator.out_structure().dtype)
    )
    base_solve = ft.partial(solve, operator.base)
    # ``unique_indices`` (true: they come from a mask) makes the scatter
    # transposable, which reverse mode needs for the implicit JVP below.
    scattered_rhs = full_zeros.at[masked_in_indices].set(vector, unique_indices=True)
    # y on the full grid, through the capacitance solver
    full_solution = operator.capacitance(scattered_rhs, base_solve=base_solve)
    # x = y[m]
    return full_solution[masked_in_indices]


@_solve_masked.def_jvp
def _solve_masked_jvp(primals, tangents):
    """Implicit JVP of the masked solve, dx = A⁻¹ (db − dA x) (gh-290).

        dx = A⁻¹ (db − dA x),   A = A₀[m, m]

    The capacitance factorisation is cached from the base, so differentiating
    through it would miss the base's contribution to the capacitance; the
    solution depends on A₀ only through A, so the implicit rule is exact. It
    costs one more masked solve and the tangent operator's matvec.

    Args:
        primals: The pair ``(operator, vector)``, A and b.
        tangents: Their tangents ``(t_operator, t_vector)``, dA and db.

    Returns:
        Tuple ``(solution, tangent_solution)`` holding x and dx.
    """
    operator, vector = primals
    t_operator, t_vector = tangents
    # x = A⁻¹ b
    solution = _solve_masked(operator, vector)
    tangent_rhs = jnp.zeros_like(solution) if t_vector is None else t_vector
    if jax.tree_util.tree_leaves(t_operator):
        # dA x
        _, tangent_product = eqx.filter_jvp(
            lambda op: op.mv(solution), (operator,), (t_operator,)
        )
        # db − dA x
        tangent_rhs = tangent_rhs - tangent_product
    # dx = A⁻¹ (db − dA x)
    return solution, _solve_masked(operator, tangent_rhs)


def _solve_diagonalised(
    operator: DiagonalizedOperator,
    vector: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Solve A x = b for A = W⁻¹ diag(γ) W in the transform domain.

        x = W⁻¹ (W b / γ),   with 1 / γ_i replaced by 0 where γ_i = 0

    For a normal operator (unitary W) that is the Moore-Penrose
    pseudo-inverse, i.e. the minimum-norm solution with the null-space
    component of b projected out: the usual gauge for periodic or
    pure-Neumann Poisson problems.

    Args:
        operator: Diagonalised operator A = W⁻¹ diag(γ) W.
        vector: Right-hand side b.

    Returns:
        The (pseudo-inverse) solution x, real when the operator's output is.
    """
    eigenvalues = operator.eigenvalues_flat()
    is_zero = eigenvalues == 0
    # 1 / γ, with exactly-zero eigenvalues mapped to 0
    safe_eigenvalues = jnp.where(is_zero, 1.0, eigenvalues)
    inverse_eigenvalues = jnp.where(is_zero, 0.0, 1.0 / safe_eigenvalues)
    # W b
    transformed = operator.forward_flat(vector)
    # W b / γ
    scaled = transformed * inverse_eigenvalues
    # x = W⁻¹ (W b / γ)
    solution = operator.inverse_flat(scaled)
    return jnp.real(solution) if operator.real_output else solution


def _solve_diagonal(
    operator: lx.DiagonalLinearOperator,
    vector: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Solve diag(a) x = b elementwise.

        x_i = b_i / a_i

    Args:
        operator: Diagonal operator diag(a).
        vector: Right-hand side b.

    Returns:
        The solution x.
    """
    diagonal = lx.diagonal(operator)
    # x_i = b_i / a_i
    return vector / diagonal


def _solve_block_diag(
    operator: BlockDiag,
    vector: Float[Array, " n"],
    solver: lx.AbstractLinearSolver | None,
) -> Float[Array, " n"]:
    """Solve a block-diagonal system one block at a time.

        x = [A_1⁻¹ b_1; …; A_D⁻¹ b_D]

    Args:
        operator: Block-diagonal operator with blocks A_d.
        vector: Right-hand side b, split into the blocks b_d.
        solver: Lineax solver threaded into each block solve, or ``None``.

    Returns:
        The stacked solution x.
    """
    block_solutions = []
    offset = 0
    for block_operator in operator.operators:
        size = block_operator.in_size()
        # b_d
        block_rhs = jax.lax.dynamic_slice(vector, (offset,), (size,))
        # x_d = A_d⁻¹ b_d
        block_solutions.append(solve(block_operator, block_rhs, solver=solver))
        offset += size
    # x = [x_1; …; x_D]
    return jnp.concatenate(block_solutions)


def _solve_kronecker(
    operator: Kronecker,
    vector: Float[Array, " n"],
    solver: lx.AbstractLinearSolver | None,
) -> Float[Array, " n"]:
    """Solve (A_1 ⊗ … ⊗ A_D) x = b with one solve per factor.

        (A_1 ⊗ … ⊗ A_D)⁻¹ = A_1⁻¹ ⊗ … ⊗ A_D⁻¹

    Algorithm:
        1. For d = D, …, 1, unfold the running vector into a matrix whose
           columns run over axis d.
        2. Solve every row against A_d, which applies A_d⁻¹ along axis d.
        3. Flatten with axis d moved to the front, so that the next factor's
           axis is last; after D steps the axes are back in order.

    Args:
        operator: Kronecker product A_1 ⊗ … ⊗ A_D.
        vector: Right-hand side b.
        solver: Lineax solver threaded into each factor solve, or ``None``.

    Returns:
        The solution x.
    """
    solution = vector
    for index in range(len(operator.operators) - 1, -1, -1):
        factor = operator.operators[index]
        factor_size = factor.in_size()
        # X: the running vector with axis d as its columns
        unfolded = rearrange(solution, "(rest col) -> rest col", col=factor_size)
        solve_factor = ft.partial(solve, factor, solver=solver)
        # X A_d⁻ᵀ, one solve A_d y = X_r per row
        factor_solved = jax.vmap(solve_factor)(unfolded)
        # vec((X A_d⁻ᵀ)ᵀ), axis d now leading
        solution = rearrange(factor_solved, "rest col -> (col rest)")
    return solution


def _low_rank_capacitance(
    operator: LowRankUpdate,
    solver: lx.AbstractLinearSolver | None,
) -> tuple[Float[Array, "n k"], Float[Array, "k k"]]:
    """Shared Woodbury pieces of A = A₀ + U D Vᵀ: A₀⁻¹ U and C.

        C = I + D Vᵀ A₀⁻¹ U

    C is the capacitance D⁻¹ + Vᵀ A₀⁻¹ U scaled by D, so no reciprocal of
    the weights is ever taken and a zero weight (a truncated SVD or Nyström
    factor past its numerical rank) stays exact (gh-307). Used by the
    low-rank ``solve``, ``logdet`` and ``inv`` primitives so the capacitance
    construction lives in exactly one place.

    Args:
        operator: Low-rank update A₀ + U D Vᵀ.
        solver: Lineax solver for the base solves, or ``None``.

    Returns:
        Tuple ``(base_inv_factor, capacitance)`` holding A₀⁻¹ U and C.
    """
    # A₀⁻¹ U, one solve per column of U
    base_inv_factor = jax.vmap(
        lambda column: solve(operator.base, column, solver=solver),
        in_axes=1,
        out_axes=1,
    )(operator.U)
    # Vᵀ A₀⁻¹ U
    projected = einsum(operator.V, base_inv_factor, "row left, row right -> left right")
    # D Vᵀ A₀⁻¹ U
    scaled = einx.multiply("left, left right -> left right", operator.d, projected)
    identity = jnp.eye(operator.rank, dtype=projected.dtype)
    # C = I + D Vᵀ A₀⁻¹ U
    capacitance = identity + scaled
    return base_inv_factor, capacitance


def _solve_low_rank(
    operator: LowRankUpdate,
    vector: Float[Array, " n"],
    solver: lx.AbstractLinearSolver | None,
) -> Float[Array, " n"]:
    """Solve (A₀ + U D Vᵀ) x = b by the Woodbury identity.

        (A₀ + U D Vᵀ)⁻¹ = A₀⁻¹ − A₀⁻¹ U C⁻¹ D Vᵀ A₀⁻¹,   C = I + D Vᵀ A₀⁻¹ U

    C is k × k and free of D⁻¹.

    Args:
        operator: Low-rank update A₀ + U D Vᵀ.
        vector: Right-hand side b.
        solver: Lineax solver for the base solves, or ``None``.

    Returns:
        The solution x.
    """
    # A₀⁻¹ b
    base_solution = solve(operator.base, vector, solver=solver)
    # A₀⁻¹ U and C
    base_inv_factor, capacitance = _low_rank_capacitance(operator, solver)
    # Vᵀ A₀⁻¹ b
    projected = einsum(operator.V, base_solution, "row rank, row -> rank")
    # D Vᵀ A₀⁻¹ b
    scaled_projected = operator.d * projected
    # C⁻¹ D Vᵀ A₀⁻¹ b
    coefficients = jnp.linalg.solve(capacitance, scaled_projected)
    # A₀⁻¹ U C⁻¹ D Vᵀ A₀⁻¹ b
    correction = base_inv_factor @ coefficients
    # x = A₀⁻¹ b − A₀⁻¹ U C⁻¹ D Vᵀ A₀⁻¹ b
    return base_solution - correction


def _solve_low_rank_orthonormal(
    operator: LowRankUpdate,
    c: Float[Array, ""],
    vector: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Solve (c I + U D Uᵀ) x = b in closed form for orthonormal U.

        x = (b − U diag(d / (c + d)) Uᵀ b) / c,   Uᵀ U = I

    O(N k), with no k × k factorisation (gh-333).

    Args:
        operator: Low-rank update c I + U D Uᵀ with orthonormal U.
        c: The scalar shift c of the base.
        vector: Right-hand side b.

    Returns:
        The solution x.
    """
    factor, weights = operator.U, operator.d
    # Uᵀ b
    projected = einsum(factor, vector, "row rank, row -> rank")
    # d / (c + d)
    shrinkage = weights / (c + weights)
    # diag(d / (c + d)) Uᵀ b
    scaled_projected = shrinkage * projected
    # U diag(d / (c + d)) Uᵀ b
    correction = factor @ scaled_projected
    # x = (b − U diag(d / (c + d)) Uᵀ b) / c
    return (vector - correction) / c


def _solve_sum_of_kroneckers(
    operator: SumOfKroneckers,
    vector: Float[Array, " n"],
    solver: lx.AbstractLinearSolver | None,
) -> Float[Array, " n"]:
    """Solve (∑_j A_j ⊗ B_j) x = b by simultaneous diagonalisation.

        (A_1 ⊗ B_1 + A_2 ⊗ B_2) x = b

    Exact for two terms with one of them positive definite, the
    B ⊗ C + σ² I and B₁ ⊗ C₁ + B₂ ⊗ C₂ shapes of multi-output GP train
    covariances, at O(N_A³ + N_B³) rather than the O((N_A N_B)³) of the
    dense fallback. Three or more terms have no closed form and keep the
    fallback; drive `SumOfKroneckers.mv` with ``solver=lineax.CG(...)`` to
    avoid materialising there.

    Args:
        operator: Sum of Kronecker products ∑_j A_j ⊗ B_j.
        vector: Right-hand side b.
        solver: Lineax solver for the fallback, or ``None``.

    Returns:
        The solution x.
    """
    solution = _sum_of_kroneckers_solve(operator, vector)
    if solution is None:
        if solver is None:
            _warn_sum_of_kroneckers_solve()
        return _solve_fallback(operator, vector, solver)
    return solution


def _warn_sum_of_kroneckers_solve() -> None:
    """Warn that a sum of Kronecker products is about to be materialised."""
    warn_dense_fallback(
        "solve(SumOfKroneckers) has no closed form here (three or more "
        "terms, or no positive-definite anchor) and materialises the "
        "operator; pass solver=lineax.CG(...) for a matrix-free solve."
    )


def _solve_kronecker_sum(
    operator: KroneckerSum,
    vector: Float[Array, " n"],
    solver: lx.AbstractLinearSolver | None,
) -> Float[Array, " n"]:
    """Solve (A_1 ⊕ A_2) x = b in the factors' eigenbases.

        A_1 ⊕ A_2 = (Q_1 ⊗ Q_2) diag(γ_1,i + γ_2,j) (Q_1 ⊗ Q_2)ᵀ

    When every factor is a `DiagonalizedOperator` (recursively), the solve
    goes through the composed transforms with no eigendecomposition at all;
    this covers non-symmetric factors via
    `DiagonalizedOperator.from_eigen_factorization`.

    The formula uses Qᵀ as the inverse rotation, which is only valid for
    symmetric factors, so the structured path is taken only when both
    factors are tagged symmetric (or are diagonal). Otherwise the operator
    goes to the generic lineax fallback: treating an untagged non-symmetric
    factor as symmetric used to return a silently wrong answer. For a fast
    solve with non-symmetric but diagonalisable factors, precompute
    `gaussx.EigenDecomposition` per factor and call
    `gaussx.kronecker_sum_solve`.

    Args:
        operator: Kronecker sum A_1 ⊕ A_2.
        vector: Right-hand side b.
        solver: Lineax solver for the fallback, or ``None``.

    Returns:
        The solution x.
    """
    diagonalised = as_diagonalized(operator)
    if diagonalised is not None:
        return _solve_diagonalised(diagonalised, vector)
    if not (lx.is_symmetric(operator.A) and lx.is_symmetric(operator.B)):
        if solver is None:
            warn_dense_fallback(
                "solve(KroneckerSum) with factors not tagged symmetric "
                "materialises the operator. Tag both factors with "
                "lineax.symmetric_tag, or wrap them as DiagonalizedOperator "
                "(e.g. DiagonalizedOperator.from_eigen_factorization)."
            )
        return _solve_fallback(operator, vector, solver)

    return _eigen_solve_kronecker_sum(operator, vector)


@eqx.filter_custom_jvp
def _eigen_solve_kronecker_sum(
    operator: KroneckerSum,
    vector: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Solve (A_1 ⊕ A_2) x = b for symmetric factors, in their eigenbases.

        X = Q_1 [(Q_1ᵀ B Q_2) ⊘ (γ_1,i + γ_2,j)] Q_2ᵀ,   b = vec(B), x = vec(X)

    Args:
        operator: Kronecker sum A_1 ⊕ A_2 with symmetric factors.
        vector: Right-hand side b.

    Returns:
        The solution x.
    """
    # A_1 = Q_1 diag(γ_1) Q_1ᵀ
    eigenvalues_1, eigenvectors_1 = _eigh_factor(operator.A)
    # A_2 = Q_2 diag(γ_2) Q_2ᵀ
    eigenvalues_2, eigenvectors_2 = _eigh_factor(operator.B)
    size_1, size_2 = operator._n_a, operator._n_b
    # B = unvec(b)
    rhs_matrix = rearrange(vector, "(a b) -> a b", a=size_1, b=size_2)
    # Q_1ᵀ B Q_2
    rotated = einsum(
        eigenvectors_1,
        rhs_matrix,
        eigenvectors_2,
        "row_a col_a, row_a row_b, row_b col_b -> col_a col_b",
    )
    # γ_1,i + γ_2,j
    eigenvalue_sums = einx.add("a, b -> a b", eigenvalues_1, eigenvalues_2)
    # (Q_1ᵀ B Q_2) ⊘ (γ_1,i + γ_2,j)
    scaled = rotated / eigenvalue_sums
    # X = Q_1 [(Q_1ᵀ B Q_2) ⊘ (γ_1,i + γ_2,j)] Q_2ᵀ
    solution_matrix = einsum(
        eigenvectors_1,
        scaled,
        eigenvectors_2,
        "row_a col_a, col_a col_b, row_b col_b -> row_a row_b",
    )
    # x = vec(X)
    return rearrange(solution_matrix, "a b -> (a b)")


@_eigen_solve_kronecker_sum.def_jvp
def _eigen_solve_kronecker_sum_jvp(primals, tangents):
    """Implicit JVP of the Kronecker-sum solve, dx = A⁻¹ (db − dA x) (gh-295).

        dx = A⁻¹ (db − dA x),   A = A_1 ⊕ A_2

    Differentiating through the factor ``eigh`` calls gives NaN whenever a
    factor has a repeated eigenvalue (an isotropic s I factor, say): the
    eigenvector derivative divides by eigenvalue gaps. The solution does not
    depend on how a degenerate eigenbasis is chosen, so the derivative
    written in terms of the operator alone is exact, at the cost of one more
    structured solve and the tangent operator's matvec. Same rule as
    `SumOfKroneckers`' ``_eigen_solve``.

    Args:
        primals: The pair ``(operator, vector)``, A and b.
        tangents: Their tangents ``(t_operator, t_vector)``, dA and db.

    Returns:
        Tuple ``(solution, tangent_solution)`` holding x and dx.
    """
    operator, vector = primals
    t_operator, t_vector = tangents
    # x = A⁻¹ b
    solution = _eigen_solve_kronecker_sum(operator, vector)
    tangent_rhs = jnp.zeros_like(vector) if t_vector is None else t_vector
    if jax.tree_util.tree_leaves(t_operator):
        # dA x
        _, tangent_product = eqx.filter_jvp(
            lambda op: op.mv(solution), (operator,), (t_operator,)
        )
        # db − dA x
        tangent_rhs = tangent_rhs - tangent_product
    # dx = A⁻¹ (db − dA x)
    return solution, _eigen_solve_kronecker_sum(operator, tangent_rhs)


def _solve_block_tridiag(
    operator: BlockTriDiag,
    vector: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Solve a symmetric block-tridiagonal system by banded Cholesky.

        A = L Lᵀ,   L y = b,   Lᵀ x = y

    Args:
        operator: Symmetric positive-definite block-tridiagonal A.
        vector: Right-hand side b.

    Returns:
        The solution x.
    """
    # A = L Lᵀ, L lower block-bidiagonal
    factor = cholesky(operator)
    # y = L⁻¹ b, forward substitution
    intermediate = solve(factor, vector)
    # x = L⁻ᵀ y, backward substitution
    return solve(factor.T, intermediate)


def _solve_lower_block_tridiag(
    operator: LowerBlockTriDiag,
    vector: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Forward substitution for a lower block-bidiagonal system.

        x_k = L_k⁻¹ (b_k − B_k−1 x_k−1),   k = 1, …, N_block

    L_k are the triangular diagonal blocks and B_k−1 the sub-diagonal block
    in block position (k, k−1); the k = 1 step has no B term.

    Args:
        operator: Lower block-bidiagonal operator with lower-triangular
            diagonal blocks L_k.
        vector: Right-hand side b, stacked over the blocks b_k.

    Returns:
        The solution x, stacked over the blocks x_k.
    """
    num_blocks = operator._num_blocks
    block_size = operator._block_size

    if num_blocks == 1:
        # Both lax.cond branches are traced, and the k > 0 one indexes the
        # empty sub-diagonal (gh-304).
        # x = L_1⁻¹ b
        return jax.scipy.linalg.solve_triangular(
            operator.diagonal[0], vector, lower=True
        )
    rhs_blocks = rearrange(
        vector, "(block d) -> block d", block=num_blocks, d=block_size
    )

    def forward_step(previous_solution, index):
        # B_k−1 x_k−1, zero at the first block
        coupling = jax.lax.cond(
            index > 0,
            lambda: operator.sub_diagonal[index - 1] @ previous_solution,
            lambda: jnp.zeros(block_size, dtype=rhs_blocks.dtype),
        )
        # b_k − B_k−1 x_k−1
        block_rhs = rhs_blocks[index] - coupling
        # x_k = L_k⁻¹ (b_k − B_k−1 x_k−1)
        block_solution = jax.scipy.linalg.solve_triangular(
            operator.diagonal[index], block_rhs, lower=True
        )
        return block_solution, block_solution

    _, solution_blocks = jax.lax.scan(
        forward_step,
        jnp.zeros(block_size, dtype=rhs_blocks.dtype),
        jnp.arange(num_blocks),
    )
    return rearrange(solution_blocks, "block d -> (block d)")


def _solve_upper_block_tridiag(
    operator: UpperBlockTriDiag,
    vector: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Backward substitution for an upper block-bidiagonal system.

        x_k = U_k⁻¹ (b_k − B_k x_k+1),   k = N_block, …, 1

    U_k are the upper-triangular diagonal blocks and B_k the super-diagonal
    block in block position (k, k+1); the k = N_block step has no B term.

    Args:
        operator: Upper block-bidiagonal operator with upper-triangular
            diagonal blocks U_k.
        vector: Right-hand side b, stacked over the blocks b_k.

    Returns:
        The solution x, stacked over the blocks x_k.
    """
    num_blocks = operator._num_blocks
    block_size = operator._block_size

    if num_blocks == 1:
        # Both lax.cond branches are traced; see _solve_lower_block_tridiag.
        # x = U_1⁻¹ b
        return jax.scipy.linalg.solve_triangular(
            operator.diagonal[0], vector, lower=False
        )
    rhs_blocks = rearrange(
        vector, "(block d) -> block d", block=num_blocks, d=block_size
    )

    def backward_step(next_solution, step):
        index = num_blocks - 1 - step
        # B_k x_k+1, zero at the last block
        coupling = jax.lax.cond(
            index < num_blocks - 1,
            lambda: operator.super_diagonal[index] @ next_solution,
            lambda: jnp.zeros(block_size, dtype=rhs_blocks.dtype),
        )
        # b_k − B_k x_k+1
        block_rhs = rhs_blocks[index] - coupling
        # x_k = U_k⁻¹ (b_k − B_k x_k+1)
        block_solution = jax.scipy.linalg.solve_triangular(
            operator.diagonal[index], block_rhs, lower=False
        )
        return block_solution, block_solution

    _, reversed_blocks = jax.lax.scan(
        backward_step,
        jnp.zeros(block_size, dtype=rhs_blocks.dtype),
        jnp.arange(num_blocks),
    )
    # the scan ran k = N_block, …, 1, so put the blocks back in order
    solution_blocks = einx.flip("[block] d", reversed_blocks)
    return rearrange(solution_blocks, "block d -> (block d)")


def _solve_tagged(
    operator: lx.TaggedLinearOperator,
    vector: Float[Array, " n"],
    solver: lx.AbstractLinearSolver | None,
) -> Float[Array, " n"]:
    """Unwrap a `lineax.TaggedLinearOperator` when the inner one is structured.

    Untagged inner operators stay wrapped so the lineax fallback solver can
    still exploit the tags (e.g. PSD → Cholesky).

    Args:
        operator: Tagged operator around A.
        vector: Right-hand side b.
        solver: Lineax solver threaded into the inner solve, or ``None``.

    Returns:
        The solution x = A⁻¹ b.
    """
    if isinstance(operator.operator, SumOfKroneckers | lx.AddLinearOperator):
        # ``sum_operator(..., tags=...)`` wraps the sum; unwrapping
        # unconditionally would cost the fallback its PSD tag, so only take
        # the structured path when the exact reduction actually applies.
        solution = _sum_of_kroneckers_solve(operator.operator, vector)
        if solution is not None:
            return solution
        if isinstance(operator.operator, SumOfKroneckers) and solver is None:
            _warn_sum_of_kroneckers_solve()
    structured = (
        lx.IdentityLinearOperator,
        lx.DiagonalLinearOperator,
        lx.TaggedLinearOperator,
        lx.MulLinearOperator,
        lx.DivLinearOperator,
        lx.NegLinearOperator,
        lx.ComposedLinearOperator,
        BlockDiag,
        Kronecker,
        LowRankUpdate,
        KroneckerSum,
        KroneckerSumSqrt,
        DiagonalizedOperator,
        BlockTriDiag,
        LowerBlockTriDiag,
        UpperBlockTriDiag,
        # ``solve`` falls back itself when there is no capacitance (gh-391).
        MaskedOperator,
    )
    if isinstance(operator.operator, structured):
        return solve(operator.operator, vector, solver=solver)
    return _solve_fallback(operator, vector, solver)


def _solve_sparse(
    operator: SparseOperator,
    vector: Float[Array, " n"],
    solver: lx.AbstractLinearSolver | None,
) -> Float[Array, " n"]:
    """Solve a sparse system by the `AutoSolver` rules: CG when large and PSD.

    An explicit ``solver`` always wins; otherwise CG runs when N exceeds the
    `AutoSolver` size threshold and A is positive semi-definite, and the
    dense fallback runs everywhere else. The exact sparse Cholesky path is
    the `SparseCholeskySolver` strategy, which callers pass explicitly.

    Args:
        operator: Sparse operator A.
        vector: Right-hand side b.
        solver: Lineax solver, or ``None`` for the rules above.

    Returns:
        The solution x.
    """
    # lazy import, cycle: _strategies._auto -> _primitives -> _primitives._solve
    from gaussx._strategies._auto import AutoSolver

    # lazy import, cycle: _strategies._cg -> _strategies -> _strategies._dense ->
    #   _primitives._solve
    from gaussx._strategies._cg import CGSolver

    if (
        solver is None
        and operator.in_size() > AutoSolver().size_threshold
        and lx.is_positive_semidefinite(operator)
    ):
        return CGSolver().solve(operator, vector)
    return _solve_fallback(operator, vector, solver)


def _solve_fallback(
    operator: lx.AbstractLinearOperator,
    vector: Float[Array, " n"],
    solver: lx.AbstractLinearSolver | None,
) -> Float[Array, " n"]:
    """Solve A x = b with no structure: dense direct, or the given solver.

    Args:
        operator: The operator A.
        vector: Right-hand side b.
        solver: Lineax solver, or ``None`` for `_dense_direct_solve`.

    Returns:
        The solution x.
    """
    if solver is None:
        return _dense_direct_solve(operator, vector)
    # An explicit solver keeps lineax's error on a singular system or an
    # unconverged iterative solve (see the `solve` docstring).
    return lx.linear_solve(operator, vector, solver).value


def _has_tag(predicate, operator: lx.AbstractLinearOperator) -> bool:
    """A lineax structure predicate, False where lineax does not know.

    Args:
        predicate: A lineax predicate such as `lineax.is_diagonal`.
        operator: The operator to test.

    Returns:
        The predicate's value, or ``False`` if it is not implemented.
    """
    try:
        return bool(predicate(operator))
    except NotImplementedError:
        return False


def _dense_direct_solve(
    operator: lx.AbstractLinearOperator,
    vector: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Dense direct solve of A x = b with numpy semantics (gh-302).

        x = A⁻¹ b,   by A = L Lᵀ (PSD), a triangular solve, or LU

    Picks the factorisation `lineax.AutoLinearSolver(well_posed=True)` would,
    but through ``jnp`` / ``jax.scipy``, which return ``nan``/``inf`` for a
    singular matrix. ``lineax.linear_solve(..., throw=False)`` does that for
    the value only: its JVP and transpose rules solve with ``throw=True``,
    so a gradient through a singular operator would still raise.

    Args:
        operator: The operator A, materialised here.
        vector: Right-hand side b.

    Returns:
        The solution x, non-finite when A is singular.

    Raises:
        ValueError: If A is not square.
    """
    matrix = operator.as_matrix()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"solve needs a square operator without a solver, got {matrix.shape}; "
            "pass e.g. solver=lineax.AutoLinearSolver(well_posed=False)."
        )
    if _has_tag(lx.is_diagonal, operator):
        # x_i = b_i / A_ii
        return vector / jnp.diagonal(matrix)
    if _has_tag(lx.is_lower_triangular, operator):
        # x = A⁻¹ b, forward substitution
        return jax.scipy.linalg.solve_triangular(matrix, vector, lower=True)
    if _has_tag(lx.is_upper_triangular, operator):
        # x = A⁻¹ b, backward substitution
        return jax.scipy.linalg.solve_triangular(matrix, vector, lower=False)
    if _has_tag(lx.is_positive_semidefinite, operator):
        # A = L Lᵀ
        factor = jnp.linalg.cholesky(matrix)
        # x = L⁻ᵀ L⁻¹ b
        return jax.scipy.linalg.cho_solve((factor, True), vector)
    # x = A⁻¹ b by LU
    return jnp.linalg.solve(matrix, vector)
