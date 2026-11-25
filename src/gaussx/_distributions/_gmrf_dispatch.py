"""How a GMRF is sampled and solved, chosen from the structure of its precision.

The dispatch shared by `GaussianMRF`, `ConstrainedGMRF` and `IntrinsicGMRF`: a draw
map f with f(z) ~ 𝒩(0, Λ⁻¹) for z ~ 𝒩(0, I) (Cholesky, factored eigenbasis or
perturbation-optimisation), the solve with Λ, and the quadratic form.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from gaussx._distributions._utils import _reshape_batch, _reshape_samples
from gaussx._einx import einsum, rearrange
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._diagonalised import DiagonalizedOperator
from gaussx._operators._factored_eigen import FactoredEigen, _reciprocal, factored_eigen
from gaussx._operators._grounding import push_scalar
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum, KroneckerSumSqrt
from gaussx._operators._sparse import SparseOperator
from gaussx._operators._spectral_function import SpectralFunction
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._solve import solve
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._cg import CGSolver
from gaussx._strategies._dispatch import dispatch_solve
from gaussx._strategies._sparse_cholesky import SparseCholeskySolver


_PSD = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
# Operators sampled in their factored eigenbasis (dispatch branch 2).
_EIGEN = (Kronecker, KroneckerSum, SpectralFunction, DiagonalizedOperator)
# Operators factored by Cholesky even when factors are given (branch 1).
_FACTORABLE = (
    lx.MatrixLinearOperator,
    lx.DiagonalLinearOperator,
    lx.IdentityLinearOperator,
)

# f(z): a zero-mean draw from standard normal noise z.
_Draw = Callable[[Array], Array]


# ---------------------------------------------------------------------------
# Sampling dispatch
# ---------------------------------------------------------------------------


def _unwrap(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Strip every `lineax.TaggedLinearOperator` wrapper from an operator.

    Args:
        operator: The operator, possibly tagged.

    Returns:
        The innermost untagged operator.
    """
    while isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    return operator


def _is_factorable(operator: lx.AbstractLinearOperator) -> bool:
    """Whether Cholesky is the cheapest exact path for a precision Λ.

    True for a dense, diagonal or identity Λ and a symmetric `BlockTriDiag`.

    Args:
        operator: The untagged precision Λ.

    Returns:
        ``True`` if Λ should be sampled through its Cholesky factor.
    """
    return isinstance(operator, _FACTORABLE) or (
        isinstance(operator, BlockTriDiag) and operator.symmetric
    )


def _cholesky_draw(operator: lx.AbstractLinearOperator) -> _Draw:
    """Draw map z ↦ L_Λ⁻ᵀ z from the structured Cholesky factor of Λ.

        Λ = L_Λ L_Λᵀ,   L_Λ⁻ᵀ z ~ 𝒩(0, Λ⁻¹)

    Args:
        operator: The precision Λ, factored here once.

    Returns:
        The map z ↦ L_Λ⁻ᵀ z.
    """
    # Λ = L_Λ L_Λᵀ
    cholesky_factor = cholesky(operator)
    if isinstance(cholesky_factor, lx.MatrixLinearOperator):
        factor_matrix = cholesky_factor.matrix
        # L_Λ⁻ᵀ z, by a dense triangular solve
        return lambda noise: jsl.solve_triangular(
            factor_matrix, noise, lower=True, trans="T"
        )
    # L_Λᵀ
    upper_factor = cholesky_factor.T
    # L_Λ⁻ᵀ z, by the structured solve
    return lambda noise: solve(upper_factor, noise)


def _eigen_draw(eigen: FactoredEigen, z: Array, *, pinv: bool = False) -> Array:
    """Symmetric-root draw Λ^(−½) z in the factored eigenbasis of Λ.

        Λ = W⁻¹ diag(γ) W,   Λ^(−½) z = W⁻¹ diag(γ^(−½)) W z

    W is the joint forward transform of `FactoredEigen` and γ its eigenvalue
    tensor. With ``pinv``, numerically zero eigenvalues get a zero root, so
    the draw is from 𝒩(0, Λ⁺).

    Args:
        eigen: Factored eigendecomposition of Λ, holding W and γ.
        z: Standard normal noise z.
        pinv: Use the pseudo-inverse γ⁺ in place of γ⁻¹.

    Returns:
        The draw Λ^(−½) z, or (Λ⁺)^½ z with ``pinv``.
    """
    # γ⁻¹, or γ⁺ with pinv
    inverse_eigenvalues = _reciprocal(eigen.eigenvalues, pinv=pinv)
    # γ^(−½)
    inverse_root_eigenvalues = jnp.sqrt(inverse_eigenvalues)
    # W z
    coeffs = eigen.to_eigenbasis(z)
    # diag(γ^(−½)) W z
    scaled_coeffs = coeffs * inverse_root_eigenvalues
    # Λ^(−½) z = W⁻¹ diag(γ^(−½)) W z
    draw = eigen.from_eigenbasis(scaled_coeffs)
    # Drop the imaginary part a real-by-construction basis leaves behind.
    return eigen._output(draw, z)


def _default_cg(vector: Array, size: int) -> CGSolver:
    """The CG strategy for Λ x = b when the GMRF holds none.

        tol_rel = tol_abs = √eps of b's dtype,   at most 10 N steps

    CG converges on a singular Λ too, when b lies in range(Λ).

    Args:
        vector: Right-hand side b, whose dtype sets the tolerance.
        size: N, the size of Λ.

    Returns:
        A `gaussx.CGSolver` at √eps with lineax's default step limit.
    """
    # √eps of b's dtype
    tolerance = float(jnp.finfo(vector.dtype).eps) ** 0.5
    return CGSolver(rtol=tolerance, atol=tolerance, max_steps=10 * size)


def _perturbation_draw(
    precision: lx.AbstractLinearOperator,
    factors: tuple[lx.AbstractLinearOperator, ...],
    solver: AbstractSolverStrategy | None,
) -> tuple[int, _Draw]:
    """Perturbation-optimisation draw map: solve Λ x = ∑_k F_kᵀ z_k per draw.

        Λ = ∑_k F_kᵀ F_k,   ∑_k F_kᵀ z_k ~ 𝒩(0, Λ),   x ~ 𝒩(0, Λ⁻¹)

    Algorithm:
        1. Split the noise z into one block z_k per factor F_k.
        2. Form ∑_k F_kᵀ z_k, whose covariance is Λ.
        3. Solve Λ x = ∑_k F_kᵀ z_k with ``solver``, or by CG without one.

    Args:
        precision: The precision Λ.
        factors: The factors F_k with Λ = ∑_k F_kᵀ F_k.
        solver: Strategy for the solve with Λ; ``None`` runs CG.

    Returns:
        Tuple ``(noise_size, draw)``: the length of z (the total number of
        rows of the F_k) and the map z ↦ Λ⁻¹ ∑_k F_kᵀ z_k.
    """
    # rows of each F_k, the length of its noise block z_k
    factor_sizes = [factor.out_size() for factor in factors]
    split_points = np.cumsum(factor_sizes)[:-1].tolist()
    psd_precision = lx.TaggedLinearOperator(precision, _PSD)

    def draw(noise: Array) -> Array:
        # z = (z_1, z_2, …), one block per factor
        noise_blocks = jnp.split(noise, split_points)
        # ∑_k F_kᵀ z_k
        perturbed_rhs = sum(
            factor.transpose().mv(block)
            for factor, block in zip(factors, noise_blocks, strict=True)
        )
        strategy = solver
        if strategy is None:
            strategy = _default_cg(perturbed_rhs, psd_precision.in_size())
        # x = Λ⁻¹ ∑_k F_kᵀ z_k, by the given strategy or CG
        return strategy.solve(psd_precision, perturbed_rhs)

    return sum(factor_sizes), draw


def _precision_sampler(
    precision: lx.AbstractLinearOperator,
    solver: AbstractSolverStrategy | None,
    factors: tuple[lx.AbstractLinearOperator, ...] | None,
) -> tuple[int, _Draw]:
    """Draw map f with f(z) ~ 𝒩(0, Λ⁻¹), chosen from the structure of Λ.

        f(z) ~ 𝒩(0, Λ⁻¹)   for   z ~ 𝒩(0, I)

    The factorisation is done here, once, outside any ``vmap`` over draws.

    Algorithm:
        1. A `SparseOperator` with no factors, or with a sparse Cholesky
           solver: f(z) = Pᵀ L⁻ᵀ z from the sparse factor P Λ Pᵀ = L Lᵀ.
        2. Dense, diagonal or symmetric `BlockTriDiag`: f(z) = L_Λ⁻ᵀ z.
        3. Kronecker-structured with a factored eigenbasis: f(z) = Λ^(−½) z.
        4. Factors F_k given: perturbation-optimisation.
        5. Anything else: the structured Cholesky, f(z) = L_Λ⁻ᵀ z.

    Args:
        precision: The precision Λ.
        solver: Solve strategy; a `SparseCholeskySolver` keeps a sparse Λ
            on branch 1 even when factors are given; on branch 4 it does
            the solve with Λ (CG when ``None``).
        factors: The factors F_k with Λ = ∑_k F_kᵀ F_k, or ``None``.

    Returns:
        Tuple ``(noise_size, draw)``: the length of z and the map f.
    """
    inner = _unwrap(precision)
    event_size = precision.in_size()
    if isinstance(inner, SparseOperator) and (
        factors is None or isinstance(solver, SparseCholeskySolver)
    ):
        if not isinstance(solver, SparseCholeskySolver):
            solver = SparseCholeskySolver()
        # Pᵀ L⁻ᵀ z with P Λ Pᵀ = L Lᵀ
        return event_size, solver.factor(inner).solve_lower_transpose
    if _is_factorable(inner):
        # L_Λ⁻ᵀ z
        return event_size, _cholesky_draw(inner)
    if isinstance(inner, _EIGEN):
        # Λ = W⁻¹ diag(γ) W
        eigen = factored_eigen(inner)
        if eigen is not None:
            # Λ^(−½) z
            return event_size, lambda noise: _eigen_draw(eigen, noise)
    if factors is not None:
        # Λ⁻¹ ∑_k F_kᵀ z_k
        return _perturbation_draw(precision, factors, solver)
    # L_Λ⁻ᵀ z
    return event_size, _cholesky_draw(inner)


def _draw_samples(
    key: jax.Array | None,
    sample_shape: tuple[int, ...],
    noise_size: int,
    draw: _Draw,
    dtype: jnp.dtype,
) -> Float[Array, "*sample N"]:
    """Zero-mean draws f(z), one ``normal`` call for every sample.

        z ~ 𝒩(0, I),   f(z) ~ 𝒩(0, Λ⁻¹)

    Args:
        key: PRNG key; ``None`` means ``jax.random.PRNGKey(0)``.
        sample_shape: Leading sample shape.
        noise_size: Length of the noise z of one draw.
        draw: The draw map f.
        dtype: Dtype of the noise z.

    Returns:
        Draws f(z), shape ``sample_shape + (N,)``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    num_draws = math.prod(sample_shape)
    # z ~ 𝒩(0, I), one row per draw
    noise = jax.random.normal(key, (num_draws, noise_size), dtype=dtype)
    # f(z), one row per draw
    draws = jax.vmap(draw)(noise)
    return _reshape_samples(draws, sample_shape)


def _flat(value: Float[Array, "*sample N"]) -> Float[Array, "S N"]:
    """Flatten every leading sample axis into one.

    Args:
        value: Points, shape ``(..., N)``.

    Returns:
        The points, one row per sample.
    """
    return rearrange(value, "... state -> (...) state")


def _quadratic(operator: lx.AbstractLinearOperator, residual: Array) -> Array:
    """Quadratic form rᵀ Λ r for each row r of ``residual``.

        rᵀ Λ r,   r = x − μ

    Args:
        operator: The symmetric operator Λ (a precision, or a structure R).
        residual: Residuals r, shape ``(..., N)``.

    Returns:
        The quadratic forms, shape ``(...)``.
    """
    flat_residual = _flat(residual)

    def quadratic_form(row: Array) -> Array:
        # Λ r
        weighted_row = operator.mv(row)
        # rᵀ Λ r
        return einsum(row, weighted_row, "state, state ->")

    values = jax.vmap(quadratic_form)(flat_residual)
    return _reshape_batch(values, residual.shape[:-1])


def _solve_strategy(
    precision: lx.AbstractLinearOperator, solver: AbstractSolverStrategy | None
) -> AbstractSolverStrategy | None:
    """The solve strategy for Λ: ``solver``, or sparse Cholesky for a sparse Λ.

    Args:
        precision: The precision Λ.
        solver: The caller's strategy, or ``None``.

    Returns:
        ``solver``, or a `SparseCholeskySolver` when it is ``None`` and Λ is
        a `SparseOperator`.
    """
    if solver is None and isinstance(_unwrap(precision), SparseOperator):
        return SparseCholeskySolver()
    return solver


def _solve_fn(
    precision: lx.AbstractLinearOperator, solver: AbstractSolverStrategy | None
) -> Callable[[Array], Array]:
    """Solve map b ↦ Λ⁻¹ b, factoring a sparse precision once for many solves.

    Args:
        precision: The precision Λ.
        solver: The caller's strategy, or ``None`` for structural dispatch.

    Returns:
        The map b ↦ Λ⁻¹ b.
    """
    strategy = _solve_strategy(precision, solver)
    if isinstance(strategy, SparseCholeskySolver):
        # P Λ Pᵀ = L Lᵀ, factored once
        return strategy.factor(precision).solve
    # Λ⁻¹ b
    return lambda rhs: dispatch_solve(precision, rhs, strategy)


class _Sqrt(eqx.Module):
    """Square root γ ↦ √f(γ) of a spectral function f.

    A pytree, so the parameters of f stay leaves.

    Attributes:
        fn: The spectral function f.
    """

    fn: Callable[[Array], Array]

    def __call__(self, eigenvalues: Array) -> Array:
        """Evaluate √f(γ).

        Args:
            eigenvalues: Eigenvalue tensor γ.

        Returns:
            The tensor √f(γ).
        """
        # f(γ)
        values = self.fn(eigenvalues)
        # √f(γ)
        return jnp.sqrt(values)


def _symmetric_root(
    operator: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator | None:
    """Symmetric square root Λ^½ of a grid precision.

        (Λ^½)ᵀ Λ^½ = Λ

    Args:
        operator: The untagged precision Λ.

    Returns:
        Λ^½ for a `SpectralFunction` or a `KroneckerSum`, else ``None``.
    """
    if isinstance(operator, SpectralFunction):
        # f(A)^½ = W⁻¹ diag(√f(γ)) W
        return SpectralFunction(
            operator.base, _Sqrt(operator.fn), _eigen=operator.eigen
        )
    if isinstance(operator, KroneckerSum):
        # (A ⊕ B)^½
        return KroneckerSumSqrt(operator.A, operator.B)
    return None


def _as_operator(A: lx.AbstractLinearOperator | ArrayLike) -> lx.AbstractLinearOperator:
    """Wrap a dense matrix as a `lineax.MatrixLinearOperator`.

    Args:
        A: An operator, returned unchanged, or a dense matrix.

    Returns:
        The operator.
    """
    if isinstance(A, lx.AbstractLinearOperator):
        return A
    return lx.MatrixLinearOperator(jnp.asarray(A))


def _scaled(
    operator: lx.AbstractLinearOperator, scale: Float[Array, ""]
) -> lx.AbstractLinearOperator:
    """Scaled structure τ R, keeping a sparse, banded or dense structure.

    Args:
        operator: The structure R.
        scale: The precision scale τ.

    Returns:
        τ R, tagged symmetric positive semidefinite.
    """
    inner = push_scalar(operator)
    if isinstance(inner, SparseOperator):
        # τ R on R's pattern
        return SparseOperator(scale * inner.values, inner.pattern, tags=_PSD)
    if isinstance(inner, BlockTriDiag):
        # τ R, block by block
        return BlockTriDiag(
            scale * inner.diagonal,
            scale * inner.sub_diagonal,
            symmetric=inner.symmetric,
            tags=_PSD,
        )
    if isinstance(inner, lx.MatrixLinearOperator):
        # τ R, dense
        return lx.MatrixLinearOperator(scale * inner.matrix, _PSD)
    # τ R, lazily
    return lx.TaggedLinearOperator(scale * operator, _PSD)
