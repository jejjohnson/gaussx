"""Structured lazy inverse with dispatch on operator type."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import rearrange
from gaussx._operators._block_diag import BlockDiag, _resolve_dtype
from gaussx._operators._diagonalised import DiagonalisedOperator
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._low_rank_update import (
    LowRankUpdate,
    orthonormal_scaled_identity,
    scaled_identity_like,
)
from gaussx._operators._utils import register_lineax_structure_functions


def inv(
    operator: lx.AbstractLinearOperator,
    *,
    solver: lx.AbstractLinearSolver | None = None,
) -> lx.AbstractLinearOperator:
    """Return a lazy inverse operator A^{-1}.

    The returned operator computes A^{-1} v via ``solve(A, v)``
    when ``mv`` is called. For structured operators, the inverse
    preserves structure.

    Related to ``lineax.invert`` (lineax >= 0.1.1), which wraps
    ``lx.linear_solve`` in a ``FunctionLinearOperator``. The gaussx
    fallback ``InverseOperator`` differs in that its matvec routes
    through the *structured* gaussx ``solve`` dispatch, and its
    ``as_matrix`` uses a Cholesky path for PSD operators.

    Args:
        operator: An invertible linear operator.
        solver: Optional lineax solver for the fallback InverseOperator.

    Returns:
        An operator representing A^{-1}.
    """
    if isinstance(operator, lx.IdentityLinearOperator):
        return operator
    if isinstance(operator, InverseOperator):
        # (A⁻¹)⁻¹ = A (gh-349).
        return operator.original
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _inv_diagonal(operator)
    if isinstance(operator, DiagonalisedOperator):
        return _inv_diagonalised(operator)
    if isinstance(operator, BlockDiag):
        return _inv_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _inv_kronecker(operator)
    if isinstance(operator, LowRankUpdate) and _is_square(operator.base):
        if operator.rank == 0:
            return inv(operator.base, solver=solver)
        c = orthonormal_scaled_identity(operator)
        if c is not None:
            return _inv_low_rank_orthonormal(operator, c)
        # Decided from static structure only (gh-328): a value check on
        # ``U == V`` is lost once the operator crosses a jit boundary.
        if lx.is_symmetric(operator) and operator.symmetric_factors:
            return _inv_low_rank_symmetric(operator, solver)
        return _inv_low_rank_general(operator, solver)
    if isinstance(operator, lx.MulLinearOperator):
        return (1.0 / operator.scalar) * inv(operator.operator, solver=solver)
    if isinstance(operator, lx.DivLinearOperator):
        return operator.scalar * inv(operator.operator, solver=solver)
    if isinstance(operator, lx.NegLinearOperator):
        return -inv(operator.operator, solver=solver)
    if isinstance(operator, lx.ComposedLinearOperator) and (
        operator.operator1.in_size() == operator.operator1.out_size()
        and operator.operator2.in_size() == operator.operator2.out_size()
    ):
        # (A B)^{-1} = B^{-1} A^{-1}
        return inv(operator.operator2, solver=solver) @ inv(
            operator.operator1, solver=solver
        )
    return InverseOperator(operator, solver)


def _inv_diagonal(
    operator: lx.DiagonalLinearOperator,
) -> lx.DiagonalLinearOperator:
    diag = lx.diagonal(operator)
    return lx.DiagonalLinearOperator(1.0 / diag)


def _inv_diagonalised(operator: DiagonalisedOperator) -> DiagonalisedOperator:
    """Same basis with ``1/λ`` (zero eigenvalues map to zero: pseudo-inverse)."""
    lam = operator.eigenvalues
    zero = lam == 0
    return operator.with_eigenvalues(
        jnp.where(zero, 0.0, 1.0 / jnp.where(zero, 1.0, lam))
    )


def _inv_block_diag(operator: BlockDiag) -> BlockDiag:
    return BlockDiag(*(inv(op) for op in operator.operators))


def _inv_kronecker(operator: Kronecker) -> Kronecker:
    return Kronecker(*(inv(op) for op in operator.operators))


def _inv_low_rank_symmetric(
    operator: LowRankUpdate,
    solver: lx.AbstractLinearSolver | None,
) -> LowRankUpdate:
    """Woodbury inverse of a symmetric low-rank update, kept low-rank.

    With K = I + D U^T L^{-1} U (no D^{-1}, so zero weights are fine),

        (L + U D U^T)^{-1} = L^{-1} - L^{-1} U M U^T L^{-1},  M = K^{-1} D.

    M = D (I + S D)^{-1} is symmetric by the push-through identity, so
    eigendecomposing M = W diag(m) W^T turns the correction into a
    diagonal-middle low-rank update and the result is again a
    ``LowRankUpdate``:

        (L + U D U^T)^{-1} = L^{-1} + Z diag(-m) Z^T,  Z = L^{-1} U W.

    A zero weight gives a zero m, never a reciprocal (gh-307). Only k x k
    matrices are ever factorised.
    """
    from gaussx._primitives._solve import _low_rank_capacitance

    Linv_U, K = _low_rank_capacitance(operator, solver)
    M = jnp.linalg.solve(K, jnp.diag(operator.d))
    m, W = jnp.linalg.eigh(0.5 * (M + rearrange(M, "i j -> j i")))
    Z = Linv_U @ W
    return LowRankUpdate(
        inv(operator.base, solver=solver),
        Z,
        -m,
        Z,
        tags=_inverse_tags(operator),
    )


def _inv_low_rank_orthonormal(
    operator: LowRankUpdate, c: Float[Array, ""]
) -> LowRankUpdate:
    """``(cI + U D Uᵀ)⁻¹ = c⁻¹ I + U diag(−d / (c (c + d))) Uᵀ`` for ``UᵀU = I``.

    Same orthonormal factor, so the result takes the fast path again
    (gh-333). No factorisation at all.
    """
    d = operator.d
    return LowRankUpdate(
        scaled_identity_like(operator.base, 1 / c, _inverse_tags(operator.base)),
        operator.U,
        -d / (c * (c + d)),
        tags=_inverse_tags(operator),
        orthonormal=True,
    )


def _inv_low_rank_general(
    operator: LowRankUpdate,
    solver: lx.AbstractLinearSolver | None,
) -> LowRankUpdate:
    """Woodbury inverse of a general low-rank update, kept low-rank.

    With the scaled capacitance K = I + D V^T L^{-1} U (no D^{-1}, so zero
    weights are fine),

        (L + U D V^T)^{-1} = L^{-1} - L^{-1} U K^{-1} D V^T L^{-1}
                           = L^{-1} + (L^{-1} U) I (-L^{-T} V D K^{-T})^T,

    so the result is a ``LowRankUpdate`` with unit weights. Only the
    k x k matrix K is ever factorised. Like the structured ``solve``, this
    needs an invertible base L.
    """
    from gaussx._primitives._solve import _low_rank_capacitance, solve

    Linv_U, K = _low_rank_capacitance(operator, solver)
    LinvT_V = jax.vmap(
        lambda col: solve(operator.base.T, col, solver=solver), in_axes=1, out_axes=1
    )(operator.V)
    # -L^{-T} V D K^{-T} = -(K^{-1} D (L^{-T} V)^T)^T
    scaled = operator.d[:, None] * rearrange(LinvT_V, "n k -> k n")
    right = -rearrange(jnp.linalg.solve(K, scaled), "k n -> n k")
    ones = jnp.ones(operator.rank, dtype=K.dtype)
    return LowRankUpdate(
        inv(operator.base, solver=solver),
        Linv_U,
        ones,
        right,
        tags=_inverse_tags(operator),
    )


def _inverse_tags(operator: lx.AbstractLinearOperator) -> frozenset[object]:
    """Tags that an operator's inverse inherits: symmetry and definiteness."""
    queries = (
        (lx.is_symmetric, lx.symmetric_tag),
        (lx.is_positive_semidefinite, lx.positive_semidefinite_tag),
        (lx.is_negative_semidefinite, lx.negative_semidefinite_tag),
    )
    return frozenset(tag for query, tag in queries if query(operator))


def _is_square(operator: lx.AbstractLinearOperator) -> bool:
    return operator.in_size() == operator.out_size()


class InverseOperator(lx.AbstractLinearOperator):
    """Lazy inverse: ``mv`` computes A^{-1} v via solve."""

    original: lx.AbstractLinearOperator
    _solver: lx.AbstractLinearSolver | None = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)

    def __init__(
        self,
        original: lx.AbstractLinearOperator,
        solver: lx.AbstractLinearSolver | None = None,
    ) -> None:
        self.original = original
        self._solver = solver
        self._dtype = _resolve_dtype(original)

    def mv(self, vector):
        from gaussx._primitives._solve import solve

        return solve(self.original, vector, solver=self._solver)

    def as_matrix(self):
        # PSD path: A = L L^T => A^{-1} = L^{-T} L^{-1}, computed via two
        # triangular solves. More stable and faster than jnp.linalg.inv.
        # Handles a leading batch shape (..., n, n) — ``L.shape[-1]`` and a
        # broadcast identity keep the cho_solve call rank-correct.
        if lx.is_positive_semidefinite(self.original):
            from gaussx._primitives._cholesky import cholesky

            factor = cholesky(self.original)
            if not isinstance(factor, lx.AbstractLinearOperator):
                # A `SparseCholeskyFactor` is permuted and has no dense form.
                return jnp.linalg.inv(self.original.as_matrix())
            L = factor.as_matrix()
            n = L.shape[-1]
            identity = jnp.broadcast_to(
                jnp.eye(n, dtype=L.dtype), (*L.shape[:-2], n, n)
            )
            return jax.scipy.linalg.cho_solve((L, True), identity)
        return jnp.linalg.inv(self.original.as_matrix())

    def transpose(self):
        return InverseOperator(self.original.T, self._solver)

    def in_structure(self):
        # Inverse swaps in/out, but for square operators they're the same
        return self.original.out_structure()

    def out_structure(self):
        return self.original.in_structure()


# Register InverseOperator with lineax's singledispatch tag queries.
# Inverse preserves symmetry but not triangularity direction.

for _check in (
    lx.is_symmetric,
    lx.is_diagonal,
    lx.is_positive_semidefinite,
    lx.is_negative_semidefinite,
):

    @_check.register(InverseOperator)
    def _(operator, check=_check):
        return check(operator.original)


@lx.is_lower_triangular.register(InverseOperator)
def _(operator):
    return False


@lx.is_upper_triangular.register(InverseOperator)
def _(operator):
    return False


@lx.is_tridiagonal.register(InverseOperator)
def _(operator):
    return False


@lx.has_unit_diagonal.register(InverseOperator)
def _(operator):
    return False


# lineax.linearise / materialise / conj (gh-410), and lineax.diagonal: the
# predicates above delegate ``is_diagonal`` to the original, so lineax's
# AutoLinearSolver picks its Diagonal solver for the inverse of any 1 x 1 or
# diagonal-tagged operator and then needs ``lineax.diagonal`` (gh-349).
register_lineax_structure_functions(InverseOperator)


@lx.diagonal.register(InverseOperator)
def _(operator):
    if lx.is_diagonal(operator.original):
        return 1.0 / lx.diagonal(operator.original)
    return jnp.diag(operator.as_matrix())
