"""Structured lazy inverse A⁻¹, dispatched on the operator type of A.

A is a square invertible N × N operator and A⁻¹ its inverse, returned as an
operator rather than a matrix. A Kronecker product is A = A_1 ⊗ … ⊗ A_D with
square factors A_d, d = 1, …, D, and a block-diagonal operator is
blockdiag(A_1, …, A_D). A diagonalised operator is A = W⁻¹ diag(γ) W for a
forward transform W (an FFT, say) and eigenvalues γ. A low-rank update is
A = A₀ + U D Vᵀ, with base A₀, N × k factors U and V and D = diag(d), and
C = I + D Vᵀ A₀⁻¹ U is its k × k capacitance scaled by D. L is a lower
Cholesky factor, A = L Lᵀ.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._operators._block_diag import BlockDiag, _resolve_dtype
from gaussx._operators._diagonalised import DiagonalizedOperator
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._low_rank_update import (
    LowRankUpdate,
    orthonormal_scaled_identity,
    scaled_identity_like,
)
from gaussx._operators._utils import register_lineax_structure_functions
from gaussx._primitives._cholesky import cholesky


def inv(
    operator: lx.AbstractLinearOperator,
    *,
    solver: lx.AbstractLinearSolver | None = None,
) -> lx.AbstractLinearOperator:
    """Lazy inverse A⁻¹ of an operator, structured where A is.

        A⁻¹ A = I

    The result is an operator, not a matrix: a structured A gives an inverse
    of the same structure, and anything else becomes an `InverseOperator`
    whose ``mv`` is ``gaussx.solve(A, v)``.

    Related to ``lineax.invert`` (lineax >= 0.1.1), which wraps
    ``lx.linear_solve`` in a ``FunctionLinearOperator``. The gaussx
    fallback `InverseOperator` differs in that its matvec routes through the
    *structured* gaussx ``solve`` dispatch, and its ``as_matrix`` uses a
    Cholesky path for PSD operators.

    Algorithm:
        1. Take the first structural rule that matches A: the identity and
           an `InverseOperator` invert in closed form, a diagonal or
           diagonalised operator takes reciprocal eigenvalues, a
           block-diagonal or Kronecker operator inverts per block or factor,
           and a square low-rank update applies the Woodbury identity and
           stays a `LowRankUpdate`.
        2. Scaled, negated and composed operators recurse into their parts,
           (c A)⁻¹ = c⁻¹ A⁻¹ and (A B)⁻¹ = B⁻¹ A⁻¹.
        3. Otherwise wrap A in a lazy `InverseOperator`.

    Args:
        operator: The square invertible operator A.
        solver: Lineax solver for the base solves of a low-rank update and
            for the fallback `InverseOperator`; ``None`` uses gaussx's
            structural ``solve``.

    Returns:
        An operator for A⁻¹: of A's structured type where a rule applies,
        otherwise an `InverseOperator`.

    Examples:
        A Kronecker product of diagonals inverts factor by factor:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A_1 = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> A_2 = lx.DiagonalLinearOperator(jnp.array([4.0, 5.0]))
        >>> A = gaussx.Kronecker(A_1, A_2)  # diag(4, 5, 8, 10)
        >>> A_inv = gaussx.inv(A)  # A_1⁻¹ ⊗ A_2⁻¹, never densified
        >>> type(A_inv).__name__
        'Kronecker'
        >>> [round(float(v), 4) for v in A_inv.mv(jnp.ones(4))]
        [0.25, 0.2, 0.125, 0.1]

        A low-rank update stays low-rank through the Woodbury identity:

        >>> base = lx.DiagonalLinearOperator(jnp.array([2.0, 3.0, 4.0]))
        >>> U = jnp.array([[1.0], [0.5], [0.0]])
        >>> A = gaussx.LowRankUpdate(base, U)
        >>> A_inv = gaussx.inv(A)
        >>> type(A_inv).__name__
        'LowRankUpdate'
        >>> identity = (A_inv @ A).as_matrix()
        >>> bool(jnp.allclose(identity, jnp.eye(3), atol=1e-5))
        True

        Inverting an inverse returns the original operator:

        >>> dense = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 3.0]]))
        >>> gaussx.inv(gaussx.inv(dense)) is dense
        True
    """
    if isinstance(operator, lx.IdentityLinearOperator):
        # I⁻¹ = I
        return operator
    if isinstance(operator, InverseOperator):
        # (A⁻¹)⁻¹ = A (gh-349).
        return operator.original
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _inv_diagonal(operator)
    if isinstance(operator, DiagonalizedOperator):
        return _inv_diagonalised(operator)
    if isinstance(operator, BlockDiag):
        return _inv_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _inv_kronecker(operator)
    if isinstance(operator, LowRankUpdate) and _is_square(operator.base):
        if operator.rank == 0:
            # A⁻¹ = A₀⁻¹ when k = 0
            return inv(operator.base, solver=solver)
        shift = orthonormal_scaled_identity(operator)
        if shift is not None:
            return _inv_low_rank_orthonormal(operator, shift)
        # Decided from static structure only (gh-328): a value check on
        # ``U == V`` is lost once the operator crosses a jit boundary.
        if lx.is_symmetric(operator) and operator.symmetric_factors:
            return _inv_low_rank_symmetric(operator, solver)
        return _inv_low_rank_general(operator, solver)
    if isinstance(operator, lx.MulLinearOperator):
        # 1 / c
        reciprocal = 1.0 / operator.scalar
        inner_inverse = inv(operator.operator, solver=solver)
        # (c A)⁻¹ = c⁻¹ A⁻¹
        return reciprocal * inner_inverse
    if isinstance(operator, lx.DivLinearOperator):
        inner_inverse = inv(operator.operator, solver=solver)
        # (A / c)⁻¹ = c A⁻¹
        return operator.scalar * inner_inverse
    if isinstance(operator, lx.NegLinearOperator):
        inner_inverse = inv(operator.operator, solver=solver)
        # (−A)⁻¹ = −A⁻¹
        return -inner_inverse
    if isinstance(operator, lx.ComposedLinearOperator) and (
        operator.operator1.in_size() == operator.operator1.out_size()
        and operator.operator2.in_size() == operator.operator2.out_size()
    ):
        # B⁻¹, for the composition A B
        second_inverse = inv(operator.operator2, solver=solver)
        # A⁻¹
        first_inverse = inv(operator.operator1, solver=solver)
        # (A B)⁻¹ = B⁻¹ A⁻¹
        return second_inverse @ first_inverse
    return InverseOperator(operator, solver)


def _inv_diagonal(
    operator: lx.DiagonalLinearOperator,
) -> lx.DiagonalLinearOperator:
    """Inverse of a diagonal operator diag(a), entry by entry.

        diag(a)⁻¹ = diag(1 / a)

    Args:
        operator: Diagonal operator diag(a).

    Returns:
        The diagonal operator diag(1 / a).
    """
    diagonal = lx.diagonal(operator)
    # 1 / a_i
    inverse_diagonal = 1.0 / diagonal
    return lx.DiagonalLinearOperator(inverse_diagonal)


def _inv_diagonalised(operator: DiagonalizedOperator) -> DiagonalizedOperator:
    """Inverse of A = W⁻¹ diag(γ) W in the same transform basis.

        A⁺ = W⁻¹ diag(γ⁺) W,   γ⁺_i = 1 / γ_i, or 0 where γ_i = 0

    Exactly-zero eigenvalues map to zero, which for a normal operator
    (unitary W) is the Moore-Penrose pseudo-inverse.

    Args:
        operator: Diagonalised operator A = W⁻¹ diag(γ) W.

    Returns:
        The diagonalised operator with eigenvalues γ⁺.
    """
    eigenvalues = operator.eigenvalues
    is_zero = eigenvalues == 0
    safe_eigenvalues = jnp.where(is_zero, 1.0, eigenvalues)
    # 1 / γ, with exactly-zero eigenvalues mapped to 0
    reciprocal_eigenvalues = 1.0 / safe_eigenvalues
    inverse_eigenvalues = jnp.where(is_zero, 0.0, reciprocal_eigenvalues)
    return operator.with_eigenvalues(inverse_eigenvalues)


def _inv_block_diag(operator: BlockDiag) -> BlockDiag:
    """Inverse of a block-diagonal operator, block by block.

        blockdiag(A_1, …, A_D)⁻¹ = blockdiag(A_1⁻¹, …, A_D⁻¹)

    Args:
        operator: Block-diagonal operator with invertible blocks A_d.

    Returns:
        The block-diagonal inverse.
    """
    # A_d⁻¹, per block
    return BlockDiag(*(inv(block) for block in operator.operators))


def _inv_kronecker(operator: Kronecker) -> Kronecker:
    """Inverse of a Kronecker product, factor by factor.

        (A_1 ⊗ … ⊗ A_D)⁻¹ = A_1⁻¹ ⊗ … ⊗ A_D⁻¹

    Args:
        operator: Kronecker product of invertible factors A_d.

    Returns:
        The Kronecker inverse.
    """
    # A_d⁻¹, per factor
    return Kronecker(*(inv(factor) for factor in operator.operators))


def _inv_low_rank_symmetric(
    operator: LowRankUpdate,
    solver: lx.AbstractLinearSolver | None,
) -> LowRankUpdate:
    """Woodbury inverse of a symmetric low-rank update, kept low-rank.

        (A₀ + U D Uᵀ)⁻¹ = A₀⁻¹ + Z diag(−m) Zᵀ,   Z = A₀⁻¹ U Q_M

    With the scaled capacitance C = I + D Uᵀ A₀⁻¹ U (no D⁻¹, so a zero
    weight stays exact), the Woodbury identity reads
    (A₀ + U D Uᵀ)⁻¹ = A₀⁻¹ − A₀⁻¹ U M Uᵀ A₀⁻¹ with M = C⁻¹ D. M is
    symmetric by the push-through identity, so its eigendecomposition
    M = Q_M diag(m) Q_Mᵀ turns the correction into a diagonal-middle
    low-rank update and the result is again a ``LowRankUpdate``. A zero
    weight gives a zero m, never a reciprocal (gh-307), and only k × k
    matrices are ever factorised.

    Algorithm:
        1. Form A₀⁻¹ U and the capacitance C.
        2. Solve M = C⁻¹ D and symmetrise it.
        3. Eigendecompose M = Q_M diag(m) Q_Mᵀ and rotate Z = A₀⁻¹ U Q_M.

    Args:
        operator: Symmetric low-rank update A₀ + U D Uᵀ.
        solver: Lineax solver for the base solves, or ``None``.

    Returns:
        The inverse as a low-rank update of A₀⁻¹ with factor Z and weights
        −m, carrying A's symmetry and definiteness tags.
    """
    # lazy import, cycle: _primitives._solve -> _primitives._inv
    from gaussx._primitives._solve import _low_rank_capacitance

    # A₀⁻¹ U and C = I + D Uᵀ A₀⁻¹ U
    base_inv_factor, capacitance = _low_rank_capacitance(operator, solver)
    weights = jnp.diag(operator.d)
    # M = C⁻¹ D
    middle = jnp.linalg.solve(capacitance, weights)
    # Mᵀ
    middle_transpose = rearrange(middle, "row col -> col row")
    # ½ (M + Mᵀ), symmetric up to round-off
    symmetric_middle = 0.5 * (middle + middle_transpose)
    # M = Q_M diag(m) Q_Mᵀ
    middle_eigenvalues, middle_eigenvectors = jnp.linalg.eigh(symmetric_middle)
    # Z = A₀⁻¹ U Q_M
    rotated_factor = einsum(
        base_inv_factor, middle_eigenvectors, "row rank, rank col -> row col"
    )
    # A₀⁻¹
    base_inverse = inv(operator.base, solver=solver)
    # −m
    negated_eigenvalues = -middle_eigenvalues
    # A⁻¹ = A₀⁻¹ + Z diag(−m) Zᵀ
    return LowRankUpdate(
        base_inverse,
        rotated_factor,
        negated_eigenvalues,
        rotated_factor,
        tags=_inverse_tags(operator),
    )


def _inv_low_rank_orthonormal(
    operator: LowRankUpdate, shift: Float[Array, ""]
) -> LowRankUpdate:
    """Inverse of c I + U D Uᵀ in closed form for orthonormal U.

        (c I + U D Uᵀ)⁻¹ = c⁻¹ I + U diag(−d / (c (c + d))) Uᵀ,   Uᵀ U = I

    Same orthonormal factor, so the result takes the fast path again
    (gh-333). No factorisation at all.

    Args:
        operator: Low-rank update c I + U D Uᵀ with orthonormal U.
        shift: The scalar shift c of the base.

    Returns:
        The inverse, an orthonormal low-rank update on c⁻¹ I.
    """
    weights = operator.d
    # c⁻¹
    reciprocal_shift = 1 / shift
    base_inverse = scaled_identity_like(
        operator.base, reciprocal_shift, _inverse_tags(operator.base)
    )
    # −d
    negated_weights = -weights
    # c + d
    shifted_weights = shift + weights
    # c (c + d)
    denominator = shift * shifted_weights
    # −d / (c (c + d))
    inverse_weights = negated_weights / denominator
    return LowRankUpdate(
        base_inverse,
        operator.U,
        inverse_weights,
        tags=_inverse_tags(operator),
        orthonormal=True,
    )


def _inv_low_rank_general(
    operator: LowRankUpdate,
    solver: lx.AbstractLinearSolver | None,
) -> LowRankUpdate:
    """Woodbury inverse of a general low-rank update, kept low-rank.

        (A₀ + U D Vᵀ)⁻¹ = A₀⁻¹ + (A₀⁻¹ U) I (−A₀⁻ᵀ V D C⁻ᵀ)ᵀ

    With the scaled capacitance C = I + D Vᵀ A₀⁻¹ U (no D⁻¹, so a zero
    weight stays exact), the Woodbury identity
    (A₀ + U D Vᵀ)⁻¹ = A₀⁻¹ − A₀⁻¹ U C⁻¹ D Vᵀ A₀⁻¹ regroups into a
    ``LowRankUpdate`` with unit weights. Only the k × k matrix C is ever
    factorised. Like the structured ``solve``, this needs an invertible
    base A₀.

    Algorithm:
        1. Form A₀⁻¹ U and the capacitance C.
        2. Solve A₀ᵀ against each column of V.
        3. Solve the right factor −A₀⁻ᵀ V D C⁻ᵀ = −(C⁻¹ D (A₀⁻ᵀ V)ᵀ)ᵀ.

    Args:
        operator: Low-rank update A₀ + U D Vᵀ.
        solver: Lineax solver for the base solves, or ``None``.

    Returns:
        The inverse as a unit-weight low-rank update of A₀⁻¹, carrying A's
        symmetry and definiteness tags.
    """
    # lazy import, cycle: _primitives._solve -> _primitives._inv
    from gaussx._primitives._solve import _low_rank_capacitance, solve

    # A₀⁻¹ U and C = I + D Vᵀ A₀⁻¹ U
    base_inv_factor, capacitance = _low_rank_capacitance(operator, solver)
    # A₀⁻ᵀ V, one solve per column of V
    base_inv_t_right = jax.vmap(
        lambda column: solve(operator.base.T, column, solver=solver),
        in_axes=1,
        out_axes=1,
    )(operator.V)
    # D (A₀⁻ᵀ V)ᵀ
    scaled = einx.multiply("rank, row rank -> rank row", operator.d, base_inv_t_right)
    # C⁻¹ D (A₀⁻ᵀ V)ᵀ
    solved = jnp.linalg.solve(capacitance, scaled)
    # (C⁻¹ D (A₀⁻ᵀ V)ᵀ)ᵀ = A₀⁻ᵀ V D C⁻ᵀ
    solved_transpose = rearrange(solved, "rank row -> row rank")
    # −A₀⁻ᵀ V D C⁻ᵀ
    right_factor = -solved_transpose
    ones = jnp.ones(operator.rank, dtype=capacitance.dtype)
    # A₀⁻¹
    base_inverse = inv(operator.base, solver=solver)
    # A⁻¹ = A₀⁻¹ + (A₀⁻¹ U) I (−A₀⁻ᵀ V D C⁻ᵀ)ᵀ
    return LowRankUpdate(
        base_inverse,
        base_inv_factor,
        ones,
        right_factor,
        tags=_inverse_tags(operator),
    )


def _inverse_tags(operator: lx.AbstractLinearOperator) -> frozenset[object]:
    """Tags that an operator's inverse inherits: symmetry and definiteness.

    Args:
        operator: The operator A being inverted.

    Returns:
        The symmetric, PSD and NSD tags that hold for A, hence for A⁻¹.
    """
    queries = (
        (lx.is_symmetric, lx.symmetric_tag),
        (lx.is_positive_semidefinite, lx.positive_semidefinite_tag),
        (lx.is_negative_semidefinite, lx.negative_semidefinite_tag),
    )
    return frozenset(tag for query, tag in queries if query(operator))


def _is_square(operator: lx.AbstractLinearOperator) -> bool:
    """Whether the operator maps a space to one of the same size."""
    return operator.in_size() == operator.out_size()


class InverseOperator(lx.AbstractLinearOperator):
    """Lazy inverse A⁻¹ whose ``mv`` solves with A.

        A⁻¹ v = x,   A x = v

    What `gaussx.inv` returns when A has no structured inverse. ``mv``
    routes through gaussx's structural ``solve``, so a structured A keeps
    its fast solve; ``as_matrix`` densifies through a Cholesky factor
    A = L Lᵀ when A is PSD, else through ``jnp.linalg.inv``.

    Args:
        original: The square invertible operator A.
        solver: Lineax solver for the solves, or ``None`` for gaussx's
            structural ``solve``.

    Examples:
        The lazy inverse of a dense matrix, applied to a vector:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from gaussx._primitives._inv import InverseOperator
        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> A_inv = InverseOperator(A)
        >>> x = A_inv.mv(jnp.array([1.0, 2.0]))
        >>> bool(jnp.allclose(matrix @ x, jnp.array([1.0, 2.0]), atol=1e-5))
        True
        >>> bool(jnp.allclose(A_inv.as_matrix() @ matrix, jnp.eye(2), atol=1e-5))
        True
    """

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
        # lazy import, cycle: _primitives._solve -> _primitives._inv
        from gaussx._primitives._solve import solve

        # A⁻¹ v
        return solve(self.original, vector, solver=self._solver)

    def as_matrix(self):
        # Two triangular solves with L are more stable and faster than
        # jnp.linalg.inv on a PSD A.
        if lx.is_positive_semidefinite(self.original):
            factor = cholesky(self.original)
            if not isinstance(factor, lx.AbstractLinearOperator):
                # A `SparseCholeskyFactor` is permuted and has no dense form.
                return jnp.linalg.inv(self.original.as_matrix())
            lower_factor = factor.as_matrix()
            size = lower_factor.shape[-1]
            # jnp, not einx: L may carry a leading batch shape (…, N, N), and
            # einx cannot resolve an empty operator (N = 0).
            identity = jnp.broadcast_to(
                jnp.eye(size, dtype=lower_factor.dtype),
                (*lower_factor.shape[:-2], size, size),
            )
            # A⁻¹ = L⁻ᵀ L⁻¹ I
            return jax.scipy.linalg.cho_solve((lower_factor, True), identity)
        # A⁻¹, dense
        return jnp.linalg.inv(self.original.as_matrix())

    def transpose(self):
        # (A⁻¹)ᵀ = (Aᵀ)⁻¹
        return InverseOperator(self.original.T, self._solver)

    def in_structure(self):
        # A⁻¹ swaps A's input and output spaces.
        return self.original.out_structure()

    def out_structure(self):
        return self.original.in_structure()


# A⁻¹ inherits symmetry, diagonality and definiteness from A, but neither
# triangularity direction nor a tridiagonal or unit-diagonal pattern.
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
        # diag(A⁻¹) = 1 / diag(A) for a diagonal A
        return 1.0 / lx.diagonal(operator.original)
    # diag(A⁻¹), dense
    return jnp.diag(operator.as_matrix())
