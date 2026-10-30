"""Structured gaussx operators A, lineax operators with their registrations.

Each operator subclasses ``lineax.AbstractLinearOperator``. Importing this
package registers the operator classes with the gaussx structure-tag
predicates (``is_kronecker``, ``is_block_diagonal``, …), the lineax
predicates (``is_symmetric``, ``is_diagonal``, ``is_positive_semidefinite``,
``is_negative_semidefinite``, ``is_tridiagonal``, ``is_lower_triangular``,
``is_upper_triangular``, ``has_unit_diagonal``) and the lineax structure
functions (``linearise``, ``materialise``, ``diagonal``, ``conj``), so
lineax's solvers and gaussx's dispatch can query them. A few classes
register in their own module instead (`SparseOperator`, `SpectralFunction`,
``ToeplitzCholesky``'s predicates).
"""

from __future__ import annotations

import lineax as lx

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import (
    BlockTriDiag,
    LowerBlockTriDiag,
    UpperBlockTriDiag,
)
from gaussx._operators._capacitance import CapacitanceSolver
from gaussx._operators._diagonalised import (
    DiagonalizedOperator,
    as_diagonalized,
    circulant,
    circulant_from_symbol,
)
from gaussx._operators._grid import create_grid, cubic_interpolation_weights, grid_data
from gaussx._operators._interpolated import InterpolatedOperator
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import (
    KroneckerSum,
    KroneckerSumSqrt,
    kronecker_sum_sample,
)
from gaussx._operators._lazy_algebra import (
    product_operator,
    scaled_operator,
    sum_operator,
)
from gaussx._operators._low_rank_update import (
    LowRankUpdate,
    SVDLowRankUpdate,
    low_rank_plus_diag,
    low_rank_plus_identity,
    svd_low_rank_plus_diag,
)
from gaussx._operators._masked import MaskedOperator, grid_coupling_indices
from gaussx._operators._sparse import SparseOperator, SparsityPattern
from gaussx._operators._spectral_function import SpectralFunction
from gaussx._operators._sum_kronecker import (
    SumKronecker,
    SumOfKroneckers,
    is_eigen_reducible as is_eigen_reducible,
    sum_of_kroneckers_sample as sum_of_kroneckers_sample,
    sumkronecker_sample as sumkronecker_sample,
)
from gaussx._operators._toeplitz import (
    Toeplitz,
    ToeplitzCholesky,
    _ToeplitzCholeskyTranspose,
    toeplitz_sample,
)
from gaussx._operators._utils import (
    lineax_conj,
    register_lineax_structure_functions,
)
from gaussx._tags import (
    is_block_diagonal,
    is_block_tridiagonal,
    is_kronecker,
    is_kronecker_sum,
    is_low_rank,
)


# -------------------------------------------------------------------
# gaussx structure-tag predicates: each class answers its own tag query
# -------------------------------------------------------------------


@is_kronecker.register(Kronecker)
def _(operator: Kronecker) -> bool:
    return True


@is_block_diagonal.register(BlockDiag)
def _(operator: BlockDiag) -> bool:
    return True


@is_low_rank.register(LowRankUpdate)
def _(operator: LowRankUpdate) -> bool:
    return True


@is_kronecker_sum.register(KroneckerSum)
def _(operator: KroneckerSum) -> bool:
    return True


@is_block_tridiagonal.register(BlockTriDiag)
def _(operator: BlockTriDiag) -> bool:
    return True


# lineax's is_symmetric, is_diagonal and is_positive_semidefinite, so that
# lineax's solvers and gaussx's dispatch can query these operators.


@lx.is_symmetric.register(BlockDiag)
def _(operator: BlockDiag) -> bool:
    return all(lx.is_symmetric(op) for op in operator.operators)


@lx.is_symmetric.register(Kronecker)
def _(operator: Kronecker) -> bool:
    return all(lx.is_symmetric(op) for op in operator.operators)


@lx.is_symmetric.register(LowRankUpdate)
def _(operator: LowRankUpdate) -> bool:
    return lx.symmetric_tag in operator.tags


@lx.is_symmetric.register(KroneckerSum)
def _(operator: KroneckerSum) -> bool:
    return lx.is_symmetric(operator.A) and lx.is_symmetric(operator.B)


@lx.is_symmetric.register(KroneckerSumSqrt)
def _(operator: KroneckerSumSqrt) -> bool:
    return True


@lx.is_symmetric.register(BlockTriDiag)
def _(operator: BlockTriDiag) -> bool:
    return lx.symmetric_tag in operator.tags


@lx.is_diagonal.register(BlockDiag)
def _(operator: BlockDiag) -> bool:
    return all(lx.is_diagonal(op) for op in operator.operators)


@lx.is_diagonal.register(Kronecker)
def _(operator: Kronecker) -> bool:
    return all(lx.is_diagonal(op) for op in operator.operators)


@lx.is_diagonal.register(LowRankUpdate)
def _(operator: LowRankUpdate) -> bool:
    return False


@lx.is_diagonal.register(KroneckerSum)
def _(operator: KroneckerSum) -> bool:
    return False


@lx.is_diagonal.register(KroneckerSumSqrt)
def _(operator: KroneckerSumSqrt) -> bool:
    return False


@lx.is_diagonal.register(BlockTriDiag)
def _(operator: BlockTriDiag) -> bool:
    return False


@lx.is_positive_semidefinite.register(BlockDiag)
def _(operator: BlockDiag) -> bool:
    return all(lx.is_positive_semidefinite(op) for op in operator.operators)


@lx.is_positive_semidefinite.register(Kronecker)
def _(operator: Kronecker) -> bool:
    return all(lx.is_positive_semidefinite(op) for op in operator.operators)


@lx.is_positive_semidefinite.register(LowRankUpdate)
def _(operator: LowRankUpdate) -> bool:
    return lx.positive_semidefinite_tag in operator.tags


@lx.is_positive_semidefinite.register(KroneckerSum)
def _(operator: KroneckerSum) -> bool:
    return lx.is_positive_semidefinite(operator.A) and lx.is_positive_semidefinite(
        operator.B
    )


@lx.is_positive_semidefinite.register(KroneckerSumSqrt)
def _(operator: KroneckerSumSqrt) -> bool:
    return True


@lx.is_positive_semidefinite.register(BlockTriDiag)
def _(operator: BlockTriDiag) -> bool:
    return lx.positive_semidefinite_tag in operator.tags


# LowerBlockTriDiag and UpperBlockTriDiag: block-triangular, never symmetric
# or diagonal


@lx.is_symmetric.register(LowerBlockTriDiag)
def _(operator: LowerBlockTriDiag) -> bool:
    return False


@lx.is_symmetric.register(UpperBlockTriDiag)
def _(operator: UpperBlockTriDiag) -> bool:
    return False


@lx.is_diagonal.register(LowerBlockTriDiag)
def _(operator: LowerBlockTriDiag) -> bool:
    return False


@lx.is_diagonal.register(UpperBlockTriDiag)
def _(operator: UpperBlockTriDiag) -> bool:
    return False


@lx.is_lower_triangular.register(LowerBlockTriDiag)
def _(operator: LowerBlockTriDiag) -> bool:
    return True


@lx.is_upper_triangular.register(UpperBlockTriDiag)
def _(operator: UpperBlockTriDiag) -> bool:
    return True


@lx.is_lower_triangular.register(UpperBlockTriDiag)
def _(operator: UpperBlockTriDiag) -> bool:
    return False


@lx.is_upper_triangular.register(LowerBlockTriDiag)
def _(operator: LowerBlockTriDiag) -> bool:
    return False


# MaskedOperator A = A₀[m_r, m_c]: symmetry and definiteness from its tags


@lx.is_symmetric.register(MaskedOperator)
def _(operator: MaskedOperator) -> bool:
    return lx.symmetric_tag in operator.tags


@lx.is_diagonal.register(MaskedOperator)
def _(operator: MaskedOperator) -> bool:
    return False


@lx.is_positive_semidefinite.register(MaskedOperator)
def _(operator: MaskedOperator) -> bool:
    return lx.positive_semidefinite_tag in operator.tags


# DiagonalizedOperator and Toeplitz


@lx.is_symmetric.register(DiagonalizedOperator)
def _(operator: DiagonalizedOperator) -> bool:
    return lx.symmetric_tag in operator.tags


@lx.is_diagonal.register(DiagonalizedOperator)
def _(operator: DiagonalizedOperator) -> bool:
    return False


@lx.is_positive_semidefinite.register(DiagonalizedOperator)
def _(operator: DiagonalizedOperator) -> bool:
    return lx.positive_semidefinite_tag in operator.tags


@lx.is_negative_semidefinite.register(DiagonalizedOperator)
def _(operator: DiagonalizedOperator) -> bool:
    return lx.negative_semidefinite_tag in operator.tags


@lx.is_symmetric.register(Toeplitz)
def _(operator: Toeplitz) -> bool:
    return True


@lx.is_diagonal.register(Toeplitz)
def _(operator: Toeplitz) -> bool:
    return False


@lx.is_positive_semidefinite.register(Toeplitz)
def _(operator: Toeplitz) -> bool:
    return lx.positive_semidefinite_tag in operator.tags


# sum_operator, scaled_operator and product_operator need no registrations:
# they are factories returning lineax's own Add, Mul and Composed operators,
# whose tag propagation lineax provides.


# SumOfKroneckers ∑_j A_j ⊗ B_j


@lx.is_symmetric.register(SumOfKroneckers)
def _(operator: SumOfKroneckers) -> bool:
    return all(lx.is_symmetric(kron) for kron in operator.operators)


@lx.is_diagonal.register(SumOfKroneckers)
def _(operator: SumOfKroneckers) -> bool:
    return False


@lx.is_positive_semidefinite.register(SumOfKroneckers)
def _(operator: SumOfKroneckers) -> bool:
    return lx.positive_semidefinite_tag in operator.tags


# InterpolatedOperator


@lx.is_symmetric.register(InterpolatedOperator)
def _(operator: InterpolatedOperator) -> bool:
    return lx.symmetric_tag in operator.tags


@lx.is_diagonal.register(InterpolatedOperator)
def _(operator: InterpolatedOperator) -> bool:
    return False


@lx.is_positive_semidefinite.register(InterpolatedOperator)
def _(operator: InterpolatedOperator) -> bool:
    return lx.positive_semidefinite_tag in operator.tags


# is_negative_semidefinite: lineax 0.1.1 made it a required dispatch (no
# default). No gaussx operator is negative semidefinite by construction, so
# the predicate reads the children or the tags where that is meaningful and
# is False otherwise.


@lx.is_negative_semidefinite.register(BlockDiag)
def _(operator: BlockDiag) -> bool:
    return all(lx.is_negative_semidefinite(op) for op in operator.operators)


@lx.is_negative_semidefinite.register(Kronecker)
def _(operator: Kronecker) -> bool:
    return False


@lx.is_negative_semidefinite.register(LowRankUpdate)
def _(operator: LowRankUpdate) -> bool:
    return lx.negative_semidefinite_tag in operator.tags


@lx.is_negative_semidefinite.register(KroneckerSum)
def _(operator: KroneckerSum) -> bool:
    return False


@lx.is_negative_semidefinite.register(KroneckerSumSqrt)
def _(operator: KroneckerSumSqrt) -> bool:
    return False


@lx.is_negative_semidefinite.register(BlockTriDiag)
def _(operator: BlockTriDiag) -> bool:
    return lx.negative_semidefinite_tag in operator.tags


@lx.is_negative_semidefinite.register(LowerBlockTriDiag)
def _(operator: LowerBlockTriDiag) -> bool:
    return False


@lx.is_negative_semidefinite.register(UpperBlockTriDiag)
def _(operator: UpperBlockTriDiag) -> bool:
    return False


@lx.is_negative_semidefinite.register(MaskedOperator)
def _(operator: MaskedOperator) -> bool:
    return lx.negative_semidefinite_tag in operator.tags


@lx.is_negative_semidefinite.register(Toeplitz)
def _(operator: Toeplitz) -> bool:
    return lx.negative_semidefinite_tag in operator.tags


@lx.is_negative_semidefinite.register(SumOfKroneckers)
def _(operator: SumOfKroneckers) -> bool:
    return lx.negative_semidefinite_tag in operator.tags


@lx.is_negative_semidefinite.register(InterpolatedOperator)
def _(operator: InterpolatedOperator) -> bool:
    return lx.negative_semidefinite_tag in operator.tags


# is_tridiagonal, is_lower_triangular and is_upper_triangular: lineax 0.1.1
# made every predicate required. No gaussx operator claims entrywise
# tridiagonal or triangular structure (block-tridiagonal is not entrywise
# tridiagonal), so they are False throughout, except where specialised above
# for LowerBlockTriDiag and UpperBlockTriDiag. A class in
# _ALL_TRIDIAG_DEFAULTS also gets the lineax structure functions below.

_ALL_TRIDIAG_DEFAULTS = (
    BlockDiag,
    DiagonalizedOperator,
    Kronecker,
    LowRankUpdate,
    KroneckerSumSqrt,
    KroneckerSum,
    BlockTriDiag,
    LowerBlockTriDiag,
    UpperBlockTriDiag,
    MaskedOperator,
    Toeplitz,
    SumOfKroneckers,
    InterpolatedOperator,
)

_TRI_DEFAULTS = (
    BlockDiag,
    DiagonalizedOperator,
    Kronecker,
    LowRankUpdate,
    KroneckerSum,
    KroneckerSumSqrt,
    BlockTriDiag,
    MaskedOperator,
    Toeplitz,
    SumOfKroneckers,
    InterpolatedOperator,
)

for _cls in _ALL_TRIDIAG_DEFAULTS:
    lx.is_tridiagonal.register(_cls)(lambda _operator: False)
    # lineax 0.1.1 requires has_unit_diagonal for every operator; the
    # Triangular solver queries it for triangular-tagged operators
    # (e.g. LowerBlockTriDiag under AutoLinearSolver).
    lx.has_unit_diagonal.register(_cls)(lambda _operator: False)

for _cls in _TRI_DEFAULTS:
    lx.is_lower_triangular.register(_cls)(lambda _operator: False)
    lx.is_upper_triangular.register(_cls)(lambda _operator: False)


# ``lineax.linearise``, ``materialise``, ``diagonal`` and ``conj``: lineax
# registers them only for its own classes, so every matrix-free solver
# (``lineax.CG``, and through it `CGSolver`) and ``lineax.LSMR`` raised
# ``NotImplementedError`` on gaussx operators, and ``AutoLinearSolver`` did
# too whenever ``is_diagonal`` was ``True`` (a ``Kronecker`` or ``BlockDiag``
# of diagonal factors), since it then calls ``lineax.diagonal`` (gh-410).
# Registering them is what makes the iterative route usable where no
# closed-form structural path exists, such as a `SumOfKroneckers` with three
# or more terms. ``ToeplitzCholesky`` registers its predicates in
# ``_toeplitz.py`` but needs these too; ``SparseOperator`` and
# ``SpectralFunction`` register their own, except ``conj``.

register_lineax_structure_functions(
    *_ALL_TRIDIAG_DEFAULTS, ToeplitzCholesky, _ToeplitzCholeskyTranspose
)
for _cls in (SparseOperator, SpectralFunction):
    lx.conj.register(_cls)(lineax_conj)


# LowerBlockTriDiag and UpperBlockTriDiag are block triangular, so not
# positive semidefinite in general.


@lx.is_positive_semidefinite.register(LowerBlockTriDiag)
def _(operator: LowerBlockTriDiag) -> bool:
    return False


@lx.is_positive_semidefinite.register(UpperBlockTriDiag)
def _(operator: UpperBlockTriDiag) -> bool:
    return False


__all__ = [
    "BlockDiag",
    "BlockTriDiag",
    "CapacitanceSolver",
    "DiagonalizedOperator",
    "InterpolatedOperator",
    "Kronecker",
    "KroneckerSum",
    "KroneckerSumSqrt",
    "LowRankUpdate",
    "LowerBlockTriDiag",
    "MaskedOperator",
    "SVDLowRankUpdate",
    "SparseOperator",
    "SparsityPattern",
    "SpectralFunction",
    "SumKronecker",
    "SumOfKroneckers",
    "Toeplitz",
    "ToeplitzCholesky",
    "UpperBlockTriDiag",
    "as_diagonalized",
    "circulant",
    "circulant_from_symbol",
    "create_grid",
    "cubic_interpolation_weights",
    "grid_coupling_indices",
    "grid_data",
    "is_eigen_reducible",
    "kronecker_sum_sample",
    "low_rank_plus_diag",
    "low_rank_plus_identity",
    "product_operator",
    "scaled_operator",
    "sum_of_kroneckers_sample",
    "sum_operator",
    "sumkronecker_sample",
    "svd_low_rank_plus_diag",
    "toeplitz_sample",
]
