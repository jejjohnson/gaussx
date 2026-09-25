"""GaussX operators -- extensions to lineax.AbstractLinearOperator."""

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
    Circulant,
    DiagonalisedOperator,
    as_diagonalised,
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
    ProductOperator,
    ScaledOperator,
    SumOperator,
)
from gaussx._operators._low_rank_update import (
    LowRankUpdate,
    SVDLowRankUpdate,
    low_rank_plus_diag,
    low_rank_plus_identity,
    svd_low_rank_plus_diag,
)
from gaussx._operators._masked import MaskedOperator, grid_coupling_indices
from gaussx._operators._sum_kronecker import (
    SumKronecker,
    SumOfKroneckers,
    sumkronecker_sample as sumkronecker_sample,
)
from gaussx._operators._toeplitz import Toeplitz, ToeplitzCholesky, toeplitz_sample
from gaussx._tags import (
    is_block_diagonal,
    is_block_tridiagonal,
    is_kronecker,
    is_kronecker_sum,
    is_low_rank,
)


# -------------------------------------------------------------------
# Register gaussx operators with tag query singledispatch functions
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


# Register with lineax's is_symmetric, is_diagonal, is_positive_semidefinite
# so that lineax solvers and gaussx dispatch can query these operators.


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


# LowerBlockTriDiag / UpperBlockTriDiag tag registrations


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


# MaskedOperator tag registrations


@lx.is_symmetric.register(MaskedOperator)
def _(operator: MaskedOperator) -> bool:
    return lx.symmetric_tag in operator.tags


@lx.is_diagonal.register(MaskedOperator)
def _(operator: MaskedOperator) -> bool:
    return False


@lx.is_positive_semidefinite.register(MaskedOperator)
def _(operator: MaskedOperator) -> bool:
    return lx.positive_semidefinite_tag in operator.tags


# Toeplitz tag registrations


@lx.is_symmetric.register(DiagonalisedOperator)
def _(operator: DiagonalisedOperator) -> bool:
    return lx.symmetric_tag in operator.tags


@lx.is_diagonal.register(DiagonalisedOperator)
def _(operator: DiagonalisedOperator) -> bool:
    return False


@lx.is_positive_semidefinite.register(DiagonalisedOperator)
def _(operator: DiagonalisedOperator) -> bool:
    return lx.positive_semidefinite_tag in operator.tags


@lx.is_negative_semidefinite.register(DiagonalisedOperator)
def _(operator: DiagonalisedOperator) -> bool:
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


# SumOperator / ScaledOperator / ProductOperator need no registrations:
# they are factories returning lineax-native Add/Mul/Composed operators,
# whose tag propagation lineax provides out of the box.


# SumOfKroneckers tag registrations


@lx.is_symmetric.register(SumOfKroneckers)
def _(operator: SumOfKroneckers) -> bool:
    return all(lx.is_symmetric(kron) for kron in operator.operators)


@lx.is_diagonal.register(SumOfKroneckers)
def _(operator: SumOfKroneckers) -> bool:
    return False


@lx.is_positive_semidefinite.register(SumOfKroneckers)
def _(operator: SumOfKroneckers) -> bool:
    return lx.positive_semidefinite_tag in operator.tags


# InterpolatedOperator tag registrations


@lx.is_symmetric.register(InterpolatedOperator)
def _(operator: InterpolatedOperator) -> bool:
    return lx.symmetric_tag in operator.tags


@lx.is_diagonal.register(InterpolatedOperator)
def _(operator: InterpolatedOperator) -> bool:
    return False


@lx.is_positive_semidefinite.register(InterpolatedOperator)
def _(operator: InterpolatedOperator) -> bool:
    return lx.positive_semidefinite_tag in operator.tags


# is_negative_semidefinite registrations.
# lineax 0.1.1 promoted this to a required dispatch (no default). None of the
# gaussx operators claim NSD by construction — propagate to children where it
# would be meaningful, otherwise return False.


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


# is_tridiagonal / is_lower_triangular / is_upper_triangular registrations.
# lineax 0.1.1 made all predicates required. None of the gaussx operators
# claim element-wise tridiagonal or triangular structure (block-tridiagonal
# is not the same as element tridiagonal); register False uniformly except
# where already specialised above for LowerBlockTriDiag / UpperBlockTriDiag.

_ALL_TRIDIAG_DEFAULTS = (
    BlockDiag,
    DiagonalisedOperator,
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
    DiagonalisedOperator,
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


# ``lineax.linearise`` is the identity for an operator that is already
# linear, but lineax registers it only for its own classes — so every
# matrix-free solver (``lineax.CG``, and through it `CGSolver`) raised
# ``NotImplementedError`` on gaussx operators. Registering it is what makes
# the iterative route usable where no closed-form structural path exists,
# such as a `SumOfKroneckers` with three or more terms.

for _cls in _ALL_TRIDIAG_DEFAULTS:
    lx.linearise.register(_cls)(lambda operator: operator)


# LowerBlockTriDiag / UpperBlockTriDiag: PSD/NSD defaults (block triangular,
# not generally PSD).


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
    "Circulant",
    "DiagonalisedOperator",
    "InterpolatedOperator",
    "Kronecker",
    "KroneckerSum",
    "KroneckerSumSqrt",
    "LowRankUpdate",
    "LowerBlockTriDiag",
    "MaskedOperator",
    "ProductOperator",
    "SVDLowRankUpdate",
    "ScaledOperator",
    "SumKronecker",
    "SumOfKroneckers",
    "SumOperator",
    "Toeplitz",
    "ToeplitzCholesky",
    "UpperBlockTriDiag",
    "as_diagonalised",
    "circulant_from_symbol",
    "create_grid",
    "cubic_interpolation_weights",
    "grid_coupling_indices",
    "grid_data",
    "kronecker_sum_sample",
    "low_rank_plus_diag",
    "low_rank_plus_identity",
    "sumkronecker_sample",
    "svd_low_rank_plus_diag",
    "toeplitz_sample",
]
