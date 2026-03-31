"""GaussX operators -- extensions to lineax.AbstractLinearOperator."""

from __future__ import annotations

import lineax as lx

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import (
    BlockTriDiag,
    LowerBlockTriDiag,
    UpperBlockTriDiag,
)
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._operators._low_rank_update import (
    LowRankUpdate,
    low_rank_plus_diag,
    low_rank_plus_identity,
    svd_low_rank_plus_diag,
)
from gaussx._operators._svd_low_rank_update import SVDLowRankUpdate
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


@is_low_rank.register(SVDLowRankUpdate)
def _(operator: SVDLowRankUpdate) -> bool:
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


@lx.is_symmetric.register(SVDLowRankUpdate)
def _(operator: SVDLowRankUpdate) -> bool:
    return lx.symmetric_tag in operator.tags


@lx.is_symmetric.register(KroneckerSum)
def _(operator: KroneckerSum) -> bool:
    return lx.is_symmetric(operator.A) and lx.is_symmetric(operator.B)


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


@lx.is_diagonal.register(SVDLowRankUpdate)
def _(operator: SVDLowRankUpdate) -> bool:
    return False


@lx.is_diagonal.register(KroneckerSum)
def _(operator: KroneckerSum) -> bool:
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


@lx.is_positive_semidefinite.register(SVDLowRankUpdate)
def _(operator: SVDLowRankUpdate) -> bool:
    return lx.positive_semidefinite_tag in operator.tags


@lx.is_positive_semidefinite.register(KroneckerSum)
def _(operator: KroneckerSum) -> bool:
    return lx.is_positive_semidefinite(operator.A) and lx.is_positive_semidefinite(
        operator.B
    )


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


__all__ = [
    "BlockDiag",
    "BlockTriDiag",
    "Kronecker",
    "KroneckerSum",
    "LowRankUpdate",
    "LowerBlockTriDiag",
    "SVDLowRankUpdate",
    "UpperBlockTriDiag",
    "low_rank_plus_diag",
    "low_rank_plus_identity",
    "svd_low_rank_plus_diag",
]
