"""GaussX operators -- extensions to lineax.AbstractLinearOperator."""

from __future__ import annotations

import lineax as lx

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._low_rank_update import (
    LowRankUpdate,
    low_rank_plus_diag,
    low_rank_plus_identity,
    svd_low_rank_plus_diag,
)
from gaussx._tags import is_block_diagonal, is_kronecker, is_low_rank


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


@lx.is_diagonal.register(BlockDiag)
def _(operator: BlockDiag) -> bool:
    return all(lx.is_diagonal(op) for op in operator.operators)


@lx.is_diagonal.register(Kronecker)
def _(operator: Kronecker) -> bool:
    return all(lx.is_diagonal(op) for op in operator.operators)


@lx.is_diagonal.register(LowRankUpdate)
def _(operator: LowRankUpdate) -> bool:
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


__all__ = [
    "BlockDiag",
    "Kronecker",
    "LowRankUpdate",
    "low_rank_plus_diag",
    "low_rank_plus_identity",
    "svd_low_rank_plus_diag",
]
