"""Structured linear algebra and Gaussian primitives for JAX."""

__version__ = "0.0.2"

from gaussx._operators import (
    BlockDiag as BlockDiag,
    Kronecker as Kronecker,
    LowRankUpdate as LowRankUpdate,
    low_rank_plus_diag as low_rank_plus_diag,
    svd_low_rank_plus_diag as svd_low_rank_plus_diag,
)
from gaussx._primitives import (
    cholesky as cholesky,
    diag as diag,
    inv as inv,
    logdet as logdet,
    solve as solve,
    sqrt as sqrt,
    trace as trace,
)
from gaussx._strategies import (
    AbstractSolverStrategy as AbstractSolverStrategy,
    CGSolver as CGSolver,
    DenseSolver as DenseSolver,
)
from gaussx._tags import (
    block_diagonal_tag as block_diagonal_tag,
    diagonal_tag as diagonal_tag,
    is_block_diagonal as is_block_diagonal,
    is_diagonal as is_diagonal,
    is_kronecker as is_kronecker,
    is_low_rank as is_low_rank,
    is_lower_triangular as is_lower_triangular,
    is_negative_semidefinite as is_negative_semidefinite,
    is_positive_semidefinite as is_positive_semidefinite,
    is_symmetric as is_symmetric,
    is_upper_triangular as is_upper_triangular,
    kronecker_tag as kronecker_tag,
    low_rank_tag as low_rank_tag,
    lower_triangular_tag as lower_triangular_tag,
    negative_semidefinite_tag as negative_semidefinite_tag,
    positive_semidefinite_tag as positive_semidefinite_tag,
    symmetric_tag as symmetric_tag,
    tridiagonal_tag as tridiagonal_tag,
    unit_diagonal_tag as unit_diagonal_tag,
    upper_triangular_tag as upper_triangular_tag,
)
