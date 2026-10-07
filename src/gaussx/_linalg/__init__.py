"""GaussX linear-algebra helpers -- compound ops built on primitives."""

from gaussx._linalg._diag_inv import diag_inv
from gaussx._linalg._eigen_factorization import (
    EigenDecomposition,
    kronecker_sum_solve,
)
from gaussx._linalg._linalg import (
    cov_transform,
    diag_conditional_variance,
    sandwich,
    solve_columns,
    solve_matrix,
    solve_rows,
    trace_product,
)
from gaussx._linalg._lyapunov import discrete_lyapunov_solve
from gaussx._linalg._mixed_precision import stable_squared_distances
from gaussx._linalg._safe_cholesky import safe_cholesky
from gaussx._linalg._schur import conditional_variance, schur_complement
from gaussx._linalg._selected_inverse import selected_inverse
from gaussx._linalg._symmetrize import symmetrize
from gaussx._linalg._tridiagonal import (
    tridiagonal_solve,
    tridiagonal_solve_batched,
)
from gaussx._linalg._woodbury import woodbury_solve


__all__ = [
    "EigenDecomposition",
    "conditional_variance",
    "cov_transform",
    "diag_conditional_variance",
    "diag_inv",
    "discrete_lyapunov_solve",
    "kronecker_sum_solve",
    "safe_cholesky",
    "sandwich",
    "schur_complement",
    "selected_inverse",
    "solve_columns",
    "solve_matrix",
    "solve_rows",
    "stable_squared_distances",
    "symmetrize",
    "trace_product",
    "tridiagonal_solve",
    "tridiagonal_solve_batched",
    "woodbury_solve",
]
