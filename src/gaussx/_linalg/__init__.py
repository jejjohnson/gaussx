"""GaussX linear-algebra helpers -- compound ops built on primitives."""

from gaussx._linalg._batched_matvec import (
    batched_kernel_matvec,
    batched_kernel_rmatvec,
)
from gaussx._linalg._diag_inv import diag_inv
from gaussx._linalg._linalg import (
    cov_transform,
    diag_conditional_variance,
    solve_columns,
    solve_rows,
    trace_product,
)
from gaussx._linalg._mixed_precision import (
    stable_rbf_kernel,
    stable_squared_distances,
)
from gaussx._linalg._safe_cholesky import safe_cholesky
from gaussx._linalg._schur import conditional_variance, schur_complement
from gaussx._linalg._woodbury import woodbury_solve


__all__ = [
    "batched_kernel_matvec",
    "batched_kernel_rmatvec",
    "conditional_variance",
    "cov_transform",
    "diag_conditional_variance",
    "diag_inv",
    "safe_cholesky",
    "schur_complement",
    "solve_columns",
    "solve_rows",
    "stable_rbf_kernel",
    "stable_squared_distances",
    "trace_product",
    "woodbury_solve",
]
