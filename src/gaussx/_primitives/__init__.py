"""GaussX primitives -- Layer 0 pure functions with structural dispatch."""

from gaussx._primitives._cholesky import DenseFallbackWarning, cholesky
from gaussx._primitives._diag import diag
from gaussx._primitives._eig import eig, eigvals
from gaussx._primitives._frobenius import frobenius_norm
from gaussx._primitives._inv import InverseOperator, inv
from gaussx._primitives._inv_quad_logdet import inv_quad_logdet
from gaussx._primitives._logdet import cholesky_logdet, logdet
from gaussx._primitives._root import (
    RootDecomposition,
    root_decomposition,
    root_inv_decomposition,
)
from gaussx._primitives._solve import solve
from gaussx._primitives._sqrt import SqrtOperator, SumKroneckerSqrt, sqrt
from gaussx._primitives._sqrt_matmul import (
    estimate_spectral_bounds,
    sqrt_inv_matmul,
    sqrt_matmul,
)
from gaussx._primitives._submatrix import submatrix
from gaussx._primitives._svd import svd
from gaussx._primitives._trace import trace, trace_and_diag


__all__ = [
    "DenseFallbackWarning",
    "InverseOperator",
    "RootDecomposition",
    "SqrtOperator",
    "SumKroneckerSqrt",
    "cholesky",
    "cholesky_logdet",
    "diag",
    "eig",
    "eigvals",
    "estimate_spectral_bounds",
    "frobenius_norm",
    "inv",
    "inv_quad_logdet",
    "logdet",
    "root_decomposition",
    "root_inv_decomposition",
    "solve",
    "sqrt",
    "sqrt_inv_matmul",
    "sqrt_matmul",
    "submatrix",
    "svd",
    "trace",
    "trace_and_diag",
]
