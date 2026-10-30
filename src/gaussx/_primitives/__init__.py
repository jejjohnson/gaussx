"""Layer 0 primitives: pure functions of an operator A with structural dispatch.

The structured primitives (solve A x = b, log det A, the Cholesky factor L
with A = L Lᵀ, diag(A), tr(A), ‖A‖_F, A^½, A⁻¹, the eigendecomposition
A = Q diag(γ) Qᵀ, the SVD and root decompositions A ≈ R Rᵀ) take the first
``isinstance`` rule that matches A's structure before any dense fallback.
The package also holds submatrix extraction, mixture quantiles and the
Chandrupatla root finder they use.
"""

from gaussx._primitives._chandrupatla import Chandrupatla
from gaussx._primitives._cholesky import DenseFallbackWarning, cholesky
from gaussx._primitives._diag import diag
from gaussx._primitives._eig import eig, eigh_generalized, eigvals
from gaussx._primitives._frobenius import frobenius_norm
from gaussx._primitives._inv import InverseOperator, inv
from gaussx._primitives._inv_quad_logdet import inv_quad_logdet
from gaussx._primitives._logdet import cholesky_logdet, logdet, pseudo_logdet
from gaussx._primitives._quantile import (
    mixture_quantile,
    mixture_quantile_gaussian_approx,
)
from gaussx._primitives._root import (
    RootDecomposition,
    root_decomposition,
    root_inv_decomposition,
)
from gaussx._primitives._solve import solve
from gaussx._primitives._sqrt import (
    SqrtOperator,
    SumKroneckerSqrt,
    SumOfKroneckersSqrt,
    sqrt,
)
from gaussx._primitives._sqrt_matmul import (
    estimate_spectral_bounds,
    sqrt_inv_matmul,
    sqrt_matmul,
)
from gaussx._primitives._submatrix import submatrix
from gaussx._primitives._svd import svd
from gaussx._primitives._trace import trace, trace_and_diag


__all__ = [
    "Chandrupatla",
    "DenseFallbackWarning",
    "InverseOperator",
    "RootDecomposition",
    "SqrtOperator",
    "SumKroneckerSqrt",
    "SumOfKroneckersSqrt",
    "cholesky",
    "cholesky_logdet",
    "diag",
    "eig",
    "eigh_generalized",
    "eigvals",
    "estimate_spectral_bounds",
    "frobenius_norm",
    "inv",
    "inv_quad_logdet",
    "logdet",
    "mixture_quantile",
    "mixture_quantile_gaussian_approx",
    "pseudo_logdet",
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
