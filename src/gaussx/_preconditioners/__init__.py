"""Preconditioners: approximate inverses M⁻¹ ≈ A⁻¹ for iterative solvers.

A conjugate-gradient (CG) solve of A x = b applies M⁻¹ to each residual r_j,
z_j = M⁻¹ r_j. Each preconditioner here builds M⁻¹ as a PSD lineax operator:
Jacobi (M = diag(A)), a rank-k randomized Nyström approximation, a rank-k
pivoted partial Cholesky factor, or an approximate inverse supplied by the
caller.
"""

from gaussx._preconditioners._base import AbstractPreconditioner
from gaussx._preconditioners._jacobi import JacobiPreconditioner
from gaussx._preconditioners._nystrom import NystromPreconditioner
from gaussx._preconditioners._operator import OperatorPreconditioner
from gaussx._preconditioners._partial_cholesky import PartialCholeskyPreconditioner


__all__ = [
    "AbstractPreconditioner",
    "JacobiPreconditioner",
    "NystromPreconditioner",
    "OperatorPreconditioner",
    "PartialCholeskyPreconditioner",
]
