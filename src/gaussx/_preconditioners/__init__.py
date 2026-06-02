"""GaussX preconditioners -- approximate inverses for iterative solvers."""

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
