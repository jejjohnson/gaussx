"""GaussX primitives -- Layer 0 pure functions with structural dispatch."""

from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._diag import diag
from gaussx._primitives._eig import eig, eigvals
from gaussx._primitives._inv import InverseOperator, inv
from gaussx._primitives._logdet import cholesky_logdet, logdet
from gaussx._primitives._solve import solve
from gaussx._primitives._sqrt import SqrtOperator, sqrt
from gaussx._primitives._svd import svd
from gaussx._primitives._trace import trace


__all__ = [
    "InverseOperator",
    "SqrtOperator",
    "cholesky",
    "cholesky_logdet",
    "diag",
    "eig",
    "eigvals",
    "inv",
    "logdet",
    "solve",
    "sqrt",
    "svd",
    "trace",
]
