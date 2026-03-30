"""GaussX primitives -- Layer 0 pure functions with structural dispatch."""

from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._diag import diag
from gaussx._primitives._inv import InverseOperator, inv
from gaussx._primitives._logdet import logdet
from gaussx._primitives._solve import solve
from gaussx._primitives._sqrt import sqrt
from gaussx._primitives._trace import trace


__all__ = [
    "InverseOperator",
    "cholesky",
    "diag",
    "inv",
    "logdet",
    "solve",
    "sqrt",
    "trace",
]
