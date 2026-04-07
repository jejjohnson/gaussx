"""GaussX solver strategies -- encapsulate solve + logdet algorithms."""

from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import (
    AbstractLogdetStrategy,
    AbstractSolverStrategy,
    AbstractSolveStrategy,
)
from gaussx._strategies._bbmm import BBMMSolver
from gaussx._strategies._cg import CGSolver
from gaussx._strategies._composed import ComposedSolver
from gaussx._strategies._dense import DenseSolver
from gaussx._strategies._lsmr import LSMRSolver
from gaussx._strategies._minres import MINRESSolver
from gaussx._strategies._precond_cg import PreconditionedCGSolver
from gaussx._strategies._slq_logdet import (
    DenseLogdet,
    IndefiniteSLQLogdet,
    SLQLogdet,
)


__all__ = [
    "AbstractLogdetStrategy",
    "AbstractSolveStrategy",
    "AbstractSolverStrategy",
    "AutoSolver",
    "BBMMSolver",
    "CGSolver",
    "ComposedSolver",
    "DenseLogdet",
    "DenseSolver",
    "IndefiniteSLQLogdet",
    "LSMRSolver",
    "MINRESSolver",
    "PreconditionedCGSolver",
    "SLQLogdet",
]
