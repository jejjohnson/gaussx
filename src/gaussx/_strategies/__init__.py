"""GaussX solver strategies -- encapsulate solve + logdet algorithms."""

from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._bbmm import BBMMSolver
from gaussx._strategies._cg import CGSolver
from gaussx._strategies._composed import ComposedSolver
from gaussx._strategies._dense import DenseSolver
from gaussx._strategies._lsmr import LSMRSolver
from gaussx._strategies._minres import MINRESSolver
from gaussx._strategies._precond_cg import PreconditionedCGSolver


__all__ = [
    "AbstractSolverStrategy",
    "AutoSolver",
    "BBMMSolver",
    "CGSolver",
    "ComposedSolver",
    "DenseSolver",
    "LSMRSolver",
    "MINRESSolver",
    "PreconditionedCGSolver",
]
