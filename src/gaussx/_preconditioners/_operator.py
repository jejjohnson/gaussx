"""Adapter that turns an approximate inverse into a preconditioner.

This is the slot through which PDE-specific approximate inverses enter gaussx:
a spectral Helmholtz solve or a multigrid V-cycle (built in finitevolX /
spectraldiffx) is wrapped here as a preconditioner and passed to a gaussx
solver, without gaussx importing those packages.
"""

from __future__ import annotations

from collections.abc import Callable

import lineax as lx

from gaussx._preconditioners._base import AbstractPreconditioner


class OperatorPreconditioner(AbstractPreconditioner):
    """Use an externally supplied approximate inverse as a preconditioner.

    Attributes:
        approx_inverse: The approximate inverse ``M^{-1}``, either as a lineax
            operator or as a callable ``v -> M^{-1} v``.
        in_structure: Input structure for the callable form. When ``None`` it is
            taken from the system operator passed to :meth:`as_operator`.
    """

    approx_inverse: lx.AbstractLinearOperator | Callable
    in_structure: object = None

    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """Return the approximate inverse as a PSD lineax operator."""
        ai = self.approx_inverse
        if isinstance(ai, lx.AbstractLinearOperator):
            if lx.is_positive_semidefinite(ai):
                return ai
            return lx.TaggedLinearOperator(ai, lx.positive_semidefinite_tag)

        structure = self.in_structure
        if structure is None:
            if operator is None:
                raise ValueError(
                    "OperatorPreconditioner with a callable approximate inverse "
                    "needs `in_structure` or a system operator to infer it."
                )
            structure = operator.out_structure()
        return lx.FunctionLinearOperator(ai, structure, lx.positive_semidefinite_tag)
