"""Preconditioner protocol.

A preconditioner supplies an approximate inverse ``M^{-1} ~= A^{-1}`` used to
accelerate iterative solves. In gaussx a preconditioner is an
:class:`equinox.Module` that knows how to produce a *positive-semidefinite*
lineax operator applying ``M^{-1}``.

The single abstract method, :meth:`as_operator`, takes the *system* operator
``A`` as an argument. Static preconditioners (e.g. Jacobi, or an externally
supplied approximate inverse) ignore it; data-dependent ones (e.g.
partial-Cholesky) use it to build their factor lazily at solve time. This is
the slot through which PDE-specific approximate inverses (spectral solves,
multigrid V-cycles) enter gaussx -- they are wrapped by
:class:`OperatorPreconditioner` and passed in, so gaussx never needs to import
the packages that build them.
"""

from __future__ import annotations

import abc

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float


class AbstractPreconditioner(eqx.Module):
    """Protocol for preconditioners producing an approximate inverse.

    Subclasses implement :meth:`as_operator`, returning a PSD lineax operator
    that applies ``M^{-1}`` (or ``None`` to disable preconditioning).
    """

    @abc.abstractmethod
    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator | None:
        """Return ``M^{-1}`` as a PSD lineax operator.

        Args:
            operator: The system operator ``A`` being solved. Used by
                data-dependent preconditioners (e.g. partial-Cholesky); ignored
                by static ones.

        Returns:
            A positive-semidefinite operator applying ``M^{-1}``, or ``None`` to
            indicate that no preconditioning should be applied.
        """
        ...

    def __call__(
        self,
        vector: Float[Array, " n"],
        operator: lx.AbstractLinearOperator | None = None,
    ) -> Float[Array, " n"]:
        """Apply ``M^{-1}`` to *vector* (identity when disabled)."""
        op = self.as_operator(operator)
        if op is None:
            return vector
        return op.mv(vector)
