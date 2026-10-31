"""Adapter that turns an approximate inverse M⁻¹ ≈ A⁻¹ into a preconditioner.

This is the slot through which PDE-specific approximate inverses enter gaussx:
a spectral Helmholtz solve or a multigrid V-cycle (built in finitevolX /
spectraldiffx) is wrapped here as a preconditioner and passed to a gaussx
solver, without gaussx importing those packages.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import lineax as lx

from gaussx._preconditioners._base import AbstractPreconditioner


class OperatorPreconditioner(AbstractPreconditioner):
    """Use an externally supplied approximate inverse as a preconditioner.

        z = M⁻¹ r,   M⁻¹ ≈ A⁻¹ given

    Attributes:
        approx_inverse: The approximate inverse M⁻¹, either as a lineax
            operator or as a callable ``v -> M⁻¹ v``.
        in_structure: Input structure for the callable form. When ``None`` it
            is taken from the system operator passed to `as_operator`.

    Examples:
        A callable approximate inverse, its structure taken from A:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.MatrixLinearOperator(
        ...     jnp.diag(jnp.array([2.0, 4.0])), lx.positive_semidefinite_tag
        ... )
        >>> preconditioner = gaussx.OperatorPreconditioner(lambda v: v / 2.0)
        >>> M_inv = preconditioner.as_operator(A)
        >>> [float(v) for v in M_inv.mv(jnp.array([1.0, 2.0]))]
        [0.5, 1.0]
        >>> lx.is_positive_semidefinite(M_inv)
        True
    """

    approx_inverse: lx.AbstractLinearOperator | Callable
    in_structure: object = eqx.field(static=True, default=None)

    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """Return the approximate inverse M⁻¹ as a PSD lineax operator.

            M⁻¹ ≈ A⁻¹

        Args:
            operator: The system operator A; used only to infer the input
                structure of a callable ``approx_inverse``.

        Returns:
            M⁻¹, tagged positive semidefinite.

        Raises:
            ValueError: If ``approx_inverse`` is a callable and neither
                ``in_structure`` nor ``operator`` is given.
        """
        approx_inverse = self.approx_inverse
        if isinstance(approx_inverse, lx.AbstractLinearOperator):
            if lx.is_positive_semidefinite(approx_inverse):
                return approx_inverse
            return lx.TaggedLinearOperator(approx_inverse, lx.positive_semidefinite_tag)

        structure = self.in_structure
        if structure is None:
            if operator is None:
                raise ValueError(
                    "OperatorPreconditioner with a callable approximate inverse "
                    "needs `in_structure` or a system operator to infer it."
                )
            structure = operator.out_structure()
        return lx.FunctionLinearOperator(
            approx_inverse, structure, lx.positive_semidefinite_tag
        )
