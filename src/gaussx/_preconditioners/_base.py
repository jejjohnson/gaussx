"""Preconditioner protocol.

A preconditioner M of a system A x = b supplies an approximate inverse
M⁻¹ ≈ A⁻¹, which conjugate gradients (CG) applies to each residual r_j,
z_j = M⁻¹ r_j, to cut the number of steps. In gaussx a preconditioner is an
`equinox.Module` that produces a *positive-semidefinite* lineax operator
applying M⁻¹.

The single abstract method, `as_operator`, takes the *system* operator A as
an argument. Static preconditioners (e.g. Jacobi, or an externally supplied
approximate inverse) ignore it; data-dependent ones (e.g. partial Cholesky)
use it to build their factor lazily at solve time. This is the slot through
which PDE-specific approximate inverses (spectral solves, multigrid
V-cycles) enter gaussx: they are wrapped by `OperatorPreconditioner` and
passed in, so gaussx never needs to import the packages that build them.
"""

from __future__ import annotations

import abc

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float


class AbstractPreconditioner(eqx.Module):
    """Protocol for preconditioners that produce an approximate inverse.

        M⁻¹ ≈ A⁻¹,   z = M⁻¹ r

    Subclasses implement `as_operator`, returning a PSD lineax operator that
    applies M⁻¹ (or ``None`` to disable preconditioning).

    Examples:
        A concrete preconditioner applied to a vector:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> jacobi = gaussx.JacobiPreconditioner(diagonal=jnp.array([2.0, 4.0]))
        >>> isinstance(jacobi, gaussx.AbstractPreconditioner)
        True
        >>> [float(v) for v in jacobi(jnp.array([1.0, 1.0]))]
        [0.5, 0.25]
    """

    @abc.abstractmethod
    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator | None:
        """Return M⁻¹ as a PSD lineax operator.

            M⁻¹ ≈ A⁻¹

        Args:
            operator: The system operator A being solved. Used by
                data-dependent preconditioners (e.g. partial Cholesky);
                ignored by static ones.

        Returns:
            A positive-semidefinite operator applying M⁻¹, or ``None`` when
            no preconditioning should be applied.
        """
        ...

    def __call__(
        self,
        vector: Float[Array, " n"],
        operator: lx.AbstractLinearOperator | None = None,
    ) -> Float[Array, " n"]:
        """Apply M⁻¹ to a vector (the identity when disabled).

            z = M⁻¹ r

        Args:
            vector: Vector r, e.g. a CG residual.
            operator: The system operator A, passed to `as_operator`.

        Returns:
            The preconditioned vector z = M⁻¹ r, or r unchanged when
            `as_operator` returns ``None``.
        """
        precond_op = self.as_operator(operator)
        if precond_op is None:
            return vector
        # z = M⁻¹ r
        return precond_op.mv(vector)
