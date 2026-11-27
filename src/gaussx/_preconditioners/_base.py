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

A preconditioner that knows P itself, the approximation P ≈ A whose inverse
`as_operator` applies, can also take part in a stochastic log-determinant
(Gyger, Furrer & Sigrist, 2026, §4.1): `logdet` gives log det P, `sample`
draws probes z ~ 𝒩(0, P), and `approximation` returns P as an operator with
a structured solve and log-determinant, so that

    log det A = log det P + log det(P^(−½) A P^(−½))

leaves only a near-identity operator to the estimator. `supports_slq` says
whether the three are available. Note the direction: `as_operator` returns
M⁻¹ = P⁻¹, while `logdet`, `sample` and `approximation` are about P.
"""

from __future__ import annotations

import abc

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray


class AbstractPreconditioner(eqx.Module):
    """Protocol for preconditioners that produce an approximate inverse.

        M⁻¹ ≈ A⁻¹,   z = M⁻¹ r

    Subclasses implement `as_operator`, returning a PSD lineax operator that
    applies M⁻¹ (or ``None`` to disable preconditioning). A subclass that
    knows P = M itself also implements `logdet`, `sample` and
    `approximation`, and returns ``True`` from `supports_slq`; the defaults
    raise ``NotImplementedError`` and return ``False``.

    Examples:
        A concrete preconditioner applied to a vector:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> jacobi = gaussx.JacobiPreconditioner(diagonal=jnp.array([2.0, 4.0]))
        >>> isinstance(jacobi, gaussx.AbstractPreconditioner)
        True
        >>> [float(v) for v in jacobi(jnp.array([1.0, 1.0]))]
        [0.5, 0.25]

        The same preconditioner as P itself, for a log-determinant:

        >>> jacobi.supports_slq()
        True
        >>> round(float(jacobi.logdet()), 4)  # log 2 + log 4
        2.0794
        >>> jacobi.approximation().as_matrix().diagonal().tolist()
        [2.0, 4.0]
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

    def supports_slq(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> bool:
        """Whether `logdet`, `sample` and `approximation` are available.

        Static (a Python ``bool`` decided by the class), so it is safe to
        branch on under ``jit``.

        Args:
            operator: The system operator A; unused by the gaussx
                preconditioners, kept so a subclass can decide per system.

        Returns:
            ``False`` by default; ``True`` for a preconditioner that knows P.
        """
        del operator
        return False

    def logdet(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> Float[Array, ""]:
        """Log-determinant of the preconditioner P ≈ A itself.

            log det P = −log det M⁻¹

        This is log det P, **not** log det M⁻¹ of the operator `as_operator`
        returns.

        Args:
            operator: The system operator A, for preconditioners built from
                it at solve time.

        Returns:
            The scalar log det P.

        Raises:
            NotImplementedError: Unless the subclass knows P.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not know its log-determinant; "
            "check supports_slq() first."
        )

    def sample(
        self,
        key: PRNGKeyArray,
        operator: lx.AbstractLinearOperator | None = None,
        *,
        shape: tuple[int, ...] = (),
    ) -> Float[Array, "*batch n"]:
        """Draw probes z ~ 𝒩(0, P) from the preconditioner P ≈ A.

        Args:
            key: PRNG key.
            operator: The system operator A, for preconditioners built from
                it at solve time.
            shape: Batch shape of the draws; the result is ``(*shape, n)``.

        Returns:
            The draws z, of shape ``(*shape, n)``.

        Raises:
            NotImplementedError: Unless the subclass knows P.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot sample from its approximation; "
            "check supports_slq() first."
        )

    def approximation(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """The preconditioner P ≈ A itself, as a PSD operator.

            P = (M⁻¹)⁻¹

        The inverse of what `as_operator` returns. It has a structured
        solve and log-determinant, so it can be passed as the
        ``logdet_preconditioner`` of `gaussx.inv_quad_logdet`.

        Args:
            operator: The system operator A, for preconditioners built from
                it at solve time.

        Returns:
            P as a PSD lineax operator.

        Raises:
            NotImplementedError: Unless the subclass knows P.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not know its approximation P; "
            "check supports_slq() first."
        )
