"""Standalone logdet strategies: SLQ, indefinite SLQ, and dense."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import matfree.decomp
import matfree.funm
import matfree.stochtrace

from gaussx._strategies._base import AbstractLogdetStrategy


def _logabsdet_integrand(order: int):
    """Build an SLQ integrand for ``log|det(A)|`` of a symmetric operator."""
    tridiag = matfree.decomp.tridiag_sym(order, reortho="full")
    dense_funm = matfree.funm.dense_funm_sym_eigh(lambda x: jnp.log(jnp.abs(x)))
    return matfree.funm.integrand_funm_sym(dense_funm, tridiag)


class SLQLogdet(AbstractLogdetStrategy):
    """Stochastic log-determinant via Lanczos quadrature (SLQ).

    Estimates ``log det(A)`` for PSD operators using stochastic
    trace estimation: ``logdet(A) = tr(log(A))``.  Uses matfree's
    Lanczos decomposition with Rademacher probe vectors.

    Args:
        num_probes: Number of probe vectors for Hutchinson estimator.
        lanczos_order: Order of the Lanczos decomposition.
        seed: Seed for probe vector generation (used when no
            ``key`` is passed to :meth:`logdet`).
    """

    num_probes: int = 20
    lanczos_order: int = 30
    seed: int = 0

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> jnp.ndarray:
        """Stochastic ``log det(A)`` via Lanczos quadrature.

        Args:
            operator: A PSD linear operator.
            key: PRNG key for probe vector sampling.  If ``None``,
                uses ``jax.random.PRNGKey(self.seed)``.

        Returns:
            Scalar estimate of ``log det(A)``.
        """
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        n = operator.in_size()
        order = min(self.lanczos_order, n)

        tridiag = matfree.decomp.tridiag_sym(order, reortho="full")
        integrand = matfree.funm.integrand_funm_sym_logdet(tridiag)

        sample_shape = jnp.zeros(n)
        sampler = matfree.stochtrace.sampler_rademacher(
            sample_shape, num=self.num_probes
        )
        estimator = matfree.stochtrace.estimator(integrand, sampler)

        return estimator(operator.mv, key)


class IndefiniteSLQLogdet(AbstractLogdetStrategy):
    """Stochastic ``log|det(A)|`` for symmetric (possibly indefinite) operators.

    Like :class:`SLQLogdet` but uses ``log(|lambda|)`` as the matrix
    function, so it works on indefinite and negative-definite matrices.
    Supports a diagonal shift ``(A + shift * I)``.

    Args:
        num_probes: Number of probe vectors for Hutchinson estimator.
        lanczos_order: Order of the Lanczos decomposition.
        shift: Diagonal shift applied before computing the logdet.
        seed: Seed for probe vector generation.
    """

    num_probes: int = 20
    lanczos_order: int = 30
    shift: float = 0.0
    seed: int = 0

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> jnp.ndarray:
        r"""Stochastic ``log|det(A + shift I)|`` via Lanczos quadrature.

        Args:
            operator: A symmetric linear operator.
            key: PRNG key for probe vector sampling.  If ``None``,
                uses ``jax.random.PRNGKey(self.seed)``.

        Returns:
            Scalar estimate of ``log|det(A + shift I)|``.
        """
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        n = operator.in_size()
        shift = self.shift

        def matvec(v):
            return operator.mv(v) + shift * v

        order = min(self.lanczos_order, n)
        integrand = _logabsdet_integrand(order)

        sample_shape = jnp.zeros(n)
        sampler = matfree.stochtrace.sampler_rademacher(
            sample_shape, num=self.num_probes
        )
        estimator = matfree.stochtrace.estimator(integrand, sampler)

        return estimator(matvec, key)


class DenseLogdet(AbstractLogdetStrategy):
    """Dense log-determinant via gaussx structural dispatch.

    Delegates to :func:`gaussx.logdet` which automatically selects
    the best algorithm based on operator structure (Diagonal,
    BlockDiag, Kronecker, LowRankUpdate, or dense fallback).
    """

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> jnp.ndarray:
        """Compute ``log |det(A)|`` via structural dispatch.

        Args:
            operator: A linear operator.
            key: Ignored (deterministic).

        Returns:
            Scalar ``log |det(A)|``.
        """
        from gaussx._primitives._logdet import logdet as _logdet

        return _logdet(operator)
