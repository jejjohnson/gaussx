"""Standalone logdet strategies: SLQ, indefinite SLQ, and dense."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import lineax as lx
import matfree.decomp
import matfree.funm
import matfree.stochtrace
from jaxtyping import Array, Float

from gaussx._primitives._samplers import SamplerName, resolve_sampler
from gaussx._strategies._base import AbstractLogdetStrategy


def _logabsdet_integrand(order: int):
    """Build an SLQ integrand for ``log|det(A)|`` of a symmetric operator."""
    tridiag = matfree.decomp.tridiag_sym(order, reortho="full")
    dense_funm = matfree.funm.dense_funm_sym_eigh(lambda x: jnp.log(jnp.abs(x)))
    return matfree.funm.monte_carlo_funm_sym(dense_funm, tridiag)


def _slq_estimators(
    integrand,
    n: int,
    num_probes: int,
    sampler: SamplerName,
) -> tuple[Callable, Callable]:
    """Build (point, mean-and-sem) SLQ estimators sharing one sampler."""
    probe_fn = resolve_sampler(sampler, n, num_probes)
    point = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    with_sem = matfree.stochtrace.estimator_monte_carlo_mean_and_sem(
        integrand, probe_fn
    )
    return point, with_sem


class SLQLogdet(AbstractLogdetStrategy):
    """Stochastic log-determinant via Lanczos quadrature (SLQ).

    Estimates ``log det(A)`` for PSD operators using stochastic
    trace estimation: ``logdet(A) = tr(log(A))``.  Uses matfree's
    Lanczos decomposition with sign-flip ("Rademacher") probe vectors
    by default.

    Attributes:
        num_probes: Number of probe vectors for Hutchinson estimator.
        lanczos_order: Order of the Lanczos decomposition.
        seed: Seed for probe vector generation (used when no
            ``key`` is passed to :meth:`logdet`).
        sampler: Probe distribution (``"signs"``, ``"normal"``,
            ``"sphere"``).
    """

    num_probes: int = 20
    lanczos_order: int = 30
    seed: int = 0
    sampler: SamplerName = "signs"

    def _integrand(self, n: int):
        order = min(self.lanczos_order, n)
        tridiag = matfree.decomp.tridiag_sym(order, reortho="full")
        return matfree.funm.monte_carlo_funm_sym_logdet(tridiag)

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
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
        point, _ = _slq_estimators(self._integrand(n), n, self.num_probes, self.sampler)
        return point(operator.mv, key)

    def logdet_and_error(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Stochastic ``log det(A)`` with its standard error.

        Args:
            operator: A PSD linear operator.
            key: PRNG key for probe vector sampling.  If ``None``,
                uses ``jax.random.PRNGKey(self.seed)``.

        Returns:
            Tuple ``(estimate, standard_error)`` where the standard
            error is the standard error of the mean across probes.
        """
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        n = operator.in_size()
        _, with_sem = _slq_estimators(
            self._integrand(n), n, self.num_probes, self.sampler
        )
        return with_sem(operator.mv, key)


class IndefiniteSLQLogdet(AbstractLogdetStrategy):
    """Stochastic ``log|det(A)|`` for symmetric (possibly indefinite) operators.

    Like :class:`SLQLogdet` but uses ``log(|lambda|)`` as the matrix
    function, so it works on indefinite and negative-definite matrices.
    Supports a diagonal shift ``(A + shift * I)``.

    Attributes:
        num_probes: Number of probe vectors for Hutchinson estimator.
        lanczos_order: Order of the Lanczos decomposition.
        shift: Diagonal shift applied before computing the logdet.
        seed: Seed for probe vector generation.
        sampler: Probe distribution (``"signs"``, ``"normal"``,
            ``"sphere"``).
    """

    num_probes: int = 20
    lanczos_order: int = 30
    shift: float = 0.0
    seed: int = 0
    sampler: SamplerName = "signs"

    def _shifted_matvec(self, operator: lx.AbstractLinearOperator):
        shift = self.shift

        def matvec(v):
            return operator.mv(v) + shift * v

        return matvec

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
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
        order = min(self.lanczos_order, n)
        point, _ = _slq_estimators(
            _logabsdet_integrand(order), n, self.num_probes, self.sampler
        )
        return point(self._shifted_matvec(operator), key)

    def logdet_and_error(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Stochastic ``log|det(A + shift I)|`` with its standard error.

        Args:
            operator: A symmetric linear operator.
            key: PRNG key for probe vector sampling.  If ``None``,
                uses ``jax.random.PRNGKey(self.seed)``.

        Returns:
            Tuple ``(estimate, standard_error)``.
        """
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        n = operator.in_size()
        order = min(self.lanczos_order, n)
        _, with_sem = _slq_estimators(
            _logabsdet_integrand(order), n, self.num_probes, self.sampler
        )
        return with_sem(self._shifted_matvec(operator), key)


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
    ) -> Float[Array, ""]:
        """Compute ``log |det(A)|`` via structural dispatch.

        Args:
            operator: A linear operator.
            key: Ignored (deterministic).

        Returns:
            Scalar ``log |det(A)|``.
        """
        from gaussx._primitives._logdet import logdet as _logdet

        return _logdet(operator)
