"""Gaussian distribution sugar: log-prob, entropy, KL, quadratic form."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._primitives._trace import trace
from gaussx._strategies._base import (
    AbstractLogdetStrategy,
    AbstractSolverStrategy,
    AbstractSolveStrategy,
)
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve


_LOG_2PI = jnp.log(2.0 * jnp.pi)


def quadratic_form(
    operator: lx.AbstractLinearOperator,
    x: Float[Array, " N"],
    *,
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, ""]:
    """Compute ``x^T A^{-1} x`` via a single solve.

    Args:
        operator: A non-singular linear operator A.
        x: Vector, shape ``(N,)``.
        solver: Optional solve strategy. When ``None``, uses
            structural dispatch.

    Returns:
        Scalar ``x^T A^{-1} x``.
    """
    return x @ dispatch_solve(operator, x, solver)


def _gaussian_log_prob_residual(
    residual: Float[Array, " N"],
    cov_operator: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> Float[Array, ""]:
    """Gaussian log-prob given a pre-computed residual ``value - loc``."""
    N = residual.shape[-1]
    alpha = dispatch_solve(cov_operator, residual, solver)
    quad = residual @ alpha
    ld = dispatch_logdet(cov_operator, solver)
    return -0.5 * (N * _LOG_2PI + ld + quad)


def gaussian_log_prob(
    loc: Float[Array, " N"],
    cov_operator: lx.AbstractLinearOperator,
    value: Float[Array, " N"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> Float[Array, ""]:
    """Multivariate normal log-probability.

    Computes::

        log N(value | loc, Sigma)
        = -0.5 * (N log(2 pi) + log|Sigma| + (value - loc)^T Sigma^{-1} (value - loc))

    All expensive operations (``solve``, ``logdet``) dispatch on
    operator structure automatically, or through an explicit *solver*.

    Args:
        loc: Mean vector, shape ``(N,)``.
        cov_operator: Covariance operator Sigma, shape ``(N, N)``.
        value: Observation vector, shape ``(N,)``.
        solver: Optional solver strategy (needs both solve and logdet).
            When ``None``, uses structural dispatch.

    Returns:
        Scalar log-probability.
    """
    return _gaussian_log_prob_residual(value - loc, cov_operator, solver=solver)


def gaussian_entropy(
    cov_operator: lx.AbstractLinearOperator,
    *,
    solver: AbstractLogdetStrategy | None = None,
) -> Float[Array, ""]:
    """Entropy of a multivariate normal ``N(mu, Sigma)``.

    Computes::

        H = 0.5 * (N * (1 + log(2 pi)) + log|Sigma|)

    Independent of the mean.

    Args:
        cov_operator: Covariance operator, shape ``(N, N)``.
        solver: Optional logdet strategy. When ``None``, uses
            structural dispatch.

    Returns:
        Scalar entropy.
    """
    N = cov_operator.in_size()
    ld = dispatch_logdet(cov_operator, solver)
    return 0.5 * (N * (1.0 + _LOG_2PI) + ld)


def kl_standard_normal(
    m: Float[Array, " N"],
    S: lx.AbstractLinearOperator,
    *,
    solver: AbstractLogdetStrategy | None = None,
) -> Float[Array, ""]:
    """KL divergence ``KL(N(m, S) || N(0, I))``.

    Computes::

        KL = 0.5 * (tr(S) + m^T m - N - log|S|)

    Ubiquitous in variational inference as the prior KL term.

    Args:
        m: Mean vector, shape ``(N,)``.
        S: Covariance operator, shape ``(N, N)``.
        solver: Optional logdet strategy. When ``None``, uses
            structural dispatch.

    Returns:
        Scalar KL divergence.
    """
    N = m.shape[-1]
    tr_S = trace(S)
    mTm = m @ m
    ld = dispatch_logdet(S, solver)
    return 0.5 * (tr_S + mTm - N - ld)


def add_jitter(
    operator: lx.AbstractLinearOperator,
    jitter: float = 1e-6,
) -> lx.AbstractLinearOperator:
    """Add diagonal jitter for numerical stability: ``A + eps * I``.

    Args:
        operator: A linear operator, shape ``(N, N)``.
        jitter: Scalar jitter value. Default ``1e-6``.

    Returns:
        ``A + jitter * I`` as a lineax ``AddLinearOperator``.
    """
    n = operator.in_size()
    dtype = operator.out_structure().dtype
    jitter_op = lx.DiagonalLinearOperator(jnp.full(n, jitter, dtype=dtype))
    return operator + jitter_op
