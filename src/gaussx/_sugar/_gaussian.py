"""Gaussian distribution sugar: log-prob, entropy, KL, quadratic form."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx

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
    x: jnp.ndarray,
    *,
    solver: AbstractSolveStrategy | None = None,
) -> jnp.ndarray:
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


def gaussian_log_prob(
    loc: jnp.ndarray,
    cov_operator: lx.AbstractLinearOperator,
    value: jnp.ndarray,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> jnp.ndarray:
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
    N = loc.shape[-1]
    residual = value - loc
    alpha = dispatch_solve(cov_operator, residual, solver)
    quad = residual @ alpha
    ld = dispatch_logdet(cov_operator, solver)
    return -0.5 * (N * _LOG_2PI + ld + quad)


def gaussian_entropy(
    cov_operator: lx.AbstractLinearOperator,
    *,
    solver: AbstractLogdetStrategy | None = None,
) -> jnp.ndarray:
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
    m: jnp.ndarray,
    S: lx.AbstractLinearOperator,
    *,
    solver: AbstractLogdetStrategy | None = None,
) -> jnp.ndarray:
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
