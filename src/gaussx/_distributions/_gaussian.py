"""Gaussian distribution sugar: log-prob, entropy, KL, quadratic form."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._primitives._logdet import _cholesky_logdet, _dense_psd_matrix
from gaussx._primitives._trace import trace
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import (
    AbstractLogdetStrategy,
    AbstractSolverStrategy,
    AbstractSolveStrategy,
)
from gaussx._strategies._dense import DenseSolver
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve
from gaussx._strategies._keyed import KeyedSolver


# A Python float, not a jnp array: it is weakly typed, so it takes the dtype
# of the array it meets, and it does not freeze the default dtype that was
# active at import (x64 is often enabled after `import gaussx`).
_LOG_2PI = math.log(2.0 * math.pi)


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
    key: jax.Array | None = None,
) -> Float[Array, ""]:
    """Gaussian log-prob given a pre-computed residual ``value - loc``."""
    N = residual.shape[-1]
    matrix = _dense_psd_matrix(cov_operator)
    if matrix is not None and _uses_structural_dispatch(cov_operator, solver):
        # Factor once (gh-329): one Cholesky gives both the quadratic form
        # and the log-determinant, where a solve plus a logdet factor twice.
        factor = jnp.linalg.cholesky(matrix)
        whitened = jax.scipy.linalg.solve_triangular(factor, residual, lower=True)
        return -0.5 * (N * _LOG_2PI + _cholesky_logdet(factor) + whitened @ whitened)
    alpha = dispatch_solve(cov_operator, residual, solver)
    quad = residual @ alpha
    ld = dispatch_logdet(cov_operator, solver, key=key)
    return -0.5 * (N * _LOG_2PI + ld + quad)


def _uses_structural_dispatch(
    operator: lx.AbstractLinearOperator,
    solver: AbstractSolverStrategy | None,
) -> bool:
    """Whether *solver* would solve *operator* by `gaussx.solve` and `logdet`.

    Only then may the factor-once path stand in for it: an explicit
    iterative or stochastic strategy must keep its own semantics.
    """
    if solver is None or isinstance(solver, DenseSolver):
        return True
    if isinstance(solver, KeyedSolver) and isinstance(
        solver.strategy, AbstractSolverStrategy
    ):
        return _uses_structural_dispatch(operator, solver.strategy)
    if isinstance(solver, AutoSolver):
        return isinstance(solver._get_strategy(operator), DenseSolver)
    return False


def gaussian_log_prob(
    loc: Float[Array, " N"],
    cov_operator: lx.AbstractLinearOperator,
    value: Float[Array, " N"],
    *,
    solver: AbstractSolverStrategy | None = None,
    key: jax.Array | None = None,
) -> Float[Array, ""]:
    """Multivariate normal log-probability.

    Computes:

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
        key: PRNG key for a stochastic logdet strategy's probes. ``None``
            uses the strategy's own seed, i.e. the same probes on every call
            (see `gaussx.KeyedSolver`). Ignored by exact strategies.

    Returns:
        Scalar log-probability.
    """
    return _gaussian_log_prob_residual(
        value - loc, cov_operator, solver=solver, key=key
    )


def gaussian_entropy(
    cov_operator: lx.AbstractLinearOperator,
    *,
    solver: AbstractLogdetStrategy | None = None,
    key: jax.Array | None = None,
) -> Float[Array, ""]:
    """Entropy of a multivariate normal ``N(mu, Sigma)``.

    Computes:

        H = 0.5 * (N * (1 + log(2 pi)) + log|Sigma|)

    Independent of the mean.

    Args:
        cov_operator: Covariance operator, shape ``(N, N)``.
        solver: Optional logdet strategy. When ``None``, uses
            structural dispatch.
        key: PRNG key for a stochastic logdet strategy's probes. ``None``
            uses the strategy's own seed, i.e. the same probes on every call
            (see `gaussx.KeyedSolver`). Ignored by exact strategies.

    Returns:
        Scalar entropy.
    """
    N = cov_operator.in_size()
    ld = dispatch_logdet(cov_operator, solver, key=key)
    return 0.5 * (N * (1.0 + _LOG_2PI) + ld)


def kl_standard_normal(
    m: Float[Array, " N"],
    S: lx.AbstractLinearOperator,
    *,
    solver: AbstractLogdetStrategy | None = None,
    key: jax.Array | None = None,
) -> Float[Array, ""]:
    """KL divergence ``KL(N(m, S) || N(0, I))``.

    Special case of `dist_kl_divergence`
    with ``q_loc = 0`` and ``q_cov = I``.  The identity prior means no
    matrix inversion is required, making this more efficient than calling
    the general form directly.

    Computes:

        KL = 0.5 * (tr(S) + m^T m - N - log|S|)

    Ubiquitous in variational inference as the prior KL term.

    Args:
        m: Mean vector, shape ``(N,)``.
        S: Covariance operator, shape ``(N, N)``.
        solver: Optional logdet strategy. When ``None``, uses
            structural dispatch.
        key: PRNG key for a stochastic logdet strategy's probes. ``None``
            uses the strategy's own seed, i.e. the same probes on every call
            (see `gaussx.KeyedSolver`). Ignored by exact strategies.

    Returns:
        Scalar KL divergence.

    See Also:
        `dist_kl_divergence`: General KL
        between two multivariate normals with arbitrary lineax covariance
        operators.
    """
    N = m.shape[-1]
    tr_S = trace(S)
    mTm = m @ m
    ld = dispatch_logdet(S, solver, key=key)
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
