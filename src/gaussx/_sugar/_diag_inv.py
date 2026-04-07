"""Diagonal of the inverse: compute diag(A⁻¹) without forming the full inverse."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._strategies._dispatch import dispatch_solve
from gaussx._sugar._safe_cholesky import safe_cholesky


def diag_inv(
    operator: lx.AbstractLinearOperator,
    *,
    method: str = "auto",
    num_probes: int = 30,
    key: jnp.ndarray | None = None,
    solver: AbstractSolveStrategy | None = None,
) -> jnp.ndarray:
    """Compute the diagonal of the inverse of a linear operator.

    Returns ``diag(A⁻¹)`` without forming the full inverse matrix.

    Args:
        operator: A linear operator representing A.
        method: Algorithm to use. One of ``"cholesky"`` (exact via
            dense Cholesky), ``"solve"`` (exact via repeated solves),
            ``"hutchinson"`` (stochastic estimator),
            or ``"auto"`` (cholesky for N ≤ 2048, hutchinson otherwise).
        num_probes: Number of Rademacher probe vectors for the
            hutchinson method.
        key: PRNG key for probe generation in the hutchinson method.
            When ``None``, defaults to ``jax.random.PRNGKey(0)``.
        solver: Optional solve strategy for ``"solve"`` and
            ``"hutchinson"`` methods.

    Returns:
        1D array of shape ``(N,)`` with the diagonal entries of A⁻¹.
    """
    n = operator.in_size()

    if method == "auto":
        method = "cholesky" if n <= 2048 else "hutchinson"

    if method == "cholesky":
        return _diag_inv_cholesky(operator)
    if method == "solve":
        return _diag_inv_solve(operator, solver=solver)
    if method == "hutchinson":
        return _diag_inv_hutchinson(
            operator, num_probes=num_probes, key=key, solver=solver
        )

    msg = (
        f"Unknown method {method!r}; expected 'cholesky', 'solve', "
        "'hutchinson', or 'auto'."
    )
    raise ValueError(msg)


def _diag_inv_cholesky(operator: lx.AbstractLinearOperator) -> jnp.ndarray:
    """Exact diagonal of A⁻¹ via Cholesky factorisation.

    Uses :func:`safe_cholesky` for robustness on ill-conditioned
    matrices, then computes L⁻¹ via triangular solve against the
    identity and returns ``sum_j (L⁻¹)_{j,i}²`` for each column i.
    """
    L = safe_cholesky(operator)
    n = L.shape[0]
    I = jnp.eye(n, dtype=L.dtype)
    L_inv = jax.scipy.linalg.solve_triangular(L, I, lower=True)
    return jnp.sum(L_inv**2, axis=0)


def _diag_inv_solve(
    operator: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolveStrategy | None,
) -> jnp.ndarray:
    """Exact diagonal of A⁻¹ via repeated solves against basis vectors."""
    n = operator.in_size()
    dtype = operator.out_structure().dtype
    I = jnp.eye(n, dtype=dtype)

    def _single_basis(e_i: jnp.ndarray) -> jnp.ndarray:
        Ainv_ei = dispatch_solve(operator, e_i, solver)
        return jnp.sum(e_i * Ainv_ei)

    return jax.vmap(_single_basis)(I)


def _diag_inv_hutchinson(
    operator: lx.AbstractLinearOperator,
    *,
    num_probes: int,
    key: jnp.ndarray | None,
    solver: AbstractSolveStrategy | None,
) -> jnp.ndarray:
    """Stochastic diagonal estimator via Hutchinson's trick.

    Generates Rademacher probe vectors z, solves A⁻¹ z, and
    estimates ``diag(A⁻¹) ≈ mean(z ⊙ A⁻¹z)`` over probes.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n = operator.in_size()
    dtype = operator.out_structure().dtype
    keys = jax.random.split(key, num_probes)

    def _single_probe(k: jnp.ndarray) -> jnp.ndarray:
        z = 2.0 * jax.random.bernoulli(k, shape=(n,)).astype(dtype) - 1.0
        Ainv_z = dispatch_solve(operator, z, solver)
        return z * Ainv_z

    samples = jax.vmap(_single_probe)(keys)
    return jnp.mean(samples, axis=0)
