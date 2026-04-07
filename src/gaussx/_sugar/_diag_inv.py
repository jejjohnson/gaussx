"""Diagonal of the inverse: compute diag(A⁻¹) without forming the full inverse."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._primitives._solve import solve


def diag_inv(
    operator: lx.AbstractLinearOperator,
    *,
    method: str = "auto",
    num_probes: int = 30,
    seed: int = 0,
) -> jnp.ndarray:
    """Compute the diagonal of the inverse of a linear operator.

    Returns ``diag(A⁻¹)`` without forming the full inverse matrix.

    Args:
        operator: A linear operator representing A.
        method: Algorithm to use. One of ``"cholesky"`` (exact via
            triangular solve), ``"hutchinson"`` (stochastic estimator),
            or ``"auto"`` (cholesky for N ≤ 2048, hutchinson otherwise).
        num_probes: Number of Rademacher probe vectors for the
            hutchinson method.
        seed: Random seed for probe generation in the hutchinson method.

    Returns:
        1D array of shape ``(N,)`` with the diagonal entries of A⁻¹.
    """
    n = operator.in_size()

    if method == "auto":
        method = "cholesky" if n <= 2048 else "hutchinson"

    if method == "cholesky":
        return _diag_inv_cholesky(operator)
    if method == "hutchinson":
        return _diag_inv_hutchinson(operator, num_probes=num_probes, seed=seed)

    msg = f"Unknown method {method!r}; expected 'cholesky', 'hutchinson', or 'auto'."
    raise ValueError(msg)


def _diag_inv_cholesky(operator: lx.AbstractLinearOperator) -> jnp.ndarray:
    """Exact diagonal of A⁻¹ via Cholesky factorisation.

    Computes L = cholesky(A), then L⁻¹ via triangular solve against
    the identity, and returns ``sum_j (L⁻¹)_{j,i}²`` for each column i.
    """
    A = operator.as_matrix()
    L = jnp.linalg.cholesky(A)
    n = L.shape[0]
    I = jnp.eye(n, dtype=L.dtype)
    L_inv = jax.scipy.linalg.solve_triangular(L, I, lower=True)
    return jnp.sum(L_inv**2, axis=0)


def _diag_inv_hutchinson(
    operator: lx.AbstractLinearOperator,
    *,
    num_probes: int,
    seed: int,
) -> jnp.ndarray:
    """Stochastic diagonal estimator via Hutchinson's trick.

    Generates Rademacher probe vectors z, solves A⁻¹ z, and
    estimates ``diag(A⁻¹) ≈ mean(z ⊙ A⁻¹z)`` over probes.
    """
    n = operator.in_size()
    dtype = operator.out_structure().dtype
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num_probes)

    def _single_probe(k: jnp.ndarray) -> jnp.ndarray:
        z = 2.0 * jax.random.bernoulli(k, shape=(n,)).astype(dtype) - 1.0
        Ainv_z = solve(operator, z)
        return z * Ainv_z

    samples = jax.vmap(_single_probe)(keys)
    return jnp.mean(samples, axis=0)
