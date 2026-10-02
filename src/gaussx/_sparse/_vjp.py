"""Custom VJPs of the log-determinant and the solve through a sparse factor.

Both take ``a``, the lower triangle of the permuted matrix ``A = P Q Pᵀ`` on
``L``'s pattern, as the differentiated input, and the factor ``L`` (computed
from ``a`` by either backend) as a non-differentiated companion. Their
cotangents never run backwards through the factorisation:

- ``log|A|``: ``Ā = Z = A⁻¹`` on the pattern, one Takahashi sweep;
- ``z = A⁻¹ y``: ``ȳ = A⁻¹ z̄`` and ``Ā = −ȳ zᵀ``, symmetrised on the pattern.

``L`` gets a symbolic-zero cotangent (``None``), so the first-order backward
pass never runs through the factorisation either. Second order
(reverse-over-reverse, e.g. a θ Hessian) differentiates these backward
passes, whose Takahashi sweep and solves depend on ``L``; with the JAX
backend ``L`` carries that dependence. ``jax.hessian`` (forward-over-reverse)
is not available: a ``custom_vjp`` has no JVP.

These are cotangents per matrix *entry*. One lower-triangle value sets both
``A_kl`` and ``A_lk``, so `_entry_to_lower` maps an entry cotangent to the
lower values by doubling off the diagonal -- in this one place, for both
backends. The scatter from an operator's stored values onto ``a`` then sends
it on: symmetric storage keeps the factor 2 (``2 Z_ij``), and general storage,
which contributes ``½ Q_ij + ½ Q_ji``, gets ``Z_ij`` per stored value.
"""

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._sparse._numeric import solve_lower, solve_upper
from gaussx._sparse._symbolic import SymbolicCholesky
from gaussx._sparse._takahashi import takahashi


def _entry_to_lower(
    sym: SymbolicCholesky, entry: Float[Array, " nnz_L"]
) -> Float[Array, " nnz_L"]:
    """Cotangent of the lower values from a symmetric per-entry cotangent."""
    off_diagonal = jnp.asarray(sym.rowidx != sym.colidx)
    return jnp.where(off_diagonal, 2 * entry, entry)


def _diagonal(sym: SymbolicCholesky, L: Array) -> Array:
    return L[jnp.asarray(sym.colptr[:-1])]


@ft.partial(jax.custom_vjp, nondiff_argnums=(0,))
def logdet(sym: SymbolicCholesky, a: Array, L: Array) -> Float[Array, ""]:
    """``log|A| = 2 Σ_j log L_jj``, with ``Ā = A⁻¹`` on the pattern."""
    del a
    return 2 * jnp.sum(jnp.log(_diagonal(sym, L)))


def _logdet_fwd(sym, a, L):
    return logdet(sym, a, L), L


def _logdet_bwd(sym, L, g):
    Z = takahashi(sym, L)
    return g * _entry_to_lower(sym, Z), None


logdet.defvjp(_logdet_fwd, _logdet_bwd)


def _solve(sym: SymbolicCholesky, L: Array, y: Array) -> Array:
    return solve_upper(sym, L, solve_lower(sym, L, y))


@ft.partial(jax.custom_vjp, nondiff_argnums=(0,))
def solve(sym: SymbolicCholesky, a: Array, L: Array, y: Array) -> Float[Array, " n"]:
    """``z = A⁻¹ y`` in the permuted order, with ``Ā = −sym(ȳ zᵀ)`` on the pattern."""
    del a
    return _solve(sym, L, y)


def _solve_fwd(sym, a, L, y):
    z = _solve(sym, L, y)
    return z, (L, z)


def _solve_bwd(sym, residuals, z_bar):
    L, z = residuals
    y_bar = _solve(sym, L, z_bar)
    rows = jnp.asarray(sym.rowidx)
    cols = jnp.asarray(sym.colidx)
    entry = -0.5 * (y_bar[rows] * z[cols] + y_bar[cols] * z[rows])
    return _entry_to_lower(sym, entry), None, y_bar


solve.defvjp(_solve_fwd, _solve_bwd)
