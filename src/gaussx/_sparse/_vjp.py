"""Custom VJPs of the log-determinant and the solve through a sparse factor.

Both take ``a``, the lower triangle of the permuted matrix P A Pᵀ = L Lᵀ on L's
pattern, as the differentiated input, and the factor L (computed from ``a`` by
either backend) as a non-differentiated companion. With Z = (P A Pᵀ)⁻¹, their
cotangents never run backwards through the factorisation:

- log det(P A Pᵀ): the cotangent on P A Pᵀ is Z on the pattern, one Takahashi sweep;
- z = Z y: ȳ = Z z̄ and the cotangent on P A Pᵀ is −ȳ zᵀ, symmetrised on the pattern.

L gets a symbolic-zero cotangent (``None``), so the first-order backward
pass never runs through the factorisation either. Second order
(reverse-over-reverse, e.g. a θ Hessian) differentiates these backward
passes, whose Takahashi sweep and solves depend on L; with the JAX
backend L carries that dependence. ``jax.hessian`` (forward-over-reverse)
is not available: a ``custom_vjp`` has no JVP.

These are cotangents per matrix *entry*. One lower-triangle value sets both
[P A Pᵀ]_kl and [P A Pᵀ]_lk, so `_entry_to_lower` maps an entry cotangent to the
lower values by doubling off the diagonal -- in this one place, for both
backends. The scatter from an operator's stored values onto ``a`` then sends
it on: symmetric storage keeps the factor 2 (2 [A⁻¹]_ij), and general storage,
which contributes ½ A_ij + ½ A_ji, gets [A⁻¹]_ij per stored value.
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
    """Cotangent of the lower values from a symmetric per-entry cotangent.

    Args:
        sym: The symbolic analysis.
        entry: The per-entry cotangent on L's CSC pattern.

    Returns:
        The cotangent of the lower values: doubled off the diagonal.
    """
    off_diagonal = jnp.asarray(sym.rowidx != sym.colidx)
    # 2 [Ā]_kl, since one lower value sets both [P A Pᵀ]_kl and [P A Pᵀ]_lk
    doubled = 2 * entry
    return jnp.where(off_diagonal, doubled, entry)


def _diagonal(sym: SymbolicCholesky, L: Array) -> Array:
    """The diagonal L_jj of the factor, the first entry of each CSC column.

    Args:
        sym: The symbolic analysis.
        L: The factor's values on its CSC pattern.

    Returns:
        The diagonal, in the permuted order.
    """
    return L[jnp.asarray(sym.colptr[:-1])]


@ft.partial(jax.custom_vjp, nondiff_argnums=(0,))
def logdet(sym: SymbolicCholesky, a: Array, L: Array) -> Float[Array, ""]:
    """Log-determinant through the factor, with the cotangent Z on the pattern.

        log det(P A Pᵀ) = 2 ∑ⱼ log L_jj

    Args:
        sym: The symbolic analysis.
        a: Lower triangle of P A Pᵀ on L's CSC pattern, the differentiated input.
        L: The factor's values on its CSC pattern.

    Returns:
        The scalar log-determinant.
    """
    del a
    diagonal = _diagonal(sym, L)
    # log L_jj
    log_diagonal = jnp.log(diagonal)
    # ∑ⱼ log L_jj
    half_logdet = jnp.sum(log_diagonal)
    # 2 ∑ⱼ log L_jj
    return 2 * half_logdet


def _logdet_fwd(
    sym: SymbolicCholesky, a: Array, L: Array
) -> tuple[Float[Array, ""], Array]:
    """Forward pass of `logdet`: the value, keeping L as the residual.

    Args:
        sym: The symbolic analysis.
        a: Lower triangle of P A Pᵀ on L's CSC pattern.
        L: The factor's values on its CSC pattern.

    Returns:
        Tuple ``(log_det, L)``.
    """
    return logdet(sym, a, L), L


def _logdet_bwd(
    sym: SymbolicCholesky, L: Array, g: Float[Array, ""]
) -> tuple[Array | None, ...]:
    """Backward pass of `logdet`: one Takahashi sweep, no reverse factorisation.

        ā = g Z on the lower values, doubled off the diagonal

    Args:
        sym: The symbolic analysis.
        L: The residual, the factor's values.
        g: Cotangent of the log-determinant.

    Returns:
        Tuple ``(a_bar, None)``: the cotangent of ``a``, and none for L.
    """
    # Z = (P A Pᵀ)⁻¹ on L's pattern
    inverse_entries = takahashi(sym, L)
    lower_cotangent = _entry_to_lower(sym, inverse_entries)
    # ā = g Z
    return g * lower_cotangent, None


logdet.defvjp(_logdet_fwd, _logdet_bwd)


def _solve(sym: SymbolicCholesky, L: Array, y: Array) -> Array:
    """Two triangular solves in the permuted order.

        z = L⁻ᵀ L⁻¹ y

    Args:
        sym: The symbolic analysis.
        L: The factor's values on its CSC pattern.
        y: Right-hand side y, in the permuted order.

    Returns:
        The solution z, in the permuted order.
    """
    # L⁻¹ y
    forward = solve_lower(sym, L, y)
    # L⁻ᵀ L⁻¹ y
    return solve_upper(sym, L, forward)


@ft.partial(jax.custom_vjp, nondiff_argnums=(0,))
def solve(sym: SymbolicCholesky, a: Array, L: Array, y: Array) -> Float[Array, " n"]:
    """Solve in the permuted order, with cotangent −sym(ȳ zᵀ) on the pattern.

        z = (P A Pᵀ)⁻¹ y

    Args:
        sym: The symbolic analysis.
        a: Lower triangle of P A Pᵀ on L's CSC pattern, a differentiated input.
        L: The factor's values on its CSC pattern.
        y: Right-hand side y, in the permuted order, a differentiated input.

    Returns:
        The solution z, in the permuted order.
    """
    del a
    return _solve(sym, L, y)


def _solve_fwd(
    sym: SymbolicCholesky, a: Array, L: Array, y: Array
) -> tuple[Array, tuple[Array, Array]]:
    """Forward pass of `solve`: the solution, keeping ``(L, z)`` as residuals.

    Args:
        sym: The symbolic analysis.
        a: Lower triangle of P A Pᵀ on L's CSC pattern.
        L: The factor's values on its CSC pattern.
        y: Right-hand side y, in the permuted order.

    Returns:
        Tuple ``(z, (L, z))``.
    """
    z = _solve(sym, L, y)
    return z, (L, z)


def _solve_bwd(
    sym: SymbolicCholesky, residuals: tuple[Array, Array], z_bar: Array
) -> tuple[Array | None, ...]:
    """Backward pass of `solve`: one adjoint solve, no reverse factorisation.

        ȳ = (P A Pᵀ)⁻¹ z̄
        [Ā]_kl = −½ (ȳ_k z_l + ȳ_l z_k)

    Args:
        sym: The symbolic analysis.
        residuals: Tuple ``(L, z)`` from the forward pass.
        z_bar: Cotangent z̄ of the solution.

    Returns:
        Tuple ``(a_bar, None, y_bar)``: the cotangents of ``a``, L and y.
    """
    L, z = residuals
    # ȳ = (P A Pᵀ)⁻¹ z̄
    y_bar = _solve(sym, L, z_bar)
    rows = jnp.asarray(sym.rowidx)
    cols = jnp.asarray(sym.colidx)
    # ȳ_k z_l
    row_term = y_bar[rows] * z[cols]
    # ȳ_l z_k
    col_term = y_bar[cols] * z[rows]
    # [Ā]_kl = −½ (ȳ_k z_l + ȳ_l z_k)
    entry = -0.5 * (row_term + col_term)
    return _entry_to_lower(sym, entry), None, y_bar


solve.defvjp(_solve_fwd, _solve_bwd)
