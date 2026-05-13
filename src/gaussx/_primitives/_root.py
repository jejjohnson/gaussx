"""Root and inverse-root decompositions for SPD operators."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float

from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._eig import eig
from gaussx._primitives._inv import inv
from gaussx._primitives._svd import svd


RootMethod = Literal["cholesky", "lanczos", "pivoted_cholesky", "svd"]


class RootDecomposition(eqx.Module):
    r"""Tall factor ``R`` with ``R Rᵀ ≈ A`` semantics.

    Attributes:
        root: Tall factor with shape ``(N, k)``.
    """

    root: Float[Array, "N k"]

    @property
    def rank(self) -> int:
        """Number of retained root directions."""
        return self.root.shape[1]

    def matmul(self, x: Float[Array, "*B k"]) -> Float[Array, "*B N"]:
        """Right-multiply by the root factor: ``x Rᵀ``."""
        return x @ self.root.T


def root_decomposition(
    operator: lx.AbstractLinearOperator,
    rank: int = 50,
    method: RootMethod = "lanczos",
    key: jax.Array | None = None,
) -> RootDecomposition:
    r"""Compute a tall factor ``R`` such that ``R Rᵀ ≈ A``.

    Args:
        operator: Square symmetric positive-definite operator ``A``.
        rank: Number of retained directions. Ignored by ``"cholesky"``,
            which returns the exact full-rank factor.
        method: Decomposition method: ``"lanczos"``, ``"cholesky"``,
            ``"pivoted_cholesky"``, or ``"svd"``.
        key: PRNG key for random-start methods.

    Returns:
        A :class:`RootDecomposition` with root shape ``(N, k)``.
    """
    _n, rank = _validate_square_rank(operator, rank, method=method)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return RootDecomposition(_diagonal_root(operator, rank, method, inverse=False))
    if method == "cholesky":
        return RootDecomposition(_cholesky_root(operator))
    if method == "lanczos":
        return RootDecomposition(_eig_root(operator, rank, key, inverse=False))
    if method == "svd":
        return RootDecomposition(_svd_root(operator, rank, key, inverse=False))
    if method == "pivoted_cholesky":
        mat = _symmetric_operator_matrix(operator)
        return RootDecomposition(_pivoted_cholesky_root(mat, rank))
    raise ValueError(f"Unknown root decomposition method: {method!r}")


def root_inv_decomposition(
    operator: lx.AbstractLinearOperator,
    rank: int = 50,
    method: RootMethod = "lanczos",
    key: jax.Array | None = None,
) -> RootDecomposition:
    r"""Compute a tall factor ``R⁻`` such that ``R⁻ (R⁻)ᵀ ≈ A⁻¹``.

    Args:
        operator: Square symmetric positive-definite operator ``A``.
        rank: Number of retained directions. Ignored by ``"cholesky"``,
            which returns the exact full-rank inverse-root factor.
        method: Decomposition method: ``"lanczos"``, ``"cholesky"``,
            ``"pivoted_cholesky"``, or ``"svd"``.
        key: PRNG key for random-start methods.

    Returns:
        A :class:`RootDecomposition` with inverse-root shape ``(N, k)``.
    """
    n, rank = _validate_square_rank(operator, rank, method=method)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return RootDecomposition(_diagonal_root(operator, rank, method, inverse=True))
    if method == "cholesky":
        L = _cholesky_root(operator)
        identity = jnp.eye(n, dtype=L.dtype)
        L_inv = jax.scipy.linalg.solve_triangular(L, identity, lower=True)
        return RootDecomposition(L_inv.T)
    if method == "lanczos":
        return RootDecomposition(_eig_root(operator, rank, key, inverse=True))
    if method == "svd":
        return RootDecomposition(_svd_root(operator, rank, key, inverse=True))
    if method == "pivoted_cholesky":
        # Dense fallback: forms A^{-1} explicitly via the lazy inverse
        # then runs pivoted Cholesky on it — O(N^3) time, O(N^2) memory.
        # Prefer ``method="lanczos"`` for large operators where structural
        # solves/matvecs are available.
        inv_mat = _symmetric_operator_matrix(inv(operator))
        return RootDecomposition(_pivoted_cholesky_root(inv_mat, rank))
    raise ValueError(f"Unknown inverse-root decomposition method: {method!r}")


def _validate_square_rank(
    operator: lx.AbstractLinearOperator,
    rank: int,
    *,
    method: RootMethod = "lanczos",
) -> tuple[int, int]:
    """Validate that ``operator`` is square and clamp ``rank`` to its size.

    ``method="cholesky"`` returns the exact full-rank factor, so ``rank``
    is documented as ignored; in that case we skip the ``rank >= 1``
    guard and just clamp the returned ``rank`` to the operator size so
    callers can pass any sentinel value.
    """
    if operator.in_size() != operator.out_size():
        raise ValueError("root decompositions require a square operator")
    n = operator.in_size()
    if method == "cholesky":
        return n, n
    if rank < 1:
        raise ValueError("rank must be at least 1")
    return n, min(rank, n)


def _diagonal_root(
    operator: lx.DiagonalLinearOperator,
    rank: int,
    method: RootMethod,
    *,
    inverse: bool,
) -> Float[Array, "N k"]:
    """Compute root factors for diagonal operators without dense eig/SVD calls."""
    diag = jnp.real(lx.diagonal(operator))
    if method == "cholesky":
        return jnp.diag(_root_scales(diag, inverse=inverse))
    if method not in ("lanczos", "svd", "pivoted_cholesky"):
        target = "inverse-root" if inverse else "root"
        raise ValueError(f"Unknown {target} decomposition method: {method!r}")

    n = diag.shape[0]
    if inverse and method == "pivoted_cholesky":
        order_scores = _safe_inverse_sqrt(diag) ** 2
    else:
        order_scores = diag
    order = jnp.argsort(order_scores)[::-1][:rank]
    selected = diag[order]
    scales = _root_scales(selected, inverse=inverse)
    root = jnp.zeros((n, rank), dtype=scales.dtype)
    return root.at[order, jnp.arange(rank)].set(scales)


def _cholesky_root(
    operator: lx.AbstractLinearOperator,
) -> Float[Array, "N N"]:
    """Compute an exact Cholesky root, preserving structure before final output."""
    return jnp.real(cholesky(operator).as_matrix())


def _eig_root(
    operator: lx.AbstractLinearOperator,
    rank: int,
    key: jax.Array | None,
    *,
    inverse: bool,
) -> Float[Array, "N k"]:
    """Build a root from the top real eigenpairs of ``operator``."""
    vals, vecs = eig(operator, rank=rank, key=key)
    vals, vecs = _top_real_eigenpairs(vals, vecs, rank)
    return vecs * _root_scales(vals, inverse=inverse)[None, :]


def _svd_root(
    operator: lx.AbstractLinearOperator,
    rank: int,
    key: jax.Array | None,
    *,
    inverse: bool,
) -> Float[Array, "N k"]:
    """Build a root from the leading left singular vectors of ``operator``."""
    U, s, _Vt = svd(operator, rank=rank, key=key)
    order = jnp.argsort(jnp.real(s))[::-1][:rank]
    U = jnp.real(U[:, order])
    s = jnp.real(s[order])
    return U * _root_scales(s, inverse=inverse)[None, :]


def _top_real_eigenpairs(
    vals: Array,
    vecs: Array,
    rank: int,
) -> tuple[Float[Array, " k"], Float[Array, "N k"]]:
    """Select the top-``rank`` real eigenpairs in descending eigenvalue order."""
    vals = jnp.real(vals)
    vecs = jnp.real(vecs)
    order = jnp.argsort(vals)[::-1][:rank]
    return vals[order], vecs[:, order]


def _safe_psd(vals: Array) -> Array:
    """Clamp eigenvalues/singular values to a positive floating-point floor."""
    floor = jnp.finfo(vals.dtype).tiny
    return jnp.maximum(vals, floor)


def _root_scales(vals: Array, *, inverse: bool) -> Array:
    """Compute root or inverse-root column scales from spectral values."""
    return _safe_inverse_sqrt(vals) if inverse else jnp.sqrt(_safe_psd(vals))


def _safe_inverse_sqrt(vals: Array) -> Array:
    """Compute a finite inverse square-root for nonnegative spectra."""
    return jax.lax.rsqrt(_safe_psd(vals))


def _symmetric_matrix(mat: Array) -> Array:
    """Symmetrize a dense covariance-like matrix after numerical operations."""
    return 0.5 * (mat + mat.T)


def _symmetric_operator_matrix(
    operator: lx.AbstractLinearOperator,
) -> Float[Array, "N N"]:
    """Materialize an operator only for dense fallback algorithms."""
    return _symmetric_matrix(operator.as_matrix())


def _pivoted_cholesky_root(
    mat: Float[Array, "N N"],
    rank: int,
) -> Float[Array, "N k"]:
    """Compute a greedy pivoted-Cholesky root from a dense fallback matrix."""
    n = mat.shape[0]
    dtype = mat.dtype
    floor = jnp.finfo(dtype).tiny
    root = jnp.zeros((n, rank), dtype=dtype)
    residual = mat
    residual_diag = jnp.maximum(jnp.diag(residual), 0.0)

    def body(i, carry):
        root, residual, residual_diag = carry
        pivot = jnp.argmax(residual_diag)
        pivot_val = residual_diag[pivot]
        pivot_col = residual[:, pivot]
        col = jax.lax.cond(
            pivot_val > floor,
            lambda _: pivot_col / jnp.sqrt(pivot_val),
            lambda _: jnp.zeros_like(pivot_col),
            operand=None,
        )
        root = root.at[:, i].set(col)
        residual = _symmetric_matrix(residual - jnp.outer(col, col))
        residual_diag = jnp.maximum(jnp.diag(residual), 0.0)
        return root, residual, residual_diag

    root, _, _ = jax.lax.fori_loop(0, rank, body, (root, residual, residual_diag))
    return root
