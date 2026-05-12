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
    _n, rank = _validate_square_rank(operator, rank)
    if method == "cholesky":
        return RootDecomposition(jnp.real(cholesky(operator).as_matrix()))
    if method == "lanczos":
        return RootDecomposition(_eig_root(operator, rank, key, inverse=False))
    if method == "svd":
        return RootDecomposition(_svd_root(operator, rank, key, inverse=False))
    if method == "pivoted_cholesky":
        mat = _symmetric_matrix(operator.as_matrix())
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
    n, rank = _validate_square_rank(operator, rank)
    if method == "cholesky":
        L = jnp.real(cholesky(operator).as_matrix())
        identity = jnp.eye(n, dtype=L.dtype)
        L_inv = jax.scipy.linalg.solve_triangular(L, identity, lower=True)
        return RootDecomposition(L_inv.T)
    if method == "lanczos":
        return RootDecomposition(_eig_root(operator, rank, key, inverse=True))
    if method == "svd":
        return RootDecomposition(_svd_root(operator, rank, key, inverse=True))
    if method == "pivoted_cholesky":
        mat = _symmetric_matrix(operator.as_matrix())
        L = jnp.linalg.cholesky(mat)
        identity = jnp.eye(n, dtype=mat.dtype)
        inv_mat = jax.scipy.linalg.cho_solve((L, True), identity)
        inv_mat = _symmetric_matrix(inv_mat)
        return RootDecomposition(_pivoted_cholesky_root(inv_mat, rank))
    raise ValueError(f"Unknown inverse-root decomposition method: {method!r}")


def _validate_square_rank(
    operator: lx.AbstractLinearOperator,
    rank: int,
) -> tuple[int, int]:
    if operator.in_size() != operator.out_size():
        raise ValueError("root decompositions require a square operator")
    if rank < 1:
        raise ValueError("rank must be at least 1")
    n = operator.in_size()
    return n, min(rank, n)


def _eig_root(
    operator: lx.AbstractLinearOperator,
    rank: int,
    key: jax.Array | None,
    *,
    inverse: bool,
) -> Float[Array, "N k"]:
    vals, vecs = eig(operator, rank=rank, key=key)
    vals, vecs = _top_real_eigenpairs(vals, vecs, rank)
    scales = _safe_inverse_sqrt(vals) if inverse else jnp.sqrt(_safe_psd(vals))
    return vecs * scales[None, :]


def _svd_root(
    operator: lx.AbstractLinearOperator,
    rank: int,
    key: jax.Array | None,
    *,
    inverse: bool,
) -> Float[Array, "N k"]:
    U, s, _Vt = svd(operator, rank=rank, key=key)
    order = jnp.argsort(jnp.real(s))[::-1][:rank]
    U = jnp.real(U[:, order])
    s = jnp.real(s[order])
    scales = _safe_inverse_sqrt(s) if inverse else jnp.sqrt(_safe_psd(s))
    return U * scales[None, :]


def _top_real_eigenpairs(
    vals: Array,
    vecs: Array,
    rank: int,
) -> tuple[Float[Array, " k"], Float[Array, "N k"]]:
    vals = jnp.real(vals)
    vecs = jnp.real(vecs)
    order = jnp.argsort(vals)[::-1][:rank]
    return vals[order], vecs[:, order]


def _safe_psd(vals: Array) -> Array:
    floor = jnp.finfo(vals.dtype).tiny
    return jnp.maximum(vals, floor)


def _safe_inverse_sqrt(vals: Array) -> Array:
    return jax.lax.rsqrt(_safe_psd(vals))


def _symmetric_matrix(mat: Array) -> Array:
    return 0.5 * (mat + mat.T)


def _pivoted_cholesky_root(
    mat: Float[Array, "N N"],
    rank: int,
) -> Float[Array, "N k"]:
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
