"""Opt-in CHOLMOD backend (``scikit-sparse``): AMD ordering, host numerics.

Only the numeric factorisation crosses into CHOLMOD, through
``jax.pure_callback``; it factors the already-permuted matrix with the
natural ordering in simplicial mode, so ``L`` lands on exactly the pattern of
the shared `SymbolicCholesky`. The triangular solves, Takahashi and the VJPs
are the same JAX code as for the JAX backend. CPU only; ``vmap`` runs the
callback once per batch element.
"""

from __future__ import annotations

import functools as ft
from typing import Any

import jax
import numpy as np
from jaxtyping import Array, Float

from gaussx._operators._sparse import SparsityPattern
from gaussx._sparse._numeric import numeric_cholesky
from gaussx._sparse._symbolic import SymbolicCholesky


def require_cholmod() -> Any:
    """Import ``sksparse.cholmod``, or raise an informative `ImportError`."""
    try:
        from sksparse import cholmod
    except ImportError as err:
        raise ImportError(
            "The CHOLMOD backend and the 'amd' ordering need scikit-sparse "
            "(and SuiteSparse): pip install scikit-sparse, or "
            "conda install -c conda-forge scikit-sparse."
        ) from err
    return cholmod


def _csc(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, n: int) -> Any:
    """Full symmetric CSC matrix from lower-triangle triplets."""
    import scipy.sparse as sp

    off = rows != cols
    return sp.csc_matrix(
        (
            np.concatenate([data, data[off]]),
            (np.concatenate([rows, cols[off]]), np.concatenate([cols, rows[off]])),
        ),
        shape=(n, n),
    )


def cholmod_amd_ordering(pattern: SparsityPattern) -> np.ndarray:
    """CHOLMOD's AMD permutation of a pattern (host)."""
    cholmod = require_cholmod()
    n = pattern.shape[0]
    rows = np.maximum(pattern.rows, pattern.cols)
    cols = np.minimum(pattern.rows, pattern.cols)
    # Structure only: a diagonally dominant stand-in for the values.
    data = np.where(rows == cols, float(n + 1), -1.0)
    A = _csc(rows, cols, data, n)
    A.sum_duplicates()
    if hasattr(cholmod, "cho_factor"):  # scikit-sparse >= 0.5
        return np.asarray(cholmod.CholeskyFactor(A, lower=True, order="amd").perm)
    return np.asarray(cholmod.analyze(A, ordering_method="amd").P())


def _host_factor(sym: SymbolicCholesky, a: np.ndarray) -> np.ndarray:
    """CHOLMOD's ``L`` of the permuted matrix, on ``sym``'s CSC pattern."""
    cholmod = require_cholmod()
    n = sym.n
    A = _csc(sym.rowidx, sym.colidx, np.asarray(a, dtype=np.float64), n)
    try:
        if hasattr(cholmod, "cho_factor"):  # scikit-sparse >= 0.5
            factor = cholmod.cho_factor(
                A, lower=True, order="natural", supernodal_mode="simplicial"
            )
            L = factor.get_factor(kind="LL", lower=True)
        else:
            L = cholmod.cholesky(A, mode="simplicial", ordering_method="natural").L()
    except cholmod.CholmodError:
        return np.full(sym.nnz, np.nan, dtype=a.dtype)
    L = L.tocoo()
    keys = sym.colidx.astype(np.int64) * n + sym.rowidx
    found = L.col.astype(np.int64) * n + L.row
    pos = np.searchsorted(keys, found)
    if np.any(pos >= sym.nnz) or np.any(keys[np.minimum(pos, sym.nnz - 1)] != found):
        raise RuntimeError("CHOLMOD's factor left the symbolic pattern of L.")
    out = np.zeros(sym.nnz, dtype=np.float64)
    out[pos] = L.data
    return out.astype(a.dtype)


def _callback(sym: SymbolicCholesky, a: Array) -> Array:
    return jax.pure_callback(
        ft.partial(_host_factor, sym),
        jax.ShapeDtypeStruct(a.shape, a.dtype),
        a,
        vmap_method="sequential",
    )


@ft.partial(jax.custom_vjp, nondiff_argnums=(0,))
def cholmod_cholesky(
    sym: SymbolicCholesky, a: Float[Array, " nnz_L"]
) -> Float[Array, " nnz_L"]:
    """Numeric factorisation by CHOLMOD; same contract as `numeric_cholesky`.

    Its VJP (only reached by gradients of quantities other than ``logdet`` and
    ``solve``, which have their own) differentiates the JAX factorisation of
    the same matrix, so both backends give identical gradients.
    """
    return _callback(sym, a)


def _cholmod_fwd(sym, a):
    return _callback(sym, a), a


def _cholmod_bwd(sym, a, L_bar):
    _, vjp = jax.vjp(ft.partial(numeric_cholesky, sym), a)
    return vjp(L_bar)


cholmod_cholesky.defvjp(_cholmod_fwd, _cholmod_bwd)
