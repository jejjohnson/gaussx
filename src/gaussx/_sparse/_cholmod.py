"""Opt-in CHOLMOD backend (``scikit-sparse``): AMD ordering, host numerics.

Only the numeric factorisation crosses into CHOLMOD, through
``jax.pure_callback``; it factors the already-permuted matrix with the
natural ordering in simplicial mode, so L lands on exactly the pattern of
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
    """Import ``sksparse.cholmod``, or raise an informative `ImportError`.

    Returns:
        The ``sksparse.cholmod`` module.

    Raises:
        ImportError: If ``scikit-sparse`` is not installed.

    Examples:
        With the import blocked, whether or not ``scikit-sparse`` is installed:

        >>> import sys
        >>> from unittest import mock
        >>> with mock.patch.dict(sys.modules, {"sksparse": None}):
        ...     require_cholmod()  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ImportError: The CHOLMOD backend and the 'amd' ordering need scikit-sparse
        (and SuiteSparse): pip install scikit-sparse, or conda install -c
        conda-forge scikit-sparse.
    """
    try:
        from sksparse import cholmod  # ty: ignore[unresolved-import]
    except ImportError as err:
        raise ImportError(
            "The CHOLMOD backend and the 'amd' ordering need scikit-sparse "
            "(and SuiteSparse): pip install scikit-sparse, or "
            "conda install -c conda-forge scikit-sparse."
        ) from err
    return cholmod


def _csc(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, n: int) -> Any:
    """Full symmetric CSC matrix from lower-triangle triplets.

    Args:
        rows: Row index of each lower-triangle entry.
        cols: Column index of each lower-triangle entry.
        data: Value of each lower-triangle entry.
        n: Matrix size N.

    Returns:
        The N × N ``scipy.sparse.csc_matrix``, with each off-diagonal mirrored.
    """
    import scipy.sparse as sp

    off_diagonal = rows != cols
    return sp.csc_matrix(
        (
            np.concatenate([data, data[off_diagonal]]),
            (
                np.concatenate([rows, cols[off_diagonal]]),
                np.concatenate([cols, rows[off_diagonal]]),
            ),
        ),
        shape=(n, n),
    )


def cholmod_amd_ordering(pattern: SparsityPattern) -> np.ndarray:
    """CHOLMOD's AMD permutation of a pattern (host).

    Args:
        pattern: Square sparsity pattern.

    Returns:
        The permutation ``perm``.

    Raises:
        ImportError: If ``scikit-sparse`` is not installed.
    """
    cholmod = require_cholmod()
    n = pattern.shape[0]
    rows = np.maximum(pattern.rows, pattern.cols)
    cols = np.minimum(pattern.rows, pattern.cols)
    # Structure only: a diagonally dominant stand-in for the values.
    data = np.where(rows == cols, float(n + 1), -1.0)
    stand_in = _csc(rows, cols, data, n)
    stand_in.sum_duplicates()
    # scikit-sparse >= 0.5
    if hasattr(cholmod, "cho_factor"):
        factor = cholmod.CholeskyFactor(stand_in, lower=True, order="amd")
        return np.asarray(factor.perm)
    return np.asarray(cholmod.analyze(stand_in, ordering_method="amd").P())


def _host_factor(sym: SymbolicCholesky, a: np.ndarray) -> np.ndarray:
    """CHOLMOD's L of the permuted matrix, on ``sym``'s CSC pattern.

    Args:
        sym: The symbolic analysis.
        a: Lower triangle of P A Pᵀ on L's CSC pattern.

    Returns:
        The values of L on its CSC pattern, in the dtype of ``a``; all NaN if
        CHOLMOD rejects the matrix.

    Raises:
        ImportError: If ``scikit-sparse`` is not installed.
        RuntimeError: If CHOLMOD's factor has an entry outside the pattern of L.
    """
    cholmod = require_cholmod()
    n = sym.n
    matrix = _csc(sym.rowidx, sym.colidx, np.asarray(a, dtype=np.float64), n)
    try:
        # scikit-sparse >= 0.5
        if hasattr(cholmod, "cho_factor"):
            factor = cholmod.cho_factor(
                matrix, lower=True, order="natural", supernodal_mode="simplicial"
            )
            lower_factor = factor.get_factor(kind="LL", lower=True)
        else:
            factor = cholmod.cholesky(
                matrix, mode="simplicial", ordering_method="natural"
            )
            lower_factor = factor.L()
    except cholmod.CholmodError:
        return np.full(sym.nnz, np.nan, dtype=a.dtype)
    lower_factor = lower_factor.tocoo()
    keys = sym.colidx.astype(np.int64) * n + sym.rowidx
    found = lower_factor.col.astype(np.int64) * n + lower_factor.row
    positions = np.searchsorted(keys, found)
    if np.any(positions >= sym.nnz) or np.any(
        keys[np.minimum(positions, sym.nnz - 1)] != found
    ):
        raise RuntimeError("CHOLMOD's factor left the symbolic pattern of L.")
    values = np.zeros(sym.nnz, dtype=np.float64)
    values[positions] = lower_factor.data
    return values.astype(a.dtype)


def _callback(sym: SymbolicCholesky, a: Array) -> Array:
    """`_host_factor` as a traced callback, sequential under ``vmap``.

    Args:
        sym: The symbolic analysis.
        a: Lower triangle of P A Pᵀ on L's CSC pattern.

    Returns:
        The values of L, with the shape and dtype of ``a``.
    """
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

        P A Pᵀ = L Lᵀ

    Its VJP (only reached by gradients of quantities other than ``logdet`` and
    ``solve``, which have their own) differentiates the JAX factorisation of
    the same matrix, so both backends give identical gradients.

    Args:
        sym: The symbolic analysis.
        a: Lower triangle of P A Pᵀ on L's CSC pattern.

    Returns:
        The values of L on its CSC pattern.

    Raises:
        ImportError: If ``scikit-sparse`` is not installed, wrapped by JAX's
            runtime error when the callback runs.
    """
    return _callback(sym, a)


def _cholmod_fwd(
    sym: SymbolicCholesky, a: Float[Array, " nnz_L"]
) -> tuple[Float[Array, " nnz_L"], Float[Array, " nnz_L"]]:
    """Forward pass of `cholmod_cholesky`: L, keeping ``a`` as the residual.

    Args:
        sym: The symbolic analysis.
        a: Lower triangle of P A Pᵀ on L's CSC pattern.

    Returns:
        Tuple ``(L, a)``.
    """
    return _callback(sym, a), a


def _cholmod_bwd(
    sym: SymbolicCholesky, a: Float[Array, " nnz_L"], L_bar: Float[Array, " nnz_L"]
) -> tuple[Float[Array, " nnz_L"]]:
    """Backward pass of `cholmod_cholesky`, through the JAX factorisation.

    Args:
        sym: The symbolic analysis.
        a: The residual, the lower triangle of P A Pᵀ.
        L_bar: Cotangent L̄ of the factor's values.

    Returns:
        The one-element tuple holding the cotangent of ``a``.
    """
    _, vjp = jax.vjp(ft.partial(numeric_cholesky, sym), a)
    return vjp(L_bar)


cholmod_cholesky.defvjp(_cholmod_fwd, _cholmod_bwd)
