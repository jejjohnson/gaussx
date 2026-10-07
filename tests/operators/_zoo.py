"""One small instance of every exported gaussx operator class.

Shared by the lineax-interop and conformance tests. Inputs come from the
`gaussx._testing` builders in the active default float, with pinned keys
derived from the caller's key: the properties checked are structural, so
any draw would do.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np

import gaussx
from gaussx._testing import (
    random_block_diag_pd,
    random_kronecker_pd,
    random_low_rank_update,
    random_pd_matrix,
    random_pd_operator,
    random_spd_block_tridiag,
    random_sum_of_kroneckers_pd,
)


def _float():
    return jnp.result_type(float)


def _toeplitz_column(n: int) -> jax.Array:
    # A decaying (exponential-kernel) column: symmetric positive definite.
    return jnp.exp(-0.7 * jnp.arange(n, dtype=_float()))


def _lower_upper_blocks(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    k1, k2 = jr.split(key)
    diagonal = jnp.stack(
        [jnp.linalg.cholesky(random_pd_matrix(k, 2)) for k in jr.split(k1, 3)]
    )
    sub_diagonal = 0.3 * jr.normal(k2, (2, 2, 2), dtype=_float())
    return diagonal, sub_diagonal


def _sparse(key: jax.Array) -> gaussx.SparseOperator:
    # Symmetric tridiagonal pattern, diagonally dominant values.
    n = 4
    rows = np.array([0, 1, 2, 3, 1, 2, 3])
    cols = np.array([0, 1, 2, 3, 0, 1, 2])
    pattern = gaussx.SparsityPattern(rows, cols, (n, n), symmetric=True)
    values = jr.uniform(key, (pattern.nnz,), dtype=_float(), minval=-0.5, maxval=0.5)
    values = jnp.where(pattern.rows == pattern.cols, values + 3.0, values)
    return gaussx.SparseOperator(values, pattern, tags=lx.positive_semidefinite_tag)


def _masked(key: jax.Array) -> gaussx.MaskedOperator:
    mask = jnp.array([True, False, True, True])
    return gaussx.MaskedOperator(random_pd_operator(key, 4), mask, mask)


def _interpolated(key: jax.Array) -> gaussx.InterpolatedOperator:
    k1, k2 = jr.split(key)
    indices = jnp.array([[0, 1], [1, 2], [2, 3]])
    values = jr.uniform(k2, (3, 2), dtype=_float())
    return gaussx.InterpolatedOperator(random_pd_operator(k1, 4), indices, values)


def _spectral_function(key: jax.Array) -> gaussx.SpectralFunction:
    return gaussx.SpectralFunction(random_kronecker_pd(key, (2, 3)), jnp.sqrt)


def _sum_kronecker_sqrt(key: jax.Array) -> gaussx.SumOfKroneckersSqrt:
    return gaussx.SumOfKroneckersSqrt(random_sum_of_kroneckers_pd(key, (2, 3)))


ZOO: dict[str, Callable[[jax.Array], lx.AbstractLinearOperator]] = {
    "block_diag": lambda k: random_block_diag_pd(k, (2, 3)),
    "block_tridiag": lambda k: random_spd_block_tridiag(k, 3, 2),
    "lower_block_tridiag": lambda k: gaussx.LowerBlockTriDiag(*_lower_upper_blocks(k)),
    # Upper-triangular diagonal blocks, as UpperBlockTriDiag requires.
    "upper_block_tridiag": lambda k: (
        gaussx.LowerBlockTriDiag(*_lower_upper_blocks(k)).T
    ),
    "circulant": lambda k: gaussx.circulant(_toeplitz_column(5)),
    "circulant_complex": lambda k: gaussx.circulant(
        jr.normal(k, (4,), dtype=_float()) * (1.0 + 0.5j)
    ),
    "interpolated": _interpolated,
    "kronecker": lambda k: random_kronecker_pd(k, (2, 3)),
    "kronecker_sum": lambda k: gaussx.KroneckerSum(
        random_pd_operator(jr.fold_in(k, 0), 2), random_pd_operator(jr.fold_in(k, 1), 3)
    ),
    "kronecker_sum_sqrt": lambda k: gaussx.KroneckerSumSqrt(
        random_pd_operator(jr.fold_in(k, 0), 2), random_pd_operator(jr.fold_in(k, 1), 3)
    ),
    "low_rank_update": lambda k: random_low_rank_update(k, 4, 2),
    "masked": _masked,
    "sparse": _sparse,
    "spectral_function": _spectral_function,
    "sum_of_kroneckers": lambda k: random_sum_of_kroneckers_pd(k, (2, 3)),
    "sum_kronecker_sqrt": _sum_kronecker_sqrt,
    "toeplitz": lambda k: gaussx.Toeplitz(_toeplitz_column(5)),
    "toeplitz_cholesky": lambda k: gaussx.ToeplitzCholesky(_toeplitz_column(4)),
}


def exported_operator_classes() -> set[type]:
    """Every class exported from `gaussx` that is a lineax operator."""
    return {
        obj
        for name in dir(gaussx)
        if inspect.isclass(obj := getattr(gaussx, name))
        and issubclass(obj, lx.AbstractLinearOperator)
    }
