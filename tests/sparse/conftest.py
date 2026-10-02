"""Reference problems for the sparse Cholesky tests (G4)."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from gaussx import SparseOperator
from gaussx._sparse._symbolic import SymbolicCholesky, _ordering


def grid_precision(
    height: int,
    width: int,
    *,
    symmetric: bool = True,
    shift: float = 0.5,
    fixed_effect: bool = False,
    seed: int = 0,
) -> SparseOperator:
    """Weighted 4-neighbour grid Laplacian plus ``shift * I``.

    With ``fixed_effect=True`` one more node is joined to every grid node
    (an arrow-shaped pattern, as a fixed effect adds to a Laplace Hessian).
    ``symmetric=False`` stores both triangles.
    """
    rng = np.random.default_rng(seed)
    i = np.arange(height * width)
    right = i[i % width != width - 1]
    down = i[i < (height - 1) * width]
    senders = np.r_[right + 1, down + width]
    receivers = np.r_[right, down]
    n = height * width
    if fixed_effect:
        senders = np.r_[senders, np.full(n, n)]
        receivers = np.r_[receivers, np.arange(n)]
        n += 1
    weights = rng.uniform(0.5, 1.5, senders.size)
    deg = np.bincount(senders, weights, n) + np.bincount(receivers, weights, n)
    diagonal = np.arange(n)
    if symmetric:
        rows, cols = np.r_[diagonal, senders], np.r_[diagonal, receivers]
        values = np.r_[deg + shift, -weights]
    else:
        rows = np.r_[diagonal, senders, receivers]
        cols = np.r_[diagonal, receivers, senders]
        values = np.r_[deg + shift, -weights, -weights]
    return SparseOperator.from_coo(
        rows, cols, jnp.asarray(values), (n, n), symmetric=symmetric
    )


def make_symbolic(
    op: SparseOperator, ordering: str = "rcm", *, banded: bool | None = None
) -> SymbolicCholesky:
    """A symbolic analysis with the layout forced (bypasses the cache)."""
    perm = _ordering(op.pattern, ordering)  # ty: ignore[invalid-argument-type]
    return SymbolicCholesky(
        op.pattern,
        ordering,  # ty: ignore[invalid-argument-type]
        "jax",
        perm,
        banded=banded,
    )


def dense_factor(sym: SymbolicCholesky, values: jnp.ndarray) -> np.ndarray:
    """The CSC values of ``L`` as a dense lower-triangular matrix."""
    L = np.zeros((sym.n, sym.n))
    L[sym.rowidx, sym.colidx] = np.asarray(values)
    return L


@pytest.fixture
def grid() -> Callable[..., SparseOperator]:
    return grid_precision


@pytest.fixture
def symbolic_for() -> Callable[..., SymbolicCholesky]:
    return make_symbolic


@pytest.fixture
def to_dense_factor() -> Callable[[SymbolicCholesky, jnp.ndarray], np.ndarray]:
    return dense_factor
