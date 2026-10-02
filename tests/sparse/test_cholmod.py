"""CHOLMOD backend (G4): integration tier, skipped without scikit-sparse."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gaussx
from gaussx import SparseOperator, SparsityPattern


pytest.importorskip("sksparse.cholmod")

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("ordering", ["rcm", "amd"])
@pytest.mark.parametrize("symmetric", [True, False])
def test_backends_agree(grid, ordering, symmetric):
    op = grid(5, 6, symmetric=symmetric, fixed_effect=True)
    jax_sym = gaussx.symbolic_cholesky(op.pattern, ordering=ordering)
    cholmod_sym = gaussx.symbolic_cholesky(
        op.pattern, ordering=ordering, backend="cholmod"
    )
    np.testing.assert_array_equal(jax_sym.perm, cholmod_sym.perm)
    a = gaussx.sparse_cholesky(op, jax_sym)
    c = gaussx.sparse_cholesky(op, cholmod_sym)
    np.testing.assert_allclose(c.values, a.values, atol=1e-12)
    np.testing.assert_allclose(c.logdet(), a.logdet())
    np.testing.assert_allclose(c.diag_inv(), a.diag_inv(), atol=1e-12)

    def logdet(values, sym):
        return gaussx.sparse_cholesky(SparseOperator(values, op.pattern), sym).logdet()

    def variances(values, sym):
        factor = gaussx.sparse_cholesky(SparseOperator(values, op.pattern), sym)
        return jnp.sum(factor.diag_inv())

    for f in (logdet, variances):
        np.testing.assert_allclose(
            jax.grad(f)(op.values, cholmod_sym),
            jax.grad(f)(op.values, jax_sym),
            atol=1e-12,
        )


def test_jit_and_sequential_vmap(grid):
    op = grid(4, 4)
    sym = gaussx.symbolic_cholesky(op.pattern, ordering="amd", backend="cholmod")

    @jax.jit
    def logdet(scale):
        values = scale * op.values
        return gaussx.sparse_cholesky(SparseOperator(values, op.pattern), sym).logdet()

    scales = jnp.array([1.0, 2.0])
    expected = [
        np.linalg.slogdet(float(s) * np.asarray(op.as_matrix()))[1] for s in scales
    ]
    np.testing.assert_allclose(jax.vmap(logdet)(scales), expected)


def test_amd_fill_on_reference_mesh():
    # 30 x 30 triangulated square: AMD 14,377 vs RCM 19,315 entries of L
    # (tests/sparse/test_symbolic.py records the RCM and natural fill).
    side = 30
    i = np.arange(side * side)
    right = i[i % side != side - 1]
    down = i[i < (side - 1) * side]
    diagonal = right[right < (side - 1) * side]  # one diagonal per cell
    rows = np.r_[right + 1, down + side, diagonal + side + 1]
    cols = np.r_[right, down, diagonal]
    pattern = SparsityPattern(rows, cols, (900, 900), symmetric=True)
    amd = gaussx.symbolic_cholesky(pattern, ordering="amd")
    rcm = gaussx.symbolic_cholesky(pattern, ordering="rcm")
    assert amd.nnz < 0.8 * rcm.nnz
    assert amd.nnz <= 15_000


def test_not_positive_definite_is_nan(grid):
    op = grid(3, 3, shift=-5.0)
    sym = gaussx.symbolic_cholesky(op.pattern, backend="cholmod")
    assert jnp.isnan(gaussx.sparse_cholesky(op, sym).logdet())
