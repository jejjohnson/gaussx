"""Tests for eig, eigvals primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import BlockDiag, Kronecker, eig, eigvals
from gaussx._einx import einsum
from gaussx._testing import random_pd_matrix, tree_allclose


def test_eig_diagonal():
    d = jnp.array([3.0, 1.0, 2.0])
    op = lx.DiagonalLinearOperator(d)
    vals, vecs = eig(op)
    assert tree_allclose(vals, d)
    assert tree_allclose(vecs, jnp.eye(3))


def test_eigvals_diagonal():
    d = jnp.array([5.0, 2.0, 7.0])
    op = lx.DiagonalLinearOperator(d)
    assert tree_allclose(eigvals(op), d)


def test_eig_symmetric(getkey):
    mat = random_pd_matrix(getkey(), 4)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
    vals, vecs = eig(op)
    # Reconstruct: A = V diag(lam) V^T
    reconstructed = vecs @ jnp.diag(vals) @ vecs.T
    assert tree_allclose(reconstructed, mat, rtol=1e-4)


def test_eig_dense(getkey):
    mat = jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3)
    op = lx.MatrixLinearOperator(mat)
    vals, vecs = eig(op)
    # Check A @ v = lam * v for each eigenpair
    for i in range(3):
        lhs = mat @ vecs[:, i]
        rhs = vals[i] * vecs[:, i]
        assert tree_allclose(lhs, rhs, atol=1e-4)


def test_eigvals_symmetric(getkey):
    mat = random_pd_matrix(getkey(), 5)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
    vals = eigvals(op)
    expected = jnp.linalg.eigvalsh(mat)
    assert tree_allclose(jnp.sort(vals), jnp.sort(expected), rtol=1e-5)


def test_eig_block_diag(getkey):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    A_op = lx.MatrixLinearOperator(A, lx.symmetric_tag)
    B_op = lx.MatrixLinearOperator(B, lx.symmetric_tag)
    bd = BlockDiag(A_op, B_op)

    vals, vecs = eig(bd)
    reconstructed = vecs @ jnp.diag(vals) @ vecs.T
    assert tree_allclose(reconstructed, bd.as_matrix(), rtol=1e-4)


def test_eigvals_block_diag():
    d1 = jnp.array([1.0, 2.0])
    d2 = jnp.array([3.0, 4.0, 5.0])
    bd = BlockDiag(
        lx.DiagonalLinearOperator(d1),
        lx.DiagonalLinearOperator(d2),
    )
    expected = jnp.concatenate([d1, d2])
    assert tree_allclose(eigvals(bd), expected)


def test_eig_kronecker(getkey):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    A_op = lx.MatrixLinearOperator(A, lx.symmetric_tag)
    B_op = lx.MatrixLinearOperator(B, lx.symmetric_tag)
    K = Kronecker(A_op, B_op)

    vals, _vecs = eig(K)
    # Check eigenvalue product property
    vals_A = jnp.linalg.eigvalsh(A)
    vals_B = jnp.linalg.eigvalsh(B)
    expected_vals = jnp.kron(vals_A, vals_B)
    assert tree_allclose(jnp.sort(vals), jnp.sort(expected_vals), rtol=1e-4)


def test_eigvals_kronecker():
    d1 = jnp.array([2.0, 3.0])
    d2 = jnp.array([4.0, 5.0])
    K = Kronecker(
        lx.DiagonalLinearOperator(d1),
        lx.DiagonalLinearOperator(d2),
    )
    expected = jnp.kron(d1, d2)
    assert tree_allclose(eigvals(K), expected)


# ----------------------------------------------------------------
# KroneckerSum dispatch
# ----------------------------------------------------------------


def test_eig_kronecker_sum_matches_dense(getkey):
    """eig(A ⊕ B) via per-factor eigendecomposition matches dense eigh."""
    import jax.numpy as jnp
    import lineax as lx

    from gaussx import KroneckerSum, eig
    from gaussx._testing import random_pd_matrix

    A = random_pd_matrix(getkey(), 3)
    B = random_pd_matrix(getkey(), 4)
    op = KroneckerSum(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    vals, _vecs = eig(op)
    vals_ref = jnp.linalg.eigvalsh(op.as_matrix())
    assert jnp.allclose(jnp.sort(vals), jnp.sort(vals_ref), atol=1e-6)


def test_eigvals_kronecker_sum_matches_dense(getkey):
    """eigvals(A ⊕ B) avoids materializing the full (n_a·n_b)² matrix."""
    import jax.numpy as jnp
    import lineax as lx

    from gaussx import KroneckerSum, eigvals
    from gaussx._testing import random_pd_matrix

    A = random_pd_matrix(getkey(), 3)
    B = random_pd_matrix(getkey(), 5)
    op = KroneckerSum(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    vals = eigvals(op)
    vals_ref = jnp.linalg.eigvalsh(op.as_matrix())
    assert jnp.allclose(jnp.sort(vals), jnp.sort(vals_ref), atol=1e-6)


def _repeated_eigenvalue_spd():
    # SPD with a repeated eigenvalue: ``eig`` there returns a non-orthonormal
    # basis of the repeated eigenspace, ``eigh`` an orthonormal one.
    Q, _ = jnp.linalg.qr(jr.normal(jr.key(1), (4, 4)))
    S = einsum(Q * jnp.array([1.0, 1.0, 2.0, 3.0]), Q, "i k, j k -> i j")
    return 0.5 * (S + S.T)


def _assert_real_orthonormal(vals, vecs, matrix):
    assert not jnp.iscomplexobj(vals)
    assert not jnp.iscomplexobj(vecs)
    n = vecs.shape[0]
    assert tree_allclose(einsum(vecs, vecs, "i k, i l -> k l"), jnp.eye(n))
    assert tree_allclose(einsum(vecs * vals, vecs, "i k, j k -> i j"), matrix)


def test_eig_psd_tagged_wrapper_takes_eigh():
    """lineax says ``is_symmetric(Tagged(X, psd))`` is False (gh-314)."""
    S = _repeated_eigenvalue_spd()
    op = lx.TaggedLinearOperator(
        lx.MatrixLinearOperator(S), lx.positive_semidefinite_tag
    )
    vals, vecs = eig(op)
    _assert_real_orthonormal(vals, vecs, S)
    ev = eigvals(op)
    assert not jnp.iscomplexobj(ev)
    assert tree_allclose(jnp.sort(ev), jnp.array([1.0, 1.0, 2.0, 3.0]))


def test_eig_tagged_kronecker_keeps_structure(getkey, monkeypatch):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    K = Kronecker(lx.MatrixLinearOperator(A), lx.MatrixLinearOperator(B))
    dense = K.as_matrix()
    op = lx.TaggedLinearOperator(K, lx.positive_semidefinite_tag)

    def _forbidden(self):
        raise AssertionError("Kronecker.as_matrix called")

    monkeypatch.setattr(Kronecker, "as_matrix", _forbidden)
    vals, _vecs = eig(op)
    ev = eigvals(op)
    monkeypatch.undo()
    assert tree_allclose(jnp.sort(jnp.real(vals)), jnp.linalg.eigvalsh(dense))
    assert tree_allclose(jnp.sort(jnp.real(ev)), jnp.linalg.eigvalsh(dense))


def test_eig_kronecker_of_psd_tagged_factors_is_real_orthonormal(getkey):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    psd = lx.positive_semidefinite_tag
    op = Kronecker(
        lx.TaggedLinearOperator(lx.MatrixLinearOperator(A), psd),
        lx.TaggedLinearOperator(lx.MatrixLinearOperator(B), psd),
    )
    vals, vecs = eig(op)
    _assert_real_orthonormal(vals, vecs, op.as_matrix())


def test_eig_psd_tag_reaches_untagged_blocks():
    """The wrapper's symmetry carries into BlockDiag blocks (review of gh-314)."""
    S = _repeated_eigenvalue_spd()
    block = lx.MatrixLinearOperator(S)  # untagged, repeated eigenvalue
    op = lx.TaggedLinearOperator(BlockDiag(block, block), lx.positive_semidefinite_tag)
    vals, vecs = eig(op)
    _assert_real_orthonormal(vals, vecs, op.as_matrix())


def test_eig_tagged_kronecker_of_rectangular_factors():
    """``A ⊗ Aᵀ`` is square with rectangular factors: decomposed whole."""
    A = jnp.array([[1.0, 2.0]])
    K = Kronecker(lx.MatrixLinearOperator(A), lx.MatrixLinearOperator(A.T))
    op = lx.TaggedLinearOperator(K, lx.symmetric_tag)
    vals = eigvals(op)
    assert tree_allclose(
        jnp.sort(vals), jnp.sort(jnp.linalg.eigvals(K.as_matrix()).real)
    )
    eig(op)


def test_eig_psd_tagged_pytree_operator():
    S = _repeated_eigenvalue_spd()
    struct = {"a": jax.ShapeDtypeStruct((4,), S.dtype)}
    inner = lx.PyTreeLinearOperator({"a": {"a": S}}, struct)
    op = lx.TaggedLinearOperator(inner, lx.positive_semidefinite_tag)
    vals, vecs = eig(op)
    _assert_real_orthonormal(vals, vecs, S)


# ---------------------------------------------------------------------------
# rank: exact structured decomposition first, then the top-k (gh-383)
# ---------------------------------------------------------------------------


def _psd(key, n):
    return lx.MatrixLinearOperator(
        random_pd_matrix(key, n), lx.positive_semidefinite_tag
    )


_RANK_OPERATORS = [
    pytest.param(
        lambda: Kronecker(_psd(jr.key(0), 3), _psd(jr.key(1), 4)), id="kronecker"
    ),
    pytest.param(
        lambda: BlockDiag(
            lx.DiagonalLinearOperator(jnp.arange(1.0, 7.0)),
            lx.DiagonalLinearOperator(jnp.arange(1.0, 7.0)),
        ),
        id="block_diag",
    ),
    pytest.param(
        lambda: lx.DiagonalLinearOperator(jnp.array([3.0, 1.0, 6.0, 2.0, 5.0, 4.0])),
        id="diagonal",
    ),
    pytest.param(lambda: _psd(jr.key(2), 12), id="dense"),
]


@pytest.mark.parametrize("build", _RANK_OPERATORS)
def test_eig_rank_shape_and_top_values(build):
    op = build()
    k = 2
    vals, vecs = eig(op, rank=k)
    ev = eigvals(op, rank=k)
    assert vals.shape == (k,)
    assert vecs.shape == (op.in_size(), k)
    assert ev.shape == (k,)
    if isinstance(op, lx.MatrixLinearOperator):
        return  # k-step Lanczos: an approximation, so only the shape is fixed
    # Structured operators: the exact top-k, descending.
    top = jnp.sort(jnp.linalg.eigvalsh(op.as_matrix()))[::-1][:k]
    assert tree_allclose(vals, top)
    assert tree_allclose(ev, top)
