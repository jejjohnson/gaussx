"""Tests for gaussx.sqrt with structural dispatch."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._einx import rearrange
from gaussx._operators import BlockDiag, Kronecker, KroneckerSum, KroneckerSumSqrt
from gaussx._primitives import solve, sqrt
from gaussx._primitives._sqrt import dense_symmetric_sqrt
from gaussx._testing import random_pd_matrix, tree_allclose


def test_sqrt_diagonal(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    S = sqrt(op)
    assert isinstance(S, lx.DiagonalLinearOperator)
    # S @ S should reconstruct A
    reconstructed = S.as_matrix() @ S.as_matrix()
    assert tree_allclose(reconstructed, op.as_matrix())


def test_sqrt_block_diag(getkey):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    bd = BlockDiag(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    S = sqrt(bd)
    assert isinstance(S, BlockDiag)
    reconstructed = S.as_matrix() @ S.as_matrix()
    assert tree_allclose(reconstructed, bd.as_matrix())


def test_sqrt_kronecker(getkey):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    K = Kronecker(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    S = sqrt(K)
    assert isinstance(S, Kronecker)
    reconstructed = S.as_matrix() @ S.as_matrix()
    assert tree_allclose(reconstructed, K.as_matrix(), rtol=1e-4)


@pytest.mark.x64_only(reason="dense-reference tolerance below float32 round-off")
def test_sqrt_kronecker_sum(getkey):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    K = KroneckerSum(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    S = sqrt(K)
    assert isinstance(S, KroneckerSumSqrt)
    reconstructed = S.as_matrix() @ S.as_matrix()
    assert tree_allclose(reconstructed, K.as_matrix(), rtol=1e-4)


def test_kronecker_sum_sqrt_rejects_nonsymmetric_factor(getkey):
    A = jr.normal(getkey(), (2, 2))  # untagged: not symmetric
    B = random_pd_matrix(getkey(), 3)
    K = KroneckerSum(
        lx.MatrixLinearOperator(A),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    with pytest.raises(ValueError, match="symmetric"):
        sqrt(K)


def test_sqrt_kronecker_sum_solve(getkey):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    K = KroneckerSum(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    S = sqrt(K)
    x = jr.normal(getkey(), (K.in_size(),))
    y = S.mv(x)
    assert tree_allclose(solve(S, y), x, rtol=1e-4)


def test_sqrt_dense(getkey):
    A = random_pd_matrix(getkey(), 4)
    op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
    S = sqrt(op)
    reconstructed = S.as_matrix() @ S.as_matrix()
    assert tree_allclose(reconstructed, op.as_matrix(), rtol=1e-4)


class TestDenseSqrtGradients:
    """`sqrt`'s dense branch must be differentiable at repeated eigenvalues.

    Differentiating straight through `jnp.linalg.eigh` divides by eigenvalue
    gaps, so an isotropic matrix -- nothing *but* repeated eigenvalues, and the
    single most common covariance there is -- would give ``NaN``. The square
    root itself is smooth there; only the naive chain rule is not.
    """

    def test_isotropic_gradient_is_finite_and_correct(self):
        # d/ds sum(sqrt(s I_3)) = d/ds 3 sqrt(s) = 3 / (2 sqrt(s)).
        grad = jax.grad(lambda s: jnp.sum(dense_symmetric_sqrt(s * jnp.eye(3))))(2.0)
        assert jnp.isfinite(grad)
        assert tree_allclose(grad, 3.0 / (2.0 * jnp.sqrt(jnp.asarray(2.0))))

    @pytest.mark.parametrize(
        "matrix",
        [
            3.0 * jnp.eye(4),  # one eigenvalue, multiplicity 4
            jnp.diag(jnp.array([2.0, 2.0, 5.0, 5.0])),  # two degenerate pairs
            jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0])),  # fully distinct
        ],
        ids=["isotropic", "degenerate-pairs", "distinct"],
    )
    @pytest.mark.x64_only(reason="finite-difference gradient check needs float64 steps")
    def test_jvp_matches_finite_differences(self, matrix, getkey):
        direction = random_pd_matrix(getkey(), matrix.shape[0])
        step = 1e-6
        expected = (
            dense_symmetric_sqrt(matrix + step * direction)
            - dense_symmetric_sqrt(matrix - step * direction)
        ) / (2.0 * step)
        _, tangent = jax.jvp(dense_symmetric_sqrt, (matrix,), (direction,))
        assert tree_allclose(tangent, expected, atol=1e-6, rtol=1e-5)

    def test_singular_matrix_gradient_stays_finite(self):
        """A zero eigenvalue is non-differentiable; it must not poison the rest.

        ``sqrt`` at the origin has an infinite derivative, so the entries of
        the null direction are returned as zero rather than ``NaN`` -- the
        identified directions keep a usable gradient.
        """
        matrix = jnp.diag(jnp.array([1.0, 2.0, 0.0]))
        direction = jnp.diag(jnp.array([1.0, 0.0, 0.0]))
        _, tangent = jax.jvp(dense_symmetric_sqrt, (matrix,), (direction,))
        assert jnp.all(jnp.isfinite(tangent))
        assert tree_allclose(tangent[0, 0], jnp.asarray(0.5))


def _isotropic_kronecker_sum(s, B):
    psd = lx.positive_semidefinite_tag
    return KroneckerSum(
        lx.MatrixLinearOperator(s * jnp.eye(3), psd),
        lx.MatrixLinearOperator(B, psd),
    )


def _dense_isotropic_root(s, B):
    dense = jnp.kron(s * jnp.eye(3), jnp.eye(2)) + jnp.kron(jnp.eye(3), B)
    return dense_symmetric_sqrt(dense)


@pytest.mark.parametrize("inverse", [False, True], ids=["mv", "solve"])
@pytest.mark.x64_only(reason="dense-reference tolerance below float32 round-off")
def test_sqrt_kronecker_sum_grad_with_repeated_eigenvalue(inverse):
    # gh-295: KroneckerSumSqrt exposed its eigh eigenvectors to autodiff,
    # so the gradient was NaN for a factor with a repeated eigenvalue.
    B = random_pd_matrix(jr.key(0), 2)
    v = jnp.arange(1.0, 7.0)

    def structured(s):
        root = sqrt(_isotropic_kronecker_sum(s, B))
        return (solve(root, v) if inverse else root.mv(v)).sum()

    def dense(s):
        root = _dense_isotropic_root(s, B)
        return (jnp.linalg.solve(root, v) if inverse else root @ v).sum()

    grad = jax.grad(structured)(1.5)
    assert jnp.isfinite(grad)
    assert jnp.allclose(grad, jax.grad(dense)(1.5), rtol=1e-8, atol=1e-8)
    assert jnp.allclose(jax.jit(jax.grad(structured))(1.5), grad, rtol=1e-12)


@pytest.mark.parametrize("inverse", [False, True], ids=["mv", "solve"])
def test_kronecker_sum_sqrt_jvp_matches_dense_root(inverse):
    # The structured Sylvester JVP of KroneckerSumSqrt's action against
    # dense_symmetric_sqrt's JVP on the materialised A ⊕ B, for random
    # symmetric tangents of the factors and of the vector.
    keys = jr.split(jr.key(0), 5)
    a, b = random_pd_matrix(keys[0], 3), random_pd_matrix(keys[1], 2)
    ta = jr.normal(keys[2], (3, 3))
    tb = jr.normal(keys[3], (2, 2))
    ta, tb = ta + ta.T, tb + tb.T
    v = jr.normal(keys[4], (6,))
    tv = jnp.ones(6)
    psd = lx.positive_semidefinite_tag

    def structured(a, b, v):
        root = KroneckerSumSqrt(
            lx.MatrixLinearOperator(a, psd), lx.MatrixLinearOperator(b, psd)
        )
        return root.solve(v) if inverse else root.mv(v)

    def dense(a, b, v):
        root = dense_symmetric_sqrt(jnp.kron(a, jnp.eye(2)) + jnp.kron(jnp.eye(3), b))
        return jnp.linalg.solve(root, v) if inverse else root @ v

    args, tangents = (a, b, v), (ta, tb, tv)
    out, tangent = jax.jvp(structured, args, tangents)
    dense_out, dense_tangent = jax.jvp(dense, args, tangents)
    assert jnp.allclose(out, dense_out, atol=1e-10)
    assert jnp.allclose(tangent, dense_tangent, atol=1e-10)


@pytest.mark.parametrize("inverse", [False, True], ids=["mv", "solve"])
def test_kronecker_sum_sqrt_second_derivative(inverse):
    # The JVP recomputes the spectrum from the factors, so a Hessian sees
    # its dependence: with A = [s], B = [1], z = [1], S z = sqrt(s + 1).
    psd = lx.positive_semidefinite_tag

    def f(s):
        root = KroneckerSumSqrt(
            lx.MatrixLinearOperator(jnp.array([[s]]), psd),
            lx.MatrixLinearOperator(jnp.ones((1, 1)), psd),
        )
        z = jnp.ones(1)
        return (root.solve(z) if inverse else root.mv(z))[0]

    s = 1.5
    # d²/ds² (s+1)^{±1/2}
    expected = 0.75 * (s + 1) ** -2.5 if inverse else -0.25 * (s + 1) ** -1.5
    assert jnp.allclose(jax.grad(jax.grad(f))(s), expected, rtol=1e-10)


def test_kronecker_sum_sqrt_keeps_diagonal_factors_lazy():
    class LazyDiagonal(lx.DiagonalLinearOperator):
        def as_matrix(self):
            raise NotImplementedError("dense materialization unavailable")

    da, db = jnp.array([1.0, 2.0, 3.0]), jnp.array([0.5, 1.5])
    root = KroneckerSumSqrt(LazyDiagonal(da), LazyDiagonal(db))
    v = jnp.arange(1.0, 7.0)
    expected = jnp.sqrt(rearrange(da[:, None] + db[None, :], "a b -> (a b)")) * v
    assert jnp.allclose(root.mv(v), expected)
    grad = jax.grad(
        lambda d: KroneckerSumSqrt(LazyDiagonal(d), LazyDiagonal(db)).mv(v).sum()
    )(da)
    dense_grad = jax.grad(
        lambda d: (
            jnp.sqrt(rearrange(d[:, None] + db[None, :], "a b -> (a b)")) * v
        ).sum()
    )(da)
    assert jnp.allclose(grad, dense_grad)


@pytest.mark.parametrize("cls", [Kronecker, BlockDiag])
@pytest.mark.parametrize(
    ("wrap", "factor"),
    [
        pytest.param(lambda op, c: c * op, lambda c: c, id="mul"),
        pytest.param(lambda op, c: op / c, lambda c: 1 / c, id="div"),
    ],
)
def test_sqrt_scalar_multiple_keeps_structure(getkey, monkeypatch, cls, wrap, factor):
    """``√(c A) = √c √A`` without materialising ``A`` (gh-326)."""
    k1, k2 = jr.split(getkey())
    a = lx.MatrixLinearOperator(random_pd_matrix(k1, 2), lx.positive_semidefinite_tag)
    b = lx.MatrixLinearOperator(random_pd_matrix(k2, 3), lx.positive_semidefinite_tag)
    op = cls(a, b)
    expected = factor(2.0) * op.as_matrix()

    def _forbidden(self):
        raise AssertionError(f"{cls.__name__}.as_matrix called")

    monkeypatch.setattr(cls, "as_matrix", _forbidden)
    S = sqrt(wrap(op, 2.0))
    monkeypatch.undo()
    assert isinstance(S, cls)
    Sm = S.as_matrix()
    assert tree_allclose(Sm @ Sm, expected)


def test_sqrt_negated_operator_raises(getkey):
    op = lx.MatrixLinearOperator(
        random_pd_matrix(getkey(), 3), lx.positive_semidefinite_tag
    )
    with pytest.raises(ValueError, match="positive semi-definite"):
        sqrt(-op)
