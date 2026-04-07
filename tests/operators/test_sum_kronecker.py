"""Tests for the SumKronecker operator."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import Kronecker, SumKronecker
from gaussx._testing import tree_allclose


def _make_psd(key, n):
    A = jr.normal(key, (n, n))
    return A @ A.T + 0.1 * jnp.eye(n)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic(self, getkey):
        A1 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B1 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        A2 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B2 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        SK = SumKronecker(Kronecker(A1, B1), Kronecker(A2, B2))
        assert SK.in_size() == 6
        assert SK.out_size() == 6

    def test_rejects_three_factor_kronecker(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        C = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        k3 = Kronecker(A, B, C)
        k2 = Kronecker(A, B)
        with pytest.raises(ValueError, match="two-factor"):
            SumKronecker(k3, k2)

    def test_rejects_size_mismatch(self, getkey):
        k1 = Kronecker(
            lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2))),
            lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3))),
        )
        k2 = Kronecker(
            lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2))),
            lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4))),
        )
        with pytest.raises(ValueError, match="same size"):
            SumKronecker(k1, k2)


# ---------------------------------------------------------------------------
# mv correctness
# ---------------------------------------------------------------------------


class TestMv:
    def test_mv_matches_dense(self, getkey):
        A1 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B1 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        A2 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B2 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        SK = SumKronecker(Kronecker(A1, B1), Kronecker(A2, B2))
        v = jr.normal(getkey(), (6,))
        assert tree_allclose(SK.mv(v), SK.as_matrix() @ v)

    def test_mv_scalar_identity_noise(self, getkey):
        """K_task kron K_spatial + sigma^2 I kron I (common GP pattern)."""
        A = lx.MatrixLinearOperator(_make_psd(getkey(), 2))
        B = lx.MatrixLinearOperator(_make_psd(getkey(), 3))
        sigma2 = 0.1
        I_a = lx.MatrixLinearOperator(jnp.sqrt(sigma2) * jnp.eye(2))
        I_b = lx.MatrixLinearOperator(jnp.sqrt(sigma2) * jnp.eye(3))
        SK = SumKronecker(Kronecker(A, B), Kronecker(I_a, I_b))
        v = jr.normal(getkey(), (6,))
        expected = SK.as_matrix() @ v
        assert tree_allclose(SK.mv(v), expected)


# ---------------------------------------------------------------------------
# as_matrix
# ---------------------------------------------------------------------------


class TestAsMatrix:
    def test_as_matrix(self, getkey):
        A1_mat = jr.normal(getkey(), (2, 2))
        B1_mat = jr.normal(getkey(), (3, 3))
        A2_mat = jr.normal(getkey(), (2, 2))
        B2_mat = jr.normal(getkey(), (3, 3))
        SK = SumKronecker(
            Kronecker(
                lx.MatrixLinearOperator(A1_mat),
                lx.MatrixLinearOperator(B1_mat),
            ),
            Kronecker(
                lx.MatrixLinearOperator(A2_mat),
                lx.MatrixLinearOperator(B2_mat),
            ),
        )
        expected = jnp.kron(A1_mat, B1_mat) + jnp.kron(A2_mat, B2_mat)
        assert tree_allclose(SK.as_matrix(), expected)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


class TestTranspose:
    def test_transpose(self, getkey):
        A1 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B1 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        A2 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B2 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        SK = SumKronecker(Kronecker(A1, B1), Kronecker(A2, B2))
        assert tree_allclose(SK.T.as_matrix(), SK.as_matrix().T)

    def test_transpose_mv(self, getkey):
        A1 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B1 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        A2 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B2 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        SK = SumKronecker(Kronecker(A1, B1), Kronecker(A2, B2))
        v = jr.normal(getkey(), (6,))
        assert tree_allclose(SK.T.mv(v), SK.as_matrix().T @ v)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TestTags:
    def test_symmetric_when_both_symmetric(self, getkey):
        A1 = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
        B1 = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        A2 = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
        B2 = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        SK = SumKronecker(Kronecker(A1, B1), Kronecker(A2, B2))
        assert lx.is_symmetric(SK) is True

    def test_not_symmetric_when_factor_not_symmetric(self, getkey):
        A1 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B1 = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        A2 = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
        B2 = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        SK = SumKronecker(Kronecker(A1, B1), Kronecker(A2, B2))
        assert lx.is_symmetric(SK) is False

    def test_not_diagonal(self, getkey):
        A1 = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
        B1 = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        A2 = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
        B2 = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        SK = SumKronecker(Kronecker(A1, B1), Kronecker(A2, B2))
        assert lx.is_diagonal(SK) is False


# ---------------------------------------------------------------------------
# Eigendecompose
# ---------------------------------------------------------------------------


class TestEigendecompose:
    def test_eigendecompose_matches_dense(self, getkey):
        A1 = _make_psd(getkey(), 2)
        B1 = _make_psd(getkey(), 3)
        A2 = _make_psd(getkey(), 2)
        B2 = _make_psd(getkey(), 3)
        SK = SumKronecker(
            Kronecker(
                lx.MatrixLinearOperator(A1, lx.positive_semidefinite_tag),
                lx.MatrixLinearOperator(B1, lx.positive_semidefinite_tag),
            ),
            Kronecker(
                lx.MatrixLinearOperator(A2, lx.positive_semidefinite_tag),
                lx.MatrixLinearOperator(B2, lx.positive_semidefinite_tag),
            ),
        )
        evals, Q = SK.eigendecompose()
        # Reconstruct: Q diag(evals) Q^T should match as_matrix
        reconstructed = Q @ jnp.diag(evals) @ Q.T
        assert tree_allclose(reconstructed, SK.as_matrix(), rtol=1e-4)

    def test_eigenvalues_positive_for_psd(self, getkey):
        A1 = _make_psd(getkey(), 3)
        B1 = _make_psd(getkey(), 2)
        A2 = _make_psd(getkey(), 3)
        B2 = _make_psd(getkey(), 2)
        SK = SumKronecker(
            Kronecker(
                lx.MatrixLinearOperator(A1, lx.positive_semidefinite_tag),
                lx.MatrixLinearOperator(B1, lx.positive_semidefinite_tag),
            ),
            Kronecker(
                lx.MatrixLinearOperator(A2, lx.positive_semidefinite_tag),
                lx.MatrixLinearOperator(B2, lx.positive_semidefinite_tag),
            ),
        )
        evals, _ = SK.eigendecompose()
        assert jnp.all(evals > -1e-6)

    def test_logdet_via_eigendecompose(self, getkey):
        A1 = _make_psd(getkey(), 2)
        B1 = _make_psd(getkey(), 3)
        A2 = _make_psd(getkey(), 2)
        B2 = _make_psd(getkey(), 3)
        SK = SumKronecker(
            Kronecker(
                lx.MatrixLinearOperator(A1, lx.positive_semidefinite_tag),
                lx.MatrixLinearOperator(B1, lx.positive_semidefinite_tag),
            ),
            Kronecker(
                lx.MatrixLinearOperator(A2, lx.positive_semidefinite_tag),
                lx.MatrixLinearOperator(B2, lx.positive_semidefinite_tag),
            ),
        )
        evals, _ = SK.eigendecompose()
        ld_eigen = jnp.sum(jnp.log(evals))
        ld_dense = jnp.linalg.slogdet(SK.as_matrix())[1]
        assert tree_allclose(ld_eigen, ld_dense, rtol=1e-4)


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------


class TestJAX:
    def test_jit(self, getkey):
        A1 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B1 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        A2 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B2 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        SK = SumKronecker(Kronecker(A1, B1), Kronecker(A2, B2))
        v = jr.normal(getkey(), (6,))

        @eqx.filter_jit
        def f(op, v):
            return op.mv(v)

        assert tree_allclose(f(SK, v), SK.as_matrix() @ v)

    def test_grad(self, getkey):
        B1_mat = jr.normal(getkey(), (3, 3))
        A2_mat = jr.normal(getkey(), (2, 2))
        B2_mat = jr.normal(getkey(), (3, 3))
        v = jr.normal(getkey(), (6,))

        def loss(a1_mat):
            SK = SumKronecker(
                Kronecker(
                    lx.MatrixLinearOperator(a1_mat),
                    lx.MatrixLinearOperator(B1_mat),
                ),
                Kronecker(
                    lx.MatrixLinearOperator(A2_mat),
                    lx.MatrixLinearOperator(B2_mat),
                ),
            )
            return jnp.sum(SK.mv(v) ** 2)

        A1_mat = jr.normal(getkey(), (2, 2))
        g = jax.grad(loss)(A1_mat)
        assert g.shape == (2, 2)
        assert jnp.all(jnp.isfinite(g))
