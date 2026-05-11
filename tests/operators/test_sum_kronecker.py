"""Tests for the SumKronecker operator."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import Kronecker, SumKronecker, sumkronecker_sample
from gaussx._primitives import DenseFallbackWarning, SumKroneckerSqrt, cholesky, sqrt
from gaussx._testing import tree_allclose


def _make_psd(key, n):
    A = jr.normal(key, (n, n))
    return A @ A.T + 0.1 * jnp.eye(n)


def _make_psd_sum_kronecker(getkey):
    A1 = _make_psd(getkey(), 2)
    B1 = _make_psd(getkey(), 3)
    A2 = _make_psd(getkey(), 2)
    B2 = _make_psd(getkey(), 3)
    return SumKronecker(
        Kronecker(
            lx.MatrixLinearOperator(A1, lx.positive_semidefinite_tag),
            lx.MatrixLinearOperator(B1, lx.positive_semidefinite_tag),
        ),
        Kronecker(
            lx.MatrixLinearOperator(A2, lx.positive_semidefinite_tag),
            lx.MatrixLinearOperator(B2, lx.positive_semidefinite_tag),
        ),
        tags=lx.positive_semidefinite_tag,
    )


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

    def test_accepts_more_than_two_terms(self, getkey):
        terms = [
            Kronecker(
                lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2))),
                lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3))),
            )
            for _ in range(3)
        ]
        SK = SumKronecker(*terms)
        expected = terms[0].as_matrix()
        for term in terms[1:]:
            expected = expected + term.as_matrix()
        assert tree_allclose(SK.as_matrix(), expected)

    def test_rejects_three_factor_kronecker(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        C = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        k3 = Kronecker(A, B, C)
        k2 = Kronecker(A, B)
        with pytest.raises(ValueError, match="two-factor"):
            SumKronecker(k3, k2)

    def test_rejects_input_size_mismatch(self, getkey):
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

    def test_rejects_output_size_mismatch(self, getkey):
        k1 = Kronecker(
            lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2))),
            lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3))),
        )
        k2 = Kronecker(
            lx.MatrixLinearOperator(jr.normal(getkey(), (3, 2))),
            lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3))),
        )
        with pytest.raises(ValueError, match="output size"):
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


def test_eigendecompose_supports_diagonal_kron2_factors(getkey, monkeypatch):
    """SumKronecker.eigendecompose accepts Diagonal kron2 factors and
    routes their per-factor eig through the structural primitive,
    *not* through ``jnp.linalg.eigh`` on a materialized matrix.

    Asserts both correctness (Q diag(evals) Q^T == SK.as_matrix()) and
    that the Diagonal-factor path was taken (no fallback to
    ``jnp.linalg.eigh`` on those factors). The joint diagonalization
    step still calls ``jnp.linalg.eigh`` on the ``(n_c·n_d, n_c·n_d)``
    transformed matrix, which is unavoidable.
    """
    import jax.numpy as _jnp

    from gaussx._testing import random_pd_matrix

    A1_mat = random_pd_matrix(getkey(), 2)
    B1_mat = random_pd_matrix(getkey(), 3)
    A2_diag = jnp.abs(jr.normal(getkey(), (2,))) + 1.0
    B2_diag = jnp.abs(jr.normal(getkey(), (3,))) + 1.0

    A1 = lx.MatrixLinearOperator(A1_mat, lx.symmetric_tag)
    B1 = lx.MatrixLinearOperator(B1_mat, lx.symmetric_tag)
    A2 = lx.DiagonalLinearOperator(A2_diag)
    B2 = lx.DiagonalLinearOperator(B2_diag)
    SK = SumKronecker(Kronecker(A1, B1), Kronecker(A2, B2))

    # Wrap jnp.linalg.eigh so we can count calls and inspect argument
    # shapes. The Diagonal-factor path should produce *exactly one*
    # eigh call — on the transformed (n_c·n_d, n_c·n_d) matrix.
    eigh_calls: list[tuple[int, int]] = []
    real_eigh = _jnp.linalg.eigh

    def spy_eigh(mat, *args, **kwargs):
        eigh_calls.append(mat.shape)
        return real_eigh(mat, *args, **kwargs)

    monkeypatch.setattr(_jnp.linalg, "eigh", spy_eigh)

    evals, Q = SK.eigendecompose()
    reconstructed = Q @ jnp.diag(evals) @ Q.T
    assert tree_allclose(reconstructed, SK.as_matrix(), rtol=1e-4, atol=1e-6)

    # No (2, 2) or (3, 3) eigh calls on the Diagonal kron2 factors.
    assert (2, 2) not in eigh_calls
    assert (3, 3) not in eigh_calls
    # The single remaining eigh is on the (6, 6) transformed matrix.
    assert (6, 6) in eigh_calls


def test_eigendecompose_rejects_nonsymmetric_kron1(getkey):
    """Non-symmetric kron1 factors should raise ValueError, not silently
    return wrong results from ``eigh`` on a non-symmetric matrix."""
    A1 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))  # not tagged symmetric
    B1 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    A2_mat = jr.normal(getkey(), (2, 2))
    B2_mat = jr.normal(getkey(), (3, 3))
    A2 = lx.MatrixLinearOperator(A2_mat @ A2_mat.T, lx.symmetric_tag)
    B2 = lx.MatrixLinearOperator(B2_mat @ B2_mat.T, lx.symmetric_tag)
    SK = SumKronecker(Kronecker(A1, B1), Kronecker(A2, B2))

    with pytest.raises(ValueError, match="kron1 factors"):
        SK.eigendecompose()


def test_sqrt_sumkronecker_returns_lanczos_operator(getkey, monkeypatch):
    SK = _make_psd_sum_kronecker(getkey)
    v = jr.normal(getkey(), (SK.in_size(),))

    def fail_as_matrix(self):
        raise AssertionError("Lanczos sqrt should use SumKronecker.mv")

    monkeypatch.setattr(SumKronecker, "as_matrix", fail_as_matrix)
    sqrt_op = sqrt(SK, lanczos_order=SK.in_size())
    assert isinstance(sqrt_op, SumKroneckerSqrt)
    result = sqrt_op.mv(v)
    assert result.shape == v.shape
    assert jnp.all(jnp.isfinite(result))


def test_sumkronecker_sample_matches_dense_reference(getkey):
    SK = _make_psd_sum_kronecker(getkey)
    key = getkey()
    num_samples = 3
    samples = sumkronecker_sample(
        SK,
        key=key,
        num_samples=num_samples,
        lanczos_order=SK.in_size(),
    )

    eps = jr.normal(key, (num_samples, SK.in_size()), dtype=SK.in_structure().dtype)
    vals, vecs = jnp.linalg.eigh(SK.as_matrix())
    dense_sqrt = vecs @ jnp.diag(jnp.sqrt(jnp.maximum(vals, 0.0))) @ vecs.T
    expected = jax.vmap(lambda e: dense_sqrt @ e)(eps)

    assert samples.shape == (num_samples, SK.in_size())
    assert tree_allclose(samples, expected, rtol=0.1, atol=1e-5)


def test_sumkronecker_sample_reproducible(getkey):
    SK = _make_psd_sum_kronecker(getkey)
    key = getkey()
    samples1 = sumkronecker_sample(SK, key=key, num_samples=2, lanczos_order=4)
    samples2 = sumkronecker_sample(SK, key=key, num_samples=2, lanczos_order=4)
    assert tree_allclose(samples1, samples2)


def test_cholesky_sumkronecker_warns_dense_fallback(getkey):
    SK = _make_psd_sum_kronecker(getkey)
    with pytest.warns(DenseFallbackWarning, match="sumkronecker_sample"):
        L = cholesky(SK)
    assert tree_allclose(L.as_matrix() @ L.as_matrix().T, SK.as_matrix(), rtol=1e-4)
