"""Tests for the temporal precision builders."""

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import gaussx


def path_laplacian(weights: np.ndarray) -> np.ndarray:
    """Weighted path-graph Laplacian, edge i joining nodes i and i + 1."""
    n = weights.shape[0] + 1
    L = np.zeros((n, n))
    for i, w in enumerate(weights):
        L[i, i] += w
        L[i + 1, i + 1] += w
        L[i, i + 1] -= w
        L[i + 1, i] -= w
    return L


def second_difference(n: int, cyclic: bool = False) -> np.ndarray:
    rows = n if cyclic else n - 2
    D = np.zeros((rows, n))
    for r in range(rows):
        for offset, value in zip((0, 1, 2), (1.0, -2.0, 1.0), strict=True):
            D[r, (r + offset) % n] += value
    return D


def gram(D: np.ndarray) -> np.ndarray:
    return einx.dot("k i, k j -> i j", D, D)


def test_iid_precision():
    Q = gaussx.iid_precision(4, 2.5)
    assert isinstance(Q, lx.DiagonalLinearOperator)
    assert jnp.allclose(Q.as_matrix(), 2.5 * jnp.eye(4))


class TestRW1:
    def test_equals_path_laplacian(self):
        R = gaussx.rw1_structure(6)
        assert isinstance(R, gaussx.BlockTriDiag)
        assert lx.is_positive_semidefinite(R)
        assert jnp.allclose(R.as_matrix(), path_laplacian(np.ones(5)))
        assert jnp.allclose(R.mv(jnp.ones(6)), 0.0)

    def test_irregular_spacing_uses_inverse_gaps(self):
        gaps = np.array([0.5, 1.0, 2.0, 0.25])
        R = gaussx.rw1_structure(5, spacing=gaps)
        assert jnp.allclose(R.as_matrix(), path_laplacian(1.0 / gaps))

    def test_cyclic(self):
        R = gaussx.rw1_structure(5, cyclic=True)
        assert isinstance(R, gaussx.SparseOperator)
        expected = path_laplacian(np.ones(4))
        expected[0, 0] += 1.0
        expected[-1, -1] += 1.0
        expected[0, -1] = expected[-1, 0] = -1.0
        assert jnp.allclose(R.as_matrix(), expected)

    def test_selected_inverse_dispatch(self):
        R = gaussx.rw1_structure(8)
        H = gaussx.BlockTriDiag(R.diagonal + 0.5, R.sub_diagonal)
        expected = jnp.diag(jnp.linalg.inv(H.as_matrix()))
        assert jnp.allclose(gaussx.diag_inv(H), expected)


class TestRW2:
    def test_equals_second_difference_gram(self):
        R = gaussx.rw2_structure(8)
        D = second_difference(8)
        assert R.diagonal.shape == (4, 2, 2)
        assert jnp.allclose(R.as_matrix(), gram(D))

    @pytest.mark.parametrize("n", [7, 8])
    def test_null_space(self, n):
        R = gaussx.rw2_structure(n)
        size = R.in_size()
        t = jnp.zeros(size).at[:n].set(jnp.arange(n, dtype=float))
        ones = jnp.zeros(size).at[:n].set(1.0)
        assert jnp.allclose(R.mv(ones), 0.0)
        assert jnp.allclose(R.mv(t), 0.0)
        eigenvalues = jnp.linalg.eigvalsh(R.as_matrix())
        assert int(jnp.sum(eigenvalues < 1e-9)) == 2

    def test_odd_n_padding_is_decoupled(self):
        R = gaussx.rw2_structure(7)
        assert R.in_size() == 8
        M = R.as_matrix()
        D = second_difference(7)
        assert jnp.allclose(M[:7, :7], gram(D))
        assert jnp.allclose(M[7], jnp.zeros(8).at[7].set(1.0))
        # Solves strip cleanly: the padding does not touch the first n entries.
        H = gaussx.BlockTriDiag(R.diagonal + jnp.eye(2), R.sub_diagonal)
        b = jnp.arange(8.0).at[7].set(0.0)
        x = gaussx.solve(H, b)
        dense = jnp.linalg.solve(gram(D) + jnp.eye(7), b[:7])
        assert jnp.allclose(x[:7], dense)

    def test_cyclic(self):
        R = gaussx.rw2_structure(7, cyclic=True)
        D = second_difference(7, cyclic=True)
        assert R.in_size() == 7
        assert jnp.allclose(R.as_matrix(), gram(D))
        assert jnp.allclose(R.mv(jnp.ones(7)), 0.0)

    def test_too_small_raises(self):
        with pytest.raises(ValueError, match="n >= 3"):
            gaussx.rw2_structure(2)


class TestAR1:
    def test_matches_stationary_covariance(self):
        n, rho, tau = 6, 0.7, 4.0
        Q = gaussx.ar1_precision(n, rho, tau)
        lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        covariance = rho**lags / tau
        assert jnp.allclose(jnp.linalg.inv(Q.as_matrix()), covariance)

    def test_marginal_variance(self):
        Q = gaussx.ar1_precision(50, rho=0.8, tau=10.0)
        assert jnp.allclose(gaussx.diag_inv(Q), 0.1)

    def test_traced_parameters(self):
        def logdet(rho, tau):
            return gaussx.logdet(gaussx.ar1_precision(10, rho, tau))

        # log|Q| = n log τ − (n − 1) log(1 − ρ²) for the stationary AR(1)
        expected = 10 * jnp.log(2.0) - 9 * jnp.log(1 - 0.5**2)
        assert jnp.allclose(jax.jit(logdet)(0.5, 2.0), expected)
        grad = jax.grad(logdet)(0.5, 2.0)
        assert jnp.allclose(grad, 9 * 2 * 0.5 / (1 - 0.5**2))

    def test_dtype_follows_parameters(self):
        Q = gaussx.ar1_precision(4, jnp.float32(0.5), jnp.float32(1.0))
        assert Q.diagonal.dtype == jnp.float32
