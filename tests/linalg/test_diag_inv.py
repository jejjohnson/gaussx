"""Tests for diag_inv."""

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx import diag_inv
from gaussx._einx import rearrange
from gaussx._strategies import DenseSolver


class TestDiagInv:
    @pytest.mark.slow
    def test_cholesky_matches_dense(self, getkey):
        """Cholesky method matches jnp.diag(jnp.linalg.inv(A))."""
        N = 12
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        result = diag_inv(op, method="cholesky")
        expected = jnp.diag(jnp.linalg.inv(K))
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_hutchinson_converges(self, getkey):
        """Hutchinson estimate is close with many probes."""
        N = 12
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        key = jax.random.PRNGKey(42)
        result = diag_inv(op, method="hutchinson", num_probes=1000, key=key)
        expected = jnp.diag(jnp.linalg.inv(K))
        assert jnp.allclose(result, expected, rtol=0.2, atol=0.05)

    def test_solve_matches_dense(self, getkey):
        """Solve method matches jnp.diag(jnp.linalg.inv(A))."""
        N = 8
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        result = diag_inv(op, method="solve", solver=DenseSolver())
        expected = jnp.diag(jnp.linalg.inv(K))
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_auto_selects_cholesky_small(self, getkey):
        """Auto mode uses cholesky for small N."""
        N = 8
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        result = diag_inv(op, method="auto")
        expected = jnp.diag(jnp.linalg.inv(K))
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_output_shape(self, getkey):
        """Output shape is (N,)."""
        N = 10
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        result = diag_inv(op)
        assert result.shape == (N,)

    def test_invalid_method_raises(self, getkey):
        """Unknown method raises ValueError."""
        N = 5
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        with pytest.raises(ValueError, match="Unknown method"):
            diag_inv(op, method="bogus")


# ---------------------------------------------------------------------------
# Structural dispatch (G3). Keys are pinned: these check identities.
# ---------------------------------------------------------------------------


def _path_laplacian(n):
    """Graph Laplacian of a path: singular, constants in the null space."""
    off = -jnp.ones(n - 1)
    L = jnp.diag(jnp.concatenate([jnp.ones(1), 2.0 * jnp.ones(n - 2), jnp.ones(1)]))
    L = L + jnp.diag(off, 1) + jnp.diag(off, -1)
    return lx.MatrixLinearOperator(L, lx.symmetric_tag)


def _spd(key, n):
    m = jr.normal(key, (n, n))
    return lx.MatrixLinearOperator(m @ m.T + n * jnp.eye(n), lx.symmetric_tag)


def _identity(n):
    return lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n,), jnp.float64))


class _MatvecOnlyKroneckerSum(gaussx.KroneckerSum):
    """A spatial factor that must never be materialised."""

    def as_matrix(self):
        raise AssertionError("the spectral factor was materialised")


def _dense_diag_inv(operator):
    return jnp.diag(jnp.linalg.inv(operator.as_matrix()))


class TestStructuredDiagInv:
    @pytest.mark.slow
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_block_tridiag(self, d):
        N = 5
        k1, k2 = jr.split(jr.key(0))
        raw = jr.normal(k1, (N, d, d))
        diag = raw @ rearrange(raw, "N i j -> N j i") + 5.0 * jnp.eye(d)
        op = gaussx.BlockTriDiag(diag, 0.3 * jr.normal(k2, (N - 1, d, d)))
        assert jnp.allclose(diag_inv(op), _dense_diag_inv(op), atol=1e-12)
        assert jnp.allclose(jax.jit(diag_inv)(op), _dense_diag_inv(op), atol=1e-12)

    def test_kronecker(self):
        op = gaussx.Kronecker(_spd(jr.key(0), 3), _spd(jr.key(1), 4))
        assert jnp.allclose(diag_inv(op), _dense_diag_inv(op), atol=1e-12)

    def test_kronecker_sum(self):
        op = gaussx.KroneckerSum(_spd(jr.key(0), 3), _path_laplacian(4))
        assert jnp.allclose(diag_inv(op), _dense_diag_inv(op), atol=1e-12)

    def test_kronecker_sum_pinv_on_singular_grid_laplacian(self):
        op = gaussx.KroneckerSum(_path_laplacian(4), _path_laplacian(5))
        expected = jnp.diag(jnp.linalg.pinv(op.as_matrix()))
        assert jnp.allclose(diag_inv(op, pinv=True), expected, atol=1e-12)

    def test_kronecker_sum_never_materialised(self):
        op = _MatvecOnlyKroneckerSum(_path_laplacian(4), _spd(jr.key(0), 5))
        dense = gaussx.KroneckerSum(op.A, op.B)
        assert jnp.allclose(diag_inv(op), _dense_diag_inv(dense), atol=1e-12)

    def test_diagonalised_operator(self):
        circulant = gaussx.Circulant(jnp.array([2.5, -1.0, 0.0, 0.0, -1.0]))
        op = gaussx.KroneckerSum(_spd(jr.key(0), 3), circulant)
        assert jnp.allclose(diag_inv(circulant), _dense_diag_inv(circulant))
        assert jnp.allclose(diag_inv(op), _dense_diag_inv(op), atol=1e-12)

    def test_pinv_without_eigen_structure_raises(self):
        with pytest.raises(ValueError, match="pinv=True"):
            diag_inv(_spd(jr.key(0), 4), pinv=True)


class TestShiftedKronecker:
    """``A ⊗ B + cI`` with a spectral ``B`` that is only ever applied."""

    c = 0.7

    def _operators(self, spatial):
        temporal = _spd(jr.key(0), 3)
        n = 3 * spatial.in_size()
        shifted = gaussx.SumOfKroneckers(
            gaussx.Kronecker(temporal, spatial),
            gaussx.Kronecker(_identity(3), self.c * _identity(spatial.in_size())),
        )
        lineax_sum = gaussx.Kronecker(temporal, spatial) + self.c * _identity(n)
        return temporal, shifted, lineax_sum

    def _dense(self, temporal, spatial_matrix):
        n = temporal.in_size() * spatial_matrix.shape[0]
        return jnp.kron(temporal.as_matrix(), spatial_matrix) + self.c * jnp.eye(n)

    @pytest.mark.parametrize("form", ["sum_of_kroneckers", "lineax_sum"])
    @pytest.mark.x64_only(reason="dense-reference tolerance below float32 round-off")
    def test_matches_dense_without_materialising_spatial_factor(self, form):
        L_H, L_W = _path_laplacian(3), _path_laplacian(4)
        spatial = _MatvecOnlyKroneckerSum(L_H, L_W)
        temporal, shifted, lineax_sum = self._operators(spatial)
        op = shifted if form == "sum_of_kroneckers" else lineax_sum
        K = self._dense(temporal, gaussx.KroneckerSum(L_H, L_W).as_matrix())
        b = jr.normal(jr.key(1), (K.shape[0],))
        assert jnp.allclose(gaussx.solve(op, b), jnp.linalg.solve(K, b), atol=1e-12)
        assert jnp.allclose(gaussx.logdet(op), jnp.linalg.slogdet(K)[1], atol=1e-10)
        expected = jnp.diag(jnp.linalg.inv(K))
        assert jnp.allclose(diag_inv(op), expected, atol=1e-12)

    @pytest.mark.x64_only(reason="dense-reference tolerance below float32 round-off")
    def test_fft_spatial_factor(self):
        spatial = gaussx.Circulant(jnp.array([2.5, -1.0, 0.0, 0.0, 0.0, -1.0]))
        temporal, shifted, _ = self._operators(spatial)
        K = self._dense(temporal, spatial.as_matrix())
        b = jr.normal(jr.key(1), (K.shape[0],))
        x = gaussx.solve(shifted, b)
        assert not jnp.iscomplexobj(x)
        assert jnp.allclose(x, jnp.linalg.solve(K, b), atol=1e-12)
        assert jnp.allclose(gaussx.logdet(shifted), jnp.linalg.slogdet(K)[1])
        assert jnp.allclose(diag_inv(shifted), jnp.diag(jnp.linalg.inv(K)), atol=1e-12)

    def test_jit(self):
        spatial = gaussx.KroneckerSum(_path_laplacian(3), _path_laplacian(4))
        temporal, shifted, _ = self._operators(spatial)
        K = self._dense(temporal, spatial.as_matrix())
        got = jax.jit(diag_inv)(shifted)
        assert jnp.allclose(got, jnp.diag(jnp.linalg.inv(K)), atol=1e-12)
