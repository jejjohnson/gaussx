"""Tests for BlockTriDiag and LowerBlockTriDiag operators."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._einx import einsum
from gaussx._testing import random_pd_matrix


def _make_spd_block_tridiag(N, d, key):
    """Create a random SPD block-tridiagonal matrix.

    Makes the diagonal blocks dominant enough to ensure positive definiteness.
    """
    k1, k2 = jax.random.split(key)
    # Sub-diagonal blocks: random
    sub = 0.3 * jax.random.normal(k1, (N - 1, d, d))
    # Diagonal blocks: SPD with diagonal dominance
    raw = jax.random.normal(k2, (N, d, d))
    diags = jax.vmap(lambda M: M @ M.T)(raw) + 5.0 * jnp.eye(d)[None]
    # Make symmetric
    diags = 0.5 * (diags + jnp.swapaxes(diags, -2, -1))
    return diags, sub


@pytest.fixture()
def block_tridiag():
    """SPD block-tridiagonal operator."""
    N, d = 5, 3
    diags, sub = _make_spd_block_tridiag(N, d, jax.random.PRNGKey(42))
    return gaussx.BlockTriDiag(diags, sub)


class TestBlockTriDiag:
    @pytest.mark.slow
    def test_mv(self, block_tridiag):
        n = block_tridiag.in_size()
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (n,))
        result = block_tridiag.mv(x)
        expected = block_tridiag.as_matrix() @ x
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_as_matrix_shape(self, block_tridiag):
        mat = block_tridiag.as_matrix()
        assert mat.shape == (15, 15)

    def test_as_matrix_band_structure(self, block_tridiag):
        """Verify entries outside the band are zero."""
        mat = block_tridiag.as_matrix()
        d = block_tridiag._block_size
        N = block_tridiag._num_blocks
        for i in range(N):
            for j in range(N):
                if abs(i - j) > 1:
                    block = mat[i * d : (i + 1) * d, j * d : (j + 1) * d]
                    assert jnp.allclose(block, 0.0, atol=1e-10)

    def test_transpose(self, block_tridiag):
        mat = block_tridiag.as_matrix()
        mat_t = block_tridiag.T.as_matrix()
        assert jnp.allclose(mat.T, mat_t, atol=1e-7)

    def test_cholesky(self, block_tridiag):
        L = gaussx.cholesky(block_tridiag)
        assert isinstance(L, gaussx.LowerBlockTriDiag)
        # L L^T should reconstruct the original
        reconstructed = L.as_matrix() @ L.as_matrix().T
        expected = block_tridiag.as_matrix()
        assert jnp.allclose(reconstructed, expected, atol=1e-4)

    def test_solve(self, block_tridiag):
        key = jax.random.PRNGKey(1)
        b = jax.random.normal(key, (block_tridiag.in_size(),))
        x = gaussx.solve(block_tridiag, b)
        residual = block_tridiag.mv(x) - b
        assert jnp.allclose(residual, 0.0, atol=1e-4)

    def test_logdet(self, block_tridiag):
        ld = gaussx.logdet(block_tridiag)
        mat = block_tridiag.as_matrix()
        expected = jnp.linalg.slogdet(mat)[1]
        assert jnp.allclose(ld, expected, atol=1e-3)

    def test_diag(self, block_tridiag):
        d = gaussx.diag(block_tridiag)
        expected = jnp.diag(block_tridiag.as_matrix())
        assert jnp.allclose(d, expected, atol=1e-7)

    def test_trace(self, block_tridiag):
        tr = gaussx.trace(block_tridiag)
        expected = jnp.trace(block_tridiag.as_matrix())
        assert jnp.allclose(tr, expected, atol=1e-5)

    def test_add(self, block_tridiag):
        result = block_tridiag.add(block_tridiag)
        mat_sum = result.as_matrix()
        expected = 2.0 * block_tridiag.as_matrix()
        assert jnp.allclose(mat_sum, expected, atol=1e-7)

    def test_scalar_mul_requires_true_scalar(self, block_tridiag):
        with pytest.raises(TypeError, match="scalar"):
            block_tridiag * jnp.array([1.0, 2.0])

    def test_tags(self, block_tridiag):
        assert gaussx.is_block_tridiagonal(block_tridiag)

    def test_reports_symmetric(self, block_tridiag):
        assert lx.is_symmetric(block_tridiag)

    def test_in_out_size(self, block_tridiag):
        assert block_tridiag.in_size() == 15
        assert block_tridiag.out_size() == 15

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="3 dimensions"):
            gaussx.BlockTriDiag(jnp.ones((3, 3)), jnp.ones((2, 3, 3)))

        with pytest.raises(ValueError, match="square"):
            gaussx.BlockTriDiag(jnp.ones((3, 2, 3)), jnp.ones((2, 2, 3)))

        with pytest.raises(ValueError, match="2 blocks"):
            gaussx.BlockTriDiag(jnp.ones((3, 2, 2)), jnp.ones((3, 2, 2)))


class TestLowerBlockTriDiag:
    def test_mv(self, block_tridiag):
        L = gaussx.cholesky(block_tridiag)
        n = L.in_size()
        key = jax.random.PRNGKey(5)
        x = jax.random.normal(key, (n,))
        result = L.mv(x)
        expected = L.as_matrix() @ x
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_transpose_gives_upper(self, block_tridiag):
        L = gaussx.cholesky(block_tridiag)
        U = L.T
        assert isinstance(U, gaussx.UpperBlockTriDiag)
        assert jnp.allclose(U.as_matrix(), L.as_matrix().T, atol=1e-7)

    def test_logdet(self, block_tridiag):
        L = gaussx.cholesky(block_tridiag)
        ld = gaussx.logdet(L)
        expected = jnp.linalg.slogdet(L.as_matrix())[1]
        assert jnp.allclose(ld, expected, atol=1e-4)

    def test_solve_forward(self, block_tridiag):
        L = gaussx.cholesky(block_tridiag)
        key = jax.random.PRNGKey(3)
        b = jax.random.normal(key, (L.in_size(),))
        x = gaussx.solve(L, b)
        residual = L.mv(x) - b
        assert jnp.allclose(residual, 0.0, atol=1e-4)

    def test_solve_backward(self, block_tridiag):
        L = gaussx.cholesky(block_tridiag)
        U = L.T
        key = jax.random.PRNGKey(4)
        b = jax.random.normal(key, (U.in_size(),))
        x = gaussx.solve(U, b)
        residual = U.mv(x) - b
        assert jnp.allclose(residual, 0.0, atol=1e-4)


class TestSingleBlock:
    """gh-304: N = 1 is accepted by the constructor and is just the block D."""

    @staticmethod
    def _ops(d):
        D = random_pd_matrix(jr.key(0), d) + jnp.eye(d)
        L = jnp.linalg.cholesky(D)
        empty = jnp.zeros((0, d, d))
        return D, {
            "full": gaussx.BlockTriDiag(D[None], empty),
            "lower": gaussx.LowerBlockTriDiag(L[None], empty),
            "upper": gaussx.UpperBlockTriDiag(L.T[None], empty),
        }

    @pytest.mark.parametrize("d", [1, 3])
    @pytest.mark.parametrize("kind", ["full", "lower", "upper"])
    @pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
    def test_matches_dense(self, d, kind, jit):
        _, ops = self._ops(d)
        op = ops[kind]
        M = op.as_matrix()
        b = jnp.arange(1.0, d + 1.0)

        def run(op, b):
            return {
                "mv": op.mv(b),
                "T": op.T.as_matrix(),
                "solve": gaussx.solve(op, b),
                "logdet": gaussx.logdet(op),
                "inv": gaussx.inv(op).as_matrix(),
            }

        out = eqx.filter_jit(run)(op, b) if jit else run(op, b)
        expected = {
            "mv": M @ b,
            "T": M.T,
            "solve": jnp.linalg.solve(M, b),
            "logdet": jnp.linalg.slogdet(M)[1],
            "inv": jnp.linalg.inv(M),
        }
        for name, value in expected.items():
            assert jnp.allclose(out[name], value, rtol=1e-12, atol=1e-12), name

    @pytest.mark.parametrize("d", [1, 3])
    def test_cholesky(self, d):
        D, ops = self._ops(d)
        L = gaussx.cholesky(ops["full"])
        assert isinstance(L, gaussx.LowerBlockTriDiag)
        assert L.sub_diagonal.shape == (0, d, d)
        assert jnp.allclose(L.as_matrix(), jnp.linalg.cholesky(D), atol=1e-12)

    @pytest.mark.slow
    def test_spingp_log_likelihood_one_step(self):
        D, ops = self._ops(2)
        H = jnp.array([[1.0, 0.0]])
        R = lx.MatrixLinearOperator(jnp.array([[0.1]]))
        y = jnp.array([[0.3]])
        S = H @ jnp.linalg.inv(D) @ H.T + R.as_matrix()
        expected = jax.scipy.stats.multivariate_normal.logpdf(y[0], jnp.zeros(1), S)
        got = gaussx.spingp_log_likelihood(ops["full"], H, R, y)
        assert jnp.allclose(got, expected, rtol=1e-12)


class TestSymmetryAndTags:
    """gh-344: a truthful symmetric_tag, and tags kept through the algebra."""

    @staticmethod
    def _nonsymmetric():
        N, d = 3, 2
        kD, kA = jr.split(jr.key(0))
        D = jr.normal(kD, (N, d, d)) + 4.0 * jnp.eye(d)
        A = 0.3 * jr.normal(kA, (N - 1, d, d))
        return D, A

    @staticmethod
    def _psd():
        D, A = TestSymmetryAndTags._nonsymmetric()
        Ds = einsum(D, D, "n i j, n k j -> n i k") + 4.0 * jnp.eye(D.shape[-1])
        return gaussx.BlockTriDiag(Ds, A, tags=lx.positive_semidefinite_tag)

    def test_nonsymmetric_blocks_raise(self):
        D, A = self._nonsymmetric()
        with pytest.raises(ValueError, match="not symmetric"):
            gaussx.BlockTriDiag(D, A)

    def test_symmetric_false_solve_and_logdet_match_dense(self):
        D, A = self._nonsymmetric()
        op = gaussx.BlockTriDiag(D, A, symmetric=False)
        M = op.as_matrix()
        b = jnp.arange(1.0, op.in_size() + 1)
        assert not lx.is_symmetric(op)
        assert jnp.allclose(gaussx.solve(op, b), jnp.linalg.solve(M, b), atol=1e-10)
        assert jnp.allclose(gaussx.logdet(op), jnp.linalg.slogdet(M)[1], atol=1e-10)
        assert jnp.allclose(op.T.as_matrix(), M.T)
        assert not lx.is_symmetric(op.T)

    def test_roundoff_asymmetry_is_accepted(self):
        op = self._psd()
        noise = 1e-12 * jr.normal(jr.key(1), op.diagonal.shape)
        assert lx.is_symmetric(
            gaussx.BlockTriDiag(op.diagonal + noise, op.sub_diagonal)
        )

    def test_psd_tags_propagate(self):
        psd = self._psd()
        assert lx.is_positive_semidefinite(psd + psd)
        assert lx.is_positive_semidefinite(2.0 * psd)
        assert lx.is_negative_semidefinite(-psd)
        assert lx.is_negative_semidefinite((-1.0) * psd)
        assert not lx.is_positive_semidefinite(psd - psd)
        assert not lx.is_positive_semidefinite((-1.0) * psd)
        for op in (psd + psd, psd - psd, -psd, 2.0 * psd):
            assert lx.is_symmetric(op)

    def test_nonsymmetric_operand_drops_symmetry(self):
        D, A = self._nonsymmetric()
        nonsym = gaussx.BlockTriDiag(D, A, symmetric=False)
        assert not lx.is_symmetric(self._psd() + nonsym)

    def test_tags_under_jit(self):
        psd = self._psd()

        @eqx.filter_jit
        def run(op, scale):
            # A traced scale has no known sign, so it drops definiteness.
            return op + op, scale * op

        total, scaled = run(psd, jnp.asarray(2.0))
        assert lx.is_positive_semidefinite(total)
        assert lx.is_symmetric(scaled)
        assert not lx.is_positive_semidefinite(scaled)
