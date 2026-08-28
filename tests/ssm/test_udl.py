"""Tests for the UDL factorisation of block-tridiagonal precisions."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._ssm._ssm_natural import ssm_to_naturals
from gaussx._ssm._udl import (
    UDLDecomposition,
    udl_decomposition,
    udl_from_ssm_params,
    udl_to_ssm_params,
)


def _make_spd_block_tridiag(key, T, d, coupling=0.3):
    """Random SPD block-tridiagonal precision (diagonally dominant)."""
    k1, k2 = jr.split(key)
    raw = jr.normal(k1, (T, d, d))
    diag = jax.vmap(lambda M: M @ M.T)(raw) + 5.0 * jnp.eye(d)[None]
    sub = coupling * jr.normal(k2, (T - 1, d, d))
    return BlockTriDiag(diag, sub)


def _make_ssm(key, T, d):
    """Random stable chain in the ``Q[0] == P0`` layout."""
    k1, k2 = jr.split(key)
    A = 0.8 * jnp.eye(d)[None] + 0.05 * jr.normal(k1, (T - 1, d, d))
    raw = jr.normal(k2, (T, d, d))
    Q = jax.vmap(lambda M: M @ M.T)(raw) + 0.1 * jnp.eye(d)[None]
    return A, Q


class TestUDLDecomposition:
    def test_shapes(self):
        T, d = 6, 3
        udl = udl_decomposition(_make_spd_block_tridiag(jr.key(0), T, d))
        assert isinstance(udl, UDLDecomposition)
        assert udl.U_sub.shape == (T - 1, d, d)
        assert udl.D_diag.shape == (T, d, d)
        assert udl.chol_D.shape == (T, d, d)
        assert udl.num_blocks == T
        assert udl.block_size == d

    def test_factors_reproduce_precision_densely(self):
        """Lambda == U D~ U^T with U unit upper block-bidiagonal."""
        T, d = 5, 2
        prec = _make_spd_block_tridiag(jr.key(1), T, d)
        udl = udl_decomposition(prec)

        n = T * d
        U = jnp.eye(n)
        D = jnp.zeros((n, n))
        for k in range(T):
            r = k * d
            D = D.at[r : r + d, r : r + d].set(udl.D_diag[k])
        for k in range(T - 1):
            r, c = k * d, (k + 1) * d
            U = U.at[r : r + d, c : c + d].set(udl.U_sub[k].T)
        assert jnp.allclose(U @ D @ U.T, prec.as_matrix(), atol=1e-8)

    def test_roundtrip_as_block_tridiag(self):
        """Lambda -> UDL -> Lambda is the identity to 1e-8."""
        T, d = 7, 3
        prec = _make_spd_block_tridiag(jr.key(2), T, d)
        rebuilt = udl_decomposition(prec).as_block_tridiag()
        assert jnp.allclose(rebuilt.diagonal, prec.diagonal, atol=1e-8)
        assert jnp.allclose(rebuilt.sub_diagonal, prec.sub_diagonal, atol=1e-8)

    def test_chol_D_is_cholesky_of_D(self):
        udl = udl_decomposition(_make_spd_block_tridiag(jr.key(3), 4, 2))
        rebuilt = jax.vmap(lambda L: L @ L.T)(udl.chol_D)
        assert jnp.allclose(rebuilt, udl.D_diag, atol=1e-10)

    def test_not_the_cholesky_factor(self):
        """UDL is a distinct factorisation: D~ is not the identity."""
        udl = udl_decomposition(_make_spd_block_tridiag(jr.key(4), 4, 2))
        assert not jnp.allclose(udl.D_diag, jnp.eye(2)[None])

    def test_solve_matches_dense(self):
        T, d = 20, 3
        prec = _make_spd_block_tridiag(jr.key(5), T, d)
        rhs = jr.normal(jr.key(6), (T * d,))
        x = udl_decomposition(prec).solve(rhs)
        expected = jnp.linalg.solve(prec.as_matrix(), rhs)
        assert jnp.allclose(x, expected, atol=1e-8)

    def test_logdet_matches_dense(self):
        prec = _make_spd_block_tridiag(jr.key(7), 12, 2)
        ld = udl_decomposition(prec).logdet()
        _, expected = jnp.linalg.slogdet(prec.as_matrix())
        assert jnp.allclose(ld, expected, atol=1e-8)

    def test_single_block(self):
        """T == 1 degenerates to a plain SPD block."""
        D = jnp.array([[[2.0, 0.5], [0.5, 3.0]]])
        prec = BlockTriDiag(D, jnp.zeros((0, 2, 2)))
        udl = udl_decomposition(prec)
        assert udl.U_sub.shape == (0, 2, 2)
        assert jnp.allclose(udl.D_diag, D)
        rhs = jnp.array([1.0, -1.0])
        assert jnp.allclose(udl.solve(rhs), jnp.linalg.solve(D[0], rhs))
        assert jnp.allclose(udl.logdet(), jnp.linalg.slogdet(D[0])[1])


class TestSSMExtraction:
    def test_extracts_chain_from_its_own_precision(self):
        """A_ssm = -U^T and Q = D~^{-1} recover the generating chain."""
        T, d = 6, 2
        A, Q = _make_ssm(jr.key(10), T, d)
        # ssm_to_naturals returns eta2 = -0.5 * Lambda.
        _, theta_prec = ssm_to_naturals(A, Q, jnp.zeros(d), Q[0])
        prec = BlockTriDiag(-2.0 * theta_prec.diagonal, -2.0 * theta_prec.sub_diagonal)

        A_hat, Q_hat, chol_Q_hat = udl_to_ssm_params(udl_decomposition(prec))
        assert jnp.allclose(A_hat, A, atol=1e-8)
        assert jnp.allclose(Q_hat, Q, atol=1e-8)
        assert jnp.allclose(jax.vmap(lambda L: L @ L.T)(chol_Q_hat), Q_hat, atol=1e-10)

    def test_from_ssm_params_matches_decomposition(self):
        """udl_from_ssm_params builds the same factors udl_decomposition finds."""
        T, d = 5, 3
        A, Q = _make_ssm(jr.key(11), T, d)
        direct = udl_from_ssm_params(A, Q)
        via_prec = udl_decomposition(direct.as_block_tridiag())
        assert jnp.allclose(direct.U_sub, via_prec.U_sub, atol=1e-8)
        assert jnp.allclose(direct.D_diag, via_prec.D_diag, atol=1e-8)
        assert jnp.allclose(direct.chol_D, via_prec.chol_D, atol=1e-8)

    def test_from_ssm_params_precision_matches_ssm_to_naturals(self):
        T, d = 5, 2
        A, Q = _make_ssm(jr.key(12), T, d)
        _, theta_prec = ssm_to_naturals(A, Q, jnp.zeros(d), Q[0])
        prec = udl_from_ssm_params(A, Q).as_block_tridiag()
        assert jnp.allclose(prec.as_matrix(), -2.0 * theta_prec.as_matrix(), atol=1e-8)

    def test_roundtrip_ssm_udl_ssm(self):
        T, d = 8, 2
        A, Q = _make_ssm(jr.key(13), T, d)
        A_hat, Q_hat, _ = udl_to_ssm_params(udl_from_ssm_params(A, Q))
        assert jnp.allclose(A_hat, A, atol=1e-10)
        assert jnp.allclose(Q_hat, Q, atol=1e-8)


class TestTransforms:
    @pytest.mark.slow
    def test_jit_grad_vmap(self):
        T, d = 6, 2
        prec = _make_spd_block_tridiag(jr.key(20), T, d)
        rhs = jr.normal(jr.key(21), (T * d,))

        def loss(diag, sub):
            udl = udl_decomposition(BlockTriDiag(diag, sub))
            return udl.logdet() + rhs @ udl.solve(rhs)

        value, (g_diag, g_sub) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(
            prec.diagonal, prec.sub_diagonal
        )

        def dense_loss(diag, sub):
            M = BlockTriDiag(diag, sub).as_matrix()
            return jnp.linalg.slogdet(M)[1] + rhs @ jnp.linalg.solve(M, rhs)

        expected, (e_diag, e_sub) = jax.value_and_grad(dense_loss, argnums=(0, 1))(
            prec.diagonal, prec.sub_diagonal
        )
        assert jnp.allclose(value, expected, atol=1e-8)
        # The dense reference differentiates through the full symmetric
        # matrix, so its sub-diagonal gradient is the sum of both
        # off-diagonal contributions; the banded path sees each once.
        assert jnp.allclose(g_diag, e_diag, atol=1e-6)
        assert jnp.allclose(g_sub, e_sub, atol=1e-6)

        # vmap over a batch of precisions.
        diags = jnp.stack([prec.diagonal, 2.0 * prec.diagonal])
        subs = jnp.stack([prec.sub_diagonal, prec.sub_diagonal])
        lds = jax.vmap(lambda D, S: udl_decomposition(BlockTriDiag(D, S)).logdet())(
            diags, subs
        )
        assert lds.shape == (2,)
