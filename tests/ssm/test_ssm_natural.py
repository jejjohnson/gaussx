"""Tests for SSM <-> natural parameter transformations."""

import re

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._ssm._ssm_natural import naturals_to_ssm, ssm_to_naturals
from gaussx._testing import tree_allclose


def _make_ssm(getkey, N=4, d=2):
    """Build a simple time-invariant SSM."""
    A_single = 0.9 * jnp.eye(d) + 0.05 * jax.random.normal(getkey(), (d, d))
    A = jnp.tile(A_single[None], (N - 1, 1, 1))
    Q_single = jax.random.normal(getkey(), (d, d))
    Q_single = Q_single @ Q_single.T + 0.1 * jnp.eye(d)
    Q = jnp.tile(Q_single[None], (N, 1, 1))
    mu_0 = jax.random.normal(getkey(), (d,))
    P_0 = Q[0]
    return A, Q, mu_0, P_0


class TestSSMToNaturals:
    @pytest.mark.slow
    def test_shapes(self, getkey):
        """Output shapes should be correct."""
        N, d = 5, 3
        A, Q, mu_0, P_0 = _make_ssm(getkey, N, d)

        theta_linear, theta_prec = ssm_to_naturals(A, Q[1:], mu_0, P_0)
        assert theta_linear.shape == (N * d,)
        assert isinstance(theta_prec, BlockTriDiag)
        assert theta_prec._num_blocks == N
        assert theta_prec._block_size == d

    @pytest.mark.slow
    def test_precision_matches_dense(self, getkey):
        """Block-tridiagonal precision should match manually-built dense."""
        N, d = 3, 2
        A, Q, mu_0, P_0 = _make_ssm(getkey, N, d)

        _, theta_prec = ssm_to_naturals(A, Q[1:], mu_0, P_0)

        # Build dense precision manually
        Q_inv = jnp.linalg.inv(Q)
        Nd = N * d
        Lambda = jnp.zeros((Nd, Nd))
        for k in range(N):
            r = k * d
            block = Q_inv[k]
            if k < N - 1:
                block = block + A[k].T @ Q_inv[k + 1] @ A[k]
            Lambda = Lambda.at[r : r + d, r : r + d].set(block)
        for k in range(N - 1):
            r = (k + 1) * d
            c = k * d
            sub = -Q_inv[k + 1] @ A[k]
            Lambda = Lambda.at[r : r + d, c : c + d].set(sub)
            Lambda = Lambda.at[c : c + d, r : r + d].set(sub.T)

        # theta_prec is in eta2 convention: -0.5 * Lambda
        expected = -0.5 * Lambda
        assert jnp.allclose(theta_prec.as_matrix(), expected, atol=1e-6)

    def test_single_step_precision_is_initial_precision(self):
        # gh-359: for N = 1 the last-block write overwrote P_0^{-1} with
        # Q[0]^{-1}; with consistent inputs the block is P_0^{-1}.
        d = 2
        P_0 = jnp.array([[2.0, 0.3], [0.3, 1.0]])
        mu_0 = jnp.array([1.0, -1.0])
        A = jnp.zeros((0, d, d))
        for f in (ssm_to_naturals, jax.jit(ssm_to_naturals)):
            theta_linear, theta_prec = f(A, jnp.zeros((0, d, d)), mu_0, P_0)
            assert jnp.allclose(-2.0 * theta_prec.diagonal[0], jnp.linalg.inv(P_0))
            assert jnp.allclose(theta_linear, jnp.linalg.solve(P_0, mu_0))


class TestNaturalsToSSM:
    def test_roundtrip(self, getkey):
        """ssm -> naturals -> ssm should recover original parameters."""
        N, d = 4, 2
        A, Q, mu_0, P_0 = _make_ssm(getkey, N, d)

        theta_linear, theta_prec = ssm_to_naturals(A, Q[1:], mu_0, P_0)
        A_rec, Q_rec, mu_0_rec, P_0_rec = naturals_to_ssm(theta_linear, theta_prec)

        assert tree_allclose(A_rec, A, rtol=1e-4)
        assert tree_allclose(Q_rec, Q[1:], rtol=1e-4)
        assert tree_allclose(mu_0_rec, mu_0, rtol=1e-4)
        assert tree_allclose(P_0_rec, P_0, rtol=1e-4)

    def test_roundtrip_time_varying(self, getkey):
        """Roundtrip should work with time-varying A, Q."""
        N, d = 3, 2
        A = jax.random.normal(getkey(), (N - 1, d, d)) * 0.5
        Q_raw = jax.random.normal(getkey(), (N, d, d))
        Q = jax.vmap(lambda M: M @ M.T + 0.2 * jnp.eye(d))(Q_raw)
        mu_0 = jax.random.normal(getkey(), (d,))
        P_0 = Q[0]

        theta_linear, theta_prec = ssm_to_naturals(A, Q[1:], mu_0, P_0)
        A_rec, Q_rec, mu_0_rec, P_0_rec = naturals_to_ssm(theta_linear, theta_prec)

        assert tree_allclose(A_rec, A, rtol=1e-3)
        assert tree_allclose(Q_rec, Q[1:], rtol=1e-3)
        assert tree_allclose(P_0_rec, P_0, rtol=1e-3)
        assert tree_allclose(mu_0_rec, mu_0, rtol=1e-3)

    def test_shapes(self, getkey):
        """Recovered shapes should match original."""
        N, d = 5, 3
        A, Q, mu_0, P_0 = _make_ssm(getkey, N, d)

        theta_linear, theta_prec = ssm_to_naturals(A, Q[1:], mu_0, P_0)
        A_rec, Q_rec, mu_0_rec, P_0_rec = naturals_to_ssm(theta_linear, theta_prec)

        assert A_rec.shape == (N - 1, d, d)
        assert Q_rec.shape == (N - 1, d, d)
        assert mu_0_rec.shape == (d,)
        assert P_0_rec.shape == (d, d)


# gh-403: N block factorisations, not 2N - 1; P_0 factorised once.


def _lapack_factorisations(f, *args):
    hlo = jax.jit(f).lower(*args).compile().as_text()
    found = []
    for line in hlo.splitlines():
        match = re.search(r'custom_call_target="lapack_(\w+?)_ffi"', line)
        if match and match.group(1) in ("dpotrf", "dgetrf"):
            shape = re.search(r"=\s*\(?([a-z0-9]+\[[0-9,]*\])", line).group(1)
            found.append((match.group(1), shape))
    return found


def _pinned_ssm(N=6, d=2):
    # Pinned: the counts do not depend on the values.
    k1, k2 = jr.split(jr.key(0))
    A = jnp.tile((0.9 * jnp.eye(d) + 0.05 * jr.normal(k1, (d, d)))[None], (N - 1, 1, 1))
    L = jr.normal(k2, (d, d))
    Q = jnp.tile((L @ L.T + 0.1 * jnp.eye(d))[None], (N, 1, 1))
    return A, Q, jnp.ones(d), Q[0]


@pytest.mark.slow
@pytest.mark.skipif(jax.default_backend() != "cpu", reason="counts LAPACK calls")
def test_naturals_to_ssm_factorisation_count():
    A, Q, mu_0, P_0 = _pinned_ssm()
    theta_linear, theta_prec = ssm_to_naturals(A, Q[1:], mu_0, P_0)
    found = _lapack_factorisations(
        naturals_to_ssm,
        theta_linear,
        theta_prec,
    )
    # One d x d factorisation in the scan body (run N - 1 times) plus P_0.
    assert found == [("dpotrf", "f64[2,2]"), ("dpotrf", "f64[2,2]")]


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="counts LAPACK calls")
def test_ssm_to_naturals_factorises_p0_once():
    A, Q, mu_0, _ = _pinned_ssm()
    found = _lapack_factorisations(
        lambda A, Q: ssm_to_naturals(A, Q[1:], mu_0, Q[0]), A, Q
    )
    # P_0 once, plus one batched factorisation of the N - 1 transition noises.
    assert sorted(found) == [("dpotrf", "f64[2,2]"), ("dpotrf", "f64[5,2,2]")]
