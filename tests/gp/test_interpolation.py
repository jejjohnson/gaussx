"""Tests for conditional interpolation between time points."""

import jax
import jax.numpy as jnp
import pytest

from gaussx._gp._interpolation import conditional_interpolate, rts_interpolate
from gaussx._testing import key_sequence


class TestConditionalInterpolate:
    def test_shapes(self):
        """Output should be (d,) mean and (d, d) covariance."""
        nextkey = key_sequence(0)
        d = 3
        A_fwd = 0.9 * jnp.eye(d)
        Q_fwd = 0.1 * jnp.eye(d)
        A_bwd = 0.9 * jnp.eye(d)
        Q_bwd = 0.1 * jnp.eye(d)
        mu_prev = jax.random.normal(nextkey(), (d,))
        P_prev = 0.5 * jnp.eye(d)
        mu_next = jax.random.normal(nextkey(), (d,))
        P_next = 0.5 * jnp.eye(d)

        m, P = conditional_interpolate(
            A_fwd, Q_fwd, A_bwd, Q_bwd, mu_prev, P_prev, mu_next, P_next
        )
        assert m.shape == (d,)
        assert P.shape == (d, d)

    def test_uncertainty_reduction(self):
        """Fused estimate should have less uncertainty than forward-only."""
        d = 2
        A_fwd = jnp.eye(d)
        Q_fwd = 0.5 * jnp.eye(d)
        A_bwd = jnp.eye(d)
        Q_bwd = 0.5 * jnp.eye(d)
        mu_prev = jnp.zeros(d)
        P_prev = jnp.eye(d)
        mu_next = jnp.ones(d)
        P_next = jnp.eye(d)

        _, P_fused = conditional_interpolate(
            A_fwd, Q_fwd, A_bwd, Q_bwd, mu_prev, P_prev, mu_next, P_next
        )

        # Forward-only prediction covariance
        P_fwd = A_fwd @ P_prev @ A_fwd.T + Q_fwd

        # Fused should be tighter
        assert jnp.trace(P_fused) < jnp.trace(P_fwd)

    def test_symmetric_case(self):
        """Symmetric inputs should give mean at midpoint."""
        d = 2
        A = jnp.eye(d)
        Q = 0.1 * jnp.eye(d)
        mu_prev = jnp.array([0.0, 0.0])
        mu_next = jnp.array([2.0, 2.0])
        P = jnp.eye(d)

        m, _ = conditional_interpolate(A, Q, A, Q, mu_prev, P, mu_next, P)

        # With identical dynamics and noise, mean should be near midpoint
        assert jnp.allclose(m, jnp.array([1.0, 1.0]), atol=0.3)

    def test_psd_covariance(self):
        """Output covariance should be positive definite."""
        nextkey = key_sequence(0)
        d = 3
        A_fwd = 0.8 * jnp.eye(d) + 0.1 * jax.random.normal(nextkey(), (d, d))
        Q_fwd = jax.random.normal(nextkey(), (d, d))
        Q_fwd = Q_fwd @ Q_fwd.T + 0.1 * jnp.eye(d)
        A_bwd = 0.8 * jnp.eye(d) + 0.1 * jax.random.normal(nextkey(), (d, d))
        Q_bwd = jax.random.normal(nextkey(), (d, d))
        Q_bwd = Q_bwd @ Q_bwd.T + 0.1 * jnp.eye(d)
        mu_prev = jax.random.normal(nextkey(), (d,))
        P_prev = jax.random.normal(nextkey(), (d, d))
        P_prev = P_prev @ P_prev.T + 0.1 * jnp.eye(d)
        mu_next = jax.random.normal(nextkey(), (d,))
        P_next = jax.random.normal(nextkey(), (d, d))
        P_next = P_next @ P_next.T + 0.1 * jnp.eye(d)

        _, P_out = conditional_interpolate(
            A_fwd, Q_fwd, A_bwd, Q_bwd, mu_prev, P_prev, mu_next, P_next
        )
        eigvals = jnp.linalg.eigvalsh(P_out)
        assert jnp.all(eigvals > -1e-6)

    def test_finite(self):
        """All outputs should be finite."""
        nextkey = key_sequence(0)
        d = 2
        A = 0.9 * jnp.eye(d)
        Q = 0.2 * jnp.eye(d)
        mu = jax.random.normal(nextkey(), (d,))
        P = 0.5 * jnp.eye(d)

        m, P_out = conditional_interpolate(A, Q, A, Q, mu, P, mu, P)
        assert jnp.all(jnp.isfinite(m))
        assert jnp.all(jnp.isfinite(P_out))


# ---------------------------------------------------------------------------
# gh-288: dense references on a 3-state chain x1 -> x2 -> x3 (y at x1, x3)
# ---------------------------------------------------------------------------


def _chain_posterior(A1, Q1, A2, Q2, P1, H, R, y1, y3):
    """Exact p(x1, x2, x3 | y1, y3) for x1 ~ N(0, P1), y_i = H x_i + N(0, R)."""
    P2 = A1 @ P1 @ A1.T + Q1
    P3 = A2 @ P2 @ A2.T + Q2
    Sig = jnp.block(
        [
            [P1, P1 @ A1.T, P1 @ (A2 @ A1).T],
            [A1 @ P1, P2, P2 @ A2.T],
            [A2 @ A1 @ P1, A2 @ P2, P3],
        ]
    )
    d, p = A1.shape[0], H.shape[0]
    Z = jnp.zeros((p, d))
    Hj = jnp.block([[H, Z, Z], [Z, Z, H]])
    S = Hj @ Sig @ Hj.T + jnp.kron(jnp.eye(2), R)
    G = jnp.linalg.solve(S, Hj @ Sig).T
    return G @ jnp.concatenate([y1, y3]), Sig - G @ Hj @ Sig


def _filtered_x1(P1, H, R, y1):
    K1 = jnp.linalg.solve(H @ P1 @ H.T + R, H @ P1).T
    return K1 @ y1, P1 - K1 @ H @ P1


def _issue_chain():
    d = 2
    A = jnp.array([[1.0, 0.3], [0.0, 0.9]])
    Q, R = 0.2 * jnp.eye(d), 0.5 * jnp.eye(d)
    y1, y3 = jnp.array([0.5, -0.2]), jnp.array([1.0, 0.3])
    return A, Q, A, Q, jnp.eye(d), jnp.eye(d), R, y1, y3


def _matern32_chain():
    """Matern-3/2 SDE discretised at unequal steps (0.3, 0.7); y = f + noise."""
    lam, var = jnp.sqrt(3.0) / 0.8, 1.3
    F = jnp.array([[0.0, 1.0], [-(lam**2), -2.0 * lam]])
    Pinf = jnp.diag(jnp.array([var, lam**2 * var]))

    def disc(dt):
        A = jax.scipy.linalg.expm(F * dt)
        return A, Pinf - A @ Pinf @ A.T

    A1, Q1 = disc(0.3)
    A2, Q2 = disc(0.7)
    H, R = jnp.array([[1.0, 0.0]]), jnp.array([[0.05]])
    return A1, Q1, A2, Q2, Pinf, H, R, jnp.array([0.4]), jnp.array([-0.9])


@pytest.mark.x64_only(reason="1e-10 agreement with a float64 dense reference")
class TestInterpolationDenseReference:
    def test_two_filter_fusion_issue_chain(self):
        """Filtered x1 + backward-information message at x3 is exact."""
        A1, Q1, A2, Q2, P1, H, R, y1, y3 = _issue_chain()
        mu, Sig = _chain_posterior(A1, Q1, A2, Q2, P1, H, R, y1, y3)
        m_f, P_f = _filtered_x1(P1, H, R, y1)
        # With H = I the message p(y3 | x3) is N(y3; x3, R).
        m, P = conditional_interpolate(A1, Q1, A2, Q2, m_f, P_f, y3, R)
        assert jnp.allclose(m, mu[2:4], atol=1e-12)
        assert jnp.allclose(P, Sig[2:4, 2:4], atol=1e-12)

    def test_smoothed_inputs_are_not_valid_for_fusion(self):
        """Documented misuse: smoothed marginals at both ends double-count."""
        A1, Q1, A2, Q2, P1, H, R, y1, y3 = _issue_chain()
        mu, Sig = _chain_posterior(A1, Q1, A2, Q2, P1, H, R, y1, y3)
        _, P = conditional_interpolate(
            A1, Q1, A2, Q2, mu[:2], Sig[:2, :2], mu[4:], Sig[4:, 4:]
        )
        rel = jnp.linalg.norm(P - Sig[2:4, 2:4]) / jnp.linalg.norm(Sig[2:4, 2:4])
        assert rel > 0.1

    @pytest.mark.parametrize("chain", [_issue_chain, _matern32_chain])
    def test_rts_interpolate(self, chain):
        """Filtered x1 + smoothed x3 reproduces the dense p(x2 | y1, y3)."""
        A1, Q1, A2, Q2, P1, H, R, y1, y3 = chain()
        mu, Sig = _chain_posterior(A1, Q1, A2, Q2, P1, H, R, y1, y3)
        d = A1.shape[0]
        m_f, P_f = _filtered_x1(P1, H, R, y1)
        m, P = rts_interpolate(
            A1, Q1, A2, Q2, m_f, P_f, mu[2 * d :], Sig[2 * d :, 2 * d :]
        )
        assert jnp.allclose(m, mu[d : 2 * d], atol=1e-12)
        assert jnp.allclose(P, Sig[d : 2 * d, d : 2 * d], atol=1e-12)

    def test_two_filter_fusion_matern(self):
        """Fusion with unequal A/Q; message from a full-rank H = I observation."""
        A1, Q1, A2, Q2, P1, _, _, _, _ = _matern32_chain()
        H, R = jnp.eye(2), jnp.diag(jnp.array([0.05, 0.4]))
        y1, y3 = jnp.array([0.4, -0.1]), jnp.array([-0.9, 0.2])
        mu, Sig = _chain_posterior(A1, Q1, A2, Q2, P1, H, R, y1, y3)
        m_f, P_f = _filtered_x1(P1, H, R, y1)
        m, P = conditional_interpolate(A1, Q1, A2, Q2, m_f, P_f, y3, R)
        assert jnp.allclose(m, mu[2:4], atol=1e-12)
        assert jnp.allclose(P, Sig[2:4, 2:4], atol=1e-12)

    def test_jit_and_grad(self):
        A1, Q1, A2, Q2, P1, H, R, y1, y3 = _matern32_chain()
        mu, Sig = _chain_posterior(A1, Q1, A2, Q2, P1, H, R, y1, y3)
        m_f, P_f = _filtered_x1(P1, H, R, y1)
        args = (A1, Q1, A2, Q2, m_f, P_f, mu[4:], Sig[4:, 4:])
        m, P = jax.jit(rts_interpolate)(*args)
        m_ref, P_ref = rts_interpolate(*args)
        assert jnp.allclose(m, m_ref, atol=1e-12)
        assert jnp.allclose(P, P_ref, atol=1e-12)
        # The mean is affine in both input means (d m / d m_s = G).
        g = jax.grad(
            lambda ms, mf: rts_interpolate(*args[:4], mf, P_f, ms, args[7])[0][0],
            argnums=(0, 1),
        )
        g_s, g_f = g(mu[4:], m_f)
        assert jnp.all(jnp.isfinite(g_s)) and jnp.all(jnp.isfinite(g_f))
        e = jnp.eye(2)[0]
        fd = (rts_interpolate(*args[:4], m_f, P_f, mu[4:] + e, args[7])[0] - m_ref)[0]
        assert jnp.allclose(g_s[0], fd, atol=1e-10)
