"""Tests for the discrete Lyapunov solver."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from gaussx import discrete_lyapunov_solve
from gaussx._testing import random_pd_matrix


def _stable_matrix(key, n, scale: float = 0.5):
    """Random matrix with spectral radius < scale."""
    M = jr.normal(key, (n, n))
    spectral_radius = jnp.max(jnp.abs(jnp.linalg.eigvals(M)))
    return M * (scale / spectral_radius)


class TestDiscreteLyapunov:
    def test_satisfies_equation(self, getkey):
        N = 5
        G = _stable_matrix(getkey(), N)
        Q = random_pd_matrix(getkey(), N)
        P = discrete_lyapunov_solve(G, Q)
        residual = P - G @ P @ G.T - Q
        assert jnp.allclose(residual, jnp.zeros_like(residual), atol=1e-8)

    def test_matches_kronecker_form(self, getkey):
        """Solution must agree with the vectorized Kronecker formulation."""
        N = 4
        G = _stable_matrix(getkey(), N)
        Q = random_pd_matrix(getkey(), N)
        P = discrete_lyapunov_solve(G, Q)

        # Reference: solve (I - kron(G, G)) vec(P) = vec(Q).
        I_NN = jnp.eye(N * N)
        kron_term = jnp.kron(G, G)
        rhs = Q.reshape(-1)
        P_ref = jnp.linalg.solve(I_NN - kron_term, rhs).reshape(N, N)

        assert jnp.allclose(P, P_ref, atol=1e-6)

    def test_returns_real(self, getkey):
        """Even when G has complex eigenvalues, output is real."""
        N = 6
        G = _stable_matrix(getkey(), N)  # generic real matrix → typically complex eigs
        Q = random_pd_matrix(getkey(), N)
        P = discrete_lyapunov_solve(G, Q)
        assert P.dtype == jnp.float64 or P.dtype == jnp.float32
        assert not jnp.iscomplexobj(P)
