"""Tests for AnalyticalPsiStatistics protocol and dispatch."""

import jax.numpy as jnp
import lineax as lx
import pytest

from gaussx import (
    AnalyticalPsiStatistics,
    GaussianState,
    compute_psi_statistics,
)


class MockAnalyticalKernel:
    """Mock kernel implementing AnalyticalPsiStatistics."""

    def psi0(self, state):
        return jnp.ones(state.mean.shape[0]) if state.mean.ndim > 0 else jnp.array(1.0)

    def psi1(self, state, X_train):
        N = 1 if state.mean.ndim == 0 else state.mean.shape[0]
        M = X_train.shape[0]
        return jnp.ones((N, M))

    def psi2(self, state, X_train):
        M = X_train.shape[0]
        return jnp.eye(M)


class TestAnalyticalPsiStatistics:
    def test_isinstance_check(self):
        """MockAnalyticalKernel passes isinstance check."""
        kernel = MockAnalyticalKernel()
        assert isinstance(kernel, AnalyticalPsiStatistics)

    def test_dispatch_to_analytical(self):
        """compute_psi_statistics dispatches to analytical methods."""
        kernel = MockAnalyticalKernel()
        mean = jnp.zeros(3)
        cov = lx.MatrixLinearOperator(jnp.eye(3))
        state = GaussianState(mean=mean, cov=cov)
        X_train = jnp.ones((5, 3))

        psi0, psi1, psi2 = compute_psi_statistics(kernel, state, X_train)
        assert psi0.shape == (3,) or psi0.shape == ()
        assert psi1.shape == (1, 5) or psi1.shape == (3, 5)
        assert psi2.shape == (5, 5)

    def test_error_without_integrator(self):
        """Raises ValueError for non-analytical kernel with no integrator."""

        class PlainKernel:
            pass

        kernel = PlainKernel()
        mean = jnp.zeros(3)
        cov = lx.MatrixLinearOperator(jnp.eye(3))
        state = GaussianState(mean=mean, cov=cov)
        X_train = jnp.ones((5, 3))

        with pytest.raises(ValueError, match="AnalyticalPsiStatistics"):
            compute_psi_statistics(kernel, state, X_train)

    def test_non_analytical_fails_isinstance(self):
        """Plain object does not pass isinstance check."""

        class NotAKernel:
            pass

        assert not isinstance(NotAKernel(), AnalyticalPsiStatistics)
