"""Tests for GaussianState and PropagationResult types."""

import jax.numpy as jnp
import lineax as lx

from gaussx._uncertain._types import GaussianState, PropagationResult


class TestGaussianState:
    def test_creation(self):
        mean = jnp.zeros(3)
        cov = lx.MatrixLinearOperator(jnp.eye(3))
        state = GaussianState(mean=mean, cov=cov)
        assert state.mean.shape == (3,)
        assert state.cov.in_size() == 3

    def test_is_pytree(self):
        """GaussianState should be a valid JAX PyTree."""
        import jax

        mean = jnp.ones(2)
        cov = lx.MatrixLinearOperator(jnp.eye(2))
        state = GaussianState(mean=mean, cov=cov)
        leaves = jax.tree.leaves(state)
        assert len(leaves) > 0


class TestPropagationResult:
    def test_with_cross_cov(self):
        state = GaussianState(
            mean=jnp.zeros(2),
            cov=lx.MatrixLinearOperator(jnp.eye(2)),
        )
        cross = jnp.ones((3, 2))
        result = PropagationResult(state=state, cross_cov=cross)
        assert result.cross_cov is not None
        assert result.cross_cov.shape == (3, 2)

    def test_without_cross_cov(self):
        state = GaussianState(
            mean=jnp.zeros(2),
            cov=lx.MatrixLinearOperator(jnp.eye(2)),
        )
        result = PropagationResult(state=state, cross_cov=None)
        assert result.cross_cov is None
