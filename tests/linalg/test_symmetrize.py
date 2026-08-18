"""Tests for symmetrize."""

import jax
import jax.numpy as jnp

from gaussx import symmetrize


class TestSymmetrize:
    def test_matches_definition(self, getkey):
        """Returns 0.5 * (A + A^T)."""
        N = 5
        A = jax.random.normal(getkey(), (N, N))
        assert jnp.allclose(symmetrize(A), 0.5 * (A + A.T))

    def test_result_is_symmetric(self, getkey):
        """Output is symmetric even when the input is not."""
        N = 6
        A = jax.random.normal(getkey(), (N, N))
        S = symmetrize(A)
        assert jnp.allclose(S, S.T)

    def test_idempotent_on_symmetric_input(self, getkey):
        """A symmetric matrix round-trips unchanged."""
        N = 4
        A = jax.random.normal(getkey(), (N, N))
        S = A + A.T
        assert jnp.allclose(symmetrize(S), S)

    def test_drops_only_the_skew_part(self, getkey):
        """symmetrize(A) + skew(A) reconstructs A exactly."""
        N = 5
        A = jax.random.normal(getkey(), (N, N))
        skew = 0.5 * (A - A.T)
        assert jnp.allclose(symmetrize(A) + skew, A)

    def test_batched_over_leading_axes(self, getkey):
        """Operates on the trailing two axes of a batched stack."""
        A = jax.random.normal(getkey(), (3, 2, 4, 4))
        S = symmetrize(A)
        assert S.shape == A.shape
        assert jnp.allclose(S, jnp.swapaxes(S, -1, -2))
        # Each batch element equals the unbatched result.
        assert jnp.allclose(S[1, 0], symmetrize(A[1, 0]))

    def test_jit(self, getkey):
        """Works under jit."""
        N = 4
        A = jax.random.normal(getkey(), (N, N))
        assert jnp.allclose(jax.jit(symmetrize)(A), symmetrize(A))

    def test_vmap(self, getkey):
        """Works under vmap."""
        A = jax.random.normal(getkey(), (3, 4, 4))
        assert jnp.allclose(jax.vmap(symmetrize)(A), symmetrize(A))
