"""Tests for Bayesian Learning Rule update primitives."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gaussx._einx import einsum
from gaussx._inference._blr import (
    blr_diag_update,
    blr_full_update,
    ggn_diagonal,
    hutchinson_hessian_diag,
)
from gaussx._testing import key_sequence, random_pd_matrix


class TestBLRDiagUpdate:
    def test_shapes(self, getkey):
        """Output shapes should match input."""
        d = 5
        nat1 = jax.random.normal(getkey(), (d,))
        nat2 = -0.5 * jnp.ones(d)
        grad = jax.random.normal(getkey(), (d,))
        hess = -jnp.ones(d)

        nat1_new, nat2_new = blr_diag_update(nat1, nat2, grad, hess, lr=0.1)
        assert nat1_new.shape == (d,)
        assert nat2_new.shape == (d,)

    def test_lr_zero_no_change(self, getkey):
        """lr=0 should return original parameters."""
        d = 3
        nat1 = jax.random.normal(getkey(), (d,))
        nat2 = -0.5 * jnp.ones(d)
        grad = jax.random.normal(getkey(), (d,))
        hess = -jnp.ones(d)

        nat1_new, nat2_new = blr_diag_update(nat1, nat2, grad, hess, lr=0.0)
        assert jnp.allclose(nat1_new, nat1)
        assert jnp.allclose(nat2_new, nat2)


class TestBLRFullUpdate:
    def test_shapes(self, getkey):
        """Output shapes should match input."""
        d = 4
        nat1 = jax.random.normal(getkey(), (d,))
        nat2 = -0.5 * jnp.eye(d)
        grad = jax.random.normal(getkey(), (d,))
        hess = -jnp.eye(d)

        nat1_new, nat2_new = blr_full_update(nat1, nat2, grad, hess, lr=0.1)
        assert nat1_new.shape == (d,)
        assert nat2_new.shape == (d, d)

    def test_lr_zero_no_change(self, getkey):
        """lr=0 should return original parameters."""
        d = 3
        nat1 = jax.random.normal(getkey(), (d,))
        nat2 = -0.5 * jnp.eye(d)
        grad = jax.random.normal(getkey(), (d,))
        hess = -jnp.eye(d)

        nat1_new, nat2_new = blr_full_update(nat1, nat2, grad, hess, lr=0.0)
        assert jnp.allclose(nat1_new, nat1)
        assert jnp.allclose(nat2_new, nat2)

    def test_matches_diag_for_diagonal_hessian(self, getkey):
        """With diagonal Hessian, should match diagonal update."""
        d = 3
        nat1 = jax.random.normal(getkey(), (d,))
        nat2_diag = -0.5 * jnp.abs(jax.random.normal(getkey(), (d,)))
        nat2_full = jnp.diag(nat2_diag)
        grad = jax.random.normal(getkey(), (d,))
        hess_diag = -jnp.abs(jax.random.normal(getkey(), (d,)))
        hess_full = jnp.diag(hess_diag)
        lr = 0.3

        n1_diag, n2_diag = blr_diag_update(nat1, nat2_diag, grad, hess_diag, lr)
        n1_full, n2_full = blr_full_update(nat1, nat2_full, grad, hess_full, lr)

        assert jnp.allclose(n1_diag, n1_full, atol=1e-6)
        assert jnp.allclose(jnp.diag(n2_full), n2_diag, atol=1e-6)


# gh-401: one lr = 1 BLR step on a Gaussian log-joint
# log p(x) = -1/2 (x - m)^T P (x - m) lands on the exact natural parameters
# eta1 = P m, eta2 = -P / 2, whatever the current iterate: the gradient at the
# current mean mu is -P (mu - m) and the Hessian is -P, so the target
# eta1 = grad - H mu = P m does not depend on mu. The tolerance is round-off
# on a d = 4 problem in float64, nothing statistical.
@pytest.mark.x64_only(reason="exactness to 1e-10 needs float64 round-off")
class TestBLRGaussianExactness:
    d = 4

    def _target(self):
        nextkey = key_sequence(0)
        P = random_pd_matrix(nextkey(), self.d, jitter=1.0)
        m = jr.normal(nextkey(), (self.d,))
        return P, m, nextkey

    @pytest.mark.parametrize("start", ["standard", "random"])
    def test_full_update_lands_on_the_posterior(self, start):
        P, m, nextkey = self._target()
        if start == "standard":
            nat1, nat2 = jnp.zeros(self.d), -0.5 * jnp.eye(self.d)
        else:
            nat1 = jr.normal(nextkey(), (self.d,))
            nat2 = -0.5 * random_pd_matrix(nextkey(), self.d, jitter=1.0)
        mu = jnp.linalg.solve(-2.0 * nat2, nat1)
        grad = -einsum(P, mu - m, "i j, j -> i")

        nat1_new, nat2_new = blr_full_update(nat1, nat2, grad, -P, lr=1.0)

        assert jnp.allclose(nat2_new, -0.5 * P, atol=1e-12, rtol=0.0)
        assert jnp.allclose(nat1_new, einsum(P, m, "i j, j -> i"), atol=1e-10, rtol=0.0)

    @pytest.mark.parametrize("start", ["standard", "random"])
    def test_diag_update_lands_on_the_posterior(self, start):
        _, m, nextkey = self._target()
        p = jnp.exp(jr.normal(nextkey(), (self.d,)))  # diagonal precision
        if start == "standard":
            nat1, nat2 = jnp.zeros(self.d), -0.5 * jnp.ones(self.d)
        else:
            nat1 = jr.normal(nextkey(), (self.d,))
            nat2 = -0.5 * jnp.exp(jr.normal(nextkey(), (self.d,)))
        mu = nat1 / (-2.0 * nat2)
        grad = -p * (mu - m)

        nat1_new, nat2_new = blr_diag_update(nat1, nat2, grad, -p, lr=1.0)

        assert jnp.allclose(nat2_new, -0.5 * p, atol=1e-12, rtol=0.0)
        assert jnp.allclose(nat1_new, p * m, atol=1e-12, rtol=0.0)


class TestGGNDiagonal:
    def test_matches_dense(self, getkey):
        """Should match diag(J^T J)."""
        N, d = 10, 5
        J = jax.random.normal(getkey(), (N, d))
        result = ggn_diagonal(J)
        expected = jnp.diag(J.T @ J)
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_nonnegative(self, getkey):
        """Result should always be non-negative."""
        J = jax.random.normal(getkey(), (8, 4))
        result = ggn_diagonal(J)
        assert jnp.all(result >= 0)

    def test_shape(self, getkey):
        """Should return (d,)."""
        J = jax.random.normal(getkey(), (6, 3))
        result = ggn_diagonal(J)
        assert result.shape == (3,)


class TestHutchinsonHessianDiag:
    def test_shape(self, getkey):
        """Should return (d,)."""
        d = 5
        H = jnp.eye(d)
        result = hutchinson_hessian_diag(lambda v: H @ v, getkey(), d, n_samples=1)
        assert result.shape == (d,)

    def test_identity_converges(self, getkey):
        """For identity Hessian, should converge to ones."""
        d = 4
        H = jnp.eye(d)
        result = hutchinson_hessian_diag(lambda v: H @ v, getkey(), d, n_samples=100)
        assert jnp.allclose(result, jnp.ones(d), atol=0.2)

    def test_diagonal_matrix(self, getkey):
        """For diagonal Hessian, should recover the diagonal."""
        d = 3
        diag_vals = jnp.array([1.0, 2.0, 3.0])
        H = jnp.diag(diag_vals)
        result = hutchinson_hessian_diag(lambda v: H @ v, getkey(), d, n_samples=200)
        assert jnp.allclose(result, diag_vals, atol=0.3)

    def test_respects_probe_dtype(self, getkey):
        """Probe dtype should be configurable instead of hardcoded."""
        d = 4
        H = jnp.eye(d, dtype=jnp.float32)
        result = hutchinson_hessian_diag(
            lambda v: H @ v,
            getkey(),
            d,
            n_samples=8,
            dtype=jnp.float32,
        )
        assert result.dtype == jnp.float32
