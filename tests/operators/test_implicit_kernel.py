"""Tests for ImplicitKernelOperator."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._operators._implicit_kernel import ImplicitKernelOperator


def _rbf_kernel(x, y, lengthscale=1.0):
    diff = x - y
    return jnp.exp(-0.5 * jnp.sum(diff**2) / lengthscale**2)


def _asymmetric_kernel(x, y):
    return x[0] + 2.0 * y[0]


class TestImplicitKernelOperator:
    def test_matvec_matches_dense(self, getkey):
        """Implicit matvec should match explicit kernel matrix multiplication."""
        N, D = 10, 2
        X = jax.random.normal(getkey(), (N, D))
        v = jax.random.normal(getkey(), (N,))
        noise_var = 0.1

        op = ImplicitKernelOperator(_rbf_kernel, X, noise_var=noise_var)
        result = op.mv(v)

        # Dense reference
        K = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X))(X)
        K = K + noise_var * jnp.eye(N)
        expected = K @ v

        assert jnp.allclose(result, expected, atol=1e-5)

    def test_as_matrix(self, getkey):
        """as_matrix should produce the correct kernel matrix."""
        N, D = 8, 3
        X = jax.random.normal(getkey(), (N, D))
        noise_var = 0.05

        op = ImplicitKernelOperator(_rbf_kernel, X, noise_var=noise_var)
        K_mat = op.as_matrix()

        K_ref = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X))(X)
        K_ref = K_ref + noise_var * jnp.eye(N)
        assert jnp.allclose(K_mat, K_ref, atol=1e-6)

    def test_symmetric(self, getkey):
        """Kernel operator should be symmetric."""
        N, D = 6, 2
        X = jax.random.normal(getkey(), (N, D))
        op = ImplicitKernelOperator(_rbf_kernel, X, noise_var=0.0)
        K = op.as_matrix()
        assert jnp.allclose(K, K.T, atol=1e-6)

    def test_transpose_is_self(self, getkey):
        """Transpose should return the same operator."""
        N, D = 5, 2
        X = jax.random.normal(getkey(), (N, D))
        op = ImplicitKernelOperator(
            _rbf_kernel,
            X,
            noise_var=0.1,
            tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
        )
        assert op.transpose() is op

    def test_structure(self, getkey):
        """in/out structure should match N."""
        N, D = 7, 2
        X = jax.random.normal(getkey(), (N, D))
        op = ImplicitKernelOperator(_rbf_kernel, X, noise_var=0.0)
        assert op.in_structure().shape == (N,)
        assert op.out_structure().shape == (N,)

    def test_zero_noise(self, getkey):
        """With zero noise, should be pure kernel matrix."""
        N, D = 5, 2
        X = jax.random.normal(getkey(), (N, D))
        v = jax.random.normal(getkey(), (N,))

        op = ImplicitKernelOperator(_rbf_kernel, X, noise_var=0.0)
        result = op.mv(v)

        K = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X))(X)
        expected = K @ v
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_jit(self, getkey):
        """Should be JIT-compatible via eqx.filter_jit."""
        import equinox as eqx

        N, D = 5, 2
        X = jax.random.normal(getkey(), (N, D))
        v = jax.random.normal(getkey(), (N,))

        op = ImplicitKernelOperator(
            _rbf_kernel,
            X,
            noise_var=0.1,
            tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
        )
        r1 = op.mv(v)

        @eqx.filter_jit
        def jit_mv(operator, vec):
            return operator.mv(vec)

        r2 = jit_mv(op, v)
        assert jnp.allclose(r1, r2, atol=1e-10)

    def test_lineax_tag_queries(self, getkey):
        """Explicit structure tags should be surfaced through lineax."""
        N, D = 4, 2
        X = jax.random.normal(getkey(), (N, D))
        op = ImplicitKernelOperator(
            _rbf_kernel,
            X,
            noise_var=0.1,
            tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
        )
        assert lx.is_symmetric(op)
        assert lx.is_positive_semidefinite(op)
        assert not lx.is_diagonal(op)

    def test_no_structure_claims_without_tags(self, getkey):
        """Operators should not claim symmetry or PSD by default."""
        N, D = 4, 2
        X = jax.random.normal(getkey(), (N, D))
        op = ImplicitKernelOperator(_rbf_kernel, X, noise_var=0.1)
        assert not lx.is_symmetric(op)
        assert not lx.is_positive_semidefinite(op)

    def test_transpose_swaps_kernel_arguments(self, getkey):
        """Transpose should build the kernel with swapped arguments."""
        N, D = 3, 1
        X = jax.random.normal(getkey(), (N, D))
        op = ImplicitKernelOperator(_asymmetric_kernel, X)
        transposed = op.transpose()
        assert transposed is not op
        assert jnp.allclose(transposed.as_matrix(), op.as_matrix().T)
