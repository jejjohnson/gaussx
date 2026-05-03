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


def _build_dense_batched(kernel_fn, X):
    if X.ndim == 2:
        return jax.vmap(lambda x: jax.vmap(lambda y: kernel_fn(x, y))(X))(X)
    return jax.vmap(lambda X_batch: _build_dense_batched(kernel_fn, X_batch))(X)


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

    def test_batched_mv_and_as_matrix(self, getkey):
        """Batched inputs should preserve leading dimensions."""
        X = jax.random.normal(getkey(), (2, 3, 5, 2))
        v = jax.random.normal(getkey(), (2, 3, 5))
        noise_var = 0.1

        op = ImplicitKernelOperator(_rbf_kernel, X, noise_var=noise_var)
        K = _build_dense_batched(_rbf_kernel, X) + noise_var * jnp.eye(5)
        expected = jnp.matmul(K, v[..., None]).squeeze(-1)

        assert op.as_matrix().shape == (2, 3, 5, 5)
        assert jnp.allclose(op.as_matrix(), K, atol=1e-6)
        assert jnp.allclose(op.mv(v), expected, atol=1e-5)

    def test_batched_structure_metadata(self, getkey):
        """in_structure/out_structure must include leading batch axes
        so lineax helpers that allocate vectors from operator metadata
        produce shapes consistent with the batched ``mv`` contract."""
        X = jax.random.normal(getkey(), (2, 3, 5, 2))
        op = ImplicitKernelOperator(_rbf_kernel, X, noise_var=0.1)
        assert op.in_structure().shape == (2, 3, 5)
        assert op.out_structure().shape == (2, 3, 5)
        # Round-trip: a vector built from the declared structure must
        # match the leading-batch shape that ``mv`` enforces.
        v = jax.random.normal(getkey(), op.in_structure().shape)
        op.mv(v)  # should not raise the batch-shape ValueError

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


def test_unbatched_linear_solve_via_lineax_metadata(getkey):
    """Unbatched ImplicitKernelOperator integrates with lineax.linear_solve.

    The operator metadata (``in_structure`` / ``out_structure``) must
    line up with what lineax allocates for the RHS. This guards against
    metadata regressions in the unbatched path.
    """
    import contextlib

    import lineax as lx

    # Register the dispatches lineax CG needs for ImplicitKernelOperator
    # (mirrors the matrix-free GP notebook setup).
    with contextlib.suppress(Exception):
        lx.is_negative_semidefinite.register(ImplicitKernelOperator)(lambda _: False)
    with contextlib.suppress(Exception):
        lx.linearise.register(ImplicitKernelOperator)(lambda op: op)

    n, d = 12, 2
    X = jax.random.normal(getkey(), (n, d))
    op = ImplicitKernelOperator(
        _rbf_kernel,
        X,
        noise_var=0.5,
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
    rhs = jax.random.normal(getkey(), op.in_structure().shape)
    sol = lx.linear_solve(op, rhs, lx.CG(rtol=1e-6, atol=1e-6)).value
    # Verify the residual via op.mv
    residual = op.mv(sol) - rhs
    assert jnp.max(jnp.abs(residual)) < 1e-3
