"""Tests for SpectralFunction: f(A₁ ⊕ … ⊕ A_d) through factor eigenvectors."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._einx import einsum
from gaussx._operators._factored_eigen import factored_eigen


def path_laplacian(n):
    return lx.MatrixLinearOperator(
        gaussx.rw1_structure(n).as_matrix(), lx.symmetric_tag
    )


def spd(key, n):
    m = jr.normal(key, (n, n))
    gram = einsum(m, m, "i k, j k -> i j")
    return lx.MatrixLinearOperator(gram + n * jnp.eye(n), lx.symmetric_tag)


class Shifted(eqx.Module):
    """f(λ) = (c + λ)^p with a traced shift."""

    c: jax.Array
    p: int = eqx.field(static=True)

    def __call__(self, lam):
        return (self.c + lam) ** self.p


def dense_function(matrix, fn):
    eigenvalues, vectors = jnp.linalg.eigh(matrix)
    return einsum(vectors * fn(eigenvalues), vectors, "i k, j k -> i j")


@pytest.fixture
def operator():
    base = gaussx.KroneckerSum(path_laplacian(3), spd(jr.key(0), 4))
    return gaussx.SpectralFunction(base, Shifted(jnp.asarray(0.5), 2))


class TestSpectralFunction:
    @pytest.mark.slow
    def test_matches_dense(self, operator):
        dense = dense_function(operator.base.as_matrix(), operator.fn)
        b = jr.normal(jr.key(1), (12,))
        assert jnp.allclose(operator.as_matrix(), dense)
        assert jnp.allclose(operator.mv(b), dense @ b)
        assert jnp.allclose(gaussx.solve(operator, b), jnp.linalg.solve(dense, b))
        assert jnp.allclose(gaussx.logdet(operator), jnp.linalg.slogdet(dense)[1])
        assert jnp.allclose(gaussx.diag_inv(operator), jnp.diag(jnp.linalg.inv(dense)))
        assert jnp.allclose(gaussx.diag(operator), jnp.diag(dense))

    @pytest.mark.x64_only(reason="dense-reference tolerance below float32 round-off")
    def test_sqrt_matmul(self, operator):
        dense = dense_function(operator.base.as_matrix(), operator.fn)
        b = jr.normal(jr.key(1), (12,))
        root = operator.sqrt_matmul(operator.sqrt_matmul(b))
        assert jnp.allclose(root, dense @ b)
        inverse_root = operator.sqrt_matmul(
            operator.sqrt_matmul(b, inverse=True), inverse=True
        )
        assert jnp.allclose(inverse_root, jnp.linalg.solve(dense, b))

    @pytest.mark.slow
    def test_three_axes_and_pinv(self):
        base = gaussx.KroneckerSum(
            path_laplacian(2), gaussx.KroneckerSum(path_laplacian(3), path_laplacian(4))
        )
        op = gaussx.SpectralFunction(base, lambda lam: lam**2)  # singular: λ = 0
        expected = jnp.diag(jnp.linalg.pinv(op.as_matrix()))
        assert jnp.allclose(gaussx.diag_inv(op, pinv=True), expected, atol=1e-10)

    @pytest.mark.x64_only(reason="dense-reference tolerance below float32 round-off")
    def test_from_eigen_factorizations_matches_init(self, operator):
        factors = [
            gaussx.EigenDecomposition.from_matrix(path_laplacian(3), symmetric=True),
            gaussx.EigenDecomposition.from_matrix(operator.base.B, symmetric=True),
        ]
        built = gaussx.SpectralFunction.from_eigen_factorizations(factors, operator.fn)
        assert jnp.allclose(built.as_matrix(), operator.as_matrix())

    def test_jit_and_grad_through_fn_parameters(self, operator):
        def logdet(c):
            op = eqx.tree_at(lambda o: o.fn.c, operator, c)
            return gaussx.logdet(op)

        def dense(c):
            fn = Shifted(c, 2)
            return jnp.linalg.slogdet(dense_function(operator.base.as_matrix(), fn))[1]

        assert jnp.allclose(jax.jit(jax.grad(logdet))(0.5), jax.grad(dense)(0.5))
        assert jnp.allclose(
            jax.jit(gaussx.diag_inv)(operator), gaussx.diag_inv(operator)
        )

    def test_factored_eigen_branch(self, operator):
        fe = factored_eigen(operator)
        assert fe is not None
        assert jnp.allclose(fe.eigenvalues, operator.spectrum())
        tagged = lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)
        assert factored_eigen(tagged) is not None

    def test_lineax_predicates(self, operator):
        assert lx.is_symmetric(operator)
        assert not lx.is_diagonal(operator)
        assert not lx.is_positive_semidefinite(operator)
        assert operator.T is operator

    def test_non_symmetric_base_raises(self):
        with pytest.raises(ValueError, match="symmetric"):
            gaussx.SpectralFunction(
                lx.MatrixLinearOperator(jnp.triu(jnp.ones((3, 3)))), lambda lam: lam
            )


class _MatvecOnlySpectral(gaussx.SpectralFunction):
    """A spatial factor that must never be materialised."""

    def as_matrix(self):
        raise AssertionError("the spectral factor was materialised")


@pytest.mark.slow
def test_shifted_kronecker_keeps_the_spectral_basis():
    """``A ⊗ f(B) + cI``: exact solve, logdet and diag_inv, B never formed."""
    c = 0.7
    spatial_dense = gaussx.spde_precision_grid((3, 4), 0.8, 1.0, 2)
    spatial = _MatvecOnlySpectral(spatial_dense.base, spatial_dense.fn)
    temporal = spd(jr.key(0), 3)
    n = 3 * 12
    identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n,), jnp.float64))
    op = gaussx.Kronecker(temporal, spatial) + c * identity
    K = jnp.kron(temporal.as_matrix(), spatial_dense.as_matrix()) + c * jnp.eye(n)
    b = jr.normal(jr.key(1), (n,))
    assert jnp.allclose(gaussx.solve(op, b), jnp.linalg.solve(K, b))
    assert jnp.allclose(gaussx.logdet(op), jnp.linalg.slogdet(K)[1])
    assert jnp.allclose(gaussx.diag_inv(op), jnp.diag(jnp.linalg.inv(K)))
