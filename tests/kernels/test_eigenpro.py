"""Tests for EigenPro spectral preconditioning helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import gaussx
from gaussx._kernels._eigenpro import (
    EigenProPreconditioner,
    eigenpro_correction,
    eigenpro_preconditioner,
    eigenpro_step_size,
)


def _rbf_kernel(x, y, lengthscale=1.0):
    diff = x - y
    return jnp.exp(-0.5 * jnp.sum(diff**2) / lengthscale**2)


def _kernel_matrix(X):
    return jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X))(X)


class TestEigenProPreconditioner:
    def test_subsample_eigendecomposition_is_positive(self):
        X = jnp.linspace(-1.0, 1.0, 10)[:, None]
        K = _kernel_matrix(X) + 1e-3 * jnp.eye(X.shape[0])
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)

        precond = eigenpro_preconditioner(
            op,
            subsample_size=6,
            n_components=3,
        )

        K_mm = K[jnp.ix_(precond.subsample_indices, precond.subsample_indices)] / 6
        eigvals = jnp.linalg.eigvalsh(K_mm)[::-1]

        assert precond.V.shape == (6, 3)
        assert precond.D.shape == (3,)
        assert jnp.all(eigvals[:3] > 0.0)
        assert jnp.all(eigvals[:-1] >= eigvals[1:])
        assert jnp.allclose(precond.max_eigenvalue, eigvals[0])
        assert jnp.all(precond.D > 0.0)
        assert precond.beta > 0.0
        assert jnp.allclose(precond.V.T @ precond.V, jnp.eye(3), atol=1e-6)

    def test_implicit_kernel_operator_path(self):
        X = jnp.linspace(-1.0, 1.0, 8)[:, None]
        op = gaussx.ImplicitKernelOperator(
            _rbf_kernel,
            X,
            noise_var=1e-3,
            tags=lx.positive_semidefinite_tag,
        )

        precond = gaussx.eigenpro_preconditioner(
            op,
            subsample_size=5,
            n_components=2,
        )

        assert precond.V.shape == (5, 2)
        assert jnp.all(precond.D > 0.0)
        assert precond.max_eigenvalue > 0.0
        assert precond.beta > 0.0

    def test_invalid_arguments_raise(self):
        op = lx.MatrixLinearOperator(jnp.eye(4), lx.positive_semidefinite_tag)

        with pytest.raises(ValueError, match="n_components"):
            eigenpro_preconditioner(op, subsample_size=3, n_components=3)


class TestEigenProStepSize:
    def test_step_size_positive_for_valid_batches(self):
        precond = EigenProPreconditioner(
            V=jnp.eye(3, 2),
            D=jnp.ones(2),
            subsample_indices=jnp.arange(3),
            max_eigenvalue=jnp.array(2.0),
            beta=jnp.array(5.0),
        )

        step_sizes = jnp.array([eigenpro_step_size(precond, b) for b in (1, 2, 8)])

        assert jnp.all(step_sizes > 0.0)

    def test_step_size_jit_compatible(self):
        precond = EigenProPreconditioner(
            V=jnp.eye(3, 2),
            D=jnp.ones(2),
            subsample_indices=jnp.arange(3),
            max_eigenvalue=jnp.array(2.0),
            beta=jnp.array(5.0),
        )

        step_size = jax.jit(lambda p: eigenpro_step_size(p, 4))(precond)

        assert step_size > 0.0


class TestEigenProCorrection:
    def test_correction_matches_spectral_formula(self):
        precond = EigenProPreconditioner(
            V=jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
            D=jnp.array([0.25, 0.5]),
            subsample_indices=jnp.arange(3),
            max_eigenvalue=jnp.array(2.0),
            beta=jnp.array(1.0),
        )
        K_batch_sub = jnp.array([[1.0, 2.0, 0.0], [0.5, -1.0, 1.0]])
        gradient = jnp.array([[2.0], [-4.0]])
        step_size = 0.1

        correction = eigenpro_correction(
            precond,
            K_batch_sub,
            gradient,
            step_size,
        )
        expected = step_size * (
            precond.V
            @ (precond.D[:, None] * (precond.V.T @ (K_batch_sub.T @ gradient)))
        )

        assert jnp.allclose(correction, expected)

    def test_correction_reduces_top_subspace_residual(self):
        precond = EigenProPreconditioner(
            V=jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
            D=jnp.array([0.25, 0.5]),
            subsample_indices=jnp.arange(3),
            max_eigenvalue=jnp.array(2.0),
            beta=jnp.array(1.0),
        )
        gradient = jnp.array([[2.0], [-4.0], [1.0]])
        correction = eigenpro_correction(
            precond,
            jnp.eye(3),
            gradient,
            step_size=1.0,
        )

        before = jnp.linalg.norm(precond.V.T @ gradient)
        after = jnp.linalg.norm(precond.V.T @ (gradient - correction))

        assert after < before

    def test_correction_jit_compatible(self):
        precond = EigenProPreconditioner(
            V=jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
            D=jnp.array([0.25, 0.5]),
            subsample_indices=jnp.arange(3),
            max_eigenvalue=jnp.array(2.0),
            beta=jnp.array(1.0),
        )
        K_batch_sub = jnp.eye(3)
        gradient = jnp.ones((3, 1))

        correction = jax.jit(eigenpro_correction)(
            precond,
            K_batch_sub,
            gradient,
            0.1,
        )

        assert correction.shape == (3, 1)
