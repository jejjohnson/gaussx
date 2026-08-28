"""Tests for the contour-integral square-root matrix-vector products."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest
from scipy.special import ellipj as scipy_ellipj, ellipk as scipy_ellipk

import gaussx
from gaussx._primitives._sqrt_matmul import _ellipj, _ellipk
from gaussx._testing import random_pd_operator


def _dense_power(operator: lx.AbstractLinearOperator, power: float):
    """Reference ``A^power`` from a dense symmetric eigendecomposition."""
    values, vectors = jnp.linalg.eigh(operator.as_matrix())
    return (vectors * (values**power)[None, :]) @ vectors.T


# ---------------------------------------------------------------------------
# Elliptic special functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modulus", [0.0, 0.3, 0.9, 0.999, 1.0 - 1e-6])
def test_ellipk_matches_scipy(modulus: float) -> None:
    assert jnp.allclose(_ellipk(jnp.asarray(modulus)), scipy_ellipk(modulus))


@pytest.mark.parametrize("modulus", [0.0, 0.3, 0.9, 0.999, 1.0 - 1e-6])
def test_ellipj_matches_scipy(modulus: float) -> None:
    argument = np.linspace(0.0, 0.999 * scipy_ellipk(modulus), 11)
    sn, cn, dn = _ellipj(jnp.asarray(argument), jnp.asarray(modulus))
    sn_ref, cn_ref, dn_ref, _ = scipy_ellipj(argument, modulus)

    assert jnp.allclose(sn, sn_ref, atol=1e-10)
    assert jnp.allclose(cn, cn_ref, atol=1e-10)
    assert jnp.allclose(dn, dn_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# Spectral bounds
# ---------------------------------------------------------------------------


def test_spectral_bounds_are_exact_for_diagonal() -> None:
    diagonal = jnp.array([0.5, 4.0, 2.0, 3.0])
    lam_min, lam_max = gaussx.estimate_spectral_bounds(
        lx.DiagonalLinearOperator(diagonal)
    )

    assert jnp.allclose(lam_min, 0.5)
    assert jnp.allclose(lam_max, 4.0)


def test_partial_lanczos_bounds_are_an_inner_bracket_before_widening() -> None:
    operator = random_pd_operator(jr.key(0), 30)
    eigenvalues = jnp.linalg.eigvalsh(operator.as_matrix())

    lam_min, lam_max = gaussx.estimate_spectral_bounds(
        operator, max_lanczos_iter=20, safety=1.0
    )

    # Ritz values interlace the spectrum, so the raw estimate is an inner
    # bracket; the largest eigenvalue is the one Lanczos nails first.
    assert lam_min >= eigenvalues[0]
    assert lam_max <= eigenvalues[-1]
    assert jnp.allclose(lam_max, eigenvalues[-1], rtol=1e-6)


def test_safety_widens_partial_lanczos_bounds_to_an_outer_bracket() -> None:
    operator = random_pd_operator(jr.key(0), 30)
    eigenvalues = jnp.linalg.eigvalsh(operator.as_matrix())

    lam_min, lam_max = gaussx.estimate_spectral_bounds(operator, max_lanczos_iter=20)

    assert lam_min < eigenvalues[0]
    assert lam_max > eigenvalues[-1]


def test_safety_is_not_applied_to_an_exact_spectrum() -> None:
    diagonal = jnp.array([0.5, 4.0, 2.0, 3.0])
    lam_min, lam_max = gaussx.estimate_spectral_bounds(
        lx.DiagonalLinearOperator(diagonal), safety=100.0
    )

    assert jnp.allclose(lam_min, 0.5)
    assert jnp.allclose(lam_max, 4.0)


def test_spectral_bounds_rejects_safety_below_one() -> None:
    with pytest.raises(ValueError, match="safety"):
        gaussx.estimate_spectral_bounds(
            lx.DiagonalLinearOperator(jnp.ones(3)), safety=0.5
        )


def test_spectral_bounds_rejects_rectangular_operator() -> None:
    operator = lx.MatrixLinearOperator(jnp.ones((3, 4)))
    with pytest.raises(ValueError, match="square operator"):
        gaussx.estimate_spectral_bounds(operator)


# ---------------------------------------------------------------------------
# Square-root products
# ---------------------------------------------------------------------------


def test_sqrt_inv_matmul_matches_dense_symmetric_root() -> None:
    operator = random_pd_operator(jr.key(1), 25)
    rhs = jr.normal(jr.key(2), (25, 3))

    result = gaussx.sqrt_inv_matmul(operator, rhs)

    assert jnp.allclose(result, _dense_power(operator, -0.5) @ rhs, atol=1e-8)


def test_sqrt_matmul_matches_dense_symmetric_root() -> None:
    operator = random_pd_operator(jr.key(3), 25)
    rhs = jr.normal(jr.key(4), (25, 3))

    result = gaussx.sqrt_matmul(operator, rhs)

    assert jnp.allclose(result, _dense_power(operator, 0.5) @ rhs, atol=1e-8)


def test_sqrt_round_trip_recovers_the_right_hand_side() -> None:
    operator = random_pd_operator(jr.key(5), 20)
    rhs = jr.normal(jr.key(6), (20, 2))

    whitened = gaussx.sqrt_inv_matmul(operator, rhs)

    assert jnp.allclose(gaussx.sqrt_matmul(operator, whitened), rhs, atol=1e-6)


def test_two_inverse_roots_equal_one_inverse() -> None:
    operator = random_pd_operator(jr.key(7), 20)
    rhs = jr.normal(jr.key(8), (20, 2))

    twice = gaussx.sqrt_inv_matmul(operator, gaussx.sqrt_inv_matmul(operator, rhs))
    solved = jax.vmap(
        lambda column: gaussx.solve(operator, column), in_axes=1, out_axes=1
    )(rhs)

    assert jnp.allclose(twice, solved, atol=1e-6)


def test_two_forward_roots_equal_one_matvec() -> None:
    operator = random_pd_operator(jr.key(9), 20)
    rhs = jr.normal(jr.key(10), (20, 2))

    twice = gaussx.sqrt_matmul(operator, gaussx.sqrt_matmul(operator, rhs))
    applied = jax.vmap(operator.mv, in_axes=1, out_axes=1)(rhs)

    assert jnp.allclose(twice, applied, atol=1e-6)


def test_batched_columns_are_independent() -> None:
    operator = random_pd_operator(jr.key(11), 15)
    rhs = jr.normal(jr.key(12), (15, 4))

    batched = gaussx.sqrt_inv_matmul(operator, rhs)
    columns = [gaussx.sqrt_inv_matmul(operator, rhs[:, i : i + 1]) for i in range(4)]

    assert jnp.allclose(batched, jnp.concatenate(columns, axis=1), atol=1e-10)


def test_diagonal_operator_takes_the_structural_solve_path() -> None:
    diagonal = jnp.linspace(0.5, 4.0, 6)
    rhs = jnp.ones((6, 1))

    result = gaussx.sqrt_inv_matmul(lx.DiagonalLinearOperator(diagonal), rhs)

    assert jnp.allclose(result[:, 0], diagonal**-0.5, atol=1e-10)


@pytest.mark.parametrize(
    ("num_quadrature", "tolerance"),
    # Hale-Higham-Trefethen convergence is geometric in the node count with a
    # rate set by log(kappa); at kappa = 1e4 these are the observed accuracies
    # with an order of magnitude of headroom.
    [(5, 1e-1), (10, 1e-3), (15, 1e-6), (25, 1e-10)],
)
def test_accuracy_improves_with_more_quadrature_nodes(
    num_quadrature: int, tolerance: float
) -> None:
    diagonal = jnp.geomspace(1e-4, 1.0, 50)
    rhs = jnp.ones((50, 1))

    result = gaussx.sqrt_inv_matmul(
        lx.DiagonalLinearOperator(diagonal), rhs, num_quadrature=num_quadrature
    )

    relative = jnp.abs(result[:, 0] - diagonal**-0.5) * diagonal**0.5
    assert jnp.max(relative) < tolerance


def test_explicit_spectral_bounds_skip_the_lanczos_estimate() -> None:
    diagonal = jnp.linspace(0.25, 9.0, 12)
    operator = lx.DiagonalLinearOperator(diagonal)
    rhs = jr.normal(jr.key(13), (12, 1))

    result = gaussx.sqrt_inv_matmul(operator, rhs, spectral_bounds=(0.25, 9.0))

    assert jnp.allclose(result[:, 0], rhs[:, 0] * diagonal**-0.5, atol=1e-10)


def test_jit_matches_eager() -> None:
    operator = random_pd_operator(jr.key(14), 15)
    rhs = jr.normal(jr.key(15), (15, 2))

    jitted = jax.jit(gaussx.sqrt_inv_matmul)(operator, rhs)

    assert jnp.allclose(jitted, gaussx.sqrt_inv_matmul(operator, rhs), atol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("primitive", [gaussx.sqrt_inv_matmul, gaussx.sqrt_matmul])
def test_grad_matches_finite_differences(primitive) -> None:
    matrix = random_pd_operator(jr.key(16), 12).as_matrix()
    rhs = jr.normal(jr.key(17), (12, 1))

    def objective(mat):
        operator = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
        return jnp.sum(primitive(operator, rhs) ** 2)

    direction = jr.normal(jr.key(18), (12, 12))
    direction = (direction + direction.T) / 2
    direction = direction / jnp.linalg.norm(direction)

    gradient = jax.grad(objective)(matrix)
    step = 1e-6
    finite = (
        objective(matrix + step * direction) - objective(matrix - step * direction)
    ) / (2 * step)

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(jnp.sum(gradient * direction), finite, rtol=1e-5)


def test_rejects_non_matrix_right_hand_side() -> None:
    operator = random_pd_operator(jr.key(19), 5)
    with pytest.raises(ValueError, match=r"shape \(N, C\)"):
        gaussx.sqrt_inv_matmul(operator, jnp.ones(5))


def test_rejects_mismatched_right_hand_side() -> None:
    operator = random_pd_operator(jr.key(20), 5)
    with pytest.raises(ValueError, match="rows but operator has size"):
        gaussx.sqrt_inv_matmul(operator, jnp.ones((4, 1)))


def test_rejects_non_positive_quadrature_count() -> None:
    operator = random_pd_operator(jr.key(21), 5)
    with pytest.raises(ValueError, match="num_quadrature"):
        gaussx.sqrt_inv_matmul(operator, jnp.ones((5, 1)), num_quadrature=0)
