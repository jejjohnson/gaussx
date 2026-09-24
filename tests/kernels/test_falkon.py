"""Tests for the Falkon Nyström KRR recipe."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._testing import random_pd_matrix


def _rbf(x, z):
    return jnp.exp(-0.5 * jnp.sum((x - z) ** 2))


def _gram(a, b):
    return jax.vmap(lambda x: jax.vmap(lambda z: _rbf(x, z))(b))(a)


def _krr_problem(n: int = 100, m: int = 20, seed: int = 0):
    """RBF Nyström KRR data: X, Z = first m rows of X, K_nm, K_mm."""
    X = jr.normal(jr.key(seed), (n, 2))
    Z = X[:m]
    return X, Z, _gram(X, Z), _gram(Z, Z)


# ---------------------------------------------------------------------------
# Preconditioner
# ---------------------------------------------------------------------------


def test_factors_are_upper_triangular_choleskys() -> None:
    m, lam = 12, 0.05
    K_mm = random_pd_matrix(jr.key(1), m)

    pre = gaussx.falkon_preconditioner(K_mm, lam, jitter=0.0)

    assert jnp.allclose(pre.T, jnp.triu(pre.T))
    assert jnp.allclose(pre.A, jnp.triu(pre.A))
    assert jnp.allclose(pre.T.T @ pre.T, K_mm, atol=1e-10)
    expected = pre.T @ pre.T.T / m + lam * jnp.eye(m)
    assert jnp.allclose(pre.A.T @ pre.A, expected, atol=1e-10)


def test_preconditioner_inverts_the_nystrom_approximation() -> None:
    # P Pᵀ = n ((n/m) K² + λ n K)⁻¹: the defining property, and the reason
    # both factors must be upper triangular.
    n, m, lam = 500, 12, 0.05
    K_mm = random_pd_matrix(jr.key(2), m)

    pre = gaussx.falkon_preconditioner(K_mm, lam, jitter=0.0)
    P = pre.precondition(jnp.eye(m))

    expected = n * jnp.linalg.inv((n / m) * K_mm @ K_mm + lam * n * K_mm)
    assert jnp.allclose(P @ P.T, expected, rtol=1e-8, atol=1e-10)


def test_k_mm_cancels_in_the_preconditioned_system() -> None:
    # Pᵀ (K_nmᵀ K_nm + λ n K_mm) P = A⁻ᵀ [T⁻ᵀ K_nmᵀ K_nm T⁻¹ + λ n I] A⁻¹.
    n, lam = 100, 1e-3
    _, _, K_nm, K_mm = _krr_problem(n)
    pre = gaussx.falkon_preconditioner(K_mm, lam)
    m = K_mm.shape[0]
    K_mm_jittered = pre.T.T @ pre.T

    P = pre.precondition(jnp.eye(m))
    system = K_nm.T @ K_nm + lam * n * K_mm_jittered
    T_inv = jnp.linalg.inv(pre.T)
    A_inv = jnp.linalg.inv(pre.A)
    cancelled = (
        A_inv.T @ (T_inv.T @ K_nm.T @ K_nm @ T_inv + lam * n * jnp.eye(m)) @ A_inv
    )

    scale = jnp.max(jnp.abs(cancelled))
    assert jnp.max(jnp.abs(P.T @ system @ P - cancelled)) < 1e-8 * scale


def test_preconditioning_collapses_the_condition_number() -> None:
    # Measured: cond 6e8 -> 40 on this RBF problem.
    n, lam = 100, 1e-3
    _, _, K_nm, K_mm = _krr_problem(n)
    pre = gaussx.falkon_preconditioner(K_mm, lam)
    P = pre.precondition(jnp.eye(K_mm.shape[0]))
    system = K_nm.T @ K_nm + lam * n * K_mm

    assert jnp.linalg.cond(P.T @ system @ P) < 1e-4 * jnp.linalg.cond(system)
    assert jnp.linalg.cond(P.T @ system @ P) < 1e3


def test_transpose_application_is_the_transpose() -> None:
    m = 10
    pre = gaussx.falkon_preconditioner(random_pd_matrix(jr.key(3), m), 0.1)
    P = pre.precondition(jnp.eye(m))

    block = jr.normal(jr.key(4), (m, 3))
    assert jnp.allclose(pre.precondition_transpose(block), P.T @ block, atol=1e-10)
    assert jnp.allclose(pre.precondition(block[:, 0]), P @ block[:, 0], atol=1e-10)


def test_default_jitter_handles_duplicated_inducing_points() -> None:
    # Two identical inducing points make K_mm exactly singular.
    Z = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.5], [-0.3, 2.0]])
    pre = gaussx.falkon_preconditioner(_gram(Z, Z), 1e-3)

    assert jnp.all(jnp.isfinite(pre.T))
    assert jnp.all(jnp.isfinite(pre.A))


@pytest.mark.parametrize("amplitude", [0.0, 1e-35])
def test_default_jitter_is_positive_for_a_tiny_kernel(amplitude: float) -> None:
    # A linear kernel at all-zero inducing points has max|diag| = 0; at 1e-35
    # in float32, M * eps * max|diag| underflows to zero. Either way a jitter
    # scaled by max|diag| alone leaves the Cholesky of a singular matrix.
    K_mm = jnp.full((4, 4), amplitude, dtype=jnp.float32)
    pre = gaussx.falkon_preconditioner(K_mm, 1e-3)

    assert jnp.all(jnp.isfinite(pre.T))
    assert jnp.all(jnp.isfinite(pre.A))


def test_accepts_an_operator() -> None:
    K_mm = random_pd_matrix(jr.key(5), 6)
    from_array = gaussx.falkon_preconditioner(K_mm, 0.1)
    from_operator = gaussx.falkon_preconditioner(
        lx.MatrixLinearOperator(K_mm, lx.positive_semidefinite_tag), 0.1
    )

    assert jnp.allclose(from_array.T, from_operator.T)
    assert jnp.allclose(from_array.A, from_operator.A)


def test_preconditioner_is_jittable() -> None:
    K_mm = random_pd_matrix(jr.key(6), 6)
    eager = gaussx.falkon_preconditioner(K_mm, 0.1)
    jitted = jax.jit(gaussx.falkon_preconditioner)(K_mm, 0.1)

    assert jnp.allclose(eager.A, jitted.A)


def test_factors_share_one_promoted_dtype() -> None:
    # A float64 regularization with a float32 K_mm used to give a float32 T
    # and a float64 A.
    K_mm = random_pd_matrix(jr.key(19), 6).astype(jnp.float32)

    pre = gaussx.falkon_preconditioner(K_mm, jnp.asarray(1e-3, dtype=jnp.float64))

    assert pre.T.dtype == pre.A.dtype == jnp.float64


def test_rejects_a_non_square_k_mm() -> None:
    with pytest.raises(ValueError, match="square"):
        gaussx.falkon_preconditioner(jnp.ones((3, 4)), 0.1)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


def _direct_nystrom_krr(K_nm, K_mm, y, lam):
    """Reference α from the dense normal equations."""
    n = y.shape[0]
    return jnp.linalg.solve(K_nm.T @ K_nm + lam * n * K_mm, K_nm.T @ y)


def _targets(X):
    return jnp.sin(3.0 * X[:, 0]) + 0.1 * jr.normal(jr.key(7), (X.shape[0],))


def test_solve_matches_the_direct_nystrom_solve() -> None:
    n, m, lam = 200, 30, 1e-3
    X, _, K_nm, K_mm = _krr_problem(n, m)
    y = _targets(X)
    pre = gaussx.falkon_preconditioner(K_mm, lam)

    alpha = gaussx.falkon_solve(
        lx.MatrixLinearOperator(K_nm), y, pre, lam, max_iter=100, tol=1e-12
    )

    # The solver works with the jittered K_mm the preconditioner factored.
    expected = _direct_nystrom_krr(K_nm, pre.T.T @ pre.T, y, lam)
    assert jnp.allclose(K_nm @ alpha, K_nm @ expected, rtol=1e-6, atol=1e-8)


def test_solve_is_matrix_free_with_an_implicit_cross_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n, m, lam = 200, 30, 1e-3
    X, Z, K_nm, K_mm = _krr_problem(n, m)
    y = _targets(X)
    pre = gaussx.falkon_preconditioner(K_mm, lam)
    # Solve to convergence and compare predictions: the weights themselves
    # are ill-conditioned, so two roundings of the same solve differ there.
    options = dict(max_iter=200, tol=1e-12)
    dense = gaussx.falkon_solve(lx.MatrixLinearOperator(K_nm), y, pre, lam, **options)

    implicit = gaussx.ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=64)

    def refuse(self):
        raise AssertionError("falkon_solve materialised K_nm")

    monkeypatch.setattr(gaussx.ImplicitCrossKernelOperator, "as_matrix", refuse)
    alpha = gaussx.falkon_solve(implicit, y, pre, lam, **options)

    assert jnp.allclose(K_nm @ alpha, K_nm @ dense, rtol=1e-6, atol=1e-8)


# A 1000 x 50 problem solved twice (Falkon and plain CG): ~3 s.
@pytest.mark.slow
def test_a_small_budget_already_reaches_the_solution() -> None:
    # Falkon's point: the preconditioned system is well conditioned (here
    # cond 3.6e13 -> 38), so a small budget suffices where plain CG on the
    # normal equations is still far off. Measured at 20 iterations: 1.5e-5
    # relative prediction error against 0.3 for plain CG.
    n, m, lam, budget = 1000, 50, 1e-3, 20
    X, _, K_nm, K_mm = _krr_problem(n, m)
    y = _targets(X)
    pre = gaussx.falkon_preconditioner(K_mm, lam)
    K_mm_jittered = pre.T.T @ pre.T
    exact = K_nm @ _direct_nystrom_krr(K_nm, K_mm_jittered, y, lam)

    alpha = gaussx.falkon_solve(
        lx.MatrixLinearOperator(K_nm), y, pre, lam, max_iter=budget
    )
    system = lx.MatrixLinearOperator(
        K_nm.T @ K_nm + lam * n * K_mm_jittered, lx.positive_semidefinite_tag
    )
    plain = lx.linear_solve(
        system,
        K_nm.T @ y,
        lx.CG(rtol=1e-12, atol=0.0, max_steps=budget),
        throw=False,
    ).value

    def error(weights):
        return jnp.linalg.norm(K_nm @ weights - exact) / jnp.linalg.norm(exact)

    assert error(alpha) < 1e-4
    assert error(plain) > 100 * error(alpha)


def test_solve_is_jittable() -> None:
    n, m, lam = 100, 20, 1e-3
    X, _, K_nm, K_mm = _krr_problem(n, m)
    y = _targets(X)
    pre = gaussx.falkon_preconditioner(K_mm, lam)
    operator = lx.MatrixLinearOperator(K_nm)

    eager = gaussx.falkon_solve(operator, y, pre, lam)
    jitted = jax.jit(lambda y, pre: gaussx.falkon_solve(operator, y, pre, lam))(y, pre)

    assert jnp.allclose(eager, jitted, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    ("data_dtype", "regularization_dtype"),
    [(jnp.float32, jnp.float64), (jnp.float64, jnp.float32)],
)
def test_solve_promotes_mixed_dtypes(data_dtype, regularization_dtype) -> None:
    # A float64 regularization with float32 data used to make the CG
    # operator's input float32 and its output float64, which lineax rejects.
    n, m = 100, 20
    X = jr.normal(jr.key(20), (n, 2)).astype(data_dtype)
    Z = X[:m]
    lam = jnp.asarray(1e-3, dtype=regularization_dtype)
    pre = gaussx.falkon_preconditioner(_gram(Z, Z), lam)

    alpha = gaussx.falkon_solve(
        lx.MatrixLinearOperator(_gram(X, Z)), jnp.sin(X[:, 0]), pre, lam
    )

    assert alpha.dtype == jnp.float64
    assert jnp.all(jnp.isfinite(alpha))


def test_solve_promotes_a_wider_cross_kernel() -> None:
    # float32 preconditioner and targets, float64 K_nm: the cross-kernel
    # products are float64 and must not leak a wider dtype into the CG output.
    n, m = 100, 20
    X = jr.normal(jr.key(21), (n, 2))
    Z = X[:m]
    pre = gaussx.falkon_preconditioner(_gram(Z, Z).astype(jnp.float32), 1e-3)
    y = jnp.sin(X[:, 0]).astype(jnp.float32)

    alpha = gaussx.falkon_solve(lx.MatrixLinearOperator(_gram(X, Z)), y, pre, 1e-3)

    assert alpha.dtype == jnp.float64
    assert jnp.all(jnp.isfinite(alpha))


def test_solve_promotes_around_a_narrower_implicit_cross_kernel() -> None:
    # A float32 implicit K_nm with a float64 regularization: the operator's
    # transpose scan accumulates in float32 and used to reject the float64
    # vector the promoted solve handed it.
    n, m = 100, 20
    X = jr.normal(jr.key(22), (n, 2)).astype(jnp.float32)
    Z = X[:m]
    lam = jnp.asarray(1e-3, dtype=jnp.float64)
    pre = gaussx.falkon_preconditioner(_gram(Z, Z), lam)
    K_nm = gaussx.ImplicitCrossKernelOperator(_rbf, X, Z, 25)

    alpha = gaussx.falkon_solve(K_nm, jnp.sin(X[:, 0]), pre, lam)

    assert alpha.dtype == jnp.float64
    assert jnp.all(jnp.isfinite(alpha))


def test_solve_rejects_mismatched_shapes() -> None:
    _, _, K_nm, K_mm = _krr_problem(50, 10)
    pre = gaussx.falkon_preconditioner(K_mm, 1e-3)
    operator = lx.MatrixLinearOperator(K_nm)

    with pytest.raises(ValueError, match="K_nm must have shape"):
        gaussx.falkon_solve(operator, jnp.ones(40), pre, 1e-3)
    with pytest.raises(ValueError, match="vector"):
        gaussx.falkon_solve(operator, jnp.ones((50, 2)), pre, 1e-3)
    with pytest.raises(ValueError, match="max_iter"):
        gaussx.falkon_solve(operator, jnp.ones(50), pre, 1e-3, max_iter=0)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_matches_explicit_kernel_evaluation() -> None:
    Z = jr.normal(jr.key(8), (15, 2))
    X_test = jr.normal(jr.key(9), (40, 2))
    alpha = jr.normal(jr.key(10), (15,))

    predictions = gaussx.falkon_predict(_rbf, Z, alpha, X_test, batch_size=16)

    assert predictions.shape == (40,)
    assert jnp.allclose(predictions, _gram(X_test, Z) @ alpha, rtol=1e-10)


def test_predict_with_kernel_parameters() -> None:
    Z = jr.normal(jr.key(11), (10, 2))
    X_test = jr.normal(jr.key(12), (7, 2))
    alpha = jr.normal(jr.key(13), (10,))

    def scaled_rbf(params, x, z):
        return jnp.exp(-0.5 * jnp.sum((x - z) ** 2) / params["length"] ** 2)

    predictions = gaussx.falkon_predict(
        scaled_rbf, Z, alpha, X_test, params={"length": 2.0}
    )

    expected = _gram(X_test / 2.0, Z / 2.0) @ alpha
    assert jnp.allclose(predictions, expected, rtol=1e-10)


def test_end_to_end_regression_recovers_the_signal() -> None:
    # Fit sin(3x) from 2000 noisy points with 100 inducing points, entirely
    # matrix-free, and check held-out accuracy against the noiseless signal.
    n, m, lam = 2000, 100, 1e-5
    X = jr.uniform(jr.key(14), (n, 1), minval=-2.0, maxval=2.0)
    y = jnp.sin(3.0 * X[:, 0]) + 0.1 * jr.normal(jr.key(15), (n,))
    Z = X[:m]
    X_test = jnp.linspace(-1.8, 1.8, 50)[:, None]

    precond = gaussx.falkon_preconditioner(_gram(Z, Z), lam)
    K_nm = gaussx.ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=256)
    alpha = gaussx.falkon_solve(K_nm, y, precond, lam, max_iter=30)
    predictions = gaussx.falkon_predict(_rbf, Z, alpha, X_test)

    rmse = jnp.sqrt(jnp.mean((predictions - jnp.sin(3.0 * X_test[:, 0])) ** 2))
    assert rmse < 0.05


def test_predict_is_jittable() -> None:
    Z = jr.normal(jr.key(16), (8, 2))
    X_test = jr.normal(jr.key(17), (5, 2))
    alpha = jr.normal(jr.key(18), (8,))

    eager = gaussx.falkon_predict(_rbf, Z, alpha, X_test)
    jitted = jax.jit(lambda a: gaussx.falkon_predict(_rbf, Z, a, X_test))(alpha)

    assert jnp.allclose(eager, jitted)


def test_predict_rejects_mismatched_weights() -> None:
    Z = jnp.zeros((5, 2))
    with pytest.raises(ValueError, match="one weight per inducing point"):
        gaussx.falkon_predict(_rbf, Z, jnp.ones(4), jnp.zeros((3, 2)))
