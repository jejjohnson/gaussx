"""Gradients through preconditioned CG solves (gh-312).

A data-dependent preconditioner is built from the traced system operator, so
it carries a tangent; lineax's ``options`` used to reject it. The CG solution
does not depend on the preconditioner, so its gradient must match the dense
solve's.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx


_X = jnp.linspace(0.0, 1.0, 30)
_Y = jnp.sin(6.0 * _X)
# Tight CG tolerances so the iterative gradient matches dense to 1e-5.
_TOL = {"rtol": 1e-10, "atol": 1e-10, "max_steps": 500}


def _kernel(lengthscale):
    d = _X[:, None] - _X[None, :]
    return jnp.exp(-0.5 * d**2 / lengthscale**2) + 0.1 * jnp.eye(_X.shape[0])


def _op(lengthscale):
    return lx.MatrixLinearOperator(_kernel(lengthscale), lx.positive_semidefinite_tag)


def _strategies():
    return {
        "jacobi": lambda op: gaussx.CGSolver(
            preconditioner=gaussx.JacobiPreconditioner(), **_TOL
        ),
        "partial_cholesky": lambda op: gaussx.CGSolver(
            preconditioner=gaussx.PartialCholeskyPreconditioner(rank=5, shift=0.1),
            **_TOL,
        ),
        "preconditioned_cg": lambda op: gaussx.PreconditionedCGSolver(
            preconditioner_rank=5, shift=0.1, **_TOL
        ),
        # Built inside the differentiated function, from the traced operator.
        "nystrom": lambda op: gaussx.CGSolver(
            preconditioner=gaussx.NystromPreconditioner.from_operator(
                lx.MatrixLinearOperator(
                    op.as_matrix() - 0.1 * jnp.eye(op.in_size()),
                    lx.positive_semidefinite_tag,
                ),
                rank=5,
                shift=0.1,
                key=jr.key(0),
            ),
            **_TOL,
        ),
    }


@pytest.mark.parametrize("name", list(_strategies()))
@pytest.mark.parametrize("jit", [pytest.param(False, marks=pytest.mark.slow), True])
def test_grad_wrt_hyperparameter_matches_dense(name, jit):
    make = _strategies()[name]

    def loss(lengthscale, iterative):
        op = _op(lengthscale)
        strategy = make(op) if iterative else gaussx.DenseSolver()
        return _Y @ strategy.solve(op, _Y)

    grad = jax.grad(loss)
    if jit:
        grad = jax.jit(grad, static_argnums=1)
    expected = grad(0.3, False)
    assert jnp.allclose(grad(0.3, True), expected, rtol=1e-5)


@pytest.mark.parametrize("name", list(_strategies()))
def test_grad_wrt_rhs_matches_dense(name):
    op = _op(0.3)
    strategy = _strategies()[name](op)

    def loss(b, s):
        return jnp.sum(s.solve(op, b) ** 2)

    expected = jax.grad(loss)(_Y, gaussx.DenseSolver())
    assert jnp.allclose(jax.grad(loss)(_Y, strategy), expected, rtol=1e-5, atol=1e-8)


@pytest.mark.slow
def test_mvn_log_prob_grad_with_preconditioned_cg():
    def log_prob(lengthscale, solver):
        mvn = gaussx.MultivariateNormal(
            jnp.zeros(_X.shape[0]), _op(lengthscale), solver=solver
        )
        return mvn.log_prob(_Y)

    # The logdet is stochastic for CG, so compare against the same strategy's
    # logdet paired with a dense solve: only the solve's gradient is at issue.
    # The default lanczos_order=30 = n exhausts the Krylov space of this
    # low-rank-ish Gram; its gradient used to be NaN there (gh-520).
    pcg = gaussx.PreconditionedCGSolver(preconditioner_rank=5, shift=0.1, **_TOL)
    composed = gaussx.ComposedSolver(
        solve_strategy=gaussx.DenseSolver(), logdet_strategy=pcg
    )
    grad = jax.grad(log_prob)(0.3, pcg)
    assert jnp.isfinite(grad)
    assert jnp.allclose(grad, jax.grad(log_prob)(0.3, composed), rtol=1e-5)
