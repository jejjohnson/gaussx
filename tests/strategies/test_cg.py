"""Tests for CGSolver strategy."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._einx import einsum
from gaussx._linalg._symmetrize import symmetrize
from gaussx._operators import Kronecker
from gaussx._strategies import (
    AutoSolver,
    BBMMSolver,
    CGSolver,
    MINRESSolver,
    PreconditionedCGSolver,
    SLQLogdet,
)
from gaussx._strategies._tolerances import resolve_tolerance
from gaussx._testing import (
    dense_logdet,
    random_pd_matrix,
    random_pd_operator,
    tree_allclose,
)


# Stochastic logdet tests bound |est − exact| by the estimator's own standard
# error (gh-409, as gh-303 did for test_slq_logdet.py): the matrix and the
# probes are pinned, the strategy's logdet is checked to be that SLQ estimate,
# and lanczos_order >= n makes the quadrature exact, so there is no Lanczos
# bias term. Over 1000 random matrices |err| / SEM peaked at 3.4 (gh-303).
K_SEM = 5.0


@pytest.mark.slow
def test_solve_psd(getkey):
    cg = CGSolver(rtol=1e-8, atol=1e-8)
    mat = random_pd_matrix(getkey(), 5)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (5,))
    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(cg.solve(op, v), expected, rtol=1e-4)


def test_solve_diagonal(getkey):
    cg = CGSolver()
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(d), lx.positive_semidefinite_tag
    )
    v = jr.normal(getkey(), (4,))
    expected = v / d
    assert tree_allclose(cg.solve(op, v), expected, rtol=1e-4)


@pytest.mark.slow
def test_logdet_psd():
    """Stochastic logdet is within K_SEM standard errors of the exact one."""
    cg = CGSolver(num_probes=50, lanczos_order=20)
    op = random_pd_operator(jr.key(0), 20)
    key = jr.PRNGKey(42)
    est, sem = SLQLogdet(num_probes=50, lanczos_order=20).logdet_and_error(op, key=key)
    assert tree_allclose(cg.logdet(op, key=key), est)
    assert jnp.abs(est - dense_logdet(op)) <= K_SEM * sem


@pytest.mark.slow
def test_logdet_diagonal():
    """On a diagonal operator SLQ with sign probes is exact, not stochastic.

    Each probe gives zᵀ log(D) z = Σ log dᵢ when zᵢ² = 1, and full-order
    Lanczos is exact quadrature, so only round-off is left.
    """
    cg = CGSolver(num_probes=50, lanczos_order=10)
    d = jnp.abs(jr.normal(jr.key(0), (10,))) + 0.5
    op = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(d), lx.positive_semidefinite_tag
    )
    estimated = cg.logdet(op, key=jr.PRNGKey(123))
    assert tree_allclose(estimated, jnp.sum(jnp.log(d)), rtol=1e-8)


def test_filter_jit_solve(getkey):
    cg = CGSolver(rtol=1e-6, atol=1e-6)
    mat = random_pd_matrix(getkey(), 4)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (4,))

    @eqx.filter_jit
    def f(op, v):
        return cg.solve(op, v)

    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(f(op, v), expected, rtol=1e-4)


def test_solve_structured_operator():
    """CG drives a gaussx operator's matrix-free ``mv``.

    ``lineax.CG`` calls ``lineax.linearise``, which lineax registers only
    for its own operator classes — so this raised ``NotImplementedError``
    on every gaussx operator until the registration in
    ``gaussx._operators``. It is the route for structured covariances with
    no closed-form solve, e.g. a `SumOfKroneckers` of three or more terms.
    """
    factor = lx.MatrixLinearOperator(
        random_pd_matrix(jr.key(0), 3),
        (lx.symmetric_tag, lx.positive_semidefinite_tag),
    )
    op = Kronecker(factor, factor, tags=lx.positive_semidefinite_tag)
    v = jr.normal(jr.key(1), (9,))
    expected = jnp.linalg.solve(op.as_matrix(), v)
    assert tree_allclose(
        CGSolver(rtol=1e-10, atol=1e-10).solve(op, v), expected, rtol=1e-5
    )


def _float32_system(log_kappa, n=300):
    """The gh-327 system: eigenvalues logspace(0, log_kappa), explicitly float32."""
    q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (n, n), dtype=jnp.float32))
    ev = jnp.logspace(0, log_kappa, n, dtype=jnp.float32)
    A = symmetrize(einsum(q * ev, q, "i k, j k -> i j"))
    return A, jr.normal(jr.key(1), (n,), dtype=jnp.float32)


@pytest.mark.parametrize(
    "strategy",
    [
        CGSolver(),
        pytest.param(BBMMSolver(), marks=pytest.mark.slow),
        pytest.param(PreconditionedCGSolver(), marks=pytest.mark.slow),
        pytest.param(AutoSolver(size_threshold=100), marks=pytest.mark.slow),
    ],
    ids=["cg", "bbmm", "preconditioned_cg", "auto"],
)
def test_float32_default_tolerances_converge(strategy):
    # gh-327: at κ = 1e3 the old 1e-5 default ran out of CG steps in float32.
    # The resolved float32 default is 1e-3, so the relative residual is too.
    A, b = _float32_system(3)
    x = strategy.solve(lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag), b)
    assert x.dtype == jnp.float32
    assert jnp.linalg.norm(A @ x - b) / jnp.linalg.norm(b) <= 1e-3


def test_throw_false_returns_unconverged_under_jit():
    # gh-327: κ = 1e4 at 1e-5 cannot converge in float32; throw=False returns
    # the last iterate instead of raising, eagerly and under jit.
    A, b = _float32_system(4)
    op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
    strategy = CGSolver(rtol=1e-5, atol=1e-5, max_steps=50, throw=False)
    x = jax.jit(strategy.solve)(op, b)
    assert x.shape == b.shape and x.dtype == jnp.float32
    with pytest.raises(eqx.EquinoxRuntimeError):
        CGSolver(rtol=1e-5, atol=1e-5, max_steps=50).solve(op, b)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(jnp.float64, 1e-5), (jnp.float32, 1e-3), (jnp.float16, 1e-2)],
)
def test_default_tolerance_follows_the_dtype(dtype, expected):
    # float64 keeps the historical default, so its results are unchanged.
    assert resolve_tolerance(None, dtype, 1e-5) == expected
    assert resolve_tolerance(1e-7, dtype, 1e-5) == 1e-7


@pytest.mark.parametrize("cls", [CGSolver, MINRESSolver])
def test_float32_default_keeps_a_small_absolute_tolerance(cls):
    # Only rtol is relaxed in float32: with atol = 1e-3 a right-hand side of
    # norm 1e-4 counted as solved by the zero iterate.
    op = lx.MatrixLinearOperator(jnp.eye(4, dtype=jnp.float32), lx.symmetric_tag)
    if cls is CGSolver:
        op = lx.TaggedLinearOperator(op, lx.positive_semidefinite_tag)
    b = jnp.full(4, 5e-5, dtype=jnp.float32)
    assert tree_allclose(cls().solve(op, b), b)


@pytest.mark.parametrize(
    ("b", "scale"),
    [
        ([3.0, -1.5, 0.0], 3.0),
        ([1e-6, 0.0], 1e-6),  # tiny b: still relative, never the zero iterate
        ([0.0, 0.0], 1.0),  # zero b: the unscaled solve
        ([jnp.inf, 1.0], 1.0),  # non-finite b: the unscaled solve, as before
        ([jnp.nan, 1.0], 1.0),
    ],
)
def test_float32_default_atol_rescales_the_right_hand_side(b, scale):
    from gaussx._strategies._tolerances import rhs_scaling

    b32 = jnp.array(b, dtype=jnp.float32)
    atol, s = rhs_scaling(None, jnp.float32, 1e-5, b32)
    assert atol == pytest.approx(float(jnp.sqrt(jnp.finfo(jnp.float32).eps)))
    assert float(s) == pytest.approx(scale)
    # float64 and an explicit atol keep the unscaled solve and their constant.
    assert rhs_scaling(None, jnp.float64, 1e-5, b32.astype(jnp.float64)) == (
        1e-5,
        None,
    )
    assert rhs_scaling(2e-7, jnp.float32, 1e-5, b32) == (2e-7, None)


_FLOAT32_DEFAULTS = [CGSolver(), PreconditionedCGSolver(preconditioner_rank=2)]


@pytest.mark.parametrize("strategy", _FLOAT32_DEFAULTS, ids=["cg", "pcg"])
def test_float32_default_solves_a_tiny_right_hand_side(strategy):
    # gh-639: max|b| = 1e-6 is below the old 1e-5 floor, which accepted the
    # zero iterate; the rescaled solve is relative to b.
    A, _ = _float32_system(1, n=20)
    b = 1e-6 * jr.normal(jr.key(2), (20,), dtype=jnp.float32)
    x = strategy.solve(lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag), b)
    assert jnp.linalg.norm(A @ x - b) / jnp.linalg.norm(b) <= 1e-3


@pytest.mark.parametrize("strategy", _FLOAT32_DEFAULTS, ids=["cg", "pcg"])
def test_float32_default_gradients_match_the_dense_solve(strategy):
    # gh-639 review: no tangent may reach the lineax solver (the scale is
    # stop-gradient), and the adjoint solve must not inherit a tolerance
    # scaled from a large primal b (max|b| = 1e6 here).
    A, b = _float32_system(1, n=20)
    b = 1e6 * b

    def loss(c, v):
        op = lx.MatrixLinearOperator(c * A, lx.positive_semidefinite_tag)
        return jnp.sum(strategy.solve(op, v)) / 1e6

    def dense(c, v):
        return jnp.sum(jnp.linalg.solve(c * A, v)) / 1e6

    one = jnp.float32(1.0)
    got = jax.grad(loss, argnums=(0, 1))(one, b)
    want = jax.grad(dense, argnums=(0, 1))(one, b)
    assert tree_allclose(got, want, rtol=1e-2)


@pytest.mark.slow
def test_preconditioned_float32_default_converges_without_avx():
    # gh-639: with atol fixed at 1e-5, float32 PCG on the gh-327 system ran out
    # of steps when XLA was limited to SSE4.2 (as on some CI runners).
    import os
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import jax, jax.numpy as jnp, jax.random as jr, lineax as lx
        jax.config.update("jax_enable_x64", True)
        from gaussx._strategies import PreconditionedCGSolver
        from gaussx._einx import einsum
        from gaussx._linalg._symmetrize import symmetrize
        n = 300
        q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (n, n), dtype=jnp.float32))
        ev = jnp.logspace(0, 3, n, dtype=jnp.float32)
        A = symmetrize(einsum(q * ev, q, "i k, j k -> i j"))
        b = jr.normal(jr.key(1), (n,), dtype=jnp.float32)
        op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
        x = PreconditionedCGSolver().solve(op, b)
        r = jnp.linalg.norm(A @ x - b) / jnp.linalg.norm(b)
        assert r <= 1e-3, float(r)
        """
    )
    env = {**os.environ, "XLA_FLAGS": "--xla_cpu_max_isa=SSE4_2"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr[-2000:]


def test_default_tolerance_uses_the_narrowest_dtype():
    from gaussx._strategies._tolerances import operator_dtype

    # A float64-in, float32-out operator, or a float32 rhs: the float32
    # residual decides which default is reachable.
    op = lx.FunctionLinearOperator(
        lambda v: v.astype(jnp.float32), jax.ShapeDtypeStruct((3,), jnp.float64)
    )
    assert operator_dtype(op) == jnp.float32
    square = lx.MatrixLinearOperator(jnp.eye(3, dtype=jnp.float64))
    assert operator_dtype(square, jnp.ones(3, dtype=jnp.float32)) == jnp.float32
