"""Tests for SketchAndPrecondLSMR and sketch_and_solve (G15, gh-484).

Each solve traces and lowers lineax's LSMR loop (~1-2 s even with a warm
compile cache), so the tests that run one are in the slow lane; the fast
lane keeps sketch-and-solve, the sketch sampling, logdet and validation.

Every key is pinned: the sketches and the test problems are incidental to
the properties checked. The sketch-and-solve test bounds the error by the
sketch's own measured embedding distortion, a deterministic consequence of
the embedding property, and checks that distortion against the
Davidson-Szarek bound; both are stated where they are used.
"""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx as gx
from gaussx._einx import einsum
from gaussx._strategies._sketch_precond import _sketch
from gaussx._testing import random_pd_operator


def _ill_conditioned(key, m, n, cond=1e4):
    """A tall Gaussian matrix with column scales spanning ``cond``."""
    scales = jnp.logspace(0.0, -math.log10(cond), n)
    return einsum(jr.normal(key, (m, n)), scales, "m n, n -> m n")


def _rel_err(x, ref):
    x, ref = np.asarray(x), np.asarray(ref)
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


# References in NumPy (float64, nothing to compile).
def _lstsq(A, b):
    return np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)[0]


def _ridge(A, b, damp):
    A, b = np.asarray(A), np.asarray(b)
    n = A.shape[1]
    gram = einsum(A, A, "m i, m j -> i j") + damp**2 * np.eye(n)
    return np.linalg.solve(gram, einsum(A, b, "m n, m -> n"))


_TOL = {"atol": 1e-12, "btol": 1e-12}


@pytest.mark.slow
@pytest.mark.parametrize("sketch", ["sparse_sign", "srht", "gaussian"])
def test_matches_lstsq(sketch):
    m, n = 600, 12
    A = _ill_conditioned(jr.key(0), m, n)
    b = jr.normal(jr.key(1), (m,))
    solver = gx.SketchAndPrecondLSMR(sketch=sketch, **_TOL)
    x, inner = eqx.filter_jit(solver._solve)(lx.MatrixLinearOperator(A), b)
    # LSMR's relative stopping tolerance 1e-12 on a system preconditioned to
    # kappa <= 3 leaves ~kappa(A) * 1e-12 = 1e-8 relative error in x.
    assert _rel_err(x, _lstsq(A, b)) < 1e-6
    # gamma = 4: kappa(AM) <~ 3, so (kappa - 1) / (kappa + 1) = 0.5 per step
    # reaches 1e-12 in ~40 steps at worst; unpreconditioned LSMR needs
    # hundreds on kappa(A) = 1e4.
    assert int(inner.stats["num_steps"]) <= 40


@pytest.mark.slow
def test_ridge_solution():
    m, n, damp = 400, 10, 0.3
    A = _ill_conditioned(jr.key(2), m, n)
    b = jr.normal(jr.key(3), (m,))
    solver = gx.SketchAndPrecondLSMR(damp=damp, **_TOL)
    x = eqx.filter_jit(solver.solve)(lx.MatrixLinearOperator(A), b)
    assert _rel_err(x, _ridge(A, b, damp)) < 1e-8


@pytest.mark.slow
def test_wide_operator_with_ridge():
    """With damp > 0 the augmented system [A; dI] is tall even when A is wide."""
    A = jr.normal(jr.key(4), (6, 10))
    b = jr.normal(jr.key(5), (6,))
    solver = gx.SketchAndPrecondLSMR(damp=0.5, **_TOL)
    x = eqx.filter_jit(solver.solve)(lx.MatrixLinearOperator(A), b)
    assert _rel_err(x, _ridge(A, b, 0.5)) < 1e-8


@pytest.mark.slow
def test_matrix_free_operator_and_jit():
    m, n = 300, 8
    A = _ill_conditioned(jr.key(6), m, n, cond=1e2)
    b = jr.normal(jr.key(7), (m,))
    op = lx.FunctionLinearOperator(
        lambda v: einsum(A, v, "m n, n -> m"), jax.ShapeDtypeStruct((n,), A.dtype)
    )
    solver = gx.SketchAndPrecondLSMR(**_TOL)
    x = eqx.filter_jit(solver.solve)(op, b)
    assert _rel_err(x, _lstsq(A, b)) < 1e-8


@pytest.mark.slow
def test_grad_matches_lstsq():
    m, n = 200, 5
    A = _ill_conditioned(jr.key(8), m, n, cond=1e2)
    b = jr.normal(jr.key(9), (m,))
    solver = gx.SketchAndPrecondLSMR(**_TOL)

    def loss(A, b):
        return jnp.sum(solver.solve(lx.MatrixLinearOperator(A), b) ** 2)

    def ref(A, b):
        return jnp.sum(jnp.linalg.lstsq(A, b)[0] ** 2)

    for got, want in zip(
        jax.jit(jax.grad(loss, argnums=(0, 1)))(A, b),
        jax.grad(ref, argnums=(0, 1))(A, b),
        strict=True,
    ):
        assert _rel_err(got, want) < 1e-6


def test_sketch_and_solve_error_is_order_eps():
    m, n, d = 1000, 10, 200
    A = _ill_conditioned(jr.key(10), m, n, cond=1e3)
    b = einsum(A, jnp.ones(n), "m n, n -> m") + jr.normal(jr.key(11), (m,))
    S = gx.GaussianSketch.sample(jr.key(12), d, m)
    x_hat = eqx.filter_jit(gx.sketch_and_solve)(lx.MatrixLinearOperator(A), b, sketch=S)
    x_star = _lstsq(A, b)

    # The distortion of S on range([A b]), measured: S is an eps-embedding
    # for it with eps = max |sigma_i(S U) - 1|, U an orthonormal basis.
    A, b = np.asarray(A), np.asarray(b)
    U, _ = np.linalg.qr(einx.id("m n, m -> m (n + 1)", A, b))
    sv = np.linalg.svd(np.asarray(S.apply(jnp.asarray(U))), compute_uv=False)
    eps = float(np.max(np.abs(sv - 1)))
    # Davidson-Szarek: the singular values of a d x k Gaussian G / sqrt(d)
    # lie in 1 -+ (sqrt(k/d) + t/sqrt(d)) w.p. >= 1 - 2 exp(-t^2/2); t = 3.
    assert eps <= math.sqrt((n + 1) / d) + 3 / math.sqrt(d)

    def residual(x):
        return float(np.linalg.norm(einsum(A, np.asarray(x), "m n, n -> m") - b))

    # ||A x_hat - b|| <= (1 + eps) / (1 - eps) ||A x* - b||: deterministic
    # given the measured eps (the three-line proof in the docstring).
    assert residual(x_hat) <= (1 + eps) / (1 - eps) * residual(x_star) * (1 + 1e-12)
    assert residual(x_hat) >= residual(x_star) * (1 - 1e-12)


def test_sketch_and_solve_exact_for_orthogonal_sketch():
    """With S orthogonal (d = m), the sketched ridge problem is the ridge problem."""
    m, n, damp = 40, 6, 0.7
    A = jr.normal(jr.key(13), (m, n))
    b = jr.normal(jr.key(14), (m,))
    S = gx.OrthonormalSketch.sample(jr.key(15), m, m)
    x = gx.sketch_and_solve(lx.MatrixLinearOperator(A), b, sketch=S, damp=damp)
    assert _rel_err(x, _ridge(A, b, damp)) < 1e-10


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("sparse_sign", gx.SparseSignSketch),
        ("srht", gx.SRHTSketch),
        ("gaussian", gx.GaussianSketch),
    ],
)
def test_samples_the_named_sketch(name, cls):
    solver = gx.SketchAndPrecondLSMR(sketch=name, sampling_factor=2.5, nnz=3)
    S = solver._sample_sketch(m=100, n=8)
    assert isinstance(S, cls)
    assert (S.out_size, S.in_size) == (20, 100)  # d = ceil(2.5 * 8)
    # d is capped at m, and the sketch is a deterministic function of seed.
    assert solver._sample_sketch(m=12, n=8).out_size == 12
    assert eqx.tree_equal(S, solver._sample_sketch(m=100, n=8))


def test_logdet_delegates_to_lsmr_solver():
    op = random_pd_operator(jr.key(16), 8)
    got = gx.SketchAndPrecondLSMR(seed=3).logdet(op)
    assert jnp.allclose(got, gx.LSMRSolver(seed=3).logdet(op))


@pytest.mark.parametrize(
    "sample",
    [
        lambda key, d, m: gx.SparseSignSketch.sample(key, d, m, nnz=3),
        lambda key, d, m: gx.SRHTSketch.sample(key, d, m),
    ],
    ids=["sparse_sign", "srht"],
)
def test_matrix_free_sketch_matches_dense(sample):
    """The batched matrix-free sketch never forms S^T, but equals S A."""
    m, n, d = 100, 5, 40  # d > _ROW_BATCH: more than one batch
    A = jr.normal(jr.key(19), (m, n))
    S = sample(jr.key(20), d, m)
    op = lx.FunctionLinearOperator(
        lambda v: einsum(A, v, "m n, n -> m"), jax.ShapeDtypeStruct((n,), A.dtype)
    )
    got = eqx.filter_jit(_sketch)(op, S)
    assert jnp.allclose(got, S.apply(A), atol=1e-10)


def test_rank_deficient_undamped_raises_and_ridge_does_not():
    m, n = 200, 6
    A = jr.normal(jr.key(21), (m, n))
    A = A.at[:, -1].set(A[:, 0])  # a duplicated column
    b = jr.normal(jr.key(22), (m,))
    op = lx.MatrixLinearOperator(A)
    S = gx.GaussianSketch.sample(jr.key(23), 40, m)
    with pytest.raises(eqx.EquinoxRuntimeError, match="rank-deficient"):
        gx.sketch_and_solve(op, b, sketch=S)
    with pytest.raises(eqx.EquinoxRuntimeError, match="rank-deficient"):
        gx.SketchAndPrecondLSMR().solve(op, b)
    assert jnp.all(jnp.isfinite(gx.sketch_and_solve(op, b, sketch=S, damp=0.1)))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sketch": "dense"}, "sketch must be"),
        ({"sampling_factor": 0.5}, "sampling_factor >= 1"),
        ({"damp": -1.0}, "damp >= 0"),
    ],
)
def test_rejects_bad_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        gx.SketchAndPrecondLSMR(**kwargs)


def test_rejects_wide_undamped_and_negative_damp():
    A = lx.MatrixLinearOperator(jnp.ones((3, 5)))
    with pytest.raises(ValueError, match="tall or square"):
        gx.SketchAndPrecondLSMR().solve(A, jnp.ones(3))
    S = gx.GaussianSketch.sample(jr.key(0), 3, 3)
    with pytest.raises(ValueError, match="damp >= 0"):
        gx.sketch_and_solve(A, jnp.ones(3), sketch=S, damp=-1.0)


@pytest.mark.slow
def test_iteration_count_flat_in_m():
    """gh-484 acceptance: <= 30 LSMR steps, flat across m in {1e4, 1e5}."""
    n = 50
    steps = []
    for m in (10_000, 100_000):
        A = _ill_conditioned(jr.key(17), m, n)
        b = jr.normal(jr.key(18), (m,))
        solver = gx.SketchAndPrecondLSMR(atol=1e-10, btol=1e-10)
        x, inner = eqx.filter_jit(solver._solve)(lx.MatrixLinearOperator(A), b)
        assert _rel_err(x, _lstsq(A, b)) < 1e-5
        steps.append(int(inner.stats["num_steps"]))
    # kappa(AM) depends on d / n only, so the count does not grow with m.
    assert max(steps) <= 30, steps
    assert abs(steps[1] - steps[0]) <= 5, steps
