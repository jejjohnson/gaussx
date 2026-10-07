"""Tests for ``diag_inv(method="xdiag")`` (G16, gh-485).

Every key is pinned. The accuracy tests bound XDiag's error by its own
leave-one-out standard error (the estimator's sampling distribution), and
the comparison with Hutchinson uses the gap measured over many seeds; each
bound says where it came from. Runs in the float32 lane too: the
estimator's error (~0.1 relative) dwarfs float32 rounding.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import matfree.stochtrace
import numpy as np
import pytest

import gaussx
from gaussx import SparseOperator, diag_inv
from gaussx._einx import einsum, reduce
from gaussx._testing import default_tolerances


def _grid_precision(side: int, shift: float = 0.01) -> SparseOperator:
    """4-neighbour ``side x side`` grid Laplacian plus ``shift * I``.

    A small shift leaves the covariance with a few large-variance smooth
    modes, the case XDiag's low-rank part is for.
    """
    i = np.arange(side * side)
    right = i[i % side != side - 1]
    down = i[i < (side - 1) * side]
    senders = np.r_[right + 1, down + side]
    receivers = np.r_[right, down]
    n = side * side
    deg = np.bincount(senders, minlength=n) + np.bincount(receivers, minlength=n)
    diagonal = np.arange(n)
    rows, cols = np.r_[diagonal, senders], np.r_[diagonal, receivers]
    values = np.r_[deg + shift, -np.ones(senders.size)]
    dtype = jnp.result_type(float)  # the active default float (lane-aware)
    return SparseOperator.from_coo(
        rows,
        cols,
        jnp.asarray(values, dtype=dtype),
        (n, n),
        symmetric=True,
        tags=lx.positive_semidefinite_tag,  # CG needs it
    )


def _rel_err(x, ref):
    return float(jnp.linalg.norm(x - ref) / jnp.linalg.norm(ref))


def _loo_sem(Q, k, key):
    """Leave-one-out standard error of XDiag on dense ``Q`` with this key."""
    estimate = matfree.stochtrace.estimator_leave_one_out_mean_and_sem(
        matfree.stochtrace.leave_one_out_xdiag(),
        matfree.stochtrace.sampler_signs(jnp.zeros(Q.shape[0], Q.dtype), num=k),
    )
    return estimate(lambda v: jnp.linalg.solve(Q, v), key)


@pytest.mark.slow
def test_matches_takahashi_on_a_mesh_that_factors():
    """gh-485: agrees with the G4 selected inverse, within its own SEM."""
    op = _grid_precision(12)
    exact = diag_inv(op, solver=gaussx.SparseCholeskySolver())  # Takahashi
    k, key = 20, jr.key(0)
    solver = gaussx.PreconditionedCGSolver(preconditioner=gaussx.JacobiPreconditioner())
    estimate = jax.jit(
        lambda op: diag_inv(op, method="xdiag", num_probes=k, key=key, solver=solver)
    )(op)
    _, sem = _loo_sem(op.as_matrix(), k, key)
    # Over 100 seeds on this kind of grid, ||err|| / ||SEM|| peaked at 1.36
    # (mean 1.2); 3 leaves room for the CG tolerance (1e-3 in float32).
    assert jnp.linalg.norm(estimate - exact) <= 3 * jnp.linalg.norm(sem)


def test_beats_hutchinson_at_equal_solve_counts():
    """gh-485: XDiag with k probes (2k solves) beats Hutchinson with 2k."""
    Q = _grid_precision(15).as_matrix()
    op = lx.MatrixLinearOperator(Q, lx.positive_semidefinite_tag)
    exact = jnp.diag(jnp.linalg.inv(Q))
    k, key = 20, jr.key(1)
    xdiag = diag_inv(op, method="xdiag", num_probes=k, key=key)
    hutchinson = diag_inv(op, method="hutchinson", num_probes=2 * k, key=key)
    # On a 20 x 20 version of this grid, over 20 seeds: XDiag 0.18 +- 0.007,
    # Hutchinson 0.88 +- 0.09 relative error; a factor 2 is many sigmas.
    assert _rel_err(xdiag, exact) < 0.5 * _rel_err(hutchinson, exact)


@pytest.mark.parametrize("symmetric", [True, False])
def test_exact_when_twice_the_probes_reach_n(symmetric):
    """With 2k >= N, XDiag returns the exact diagonal from N solves."""
    n = 10
    A = jr.normal(jr.key(2), (n, n))
    gram = einsum(A, A, "i k, j k -> i j")
    A = gram + n * jnp.eye(n) if symmetric else A + n * jnp.eye(n)
    op = lx.MatrixLinearOperator(A, lx.symmetric_tag if symmetric else ())
    rtol, atol = default_tolerances(A)
    assert jnp.allclose(
        diag_inv(op, method="xdiag", num_probes=5),
        jnp.diag(jnp.linalg.inv(A)),
        rtol=rtol,
        atol=atol,
    )


def test_nonsymmetric_estimate_uses_the_transpose():
    """XDiag needs A⁻ᵀ; an untagged operator solves with Aᵀ for it."""
    n, k = 40, 12
    A = jr.normal(jr.key(3), (n, n)) + 2 * jnp.sqrt(n) * jnp.eye(n)
    op = lx.MatrixLinearOperator(A)
    key = jr.key(4)
    # N = 40 is below the sign-independence threshold: Gaussian probes.
    mean, sem = matfree.stochtrace.estimator_leave_one_out_mean_and_sem(
        matfree.stochtrace.leave_one_out_xdiag(),
        matfree.stochtrace.sampler_normal(jnp.zeros(n, A.dtype), num=k),
    )(lambda v: jnp.linalg.solve(A, v), key)
    rtol, atol = default_tolerances(A)
    got = diag_inv(op, method="xdiag", num_probes=k, key=key)
    # The same estimator, so the same estimate up to rounding in the solves.
    assert jnp.allclose(got, mean, rtol=100 * rtol, atol=100 * atol)
    exact = jnp.diag(jnp.linalg.inv(A))
    assert jnp.linalg.norm(got - exact) <= 3 * jnp.linalg.norm(sem)


def test_key_none_means_prngkey_zero():
    op = lx.MatrixLinearOperator(_grid_precision(5).as_matrix(), lx.symmetric_tag)
    assert jnp.array_equal(
        diag_inv(op, method="xdiag", num_probes=4),
        diag_inv(op, method="xdiag", num_probes=4, key=jax.random.PRNGKey(0)),
    )


@pytest.mark.parametrize(("n", "sampler"), [(8, "normal"), (80, "signs")])
def test_probe_distribution_avoids_dependent_signs(n, sampler):
    """Small N uses Gaussian probes: k signs in N dims can be dependent.

    matfree's XDiag reads a rank-deficient probe image as a low-rank operator
    and returns a biased "exact" diagonal; at N = 8 and k = 3 a dependent
    sign pair has probability ~ 3 * 2^-7. Large N keeps the signs.
    """
    A = jr.normal(jr.key(5), (n, n))
    A = einsum(A, A, "i k, j k -> i j") + n * jnp.eye(n)
    k, key = 3, jr.key(6)
    factory = {
        "normal": matfree.stochtrace.sampler_normal,
        "signs": matfree.stochtrace.sampler_signs,
    }[sampler]
    expected = matfree.stochtrace.estimator_leave_one_out(
        matfree.stochtrace.leave_one_out_xdiag(),
        factory(jnp.zeros(n, A.dtype), num=k),
    )(lambda v: jnp.linalg.solve(A, v), key)
    op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
    got = diag_inv(op, method="xdiag", num_probes=k, key=key)
    rtol, atol = default_tolerances(A)
    assert jnp.allclose(got, expected, rtol=100 * rtol, atol=100 * atol)


@pytest.mark.parametrize("k", [0, 26])
def test_rejects_num_probes_outside_one_to_n(k):
    op = lx.MatrixLinearOperator(jnp.eye(25), lx.symmetric_tag)
    with pytest.raises(ValueError, match="1 <= num_probes <= N = 25"):
        diag_inv(op, method="xdiag", num_probes=k)


@pytest.mark.slow
def test_generalized_variance_route_for_large_graphs():
    """The documented route: diag(R⁺) = diag((R + VVᵀ)⁻¹) − rowsq(V).

    For an intrinsic structure ``R`` with orthonormal null space ``V``,
    ``(R + VVᵀ)⁻¹ = R⁺ + VVᵀ``, a well-conditioned system XDiag can take
    (no O(1/ε) ridge cancellation). At 2k >= N it is exact, so the
    geometric mean equals `generalized_variance_scale`.
    """
    n = 12
    R = gaussx.rw1_structure(n)
    V = jnp.full((n, 1), 1 / jnp.sqrt(n), dtype=R.as_matrix().dtype)
    # Matrix-free R + VVᵀ, solved by CG (Woodbury would invert singular R).
    shifted = lx.TaggedLinearOperator(
        gaussx.LowRankUpdate(R, V), lx.positive_semidefinite_tag
    )
    eps = float(jnp.finfo(V.dtype).eps)
    solver = gaussx.CGSolver(rtol=100 * eps, atol=100 * eps)  # cond(R + VVᵀ) ~ 60
    variances = diag_inv(
        shifted, method="xdiag", num_probes=n // 2, solver=solver
    ) - reduce(V**2, "n k -> n", "sum")
    s = jnp.exp(jnp.mean(jnp.log(variances)))
    expected = gaussx.generalized_variance_scale(R, jnp.ones(n))
    # The reference adds the ridge e = sqrt(machine eps) * max diag(R), which
    # perturbs R's pseudo-inverse variances by O(e / lambda_2), lambda_2 the
    # smallest non-zero eigenvalue of R; the CG solves add far less.
    Rd = np.asarray(R.as_matrix(), dtype=np.float64)
    lambda_2 = np.linalg.eigvalsh(Rd)[1]
    ridge = np.sqrt(eps) * Rd.diagonal().max()
    assert jnp.allclose(s, expected, rtol=10 * ridge / lambda_2)
