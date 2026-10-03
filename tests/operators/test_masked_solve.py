"""Tests for ``solve(MaskedOperator)`` via the capacitance method (gh-265).

Base operator: the periodic five-point Laplacian on a 32×32 grid as a
`gaussx.circulant_from_symbol`, optionally shifted by −λI. References are dense
solves of the masked sub-matrix ``B[m][:, m]``. Grids and right-hand sides are
fixed (pinned keys), so tolerances are round-off bounds.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx
from gaussx import MaskedOperator, grid_coupling_indices


N = 32


def _laplacian(lam: float = 0.0):
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(N)
    s1 = 2.0 * jnp.cos(k) - 2.0
    op = gaussx.circulant_from_symbol(s1[:, None] + s1[None, :])
    if lam:
        op = op - lam * lx.IdentityLinearOperator(op.in_structure())
    return op


def _disc(cy: float, cx: float, r: float) -> np.ndarray:
    j, i = np.mgrid[:N, :N]
    return np.hypot(j - cy, i - cx) < r


def _dense_masked_solve(op, mask, f):
    idx = np.flatnonzero(mask.ravel())
    B = op.as_matrix()[np.ix_(idx, idx)]
    return jnp.linalg.solve(B, f)


def _stencil_residual(x_masked, mask, f, lam):
    """Periodic five-point ∇²ψ − λψ − f on masked-in cells, ψ = 0 outside."""
    psi = jnp.zeros(N * N).at[np.flatnonzero(mask.ravel())].set(x_masked).reshape(N, N)
    lap = (
        jnp.roll(psi, 1, 0)
        + jnp.roll(psi, -1, 0)
        + jnp.roll(psi, 1, 1)
        + jnp.roll(psi, -1, 1)
        - 4.0 * psi
    )
    return (lap - lam * psi).reshape(-1)[np.flatnonzero(mask.ravel())] - f


@pytest.fixture
def disc_mask():
    return _disc(15.5, 15.5, 10.0)


def _masked(op, mask, periodic=True):
    flat = jnp.asarray(mask.ravel())
    return MaskedOperator(
        op,
        flat,
        flat,
        coupling_indices=grid_coupling_indices(mask, periodic=periodic),
    )


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_masked_solve_matches_dense(disc_mask, lam):
    """λ = 1 (non-singular base) and λ = 0 (singular: null vector derived)."""
    op = _laplacian(lam)
    masked = _masked(op, disc_mask)
    assert masked.capacitance is not None
    f = jr.normal(jr.key(0), (int(disc_mask.sum()),))
    x = gaussx.solve(masked, f)
    assert jnp.allclose(x, _dense_masked_solve(op, disc_mask, f), atol=1e-12)
    assert float(jnp.abs(_stencil_residual(x, disc_mask, f, lam)).max()) < 1e-10


def test_singular_base_null_vector_is_derived(disc_mask):
    masked = _masked(_laplacian(0.0), disc_mask)
    r = masked.capacitance.null_vector
    assert r is not None
    assert jnp.allclose(r, r[0])  # the constant mode


@pytest.mark.slow
def test_edge_touching_mask_needs_periodic_coupling():
    """Regression for the spectraldiffx wrap bug: a mask touching the edge of
    a periodic grid couples to cells on the opposite edge."""
    # Touches row 0 only: its wrapped neighbours on row N − 1 are masked out.
    mask = _disc(1.0, 15.5, 6.0)
    assert mask[0].any() and not mask[-1].any()
    op = _laplacian(1.0)
    f = jr.normal(jr.key(1), (int(mask.sum()),))
    reference = _dense_masked_solve(op, mask, f)

    wrapped = gaussx.solve(_masked(op, mask, periodic=True), f)
    assert jnp.allclose(wrapped, reference, atol=1e-12)

    no_wrap = gaussx.solve(_masked(op, mask, periodic=False), f)
    assert float(jnp.abs(no_wrap - reference).max()) > 1e-6


def test_capacitance_is_built_once(disc_mask, monkeypatch):
    """Construction pays the |C| base solves; each solve then costs two."""
    import gaussx._primitives._solve as solve_mod

    masked = _masked(_laplacian(1.0), disc_mask)
    calls = []
    original = solve_mod._solve_diagonalised

    def counting(operator, vector):
        calls.append(1)
        return original(operator, vector)

    monkeypatch.setattr(solve_mod, "_solve_diagonalised", counting)
    f = jr.normal(jr.key(2), (int(disc_mask.sum()),))
    for _ in range(3):
        gaussx.solve(masked, f)
    assert len(calls) == 6


@pytest.mark.slow
def test_jit_and_vmap(disc_mask):
    masked = _masked(_laplacian(1.0), disc_mask)
    F = jr.normal(jr.key(3), (4, int(disc_mask.sum())))
    solve = eqx.filter_jit(lambda op, f: gaussx.solve(op, f))
    expected = gaussx.solve(masked, F[0])
    assert jnp.allclose(solve(masked, F[0]), expected, atol=1e-12)
    batched = jax.vmap(lambda f: gaussx.solve(masked, f))(F)
    assert jnp.allclose(batched[0], expected, atol=1e-12)


@pytest.mark.slow
def test_without_coupling_falls_back_to_dense(disc_mask):
    flat = jnp.asarray(disc_mask.ravel())
    op = _laplacian(1.0)
    masked = MaskedOperator(op, flat, flat)
    assert masked.capacitance is None
    f = jr.normal(jr.key(4), (int(disc_mask.sum()),))
    assert jnp.allclose(
        gaussx.solve(masked, f), _dense_masked_solve(op, disc_mask, f), atol=1e-10
    )


@pytest.mark.slow
def test_all_exterior_coupling_is_exact_for_dense_base():
    """With every masked-out index in C the method is exact for any base."""
    n = 10
    k1, k2 = jr.split(jr.key(5))
    A = jr.normal(k1, (n, n)) + n * jnp.eye(n)
    base = lx.MatrixLinearOperator(A)
    mask = jnp.array([True, True, False, True, True, False, True, True, True, False])
    masked = MaskedOperator(base, mask, mask, coupling_indices=jnp.flatnonzero(~mask))
    f = jr.normal(k2, (int(mask.sum()),))
    idx = np.flatnonzero(np.asarray(mask))
    expected = jnp.linalg.solve(A[np.ix_(idx, idx)], f)
    assert jnp.allclose(gaussx.solve(masked, f), expected, atol=1e-10)


def test_coupling_requires_square_mask():
    base = lx.MatrixLinearOperator(jnp.eye(4))
    with pytest.raises(ValueError, match="square mask"):
        MaskedOperator(
            base,
            jnp.array([True, True, False, False]),
            jnp.array([True, False, True, False]),
            coupling_indices=jnp.array([3]),
        )


def test_grid_coupling_indices_wrap_and_connectivity():
    mask = np.zeros((4, 5), dtype=bool)
    mask[0, 0] = True
    plain = set(np.asarray(grid_coupling_indices(mask)).tolist())
    assert plain == {1, 5}  # right and below only
    wrapped = set(np.asarray(grid_coupling_indices(mask, periodic=True)).tolist())
    assert wrapped == {1, 5, 4, 15}  # plus the wrapped left (0,4) and up (3,0)
    diag = set(np.asarray(grid_coupling_indices(mask, connectivity=2)).tolist())
    assert diag == {1, 5, 6}
    with pytest.raises(ValueError, match="connectivity"):
        grid_coupling_indices(mask, connectivity=3)


# ---------------------------------------------------------------------------
# gh-290: the base is a differentiable leaf; no per-instance recompiles;
# .T keeps the capacitance and is trace-safe.
# ---------------------------------------------------------------------------

_SMALL_MASK = jnp.array([True, True, False, True, True, False])
_SMALL_IDX = np.flatnonzero(np.asarray(_SMALL_MASK))
_SMALL_F = jnp.arange(1.0, 5.0)


def _small_base(symmetric=True):
    A = jr.normal(jr.key(0), (6, 6))
    B = A @ A.T + 6 * jnp.eye(6)
    if not symmetric:
        B = B + 0.5 * jr.normal(jr.key(1), (6, 6))
    return B


def _small_masked(B, coupling=True):
    return MaskedOperator(
        lx.MatrixLinearOperator(B),
        _SMALL_MASK,
        _SMALL_MASK,
        coupling_indices=jnp.array([2, 5]) if coupling else None,
    )


def _dense_small_grad(B):
    ix = np.ix_(_SMALL_IDX, _SMALL_IDX)
    return jax.grad(lambda M: jnp.linalg.solve(M[ix], _SMALL_F).sum())(B)


@pytest.mark.parametrize("symmetric", [True, False], ids=["sym", "nonsym"])
@pytest.mark.parametrize("coupling", [True, False], ids=["capacitance", "dense"])
def test_masked_solve_grad_wrt_matrix_base(symmetric, coupling):
    # The capacitance path used to return exact zeros (static closure over
    # the base); the dense path raised in lineax's VJP (transpose via
    # __init__ on traced masks).
    B = _small_base(symmetric)
    loss = eqx.filter_grad(lambda op: gaussx.solve(op, _SMALL_F).sum())
    grad = eqx.filter_jit(loss)(_small_masked(B, coupling))
    assert jnp.allclose(grad.base.matrix, _dense_small_grad(B), atol=1e-10)


@pytest.mark.slow
def test_masked_solve_grad_wrt_diagonalised_eigenvalues():
    n = 6
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n)
    s1 = 2.0 * jnp.cos(k) - 2.0
    base = gaussx.circulant_from_symbol(s1[:, None] + s1[None, :] - 1.0)
    j, i = np.mgrid[:n, :n]
    mask = np.hypot(j - 2.5, i - 2.5) < 1.9
    flat = jnp.asarray(mask.ravel())
    coupling = grid_coupling_indices(mask, periodic=True)
    idx = np.flatnonzero(mask.ravel())
    f = jr.normal(jr.key(0), (idx.size,))

    def structured(lam):
        op = MaskedOperator(
            base.with_eigenvalues(lam), flat, flat, coupling_indices=coupling
        )
        return gaussx.solve(op, f).sum()

    def dense(lam):
        M = base.with_eigenvalues(lam).as_matrix()[np.ix_(idx, idx)]
        return jnp.linalg.solve(M, f).sum()

    lam0 = base.eigenvalues
    # jit so each gradient compiles once rather than op by op.
    grad = jax.jit(jax.grad(structured))(lam0)
    assert float(jnp.abs(grad).max()) > 0.0
    assert jnp.allclose(grad, jax.jit(jax.grad(dense))(lam0), atol=1e-10)

    # Built outside, differentiated w.r.t. the operator.
    op = MaskedOperator(base, flat, flat, coupling_indices=coupling)
    grad_op = eqx.filter_jit(eqx.filter_grad(lambda o: gaussx.solve(o, f).sum()))(op)
    assert jnp.allclose(grad_op.base.eigenvalues, grad, atol=1e-10)


def test_new_instance_does_not_recompile():
    traces = []

    @eqx.filter_jit
    def solve(op, f):
        traces.append(1)
        return gaussx.solve(op, f)

    B = _small_base()
    first = solve(_small_masked(B), _SMALL_F)
    second = solve(_small_masked(B), _SMALL_F)
    assert len(traces) == 1
    assert jnp.allclose(first, second)


@pytest.mark.parametrize("symmetric", [True, False], ids=["sym", "nonsym"])
def test_transpose_keeps_capacitance(symmetric):
    B = _small_base(symmetric)
    op = _small_masked(B)
    assert op.T.capacitance is not None
    expected = jnp.linalg.solve(B[np.ix_(_SMALL_IDX, _SMALL_IDX)].T, _SMALL_F)
    assert jnp.allclose(gaussx.solve(op.T, _SMALL_F), expected, atol=1e-12)
    assert jnp.allclose(op.T.as_matrix(), op.as_matrix().T)
    assert jnp.allclose(gaussx.solve(op.T.T, _SMALL_F), gaussx.solve(op, _SMALL_F))


def test_transpose_of_singular_base_under_jit(disc_mask):
    # Bordered (null-vector) capacitance, transposed inside a trace.
    op = _laplacian(0.0)
    masked = _masked(op, disc_mask)
    f = jr.normal(jr.key(6), (int(disc_mask.sum()),))
    x = eqx.filter_jit(lambda o, f: gaussx.solve(o.T, f))(masked, f)
    idx = np.flatnonzero(disc_mask.ravel())
    B = op.as_matrix()[np.ix_(idx, idx)]
    assert jnp.allclose(x, jnp.linalg.solve(B.T, f), atol=1e-10)


@pytest.mark.parametrize(
    ("old", "new"), [(0.0, 0.5), (1.0, 0.0)], ids=["singular-to-regular", "to-singular"]
)
def test_with_base_rebuilds_the_capacitance(disc_mask, old, new):
    # The capacitance is a cache of the base; after an update (here a new
    # shift, as an optimiser step would produce) with_base rebuilds it,
    # re-deriving the null vector for the new base.
    masked = _masked(_laplacian(old), disc_mask)
    new_base = _laplacian(new)
    f = jr.normal(jr.key(7), (int(disc_mask.sum()),))
    expected = _dense_masked_solve(new_base, disc_mask, f)
    stale = eqx.tree_at(lambda o: o.base, masked, new_base)
    assert float(jnp.abs(gaussx.solve(stale, f) - expected).max()) > 1e-6
    rebuilt = masked.with_base(new_base)
    assert jnp.allclose(gaussx.solve(rebuilt, f), expected, atol=1e-10)
    assert jnp.allclose(gaussx.solve(rebuilt.T, f), expected, atol=1e-10)


def test_traced_singular_base_needs_an_explicit_null_vector(disc_mask):
    # Built under jit, the null vector of a singular base cannot be derived:
    # without one the solve raises instead of solving the wrong system, and
    # with one it is exact.
    base = _laplacian(0.0)
    flat = jnp.asarray(disc_mask.ravel())
    coupling = grid_coupling_indices(disc_mask, periodic=True)
    f = jr.normal(jr.key(8), (int(disc_mask.sum()),))

    def solve_with(null_vector):
        @jax.jit
        def run(lam):
            op = MaskedOperator(
                base.with_eigenvalues(lam),
                flat,
                flat,
                coupling_indices=coupling,
                null_vector=null_vector,
            )
            return gaussx.solve(op, f)

        return run(base.eigenvalues)

    with pytest.raises(Exception, match="null vector cannot be derived"):
        jax.block_until_ready(solve_with(None))
    x = solve_with(jnp.ones(N * N))
    assert jnp.allclose(x, _dense_masked_solve(base, disc_mask, f), atol=1e-10)
