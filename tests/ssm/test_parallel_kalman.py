"""Tests for the parallel Kalman filter and RTS smoother.

The parallel implementation is the Särkkä-García-Fernández covariance
form via :func:`jax.lax.associative_scan`. These tests pin numerical
parity against the validated sequential filter / smoother in
``_kalman.py`` and exercise the API surface (TI / TV broadcast, mask,
operator inputs, JIT, vmap, grad).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import DenseSolver, FilterState, kalman_filter, rts_smoother
from gaussx._einx import rearrange
from gaussx._ssm._parallel_kalman import (
    parallel_kalman_filter,
    parallel_rts_smoother,
)
from gaussx._testing import default_tolerances, tree_allclose


def _make_model(getkey, N=2, M=2):
    A = 0.95 * jnp.eye(N)
    H = jnp.eye(M, N)
    Q = 0.1 * jnp.eye(N)
    R = 0.5 * jnp.eye(M)
    x0 = jnp.zeros(N)
    P0 = jnp.eye(N)
    return A, H, Q, R, x0, P0


# ----------------------------------------------------------------
# Parity with the sequential filter / smoother
# ----------------------------------------------------------------


@pytest.mark.parametrize("T", [1, 2, 8, 64])
@pytest.mark.slow
def test_parallel_kf_matches_sequential(getkey, T):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (T, 2))

    seq_state = kalman_filter(A, H, Q, R, obs, x0, P0)
    par_state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)

    assert tree_allclose(par_state.filtered_means, seq_state.filtered_means, rtol=1e-4)
    assert tree_allclose(par_state.filtered_covs, seq_state.filtered_covs, rtol=1e-4)
    assert tree_allclose(
        par_state.predicted_means, seq_state.predicted_means, rtol=1e-4
    )
    assert tree_allclose(par_state.predicted_covs, seq_state.predicted_covs, rtol=1e-4)
    assert tree_allclose(par_state.log_likelihood, seq_state.log_likelihood, rtol=1e-3)


# T = 1, 2 exercise the associative combine in the fast lane; the longer
# sequences only add compile time.
@pytest.mark.parametrize(
    "T",
    [
        1,
        2,
        pytest.param(8, marks=pytest.mark.slow),
        pytest.param(64, marks=pytest.mark.slow),
    ],
)
def test_parallel_rts_matches_sequential(getkey, T):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (T, 2))

    seq_state = kalman_filter(A, H, Q, R, obs, x0, P0)
    par_state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)

    seq_means, seq_covs = rts_smoother(seq_state, A)
    par_means, par_covs = parallel_rts_smoother(par_state, A)

    assert tree_allclose(par_means, seq_means, rtol=1e-4)
    assert tree_allclose(par_covs, seq_covs, rtol=1e-4)


@pytest.mark.slow
def test_parallel_kf_returns_filter_state(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (5, 2))
    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    assert isinstance(state, FilterState)
    assert state.filtered_means.shape == (5, 2)
    assert state.filtered_covs.shape == (5, 2, 2)
    assert state.predicted_means.shape == (5, 2)
    assert state.predicted_covs.shape == (5, 2, 2)


@pytest.mark.slow
def test_parallel_kf_sqrt_matches_covariance_form(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (100, 2))

    cov_state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    sqrt_state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0, psd_project=True)

    assert tree_allclose(sqrt_state.filtered_means, cov_state.filtered_means, rtol=1e-5)
    assert tree_allclose(sqrt_state.filtered_covs, cov_state.filtered_covs, rtol=1e-5)
    assert tree_allclose(
        sqrt_state.predicted_means, cov_state.predicted_means, rtol=1e-5
    )
    assert tree_allclose(sqrt_state.predicted_covs, cov_state.predicted_covs, rtol=1e-5)
    assert tree_allclose(sqrt_state.log_likelihood, cov_state.log_likelihood, rtol=1e-5)


@pytest.mark.slow
def test_parallel_rts_sqrt_matches_covariance_form(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (64, 2))
    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0, psd_project=True)

    cov_means, cov_covs = parallel_rts_smoother(state, A)
    sqrt_means, sqrt_covs = parallel_rts_smoother(state, A, psd_project=True)

    assert tree_allclose(sqrt_means, cov_means, rtol=1e-5)
    assert tree_allclose(sqrt_covs, cov_covs, rtol=1e-5)


@pytest.mark.slow
def test_parallel_kf_sqrt_covariances_are_psd(getkey):
    dtype = jnp.float32
    A = jnp.array([[0.999, 0.01], [0.0, 0.98]], dtype=dtype)
    H = jnp.array([[1.0, 0.0]], dtype=dtype)
    Q = jnp.diag(jnp.array([1e-8, 1e-10], dtype=dtype))
    R = jnp.array([[1e-6]], dtype=dtype)
    x0 = jnp.zeros(2, dtype=dtype)
    P0 = jnp.eye(2, dtype=dtype)
    obs = jr.normal(getkey(), (128, 1), dtype=dtype)

    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0, psd_project=True)
    covs = jnp.concatenate([state.filtered_covs, state.predicted_covs], axis=0)
    min_eig = jnp.min(jnp.linalg.eigvalsh(covs))
    psd_atol_factor = 100
    atol = jnp.array(jnp.finfo(dtype).eps * psd_atol_factor, dtype=dtype)

    assert jnp.isfinite(min_eig)
    assert min_eig >= -atol


def test_parallel_kf_rejects_unknown_form(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (5, 2))

    with pytest.raises(ValueError, match="form"):
        parallel_kalman_filter(A, H, Q, R, obs, x0, P0, form="information")


def test_parallel_rts_rejects_unknown_form(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (5, 2))
    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)

    with pytest.raises(ValueError, match="form"):
        parallel_rts_smoother(state, A, form="information")


def test_parallel_rts_last_matches_filter(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    T = 8
    obs = jr.normal(getkey(), (T, 2))
    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    par_means, par_covs = parallel_rts_smoother(state, A)
    assert tree_allclose(par_means[-1], state.filtered_means[-1], rtol=1e-6)
    assert tree_allclose(par_covs[-1], state.filtered_covs[-1], rtol=1e-6)


# ----------------------------------------------------------------
# solver= kwarg passthrough (currently a no-op; pinned for API stability)
# ----------------------------------------------------------------


@pytest.mark.slow
def test_parallel_kf_with_dense_solver_matches_default(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (6, 2))

    default_state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    # gh-364: solver has no effect on the associative scan, so it warns.
    with pytest.warns(DeprecationWarning, match="solver"):
        dense_state = parallel_kalman_filter(
            A, H, Q, R, obs, x0, P0, solver=DenseSolver()
        )

    assert tree_allclose(
        dense_state.filtered_means, default_state.filtered_means, rtol=1e-5
    )
    assert tree_allclose(
        dense_state.filtered_covs, default_state.filtered_covs, rtol=1e-5
    )
    assert tree_allclose(
        dense_state.log_likelihood, default_state.log_likelihood, rtol=1e-4
    )


def test_parallel_rts_with_dense_solver_matches_default(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (5, 2))

    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    default_means, default_covs = parallel_rts_smoother(state, A)
    dense_means, dense_covs = parallel_rts_smoother(state, A, solver=DenseSolver())

    assert tree_allclose(dense_means, default_means, rtol=1e-5)
    assert tree_allclose(dense_covs, default_covs, rtol=1e-5)


# ----------------------------------------------------------------
# Operator-typed inputs (time-invariant)
# ----------------------------------------------------------------


def test_parallel_kf_obs_noise_diagonal_operator(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (5, 2))
    R_diag = jnp.diag(R)

    ref = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    op = parallel_kalman_filter(A, H, Q, lx.DiagonalLinearOperator(R_diag), obs, x0, P0)
    assert tree_allclose(ref.filtered_means, op.filtered_means, rtol=1e-5)
    assert tree_allclose(ref.log_likelihood, op.log_likelihood, rtol=1e-5)


@pytest.mark.slow
def test_parallel_kf_woodbury_innovation_matches_sequential(getkey):
    N, M, T = 4, 32, 5
    A = 0.95 * jnp.eye(N)
    H = jr.normal(getkey(), (M, N)) / jnp.sqrt(N)
    Q = 0.05 * jnp.eye(N)
    R_diag = 0.3 + 0.1 * jnp.linspace(0.0, 1.0, M)
    obs = jr.normal(getkey(), (T, M))
    x0, P0 = jnp.zeros(N), jnp.eye(N)

    ref = kalman_filter(
        A,
        H,
        Q,
        lx.DiagonalLinearOperator(R_diag),
        obs,
        x0,
        P0,
        woodbury_innovation=True,
    )
    got = parallel_kalman_filter(
        A,
        H,
        Q,
        lx.DiagonalLinearOperator(R_diag),
        obs,
        x0,
        P0,
        woodbury_innovation=True,
    )

    assert tree_allclose(got.filtered_means, ref.filtered_means, rtol=1e-6)
    assert tree_allclose(got.filtered_covs, ref.filtered_covs, rtol=1e-6)
    assert tree_allclose(got.log_likelihood, ref.log_likelihood, rtol=1e-6)


@pytest.mark.slow
def test_parallel_kf_transition_block_diag_operator(getkey):
    from gaussx import BlockDiag

    A1 = 0.9 * jnp.eye(2)
    A2 = 0.7 * jnp.eye(2)
    A_dense = jnp.block([[A1, jnp.zeros((2, 2))], [jnp.zeros((2, 2)), A2]])
    N, M, T = 4, 2, 6
    H = jr.normal(getkey(), (M, N))
    Q = 0.1 * jnp.eye(N)
    R = 0.5 * jnp.eye(M)
    obs = jr.normal(getkey(), (T, M))
    x0, P0 = jnp.zeros(N), jnp.eye(N)

    ref = parallel_kalman_filter(A_dense, H, Q, R, obs, x0, P0)
    A_op = BlockDiag(lx.MatrixLinearOperator(A1), lx.MatrixLinearOperator(A2))
    op = parallel_kalman_filter(A_op, H, Q, R, obs, x0, P0)
    assert tree_allclose(ref.filtered_means, op.filtered_means, rtol=1e-5)


def test_parallel_kf_rejects_3d_with_operator(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    T = 4
    obs = jr.normal(getkey(), (T, 2))
    H_seq = jnp.broadcast_to(H, (T, *H.shape))
    A_op = lx.MatrixLinearOperator(A)
    with pytest.raises(TypeError):
        parallel_kalman_filter(A_op, H_seq, Q, R, obs, x0, P0)


# ----------------------------------------------------------------
# Time-varying inputs and mask
# ----------------------------------------------------------------


def test_parallel_kf_ti_broadcast_matches(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    T = 6
    obs = jr.normal(getkey(), (T, 2))
    ref = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    A_seq = jnp.broadcast_to(A, (T, *A.shape))
    H_seq = jnp.broadcast_to(H, (T, *H.shape))
    Q_seq = jnp.broadcast_to(Q, (T, *Q.shape))
    R_seq = jnp.broadcast_to(R, (T, *R.shape))
    tv = parallel_kalman_filter(A_seq, H_seq, Q_seq, R_seq, obs, x0, P0)
    assert tree_allclose(ref.filtered_means, tv.filtered_means, rtol=1e-6)
    assert tree_allclose(ref.log_likelihood, tv.log_likelihood, rtol=1e-6)


@pytest.mark.slow
def test_parallel_kf_mask_predict_only(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    T = 6
    obs = jr.normal(getkey(), (T, 2))
    mask = jnp.array([True, False, True, False, True, True])
    par = parallel_kalman_filter(A, H, Q, R, obs, x0, P0, mask=mask)
    seq = kalman_filter(A, H, Q, R, obs, x0, P0, mask=mask)
    assert tree_allclose(par.filtered_means, seq.filtered_means, rtol=1e-4)
    assert tree_allclose(par.filtered_covs, seq.filtered_covs, rtol=1e-4)
    assert tree_allclose(par.log_likelihood, seq.log_likelihood, rtol=1e-3)
    idx = jnp.where(~mask)[0]
    assert tree_allclose(par.filtered_means[idx], par.predicted_means[idx], atol=1e-7)


@pytest.mark.slow
def test_parallel_rts_smoother_tv(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    T = 6
    A_seq = jnp.broadcast_to(A, (T, *A.shape))
    obs = jr.normal(getkey(), (T, 2))
    state = parallel_kalman_filter(A_seq, H, Q, R, obs, x0, P0)
    s_means, _s_covs = parallel_rts_smoother(state, A_seq)
    assert tree_allclose(s_means[-1], state.filtered_means[-1], rtol=1e-6)


# ----------------------------------------------------------------
# JIT / vmap / grad smoke tests
# ----------------------------------------------------------------


@pytest.mark.slow
def test_parallel_kf_jit(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (8, 2))

    def fn(A_, H_, Q_, R_, obs_, x0_, P0_):
        return parallel_kalman_filter(A_, H_, Q_, R_, obs_, x0_, P0_).log_likelihood

    eager = fn(A, H, Q, R, obs, x0, P0)
    jitted = jax.jit(fn)(A, H, Q, R, obs, x0, P0)
    assert tree_allclose(eager, jitted, rtol=1e-6)


@pytest.mark.slow
def test_parallel_kf_vmap(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    B, T = 4, 6
    obs_batch = jr.normal(getkey(), (B, T, 2))

    def fn(obs_):
        return parallel_kalman_filter(A, H, Q, R, obs_, x0, P0).log_likelihood

    batched = jax.vmap(fn)(obs_batch)
    sequential = jnp.stack([fn(obs_batch[b]) for b in range(B)])
    assert tree_allclose(batched, sequential, rtol=1e-5)


@pytest.mark.slow
def test_parallel_kf_grad(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (8, 2))

    def loss(log_q_diag):
        Q_ = jnp.diag(jnp.exp(log_q_diag))
        return -parallel_kalman_filter(A, H, Q_, R, obs, x0, P0).log_likelihood

    log_q = jnp.log(jnp.diag(Q))
    g_par = jax.grad(loss)(log_q)

    def loss_seq(log_q_diag):
        Q_ = jnp.diag(jnp.exp(log_q_diag))
        return -kalman_filter(A, H, Q_, R, obs, x0, P0).log_likelihood

    g_seq = jax.grad(loss_seq)(log_q)
    assert tree_allclose(g_par, g_seq, rtol=1e-3, atol=1e-5)


@pytest.mark.slow
def test_parallel_kf_sqrt_jit_vmap_grad(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    B, T = 3, 8
    obs_batch = jr.normal(getkey(), (B, T, 2))

    def fn(obs_, log_q_diag):
        Q_ = jnp.diag(jnp.exp(log_q_diag))
        return parallel_kalman_filter(
            A, H, Q_, R, obs_, x0, P0, psd_project=True
        ).log_likelihood

    log_q = jnp.log(jnp.diag(Q))
    batched = jax.jit(jax.vmap(lambda obs_: fn(obs_, log_q)))(obs_batch)
    grad = jax.grad(lambda log_q_: -fn(obs_batch[0], log_q_))(log_q)

    assert batched.shape == (B,)
    assert jnp.all(jnp.isfinite(batched))
    assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# gh-412: genuinely time-varying parity (every A_t, H_t, Q_t, R_t distinct)
# ---------------------------------------------------------------------------


def _random_tv_model():
    # Pinned: parity is an identity, any well-conditioned model will do.
    T, N, M = 9, 3, 2
    k = jr.split(jr.key(0), 8)
    A = 0.8 * jnp.eye(N) + 0.1 * jr.normal(k[0], (T, N, N))
    H = jr.normal(k[1], (T, M, N))
    Lq = 0.3 * jr.normal(k[2], (T, N, N))
    Q = Lq @ jnp.swapaxes(Lq, -1, -2) + 0.1 * jnp.eye(N)
    Lr = 0.3 * jr.normal(k[3], (T, M, M))
    R = Lr @ jnp.swapaxes(Lr, -1, -2) + 0.2 * jnp.eye(M)
    y = jr.normal(k[4], (T, M))
    m0, P0 = jr.normal(k[5], (N,)), jnp.eye(N)
    masks = {
        "none": None,
        "steps": jr.bernoulli(k[6], 0.7, (T,)).at[0].set(True),
        "channels": jr.bernoulli(k[7], 0.7, (T, M)),
    }
    return (A, H, Q, R, y, m0, P0), masks


# The unmasked cases are slow: the masked ones run the same time-varying
# path and add the mask handling on top.
_TV_CASES = [
    pytest.param(mask, psd_project, marks=[pytest.mark.slow] if mask == "none" else [])
    for mask in ("none", "steps", "channels")
    for psd_project in (False, True)
    if not (psd_project and mask == "channels")  # rejected by design
]


@pytest.mark.parametrize(("mask_name", "psd_project"), _TV_CASES)
@pytest.mark.x64_only(reason="dense-reference tolerance below float32 round-off")
def test_tv_parity_random_params(mask_name, psd_project):
    args, masks = _random_tv_model()
    A = args[0]
    # Guard against regressing to broadcast (time-invariant) inputs.
    assert jnp.abs(A[0] - A[1]).max() > 0.01
    mask = masks[mask_name]
    seq = kalman_filter(*args, mask=mask)
    par = parallel_kalman_filter(*args, mask=mask, psd_project=psd_project)
    tol = {"rtol": 1e-12, "atol": 1e-12}
    assert jnp.allclose(seq.log_likelihood, par.log_likelihood, **tol)
    for field in (
        "filtered_means",
        "filtered_covs",
        "predicted_means",
        "predicted_covs",
    ):
        assert jnp.allclose(getattr(seq, field), getattr(par, field), **tol), field
    m_seq, P_seq = rts_smoother(seq, A)
    m_par, P_par = parallel_rts_smoother(par, A, psd_project=psd_project)
    assert jnp.allclose(m_seq, m_par, **tol)
    assert jnp.allclose(P_seq, P_par, **tol)


def test_form_sqrt_is_a_deprecated_spelling_of_psd_project():
    # gh-306: form="sqrt" never was a square-root filter.
    (A, H, Q, R, y, m0, P0), _ = _random_tv_model()
    new = parallel_kalman_filter(A, H, Q, R, y, m0, P0, psd_project=True)
    with pytest.warns(DeprecationWarning, match="psd_project=True"):
        old = parallel_kalman_filter(A, H, Q, R, y, m0, P0, form="sqrt")
    assert jnp.array_equal(new.log_likelihood, old.log_likelihood)
    assert jnp.array_equal(new.filtered_covs, old.filtered_covs)
    smoothed = parallel_rts_smoother(new, A, psd_project=True)
    with pytest.warns(DeprecationWarning, match="psd_project=True"):
        smoothed_old = parallel_rts_smoother(new, A, form="sqrt")
    for a, b in zip(smoothed, smoothed_old, strict=True):
        assert jnp.array_equal(a, b)


@pytest.mark.slow
def test_psd_project_keeps_a_float32_chain_finite():
    # gh-306: a regime where the covariance form goes indefinite in float32
    # (Matérn-5/2, dt = 1e-4, R = 1e-10) while the projection stays PSD. The
    # model is built in float64 so only the filter runs in float32.
    from gaussx import MaternSDE

    T = 1000
    kern = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(0.5), order=2)
    A, Q = kern.discretise(jnp.array(1e-4))
    P0 = kern.sde_params().P_inf
    H = jnp.array([[1.0, 0.0, 0.0]])
    t = jnp.arange(T) * 1e-4
    y = (jnp.sin(6.0 * t) + 0.1 * jr.normal(jr.key(0), (T,)))[:, None]
    args32 = [x.astype(jnp.float32) for x in (A, H, Q, 1e-10 * jnp.eye(1), y)]
    m0 = jnp.zeros(3, jnp.float32)
    P0_32 = P0.astype(jnp.float32)
    projected = parallel_kalman_filter(*args32, m0, P0_32, psd_project=True)
    assert jnp.isfinite(projected.log_likelihood)
    eigs = jnp.linalg.eigvalsh(projected.filtered_covs.astype(jnp.float64))
    assert jnp.min(eigs) > -1e-9


# ---------------------------------------------------------------------------
# gh-454: square-root (factor-propagating) parallel filter
# ---------------------------------------------------------------------------

_FIELDS = (
    "filtered_means",
    "filtered_covs",
    "predicted_means",
    "predicted_covs",
    "log_likelihood",
)


# As in _TV_CASES, the unmasked case is slow: the masked ones run the same
# path with the mask handling on top.
@pytest.mark.parametrize(
    "mask_name", [pytest.param("none", marks=pytest.mark.slow), "steps", "channels"]
)
@pytest.mark.x64_only(reason="parity tolerance below float32 round-off")
def test_square_root_parity_with_covariance_form(mask_name):
    """T = 9, N = 3, M = 2, jr.key(0): every output matches both the
    covariance-form parallel filter and the sequential filter.

    The input factors carry a 4 n eps diagonal shift (12 eps relative),
    so 1e-12 still holds with room to spare.
    """
    args, masks = _random_tv_model()
    mask = masks[mask_name]
    sq = parallel_kalman_filter(*args, mask=mask, square_root=True)
    cov = parallel_kalman_filter(*args, mask=mask)
    seq = kalman_filter(*args, mask=mask)
    for field in _FIELDS:
        for ref in (cov, seq):
            assert jnp.allclose(
                getattr(sq, field), getattr(ref, field), rtol=1e-12, atol=1e-12
            ), field
    # The smoother consumes the square-root filter's output unchanged.
    A = args[0]
    m_seq, P_seq = rts_smoother(seq, A)
    m_sq, P_sq = parallel_rts_smoother(sq, A)
    assert jnp.allclose(m_sq, m_seq, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(P_sq, P_seq, rtol=1e-12, atol=1e-12)


def test_square_root_matches_covariance_form_in_default_precision():
    """Float32-lane parity on a well-conditioned model."""
    (A, H, Q, R, y, m0, P0), _ = _random_tv_model()
    dtype = jnp.result_type(float)
    args = [x.astype(dtype) for x in (A, H, Q, R, y, m0, P0)]
    sq = parallel_kalman_filter(*args, square_root=True)
    cov = parallel_kalman_filter(*args)
    rtol, atol = default_tolerances(sq.filtered_covs)
    for field in _FIELDS:
        assert getattr(sq, field).dtype == dtype
        # Two different float orderings of the same recursion; the slack
        # over default_tolerances covers 9 steps of accumulated rounding.
        assert jnp.allclose(
            getattr(sq, field), getattr(cov, field), rtol=10 * rtol, atol=10 * atol
        ), field


@pytest.mark.slow
@pytest.mark.x64_only(reason="gradient parity to 1e-6 needs float64")
def test_square_root_gradient_matches_sequential():
    """jax.grad of the LL w.r.t. every model input matches kalman_filter.

    M = 2 < N = 3, so the information factors Z are rank deficient; the
    Gram-tangent rule keeps their derivative finite and exact.
    """
    (A, H, Q, R, y, m0, P0), _ = _random_tv_model()

    def ll(fn, A, H, Q, R, m0):
        return fn(A, H, Q, R, y, m0, P0).log_likelihood

    sq = jax.jit(
        jax.grad(
            lambda *a: ll(lambda *b: parallel_kalman_filter(*b, square_root=True), *a),
            argnums=(0, 1, 2, 3, 4),
        )
    )(A, H, Q, R, m0)
    seq = jax.jit(jax.grad(lambda *a: ll(kalman_filter, *a), argnums=(0, 1, 2, 3, 4)))(
        A, H, Q, R, m0
    )
    for g_sq, g_seq in zip(sq, seq, strict=True):
        assert jnp.all(jnp.isfinite(g_sq))
        assert jnp.allclose(g_sq, g_seq, rtol=1e-6, atol=1e-9)


def test_square_root_scan_has_no_eigendecomposition():
    """The compiled filter is QR-only: no eigh anywhere (gh-454)."""
    (A, H, Q, R, y, m0, P0), _ = _random_tv_model()
    jaxpr = str(
        jax.make_jaxpr(
            lambda *a: parallel_kalman_filter(*a, square_root=True).log_likelihood
        )(A, H, Q, R, y, m0, P0)
    )
    assert "eigh" not in jaxpr
    assert "qr" in jaxpr or "householder" in jaxpr


@pytest.mark.x64_only(reason="compares against a float64 sequential reference")
def test_square_root_accepts_zero_process_noise_at_step_zero():
    """A0 = I, Q0 = 0 observes the prior directly (documented trick)."""
    (A, H, Q, R, y, m0, P0), _ = _random_tv_model()
    A = A.at[0].set(jnp.eye(3))
    Q = Q.at[0].set(jnp.zeros((3, 3)))
    sq = parallel_kalman_filter(A, H, Q, R, y, m0, P0, square_root=True)
    seq = kalman_filter(A, H, Q, R, y, m0, P0)
    assert jnp.allclose(sq.log_likelihood, seq.log_likelihood, rtol=1e-10)
    assert jnp.allclose(sq.filtered_covs, seq.filtered_covs, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs", [{"psd_project": True}, {"woodbury_innovation": True}]
)
def test_square_root_rejects_incompatible_options(kwargs):
    (A, H, Q, R, y, m0, P0), _ = _random_tv_model()
    with pytest.raises(ValueError, match="square_root=True cannot be combined"):
        parallel_kalman_filter(A, H, Q, R, y, m0, P0, square_root=True, **kwargs)


@pytest.mark.slow
@pytest.mark.x64_only(reason="builds the model and reference in float64")
def test_square_root_float32_chain_is_psd_and_accurate():
    """gh-454 acceptance: Matérn-5/2, dt = 1e-4, R = 1e-10, T = 1000.

    Model built in float64, filter run in float32. The covariance form
    returns a NaN log-likelihood here and psd_project=True has a 7.9e-4
    relative error; the square-root filter must stay finite, return PSD
    covariances, and be no less accurate than the projection.
    """
    from gaussx import MaternSDE

    T = 1000
    kern = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(0.5), order=2)
    A, Q = kern.discretise(jnp.array(1e-4))
    P0 = kern.sde_params().P_inf
    H = jnp.array([[1.0, 0.0, 0.0]])
    t = jnp.arange(T) * 1e-4
    y = rearrange(jnp.sin(6.0 * t) + 0.1 * jr.normal(jr.key(0), (T,)), "t -> t 1")
    args = (A, H, Q, 1e-10 * jnp.eye(1), y, jnp.zeros(3), P0)
    ref = kalman_filter(*args).log_likelihood
    args32 = [x.astype(jnp.float32) for x in args]

    sq = parallel_kalman_filter(*args32, square_root=True)
    proj = parallel_kalman_filter(*args32, psd_project=True)
    assert jnp.isfinite(sq.log_likelihood)
    for covs in (sq.filtered_covs, sq.predicted_covs):
        assert jnp.min(jnp.linalg.eigvalsh(covs.astype(jnp.float64))) > -1e-12
    err_sq = jnp.abs(sq.log_likelihood - ref) / jnp.abs(ref)
    err_proj = jnp.abs(proj.log_likelihood - ref) / jnp.abs(ref)
    assert err_sq <= err_proj


@pytest.mark.slow
def test_square_root_more_observations_than_states():
    """M = 3 > N = 2 compresses the information factor Z by QR."""
    k = jr.split(jr.key(1), 3)
    A, Q = 0.9 * jnp.eye(2), 0.2 * jnp.eye(2)
    H = jr.normal(k[0], (3, 2))
    R = 0.5 * jnp.eye(3)
    y = jr.normal(k[1], (5, 3))
    args = (A, H, Q, R, y, jnp.zeros(2), jnp.eye(2))
    sq = jax.jit(lambda *a: parallel_kalman_filter(*a, square_root=True))(*args)
    cov = jax.jit(parallel_kalman_filter)(*args)
    rtol, atol = default_tolerances(sq.filtered_covs)
    for field in _FIELDS:
        assert jnp.allclose(
            getattr(sq, field), getattr(cov, field), rtol=10 * rtol, atol=10 * atol
        ), field


def test_square_root_empty_window():
    A, H, Q, R, x0, P0 = _make_model(None)
    state = parallel_kalman_filter(
        A, H, Q, R, jnp.zeros((0, 2)), x0, P0, square_root=True
    )
    assert state.filtered_covs.shape == (0, 2, 2)
    assert state.log_likelihood == 0.0
