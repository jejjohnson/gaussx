"""gh-408: float32 inputs stay float32 under x64 (and float64 stays float64).

conftest.py enables x64, so every case here would have returned float64 for
float32 inputs before the fix. Each case builds its inputs in ``dtype`` and
returns the arrays (or structures) whose dtype must match it. Keys are
pinned: only dtypes are checked.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx


def _pd(dtype, n=4):
    a = jr.normal(jr.key(0), (n, n), dtype=jnp.float64).astype(dtype)
    return a @ a.T + 4 * jnp.eye(n, dtype=dtype)


def _psd_op(dtype, n=4):
    return lx.MatrixLinearOperator(_pd(dtype, n), lx.positive_semidefinite_tag)


def _logdet_identity(dtype):
    op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), dtype))
    return [gaussx.logdet(op)]


def _trace_identity(dtype):
    op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), dtype))
    return [gaussx.trace(op)]


def _submatrix_kronecker(dtype):
    a = lx.MatrixLinearOperator(jnp.array([[2.0, 0.5], [0.5, 1.0]], dtype))
    idx = jnp.array([0, 3])
    return [gaussx.submatrix(gaussx.Kronecker(a, a), idx, idx)]


def _svd_diagonal(dtype):
    return list(gaussx.svd(lx.DiagonalLinearOperator(jnp.array([1.0, -2.0], dtype))))


def _eig_partial(dtype):
    return list(gaussx.eig(_psd_op(dtype, 6), rank=2, key=jr.key(0)))


def _schur_complement(dtype):
    k_xz = jr.normal(jr.key(1), (4, 2), dtype=jnp.float64).astype(dtype)
    k_zz = lx.MatrixLinearOperator(
        2 * jnp.eye(2, dtype=dtype), lx.positive_semidefinite_tag
    )
    sc = gaussx.schur_complement(_psd_op(dtype), k_xz, k_zz)
    return [sc.d, sc.mv(jnp.ones(4, dtype)), sc.in_structure()]


def _low_rank_update(dtype):
    lru = gaussx.LowRankUpdate(
        lx.DiagonalLinearOperator(jnp.ones(4, dtype)), jnp.ones((4, 1), dtype)
    )
    return [lru.in_structure(), lru.out_structure(), lru.mv(jnp.ones(4, dtype))]


def _etkf_transform(dtype):
    particles = jr.normal(jr.key(2), (5, 3), dtype=jnp.float64).astype(dtype)
    noise = lx.DiagonalLinearOperator(jnp.ones(3, dtype))
    return list(gaussx.etkf_transform(particles, jnp.zeros(3, dtype), noise))


def _trace_stochastic(dtype):
    return [gaussx.trace(_psd_op(dtype), stochastic=True, key=jr.key(3))]


def _diag_stochastic(dtype):
    return [gaussx.diag(_psd_op(dtype), stochastic=True, key=jr.key(3))]


def _slq_logdet(dtype):
    strategy = gaussx.SLQLogdet(num_probes=4, lanczos_order=3)
    return list(strategy.logdet_and_error(_psd_op(dtype), key=jr.key(3)))


def _nystrom_preconditioner(dtype):
    pre = gaussx.NystromPreconditioner.from_operator(
        _psd_op(dtype), rank=2, shift=0.1, key=jr.key(4)
    )
    return [pre.basis, pre.eigenvalues, pre.shift]


def _uncertain_gp_predict_mc(dtype):
    state = gaussx.GaussianState(
        jnp.zeros(2, dtype),
        lx.MatrixLinearOperator(jnp.eye(2, dtype=dtype), lx.positive_semidefinite_tag),
    )
    return list(
        gaussx.uncertain_gp_predict_mc(
            lambda x: (jnp.sum(x), jnp.sum(x**2)), state, n_particles=8, key=jr.key(5)
        )
    )


CASES = [
    _logdet_identity,
    _trace_identity,
    _submatrix_kronecker,
    _svd_diagonal,
    _eig_partial,
    _schur_complement,
    _low_rank_update,
    _etkf_transform,
    _trace_stochastic,
    _diag_stochastic,
    _slq_logdet,
    _nystrom_preconditioner,
    _uncertain_gp_predict_mc,
]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("case", CASES, ids=lambda f: f.__name__.lstrip("_"))
def test_output_dtype_follows_input_dtype(case, dtype):
    for out in case(dtype):
        assert out.dtype == dtype, (case.__name__, out)


def test_low_rank_update_structure_matches_mv_for_mixed_dtypes():
    """The declared structure is the dtype ``mv`` actually returns."""
    lru = gaussx.LowRankUpdate(
        lx.DiagonalLinearOperator(jnp.ones(4, jnp.float32)),
        jnp.ones((4, 1), jnp.float64),
    )
    out = lru.mv(jnp.ones(4, jnp.float32))
    assert lru.in_structure().dtype == lru.out_structure().dtype == out.dtype
    assert out.dtype == jnp.float64
