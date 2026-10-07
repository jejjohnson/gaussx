"""Tests for NystromLogdet (G17, gh-486).

Every key is pinned. The headline test is Wenger et al.'s (2022) claim:
at equal probe counts, Nyström preconditioning collapses SLQ's variance on
a kernel matrix. Its bounds are the estimators' own spread over pinned
keys, with the measured values stated where they are used.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx as gx
from gaussx._testing import default_tolerances


def _kernel_system(n: int, shift: float):
    """RBF Gram matrix on a 1-D grid plus ``shift * I`` (covariance form)."""
    x = jnp.linspace(0.0, 20.0, n)
    K = jnp.exp(-0.5 * einx.subtract("i, j -> i j", x, x) ** 2)
    return lx.MatrixLinearOperator(
        K + shift * jnp.eye(n, dtype=K.dtype), lx.positive_semidefinite_tag
    )


def _exact_logdet(op):
    return float(np.linalg.slogdet(np.asarray(op.as_matrix(), dtype=np.float64))[1])


@pytest.mark.slow
def test_collapses_slq_variance_on_a_kernel_matrix():
    """Wenger et al. (2022): same probes, far smaller spread, still unbiased."""
    n, shift = 200, 1e-2
    op = _kernel_system(n, shift)
    exact = _exact_logdet(op)
    keys = jr.split(jr.key(0), 20)

    def spread(strategy):
        est = np.asarray(jax.jit(jax.vmap(lambda k: strategy.logdet(op, key=k)))(keys))
        return est, est.std()

    _, slq_std = spread(gx.SLQLogdet(num_probes=10))
    est, nys_std = spread(gx.NystromLogdet(shift=shift, rank=40, num_probes=10))
    # n = 300 on this kernel (d_eff = 27), 50 keys: SLQ std 11.7, Nyström
    # (rank 40) std 0.002, a factor ~5000; ask for 100.
    assert nys_std < 1e-2 * slq_std
    # Unbiased: the mean of 20 i.i.d. estimates within 5 SEM (measured |z| <
    # 0.4), plus rounding at the scale of the log-determinant.
    sem = nys_std / np.sqrt(est.size)
    assert abs(est.mean() - exact) <= 5 * sem + 1e-9 * abs(exact)


def test_exact_at_full_rank():
    """rank = n: P = A + mu I, so M = I and SLQ contributes log|I| = 0."""
    n, shift = 30, 0.1
    op = _kernel_system(n, shift)
    est = jax.jit(lambda op: gx.NystromLogdet(shift=shift, rank=n).logdet(op))(op)
    rtol, _ = default_tolerances(op.as_matrix())
    # The Nyström eigenvalues of a rank-n sketch are exact to rounding, but
    # the small ones are relative to the largest: allow 100 x the default.
    assert np.isclose(float(est), _exact_logdet(op), rtol=100 * rtol)


@pytest.mark.slow
def test_gradient_in_the_shift():
    """d/dmu log|K + mu I| = tr((K + mu I)^-1), through sketch and SLQ."""
    n, shift = 40, 0.1
    x = jnp.linspace(0.0, 20.0, n)
    K = jnp.exp(-0.5 * einx.subtract("i, j -> i j", x, x) ** 2)

    def logdet(mu):
        op = lx.MatrixLinearOperator(K + mu * jnp.eye(n), lx.positive_semidefinite_tag)
        return gx.NystromLogdet(shift=mu, rank=n).logdet(op)

    grad = jax.jit(jax.grad(logdet))(shift)
    expected = jnp.trace(jnp.linalg.inv(K + shift * jnp.eye(n)))
    assert jnp.allclose(grad, expected, rtol=1e-6)


def test_zero_covariance_part_is_exact():
    """operator = mu I (a zero kernel amplitude): exactly n log(mu), no NaN."""
    n, shift = 25, 0.3
    op = lx.MatrixLinearOperator(shift * jnp.eye(n), lx.positive_semidefinite_tag)
    est = jax.jit(lambda op: gx.NystromLogdet(shift=shift, rank=5).logdet(op))(op)
    rtol, atol = default_tolerances(op.as_matrix())
    assert jnp.allclose(est, n * jnp.log(shift), rtol=10 * rtol, atol=10 * atol)


def test_key_none_means_prngkey_seed():
    op = _kernel_system(20, 0.1)
    strategy = gx.NystromLogdet(shift=0.1, rank=5, num_probes=3, seed=7)
    default, seeded = jax.jit(
        lambda op: (strategy.logdet(op), strategy.logdet(op, key=jr.PRNGKey(7)))
    )(op)
    assert jnp.array_equal(default, seeded)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"shift": 0.0}, "shift > 0"), ({"shift": 0.1, "rank": 0}, "rank >= 1")],
)
def test_rejects_bad_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        gx.NystromLogdet(**kwargs)
