"""Test utilities for gaussx.

Provides helper functions for generating random structured operators
and comparing results. Intended for use in the gaussx test suite and
other internal tests; not part of the public, stable gaussx API.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._einx import einsum, reduce
from gaussx._operators import (
    BlockDiag,
    BlockTriDiag,
    Kronecker,
    LowRankUpdate,
    SumOfKroneckers,
)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


_DEFAULT_TOLERANCES = {
    # (rtol, atol). float64 keeps the historical defaults; float32 allows
    # ~100 eps of relative and absolute round-off, so a float32 matvec
    # compared with a float32 matmul does not fail on entries near zero.
    8: (1e-5, 1e-8),
    4: (1e-4, 1e-5),
    2: (1e-2, 1e-3),
}


def default_tolerances(*trees) -> tuple[float, float]:
    """``(rtol, atol)`` for the lowest-precision float leaf in ``trees``.

    Lets one assertion hold in both the x64 and the no-x64
    (``GAUSSX_TEST_X64=0``) lanes. Trees with no float leaves get the
    float64 defaults.
    """
    # finfo gives the component precision for complex dtypes too.
    sizes = [
        jnp.finfo(leaf.dtype).bits // 8
        for leaf in jax.tree.leaves(trees)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    return _DEFAULT_TOLERANCES.get(min(sizes, default=8), _DEFAULT_TOLERANCES[8])


def tree_allclose(
    x, y, *, rtol: float | None = None, atol: float | None = None
) -> bool | Bool[Array, ""]:
    """PyTree-aware approximate equality check.

    Wraps ``eqx.tree_equal`` with tolerance support. Matches the
    pattern used in the lineax test suite. ``rtol`` / ``atol`` left as
    ``None`` follow the inputs' precision (see `default_tolerances`).
    """
    default_rtol, default_atol = default_tolerances(x, y)
    rtol = default_rtol if rtol is None else rtol
    atol = default_atol if atol is None else atol
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Random operator generators
# ---------------------------------------------------------------------------


def _resolve_dtype(dtype):
    """``None`` means the active default float: float64 with x64, else float32.

    Following the active default (rather than hard float64) lets the same
    test run in the x64 and the no-x64 lanes.
    """
    return jnp.result_type(float) if dtype is None else dtype


def random_pd_matrix(
    key: jax.Array,
    n: int,
    *,
    dtype=None,
    jitter: float = 0.1,
) -> Float[Array, "n n"]:
    """A random positive-definite ``A Aᵀ + jitter·I``, ``A ~ N(0, 1)``.

    ``jitter`` sets the conditioning, and so what a given tolerance means:
    say it explicitly when a test's tolerance was chosen for it.
    """
    dtype = _resolve_dtype(dtype)
    A = jr.normal(key, (n, n), dtype=dtype)
    return A @ A.T + jitter * jnp.eye(n, dtype=dtype)


def random_pd_operator(
    key: jax.Array,
    n: int,
    *,
    dtype=None,
    jitter: float = 0.1,
    tags: object = lx.positive_semidefinite_tag,
) -> lx.MatrixLinearOperator:
    """`random_pd_matrix` wrapped as a tagged ``MatrixLinearOperator``."""
    mat = random_pd_matrix(key, n, dtype=dtype, jitter=jitter)
    return lx.MatrixLinearOperator(mat, tags)


def psd_operator(matrix: Float[Array, "n n"]) -> lx.MatrixLinearOperator:
    """Wrap a matrix the test knows to be PSD as a PSD-tagged operator."""
    return lx.MatrixLinearOperator(jnp.asarray(matrix), lx.positive_semidefinite_tag)


def random_kronecker_pd(
    key: jax.Array,
    sizes: tuple[int, ...],
    *,
    dtype=None,
    jitter: float = 0.1,
) -> Kronecker:
    """Generate a Kronecker product of random PSD matrices."""
    keys = jr.split(key, len(sizes))
    ops = tuple(
        random_pd_operator(k, n, dtype=dtype, jitter=jitter)
        for k, n in zip(keys, sizes, strict=True)
    )
    return Kronecker(*ops)


def random_block_diag_pd(
    key: jax.Array,
    sizes: tuple[int, ...],
    *,
    dtype=None,
    jitter: float = 0.1,
) -> BlockDiag:
    """Generate a BlockDiag of random PSD matrices."""
    keys = jr.split(key, len(sizes))
    ops = tuple(
        random_pd_operator(k, n, dtype=dtype, jitter=jitter)
        for k, n in zip(keys, sizes, strict=True)
    )
    return BlockDiag(*ops)


def random_low_rank_update(
    key: jax.Array,
    n: int,
    rank: int,
    *,
    dtype=None,
) -> LowRankUpdate:
    """Generate a random LowRankUpdate with positive diagonal base."""
    dtype = _resolve_dtype(dtype)
    k1, k2, k3 = jr.split(key, 3)
    d = jnp.abs(jr.normal(k1, (n,), dtype=dtype)) + 0.5
    U = jr.normal(k2, (n, rank), dtype=dtype) * 0.3
    diag_vals = jnp.abs(jr.normal(k3, (rank,), dtype=dtype)) + 0.1
    base = lx.DiagonalLinearOperator(d)
    return LowRankUpdate(base, U, diag_vals)


def random_sum_of_kroneckers_pd(
    key: jax.Array,
    sizes: tuple[int, ...],
    *,
    num_terms: int = 2,
    dtype=None,
    jitter: float = 0.1,
) -> SumOfKroneckers:
    """A ``SumOfKroneckers`` of ``num_terms`` Kronecker products of PSD factors.

    Each factor is `random_pd_operator` with the given ``jitter``, so the sum
    is symmetric positive definite.
    """
    keys = jr.split(key, num_terms)
    return SumOfKroneckers(
        *(random_kronecker_pd(k, sizes, dtype=dtype, jitter=jitter) for k in keys)
    )


def random_spd_block_tridiag(
    key: jax.Array,
    num_blocks: int,
    block_size: int,
    *,
    coupling: float = 0.3,
    dtype=None,
) -> BlockTriDiag:
    """A random SPD ``BlockTriDiag``, positive definite by construction.

    Built as ``L Lᵀ`` for a lower block-bidiagonal ``L`` with well-conditioned
    lower-triangular diagonal blocks ``D_k`` and sub-diagonal blocks
    ``C_k ~ coupling·N(0, 1)``: then ``A_kk = D_k D_kᵀ + C_{k-1} C_{k-1}ᵀ`` and
    ``A_{k+1,k} = C_k D_kᵀ``. Diagonal dominance, which the local builders this
    replaces relied on, is not guaranteed for arbitrary draws.
    """
    dtype = _resolve_dtype(dtype)
    k_diag, k_sub = jr.split(key)
    raw = jr.normal(k_diag, (num_blocks, block_size, block_size), dtype=dtype)
    gram = einsum(raw, raw, "N i k, N j k -> N i j") + block_size * jnp.eye(
        block_size, dtype=dtype
    )
    D = jnp.linalg.cholesky(gram)
    C = coupling * jr.normal(
        k_sub, (num_blocks - 1, block_size, block_size), dtype=dtype
    )
    diagonal = einsum(D, D, "N i k, N j k -> N i j")
    if num_blocks == 1:
        # einx rejects a zero-length axis, so the empty band skips it.
        return BlockTriDiag(diagonal, C)
    diagonal = diagonal.at[1:].add(einsum(C, C, "N i k, N j k -> N i j"))
    sub_diagonal = einsum(C, D[:-1], "N i k, N j k -> N i j")
    return BlockTriDiag(diagonal, sub_diagonal)


def empirical_moments(
    samples: Float[Array, "S N"],
) -> tuple[Float[Array, " N"], Float[Array, "N N"]]:
    """Sample mean and the unbiased sample covariance (``1 / (S - 1)``)."""
    mean = reduce(samples, "S N -> N", "mean")
    anomalies = einx.subtract("S N, N -> S N", samples, mean)
    cov = einsum(anomalies, anomalies, "S i, S j -> i j") / (samples.shape[0] - 1)
    return mean, cov


# ---------------------------------------------------------------------------
# Dense reference computations
# ---------------------------------------------------------------------------


def dense_solve(
    op: lx.AbstractLinearOperator,
    v: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Solve via dense materialization (reference implementation)."""
    return jnp.linalg.solve(op.as_matrix(), v)


def dense_logdet(op: lx.AbstractLinearOperator) -> Float[Array, ""]:
    """Log-determinant via dense materialization."""
    return jnp.linalg.slogdet(op.as_matrix())[1]


def dense_inv(op: lx.AbstractLinearOperator) -> Float[Array, "n n"]:
    """Inverse via dense materialization."""
    return jnp.linalg.inv(op.as_matrix())


def dense_diag(op: lx.AbstractLinearOperator) -> Float[Array, " n"]:
    """Diagonal via dense materialization."""
    return jnp.diag(op.as_matrix())


def dense_trace(op: lx.AbstractLinearOperator) -> Float[Array, ""]:
    """Trace via dense materialization."""
    return jnp.trace(op.as_matrix())


# ---------------------------------------------------------------------------
# Sampling-statistics assertions
# ---------------------------------------------------------------------------


def assert_sample_moments(
    samples: Float[Array, "S N"],
    mean: Float[Array, " N"],
    cov: Float[Array, "N N"],
    *,
    n_sigma: float = 7.0,
) -> None:
    r"""Assert empirical moments match their population values.

    Bounds each entry by ``n_sigma`` standard deviations of *that
    estimator's own sampling distribution*, rather than by a fixed
    absolute tolerance. For ``S`` draws from $\mathcal{N}(\mu, \Sigma)$,

    $$
    \operatorname{Var}[\bar{x}_i] = \frac{\Sigma_{ii}}{S},
    \qquad
    \operatorname{Var}[S_{ij}]
        \approx \frac{\Sigma_{ii}\Sigma_{jj} + \Sigma_{ij}^2}{S}.
    $$

    A fixed ``atol`` is the wrong shape for these assertions whenever the
    covariance under test is itself random: measured in standard
    deviations it then varies with the draw, so the test is far stricter
    on some seeds than others. On the 3x3 Wishart-style covariances used
    in the distribution tests, ``atol=0.5`` ranged from 1.9 to 193 sigma
    across seeds and failed roughly 1 run in 1300 (gh-220).

    The default ``n_sigma=7`` is set from a 20,000-seed sweep of those
    same covariances, over which the largest deviation observed was
    5.16 sigma. Seven leaves real headroom above that without weakening
    the test in any way that matters: a genuine bias or a wrong
    covariance moves the estimator by ``O(sqrt(S))`` standard
    deviations — hundreds, here — not by a handful.

    Args:
        samples: Draws, shape ``(S, N)``.
        mean: Population mean, shape ``(N,)``.
        cov: Population covariance, shape ``(N, N)``.
        n_sigma: Width of the acceptance band, in standard deviations of
            the estimator. Defaults to ``6.0``.

    Raises:
        AssertionError: If any empirical moment falls outside its band.
    """
    n_draws = samples.shape[0]
    sample_mean = jnp.mean(samples, axis=0)
    sample_cov = jnp.cov(samples.T)

    variances = jnp.diag(cov)
    mean_band = n_sigma * jnp.sqrt(variances / n_draws)
    cov_band = n_sigma * jnp.sqrt(
        (variances[:, None] * variances[None, :] + cov**2) / n_draws
    )

    mean_excess = jnp.max(jnp.abs(sample_mean - mean) - mean_band)
    assert mean_excess <= 0.0, (
        f"sample mean outside its {n_sigma}-sigma band by {float(mean_excess):.3g}"
    )

    cov_excess = jnp.max(jnp.abs(sample_cov - cov) - cov_band)
    assert cov_excess <= 0.0, (
        f"sample covariance outside its {n_sigma}-sigma band by {float(cov_excess):.3g}"
    )
