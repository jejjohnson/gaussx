"""Test utilities for gaussx.

Provides helper functions for generating random structured operators
and comparing results. Intended for use in the gaussx test suite and
other internal tests; not part of the public, stable gaussx API.

Status (decided in gh-322): the module ships in the wheel, because the test
suite imports it as ``gaussx._testing``, but it stays underscore-private. It
is not listed in ``gaussx.__all__`` or the API reference, and its helpers
may change or disappear in any release without a deprecation. Downstream
test suites should copy what they need rather than import it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

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

    Args:
        *trees: The pytrees about to be compared.

    Returns:
        Tuple ``(rtol, atol)`` for the lowest precision among the float leaves.

    Examples:
        The lowest precision wins, and a tree with no float array gets float64's:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> default_tolerances(np.ones(2), jnp.ones(2, dtype=jnp.float32))
        (0.0001, 1e-05)
        >>> default_tolerances({"count": jnp.ones(2, dtype=jnp.int32), "step": 1.0})
        (1e-05, 1e-08)
    """
    # finfo gives the component precision for complex dtypes too.
    sizes = [
        jnp.finfo(leaf.dtype).bits // 8
        for leaf in jax.tree.leaves(trees)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    lowest_size = min(sizes, default=8)
    return _DEFAULT_TOLERANCES.get(lowest_size, _DEFAULT_TOLERANCES[8])


def tree_allclose(
    x, y, *, rtol: float | None = None, atol: float | None = None
) -> bool | Bool[Array, ""]:
    """PyTree-aware approximate equality check.

        |x − y| ≤ atol + rtol |y|,   on each pair of array leaves

    Wraps ``eqx.tree_equal`` with tolerance support. Matches the
    pattern used in the lineax test suite. ``rtol`` / ``atol`` left as
    ``None`` follow the inputs' precision (see `default_tolerances`).

    Args:
        x: First pytree.
        y: Second pytree, the reference for the relative tolerance.
        rtol: Relative tolerance, or ``None`` to follow the inputs' precision.
        atol: Absolute tolerance, or ``None`` to follow the inputs' precision.

    Returns:
        ``True`` or ``False`` when a tree structure, leaf type, shape or dtype
        decides it, otherwise a boolean array, traceable under ``jit``.

    Examples:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1.0, 2.0])
        >>> bool(tree_allclose({"a": x}, {"a": x}))
        True
        >>> bool(tree_allclose(x, x + 1.0))
        False
        >>> bool(tree_allclose(x, x + 1.0, atol=2.0))
        True
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

    Args:
        dtype: A dtype, or ``None`` for the active default float.

    Returns:
        The dtype to build with.
    """
    return jnp.result_type(float) if dtype is None else dtype


def key_sequence(seed: int = 0) -> Callable[[], jax.Array]:
    """A fixed stream of PRNG keys: each call returns the next one.

        kᵢ = fold_in(k₀, i),   i = 0, 1, …

    ``nextkey = key_sequence(0)`` replaces the per-test ``getkey`` fixture
    where the randomness is incidental, so the drawn model is the same for
    every run, independent of the test's name and of ``EQX_GETKEY_SEED``.

    Args:
        seed: Integer seed of the base key k₀.

    Returns:
        A function of no arguments that returns the next key on each call.

    Examples:
        The same seed gives the same stream, and each call moves it on:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> first, second = key_sequence(0), key_sequence(0)
        >>> key = first()
        >>> bool(jnp.all(jr.key_data(key) == jr.key_data(second())))
        True
        >>> bool(jnp.all(jr.key_data(key) == jr.key_data(first())))
        False
    """
    base = jr.key(seed)
    counter = itertools.count()
    return lambda: jr.fold_in(base, next(counter))


def random_pd_matrix(
    key: jax.Array,
    n: int,
    *,
    dtype=None,
    jitter: float = 0.1,
) -> Float[Array, "n n"]:
    """A random positive-definite matrix A from a standard-normal factor.

        A = W Wᵀ + β I,   [W]_ij ~ 𝒩(0, 1)

    ``jitter`` sets the conditioning, and so what a given tolerance means:
    say it explicitly when a test's tolerance was chosen for it.

    Args:
        key: PRNG key for W.
        n: Size of A.
        dtype: Dtype of A; ``None`` is the active default float.
        jitter: Diagonal loading β.

    Returns:
        The dense matrix A.

    Examples:
        A positive-definite draw, and the empty matrix at n = 0:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> matrix = random_pd_matrix(jr.key(0), 3, dtype=jnp.float32)
        >>> matrix.shape, matrix.dtype.name
        ((3, 3), 'float32')
        >>> bool(jnp.all(jnp.linalg.eigvalsh(matrix) > 0))
        True
        >>> random_pd_matrix(jr.key(0), 0).shape
        (0, 0)
    """
    dtype = _resolve_dtype(dtype)
    factor = jr.normal(key, (n, n), dtype=dtype)
    # W Wᵀ, with @ and .T because einx rejects the zero-length axes of n = 0
    outer = factor @ factor.T
    # β I
    loading = jitter * jnp.eye(n, dtype=dtype)
    # A = W Wᵀ + β I
    matrix = outer + loading
    return matrix


def random_pd_operator(
    key: jax.Array,
    n: int,
    *,
    dtype=None,
    jitter: float = 0.1,
    tags: object = lx.positive_semidefinite_tag,
) -> lx.MatrixLinearOperator:
    """`random_pd_matrix` wrapped as a tagged ``MatrixLinearOperator``.

    Args:
        key: PRNG key for the matrix.
        n: Size of the matrix.
        dtype: Dtype of the matrix; ``None`` is the active default float.
        jitter: Diagonal loading β.
        tags: lineax tags of the operator.

    Returns:
        The tagged operator.

    Examples:
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> operator = random_pd_operator(jr.key(0), 3)
        >>> operator.in_size(), lx.is_positive_semidefinite(operator)
        (3, True)
    """
    matrix = random_pd_matrix(key, n, dtype=dtype, jitter=jitter)
    return lx.MatrixLinearOperator(matrix, tags)


def psd_operator(matrix: Float[Array, "n n"]) -> lx.MatrixLinearOperator:
    """Wrap a matrix the test knows to be PSD as a PSD-tagged operator.

    Args:
        matrix: The matrix, which this does not check is PSD.

    Returns:
        The PSD-tagged operator.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> lx.is_positive_semidefinite(psd_operator(jnp.eye(2)))
        True
    """
    dense = jnp.asarray(matrix)
    return lx.MatrixLinearOperator(dense, lx.positive_semidefinite_tag)


def random_kronecker_pd(
    key: jax.Array,
    sizes: tuple[int, ...],
    *,
    dtype=None,
    jitter: float = 0.1,
) -> Kronecker:
    """Generate a Kronecker product of random PSD matrices.

        A = A₁ ⊗ … ⊗ A_K,   each A_k from `random_pd_operator`

    Args:
        key: PRNG key, split once per factor.
        sizes: Size of each factor A_k; at least two factors.
        dtype: Dtype of the factors; ``None`` is the active default float.
        jitter: Diagonal loading β of each factor.

    Returns:
        The ``Kronecker`` operator A.

    Raises:
        ValueError: If ``sizes`` has fewer than two entries.

    Examples:
        >>> import jax.random as jr
        >>> import gaussx
        >>> operator = random_kronecker_pd(jr.key(0), (2, 3))
        >>> operator.in_size(), gaussx.is_kronecker(operator)
        (6, True)
    """
    keys = jr.split(key, len(sizes))
    ops = tuple(
        random_pd_operator(factor_key, factor_size, dtype=dtype, jitter=jitter)
        for factor_key, factor_size in zip(keys, sizes, strict=True)
    )
    return Kronecker(*ops)


def random_block_diag_pd(
    key: jax.Array,
    sizes: tuple[int, ...],
    *,
    dtype=None,
    jitter: float = 0.1,
) -> BlockDiag:
    """Generate a BlockDiag of random PSD matrices.

        A = diag(A₁, …, A_K),   each A_k from `random_pd_operator`

    Args:
        key: PRNG key, split once per block.
        sizes: Size of each block A_k; at least one block.
        dtype: Dtype of the blocks; ``None`` is the active default float.
        jitter: Diagonal loading β of each block.

    Returns:
        The ``BlockDiag`` operator A.

    Raises:
        ValueError: If ``sizes`` is empty.

    Examples:
        >>> import jax.random as jr
        >>> import gaussx
        >>> operator = random_block_diag_pd(jr.key(0), (2, 3))
        >>> operator.in_size(), gaussx.is_block_diagonal(operator)
        (5, True)
    """
    keys = jr.split(key, len(sizes))
    ops = tuple(
        random_pd_operator(block_key, block_size, dtype=dtype, jitter=jitter)
        for block_key, block_size in zip(keys, sizes, strict=True)
    )
    return BlockDiag(*ops)


def random_low_rank_update(
    key: jax.Array,
    n: int,
    rank: int,
    *,
    dtype=None,
) -> LowRankUpdate:
    """Generate a random LowRankUpdate with positive diagonal base.

        A = A₀ + U diag(d) Uᵀ,   A₀ = diag(|a| + ½),   U = 0.3 W,   d = |c| + 0.1

    with a ∈ ℝⁿ, W ∈ ℝ^(n × rank) and c ∈ ℝ^rank independent standard normals.

    Args:
        key: PRNG key, split into one key each for a, W and c.
        n: Size of A.
        rank: Number of columns of U.
        dtype: Dtype of the arrays; ``None`` is the active default float.

    Returns:
        The ``LowRankUpdate`` operator A.

    Examples:
        >>> import jax.random as jr
        >>> import gaussx
        >>> operator = random_low_rank_update(jr.key(0), 4, 2)
        >>> operator.in_size(), gaussx.is_low_rank(operator)
        (4, True)
    """
    dtype = _resolve_dtype(dtype)
    base_key, factor_key, weight_key = jr.split(key, 3)
    base_draw = jr.normal(base_key, (n,), dtype=dtype)
    # diag(A₀) = |a| + ½
    base_diagonal = jnp.abs(base_draw) + 0.5
    factor_draw = jr.normal(factor_key, (n, rank), dtype=dtype)
    # U = 0.3 W
    factor = factor_draw * 0.3
    weight_draw = jr.normal(weight_key, (rank,), dtype=dtype)
    # d = |c| + 0.1
    weights = jnp.abs(weight_draw) + 0.1
    base = lx.DiagonalLinearOperator(base_diagonal)
    return LowRankUpdate(base, factor, weights)


def random_sum_of_kroneckers_pd(
    key: jax.Array,
    sizes: tuple[int, ...],
    *,
    num_terms: int = 2,
    dtype=None,
    jitter: float = 0.1,
) -> SumOfKroneckers:
    """A ``SumOfKroneckers`` of ``num_terms`` Kronecker products of PSD factors.

        A = ∑ₜ A_t,1 ⊗ … ⊗ A_t,K

    Each factor is `random_pd_operator` with the given ``jitter``, so the sum
    is symmetric positive definite.

    Args:
        key: PRNG key, split once per term.
        sizes: Size of each factor of a term; at least two factors.
        num_terms: Number of terms in the sum; at least two.
        dtype: Dtype of the factors; ``None`` is the active default float.
        jitter: Diagonal loading β of each factor.

    Returns:
        The ``SumOfKroneckers`` operator A.

    Raises:
        ValueError: If ``sizes`` has fewer than two entries.
        TypeError: If ``num_terms`` is 0 or 1.

    Examples:
        >>> import jax.random as jr
        >>> random_sum_of_kroneckers_pd(jr.key(0), (2, 3), num_terms=3).in_size()
        6
    """
    keys = jr.split(key, num_terms)
    return SumOfKroneckers(
        *(
            random_kronecker_pd(term_key, sizes, dtype=dtype, jitter=jitter)
            for term_key in keys
        )
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

        A = L Lᵀ,   A_k,k = D_k D_kᵀ + C_k−1 C_k−1ᵀ,   A_k+1,k = C_k D_kᵀ

    Here L is lower block-bidiagonal, with well-conditioned lower-triangular
    diagonal blocks D_k and sub-diagonal blocks C_k ~ coupling·𝒩(0, 1). Diagonal
    dominance, which the local builders this replaces relied on, is not
    guaranteed for arbitrary draws.

    Algorithm:
        1. Draw W_k and factor D_k D_kᵀ = W_k W_kᵀ + d I, with d the block size.
        2. Draw C_k.
        3. Form the diagonal and sub-diagonal blocks of L Lᵀ.

    Args:
        key: PRNG key, split into one key for the W_k and one for the C_k.
        num_blocks: Number of diagonal blocks, at least one.
        block_size: Size d of each block, at least one.
        coupling: Scale of the sub-diagonal blocks C_k.
        dtype: Dtype of the blocks; ``None`` is the active default float.

    Returns:
        The ``BlockTriDiag`` operator A.

    Examples:
        Three coupled blocks, and one block with an empty band:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> operator = random_spd_block_tridiag(jr.key(0), 3, 2)
        >>> operator.in_size()
        6
        >>> bool(jnp.all(jnp.linalg.eigvalsh(operator.as_matrix()) > 0))
        True
        >>> random_spd_block_tridiag(jr.key(0), 1, 2).in_size()
        2
    """
    dtype = _resolve_dtype(dtype)
    diag_key, sub_key = jr.split(key)

    # --- blocks of L ---
    raw = jr.normal(diag_key, (num_blocks, block_size, block_size), dtype=dtype)
    # W_k W_kᵀ
    raw_outer = einsum(raw, raw, "N i k, N j k -> N i j")
    # d I
    loading = block_size * jnp.eye(block_size, dtype=dtype)
    # W_k W_kᵀ + d I
    loaded_outer = raw_outer + loading
    # D_k D_kᵀ = W_k W_kᵀ + d I
    diagonal_factor = jnp.linalg.cholesky(loaded_outer)
    sub_draw = jr.normal(sub_key, (num_blocks - 1, block_size, block_size), dtype=dtype)
    # C_k ~ coupling·𝒩(0, 1)
    sub_factor = coupling * sub_draw

    # --- blocks of A = L Lᵀ ---
    # D_k D_kᵀ
    diagonal = einsum(diagonal_factor, diagonal_factor, "N i k, N j k -> N i j")
    if num_blocks == 1:
        # einx rejects a zero-length axis, so the empty band skips it.
        return BlockTriDiag(diagonal, sub_factor)
    # C_k−1 C_k−1ᵀ
    sub_outer = einsum(sub_factor, sub_factor, "N i k, N j k -> N i j")
    # A_k,k = D_k D_kᵀ + C_k−1 C_k−1ᵀ
    diagonal = diagonal.at[1:].add(sub_outer)
    leading_factor = diagonal_factor[:-1]
    # A_k+1,k = C_k D_kᵀ
    sub_diagonal = einsum(sub_factor, leading_factor, "N i k, N j k -> N i j")
    return BlockTriDiag(diagonal, sub_diagonal)


def empirical_moments(
    samples: Float[Array, "S N"],
) -> tuple[Float[Array, " N"], Float[Array, "N N"]]:
    """Sample mean and the unbiased sample covariance (``1 / (S - 1)``).

        x̄ = (1 / S) ∑ₑ x⁽ᵉ⁾,   Σ̂_x = (1 / (S − 1)) ∑ₑ x̃⁽ᵉ⁾ x̃⁽ᵉ⁾ᵀ,   x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄

    Args:
        samples: The S draws x⁽ᵉ⁾, one per row.

    Returns:
        Tuple ``(mean, cov)`` holding x̄ and Σ̂_x. With one draw, Σ̂_x is NaN.

    Examples:
        Three draws on a line, then one draw, which leaves Σ̂_x undefined:

        >>> import jax.numpy as jnp
        >>> samples = jnp.array([[1.0, 0.0], [3.0, 2.0], [2.0, 1.0]])
        >>> mean, cov = empirical_moments(samples)
        >>> mean.tolist()
        [2.0, 1.0]
        >>> cov.tolist()
        [[1.0, 1.0], [1.0, 1.0]]
        >>> bool(jnp.isnan(empirical_moments(samples[:1])[1]).all())
        True
    """
    # x̄ = (1 / S) ∑ₑ x⁽ᵉ⁾
    mean = reduce(samples, "S N -> N", "mean")
    # x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄
    anomalies = einx.subtract("S N, N -> S N", samples, mean)
    # ∑ₑ x̃⁽ᵉ⁾ x̃⁽ᵉ⁾ᵀ
    scatter = einsum(anomalies, anomalies, "S i, S j -> i j")
    # Σ̂_x = (1 / (S − 1)) ∑ₑ x̃⁽ᵉ⁾ x̃⁽ᵉ⁾ᵀ
    cov = scatter / (samples.shape[0] - 1)
    return mean, cov


# ---------------------------------------------------------------------------
# Dense reference computations
# ---------------------------------------------------------------------------


def dense_solve(
    op: lx.AbstractLinearOperator,
    v: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Solve via dense materialization (reference implementation).

        x = A⁻¹ v

    Args:
        op: The operator A, materialised to a dense matrix.
        v: Right-hand side v.

    Returns:
        The solution x.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> operator = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> dense_solve(operator, jnp.ones(2)).tolist()
        [0.5, 0.25]
    """
    matrix = op.as_matrix()
    # x = A⁻¹ v
    return jnp.linalg.solve(matrix, v)


def dense_logdet(op: lx.AbstractLinearOperator) -> Float[Array, ""]:
    """Log-determinant via dense materialization.

        ln |det A|

    Args:
        op: The operator A, materialised to a dense matrix.

    Returns:
        The log of the absolute determinant, whatever the determinant's sign.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> operator = lx.DiagonalLinearOperator(jnp.array([2.0, -4.0]))
        >>> bool(jnp.allclose(dense_logdet(operator), jnp.log(8.0)))
        True
    """
    matrix = op.as_matrix()
    # ln |det A|
    return jnp.linalg.slogdet(matrix)[1]


def dense_inv(op: lx.AbstractLinearOperator) -> Float[Array, "n n"]:
    """Inverse via dense materialization.

        A⁻¹

    Args:
        op: The operator A, materialised to a dense matrix.

    Returns:
        The dense inverse A⁻¹.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> operator = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> dense_inv(operator).tolist()
        [[0.5, 0.0], [0.0, 0.25]]
    """
    matrix = op.as_matrix()
    # A⁻¹
    return jnp.linalg.inv(matrix)


def dense_diag(op: lx.AbstractLinearOperator) -> Float[Array, " n"]:
    """Diagonal via dense materialization.

        [A]_ii

    Args:
        op: The operator A, materialised to a dense matrix.

    Returns:
        The main diagonal of A.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> operator = lx.MatrixLinearOperator(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        >>> dense_diag(operator).tolist()
        [1.0, 4.0]
    """
    matrix = op.as_matrix()
    # [A]_ii
    return jnp.diag(matrix)


def dense_trace(op: lx.AbstractLinearOperator) -> Float[Array, ""]:
    """Trace via dense materialization.

        tr A = ∑ᵢ [A]_ii

    Args:
        op: The operator A, materialised to a dense matrix.

    Returns:
        The trace of A.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> operator = lx.MatrixLinearOperator(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        >>> dense_trace(operator).tolist()
        5.0
    """
    matrix = op.as_matrix()
    # tr A = ∑ᵢ [A]_ii
    return jnp.trace(matrix)


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
    """Assert empirical moments match their population values.

        𝕍[x̄ᵢ] = [Σ_x]_ii / S
        𝕍[[Σ̂_x]_ij] ≈ ([Σ_x]_ii [Σ_x]_jj + [Σ_x]_ij²) / S

    Bounds each entry by ``n_sigma`` standard deviations of *that
    estimator's own sampling distribution*, rather than by a fixed
    absolute tolerance, for S draws from 𝒩(μ_x, Σ_x) with sample mean x̄ and
    sample covariance Σ̂_x.

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

    Algorithm:
        1. Compute x̄ and Σ̂_x from the draws.
        2. Scale each entry's standard deviation by ``n_sigma`` to get its band.
        3. Assert that no entry of x̄ − μ_x or Σ̂_x − Σ_x leaves its band.

    Args:
        samples: The S draws, one per row.
        mean: Population mean μ_x.
        cov: Population covariance Σ_x.
        n_sigma: Width of the acceptance band, in standard deviations of
            the estimator.

    Raises:
        AssertionError: If any empirical moment falls outside its band.

    Examples:
        Draws from 𝒩(0, I) pass, and a wrong mean does not:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> samples = jr.normal(jr.key(0), (2000, 2))
        >>> assert_sample_moments(samples, jnp.zeros(2), jnp.eye(2))
        >>> wrong_mean = jnp.ones(2)
        >>> assert_sample_moments(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     samples, wrong_mean, jnp.eye(2)
        ... )
        Traceback (most recent call last):
        ...
        AssertionError: sample mean outside its 7.0-sigma band by 0.8
    """
    # axis=, .T and [:, None], not einx: one marginal of shape (S,) passes here,
    # and a zero-length axis, which einx rejects, reaches the assertions below.
    n_draws = samples.shape[0]
    # x̄
    sample_mean = jnp.mean(samples, axis=0)
    # Σ̂_x
    sample_cov = jnp.cov(samples.T)

    # [Σ_x]_ii
    variances = jnp.diag(cov)
    # n_sigma √𝕍[x̄ᵢ]
    mean_variance = variances / n_draws
    mean_band = n_sigma * jnp.sqrt(mean_variance)
    # [Σ_x]_ii [Σ_x]_jj
    variance_outer = variances[:, None] * variances[None, :]
    # 𝕍[[Σ̂_x]_ij] ≈ ([Σ_x]_ii [Σ_x]_jj + [Σ_x]_ij²) / S
    cov_variance = (variance_outer + cov**2) / n_draws
    # n_sigma √𝕍[[Σ̂_x]_ij]
    cov_band = n_sigma * jnp.sqrt(cov_variance)

    # max |x̄ᵢ − μ_x,i| − n_sigma √𝕍[x̄ᵢ]
    mean_gap = jnp.abs(sample_mean - mean)
    mean_excess = jnp.max(mean_gap - mean_band)
    assert mean_excess <= 0.0, (
        f"sample mean outside its {n_sigma}-sigma band by {float(mean_excess):.3g}"
    )

    # max |[Σ̂_x]_ij − [Σ_x]_ij| − n_sigma √𝕍[[Σ̂_x]_ij]
    cov_gap = jnp.abs(sample_cov - cov)
    cov_excess = jnp.max(cov_gap - cov_band)
    assert cov_excess <= 0.0, (
        f"sample covariance outside its {n_sigma}-sigma band by {float(cov_excess):.3g}"
    )
