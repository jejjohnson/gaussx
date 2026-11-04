"""Structure-aware sampling from multivariate normal distributions.

`sample_mvn` draws x = μ + L z from 𝒩(μ, Σ), with standard normal noise
z ~ 𝒩(0, I) and a factor L Lᵀ = Σ chosen from the structure of the
covariance operator Σ, so a sample never costs more than the structure
demands. D_x is the event size. `sample_joint_conditional` draws a
partitioned Gaussian block by block, with the free entries indexed by A and
the observed ones by B, as in `gaussx.conditional`.
"""

from __future__ import annotations

import math
from typing import Any

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._linalg._symmetrize import symmetrize
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import (
    KroneckerSum,
    _from_eigenbasis,
    _kronecker_sum_root_tangent,
    _to_eigenbasis,
)
from gaussx._operators._low_rank_update import (
    LowRankUpdate,
    _safe_query,
)
from gaussx._operators._toeplitz import Toeplitz, _circulant_embedding, toeplitz_sample
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._solve import solve
from gaussx._primitives._sqrt import dense_symmetric_sqrt
from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._strategies._dispatch import dispatch_solve


def sample_mvn(
    mean: Float[Array, "*batch N"],
    covariance: lx.AbstractLinearOperator,
    *,
    key: jax.Array,
    num_samples: int = 1,
) -> Float[Array, "num_samples *batch N"]:
    """Draw samples from 𝒩(μ, Σ), dispatching on the structure of Σ.

        x = μ + L z,   z ~ 𝒩(0, I),   L Lᵀ = Σ

    Every branch draws x *exactly* (no branch approximates the covariance),
    and only the factor L changes:

    | Covariance Σ | Factor | Cost |
    |---|---|---|
    | `lineax.DiagonalLinearOperator` | σ ⊙ z | O(D_x) |
    | `gaussx.BlockDiag` | each block by its own rule | per block |
    | `gaussx.Kronecker` | per-factor eigendecomposition | O(N_1³ + … + N_D³) |
    | `gaussx.KroneckerSum` | per-factor eigendecomposition | O(N_1³ + N_2³) |
    | `gaussx.BlockTriDiag` | block-banded Cholesky | O(B d³) |
    | `gaussx.Toeplitz` | FFT circulant embedding | O(D_x log D_x) |
    | `gaussx.LowRankUpdate` | L_A₀ z₁ + U D^½ z₂ | A₀'s cost + O(D_x k) |
    | anything else | dense symmetric square root Σ^½ | O(D_x³) |

    Here a diagonal is Σ = diag(σ²); a Kronecker product A_1 ⊗ … ⊗ A_D has
    factors of sizes N_1, …, N_D and a Kronecker sum A_1 ⊕ A_2 two; a
    block-tridiagonal Σ has B diagonal blocks of size d; and a low-rank
    update A₀ + U D Uᵀ has rank k and L_A₀ L_A₀ᵀ = A₀.

    `lineax.TaggedLinearOperator` wrappers are looked through, and a scalar
    multiple c Σ (with c > 0) samples Σ and rescales by √c, so wrapping
    never costs the structure underneath.

    The Kronecker branches work in the factors' eigenbases, where the
    eigenvalues of the whole operator are products (or sums) of the
    factors' ones. That keeps them exact when a factor is indefinite but
    the product is not, as in (−A_1) ⊗ (−A_2), and when the covariance is
    only semi-definite. The dense fallback is the symmetric square root for
    the same reason: a Cholesky factor of a singular covariance is ``NaN``.

    A `gaussx.SumOfKroneckers` takes the dense fallback: its matrix-free
    square root, `gaussx.sum_of_kroneckers_sample`, is a truncated Lanczos
    approximation, so it stays an explicit opt-in rather than a default.

    Some structured routes hold only for some values, and each takes the
    dense square root otherwise, so every covariance the operator can
    represent is sampled exactly and pathwise gradients stay finite:

    - a scalar multiple c Σ needs c > 0;
    - a `gaussx.Toeplitz` needs a circulant embedding (2, 4, 8 or 16 times
      D_x) whose spectrum is strictly positive: a zero eigenvalue there
      would put √0 in the pathwise derivative;
    - a `gaussx.LowRankUpdate` A₀ + U D Vᵀ needs V = U established (the
      same array, or equal values), a base A₀ that is a covariance in its
      own right (tagged positive semi-definite, or a strictly positive
      diagonal), and every weight in D positive: a zero weight would again
      put √0 in the derivative.

    When those values are only known at run time (under `jax.jit`), the
    choice is a `jax.lax.cond`, which runs only the branch it takes; under
    `jax.vmap` it evaluates both.

    Args:
        mean: Mean μ, with leading batch axes. Each batch element gets
            independent noise; the covariance is shared.
        covariance: Positive-definite covariance Σ, D_x × D_x.
        key: PRNG key.
        num_samples: Number of independent draws.

    Returns:
        Samples x of shape ``(num_samples, *batch, N)``.

    Raises:
        ValueError: If ``covariance`` is not square, does not match the last
            axis of ``mean``, or ``num_samples`` is below one.

    Examples:
        Five draws from a diagonal covariance:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.array([1.0, 4.0, 9.0]))
        >>> samples = gaussx.sample_mvn(
        ...     jnp.zeros(3), covariance, key=jax.random.key(0), num_samples=5
        ... )
        >>> samples.shape
        (5, 3)

        A batch of means shares the covariance:

        >>> means = jnp.zeros((4, 3))
        >>> batch = gaussx.sample_mvn(
        ...     means, covariance, key=jax.random.key(1), num_samples=2
        ... )
        >>> batch.shape
        (2, 4, 3)

        A zero covariance returns the mean:

        >>> zero = lx.DiagonalLinearOperator(jnp.zeros(3))
        >>> mean = jnp.array([1.0, 2.0, 3.0])
        >>> draws = gaussx.sample_mvn(mean, zero, key=jax.random.key(2))
        >>> bool(jnp.allclose(draws, mean))
        True
    """
    mean = jnp.asarray(mean)
    if covariance.in_size() != covariance.out_size():
        raise ValueError(
            "sample_mvn requires a square covariance, got "
            f"{covariance.out_size()}x{covariance.in_size()}."
        )
    if mean.ndim < 1 or mean.shape[-1] != covariance.in_size():
        raise ValueError(
            f"mean must have shape (*batch, {covariance.in_size()}), got {mean.shape}."
        )
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}.")

    batch_shape = mean.shape[:-1]
    num_draws = num_samples * math.prod(batch_shape)
    if num_draws == 0:
        # An empty batch: nothing to draw, but the output keeps its shape.
        # The represented dtype, which `in_structure` can understate (a
        # LowRankUpdate reports its base's). Abstract, so nothing is formed.
        matrix_dtype = jax.eval_shape(covariance.as_matrix).dtype
        dtype = jnp.result_type(mean, matrix_dtype)
        return jnp.zeros((num_samples, *mean.shape), dtype=dtype)
    # L z, one row per draw
    draws = _zero_mean_draws(covariance, key, num_draws)

    batch_axes = [f"b{index}" for index in range(len(batch_shape))]
    batch_pattern = " ".join(batch_axes)
    pattern = f"(s {batch_pattern}) n -> s {batch_pattern} n"
    batch_sizes = dict(zip(batch_axes, batch_shape, strict=True))
    # L z, one slab per sample, batch axes restored
    draws = rearrange(draws, pattern, **batch_sizes)
    dtype = jnp.result_type(mean, draws)
    # x = μ + L z
    return mean.astype(dtype) + draws.astype(dtype)


def _zero_mean_draws(
    covariance: lx.AbstractLinearOperator,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "S N"]:
    """Exact zero-mean draws L z from 𝒩(0, Σ), stacked row-wise.

        L z,   z ~ 𝒩(0, I),   L Lᵀ = Σ

    Algorithm:
        1. Look through tags; sample a scalar multiple Σ = c Σ₀ as √c L₀ z
           when c > 0, and Σ = Σ₀ / c as L₀ z / √c.
        2. Dispatch on the operator class to its structured factor (the
           table in `sample_mvn`).
        3. Fall back to the dense symmetric square root Σ^½.

    Args:
        covariance: Covariance Σ.
        key: PRNG key.
        num_draws: Number of draws.

    Returns:
        The draws L z, one row per draw.
    """
    if isinstance(covariance, lx.TaggedLinearOperator):
        return _zero_mean_draws(covariance.operator, key, num_draws)
    if isinstance(covariance, lx.MulLinearOperator | lx.DivLinearOperator):
        scalar = covariance.scalar

        def scaled() -> Float[Array, "S N"]:
            # L₀ z with L₀ L₀ᵀ = Σ₀
            inner = _zero_mean_draws(covariance.operator, key, num_draws)
            # √c
            root_scalar = jnp.sqrt(scalar)
            if isinstance(covariance, lx.MulLinearOperator):
                # √c L₀ z
                return root_scalar * inner
            # L₀ z / √c
            return inner / root_scalar

        # cK is only sampleable through K when c > 0; (-1)(-I) = I is not.
        return _structured_or_dense(scalar > 0, scaled, covariance, key, num_draws)
    if isinstance(covariance, lx.IdentityLinearOperator | lx.DiagonalLinearOperator):
        # L = diag(σ)
        factor = cholesky(covariance)
        return _factor_draws(factor, covariance, key, num_draws)
    if isinstance(covariance, BlockDiag) and _all_square(covariance.operators):
        keys = jr.split(key, len(covariance.operators))
        # L_b z_b, one block at a time
        block_draws = [
            _zero_mean_draws(block, block_key, num_draws)
            for block, block_key in zip(covariance.operators, keys, strict=True)
        ]
        # jnp, not einx: a block may be empty, and einx rejects a zero axis.
        # [L_1 z_1, …, L_B z_B]
        return jnp.concatenate(block_draws, axis=1)
    if isinstance(covariance, Kronecker) and _all_square(covariance.operators):
        return _kronecker_draws(covariance, key, num_draws)
    if isinstance(covariance, KroneckerSum):
        return _kronecker_sum_draws(covariance, key, num_draws)
    if isinstance(covariance, BlockTriDiag):
        return _block_tridiag_draws(covariance, key, num_draws)
    if isinstance(covariance, Toeplitz):
        return _toeplitz_draws(covariance, key, num_draws)
    if isinstance(covariance, LowRankUpdate) and covariance.symmetric_factors:
        return _structured_or_dense(
            _low_rank_is_sampleable(covariance),
            lambda: _low_rank_draws(covariance, key, num_draws),
            covariance,
            key,
            num_draws,
        )

    return _dense_draws(covariance, key, num_draws)


def _structured_or_dense(
    valid,
    structured,
    covariance: lx.AbstractLinearOperator,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "S N"]:
    """``structured()`` when ``valid`` holds, else the dense square root.

    ``valid`` may be a Python bool or a JAX boolean. Concrete, only one route
    is traced; traced, the choice is a ``lax.cond``.

    Args:
        valid: Whether the structured route is exact for these values.
        structured: Zero-argument callable returning the structured draws.
        covariance: Covariance Σ, for the dense fallback.
        key: PRNG key, for the dense fallback.
        num_draws: Number of draws.

    Returns:
        The draws L z, one row per draw.
    """
    try:
        known = bool(valid)
    except jax.errors.TracerBoolConversionError:
        return jax.lax.cond(
            valid, structured, lambda: _dense_draws(covariance, key, num_draws)
        )
    return structured() if known else _dense_draws(covariance, key, num_draws)


def _dense_draws(
    covariance: lx.AbstractLinearOperator,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "S N"]:
    """Draws Σ^½ z through the dense symmetric square root of the whole Σ.

    Args:
        covariance: Covariance Σ.
        key: PRNG key.
        num_draws: Number of draws.

    Returns:
        The draws Σ^½ z, one row per draw.
    """
    matrix = covariance.as_matrix()
    # Σ^½, symmetric and defined for a semi-definite Σ
    root_matrix = dense_symmetric_sqrt(matrix)
    root = lx.MatrixLinearOperator(root_matrix)
    return _factor_draws(root, covariance, key, num_draws)


def _factor_draws(
    factor: lx.AbstractLinearOperator,
    covariance: lx.AbstractLinearOperator,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "S N"]:
    """Draws L z for ``num_draws`` standard-normal z.

    Args:
        factor: Factor L, with L Lᵀ = Σ.
        covariance: Covariance Σ, which sets the size and dtype of z.
        key: PRNG key.
        num_draws: Number of draws.

    Returns:
        The draws L z, one row per draw.
    """
    dtype = covariance.in_structure().dtype
    # z ~ 𝒩(0, I), one row per draw
    noise = jr.normal(key, (num_draws, covariance.in_size()), dtype=dtype)
    # L z
    return jax.vmap(factor.mv)(noise)


def _all_square(operators: tuple[lx.AbstractLinearOperator, ...]) -> bool:
    """Whether every operator is square.

    Args:
        operators: Factors or blocks of a structured operator.

    Returns:
        ``True`` when each operator has as many rows as columns.
    """
    return all(op.in_size() == op.out_size() for op in operators)


def _kronecker_draws(
    covariance: Kronecker,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "S N"]:
    """Draws for Σ = A_1 ⊗ … ⊗ A_D through ⊗_d (sgn(tr A_d) A_d)^½.

        Σ^½ = (s_1 A_1)^½ ⊗ … ⊗ (s_D A_D)^½,   s_d = sgn(tr A_d)

    The Kronecker product of the factors' symmetric square roots is the
    symmetric square root of the product, and each factor's root carries
    ``dense_symmetric_sqrt``'s custom JVP, so pathwise gradients stay finite
    at repeated eigenvalues (an isotropic factor c I).

    A positive semi-definite product of square factors has every factor
    semi-definite, positive or negative, with the negative ones pairing up.
    Flipping each factor by the sign s_d of its trace makes it positive
    semi-definite without changing the product, so (−A_1) ⊗ (−A_2) is
    exact.

    Args:
        covariance: Kronecker covariance Σ = A_1 ⊗ … ⊗ A_D.
        key: PRNG key.
        num_draws: Number of draws.

    Returns:
        The draws Σ^½ z, one row per draw.
    """
    roots = []
    for factor in covariance.operators:
        diagonal = _base_diagonal(factor)
        if diagonal is not None:
            # tr A_d, a full reduction
            trace = jnp.sum(diagonal)
            # s_d = sgn(tr A_d)
            sign = jnp.where(trace < 0, -1.0, 1.0)
            # s_d diag(A_d), with round-off negatives clipped
            signed_diagonal = jnp.clip(sign * diagonal, 0.0, None)
            # (s_d A_d)^½
            root_diagonal = jnp.sqrt(signed_diagonal)
            roots.append(lx.DiagonalLinearOperator(root_diagonal))
            continue
        matrix = factor.as_matrix()
        # tr A_d
        trace = jnp.trace(matrix)
        # s_d = sgn(tr A_d)
        sign = jnp.where(trace < 0, -1.0, 1.0)
        # (s_d A_d)^½
        root_matrix = dense_symmetric_sqrt(sign * matrix)
        roots.append(lx.MatrixLinearOperator(root_matrix))
    # Σ^½ = ⊗_d (s_d A_d)^½
    root = Kronecker(*roots)
    return _factor_draws(root, covariance, key, num_draws)


def _kronecker_sum_draws(
    covariance: KroneckerSum,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "S N"]:
    """Draws S vec(Z) for Σ = A_1 ⊕ A_2 through its symmetric root, never formed.

        S² = A_1 ⊕ A_2,   Z ∈ ℝ^{N_1 × N_2},   vec(Z) ~ 𝒩(0, I)

    Args:
        covariance: Kronecker-sum covariance Σ = A_1 ⊕ A_2.
        key: PRNG key.
        num_draws: Number of draws.

    Returns:
        The draws S vec(Z), one row per draw.
    """
    dtype = covariance.in_structure().dtype
    size_a, size_b = covariance.A.in_size(), covariance.B.in_size()
    # Z ~ 𝒩(0, I), one N_1 × N_2 slab per draw
    noise = jr.normal(key, (num_draws, size_a, size_b), dtype=dtype)
    # S vec(Z), as one N_1 × N_2 slab per draw
    draws = _kronecker_sum_root_action(
        covariance.A.as_matrix(), covariance.B.as_matrix(), noise
    )
    # vec(·), row by row
    return rearrange(draws, "draw first second -> draw (first second)")


def _kronecker_sum_root_parts(a, b, noise):
    """Eigenbases, root spectrum and rotated noise for S vec(Z), S² = A_1 ⊕ A_2.

        A_d = Q_d diag(γ_d) Q_dᵀ,   s_ij = √(γ_1,i + γ_2,j),   Z̃ = Q_1ᵀ Z Q_2

    Args:
        a: First factor A_1, N_1 × N_1.
        b: Second factor A_2, N_2 × N_2.
        noise: Slabs Z, one N_1 × N_2 slab per draw.

    Returns:
        Tuple ``(basis_a, basis_b, roots, rotated)`` holding Q_1, Q_2, the
        root eigenvalues s_ij and the rotated slabs Z̃.
    """
    # A_1 = Q_1 diag(γ_1) Q_1ᵀ
    values_a, basis_a = jnp.linalg.eigh(a)
    # A_2 = Q_2 diag(γ_2) Q_2ᵀ
    values_b, basis_b = jnp.linalg.eigh(b)
    # γ_1,i + γ_2,j, the eigenvalues of A_1 ⊕ A_2
    eigenvalue_sums = einx.add("first, second -> first second", values_a, values_b)
    # Round-off negatives are clipped, which also covers a singular A ⊕ B.
    clipped_sums = jnp.clip(eigenvalue_sums, 0.0, None)
    # s_ij = √(γ_1,i + γ_2,j)
    roots = jnp.sqrt(clipped_sums)
    # Z̃ = Q_1ᵀ Z Q_2
    rotated = _to_eigenbasis(basis_a, basis_b, noise)
    return basis_a, basis_b, roots, rotated


@jax.custom_jvp
def _kronecker_sum_root_action(
    a: Float[Array, "A A"],
    b: Float[Array, "B B"],
    noise: Float[Array, "S A B"],
) -> Float[Array, "S A B"]:
    """Symmetric-root action S vec(Z) for each noise slab, in the eigenbases.

        A_1 ⊕ A_2 = (Q_1 ⊗ Q_2) diag(γ_1,i + γ_2,j) (Q_1 ⊗ Q_2)ᵀ
        S vec(Z) = vec(Q_1 (s ⊙ Z̃) Q_2ᵀ),   Z̃ = Q_1ᵀ Z Q_2

    The symmetric root S acts on the eigenbasis coordinates Z̃ entrywise.

    Args:
        a: First factor A_1, N_1 × N_1.
        b: Second factor A_2, N_2 × N_2.
        noise: Slabs Z, one N_1 × N_2 slab per draw.

    Returns:
        The slabs of S vec(Z), one N_1 × N_2 slab per draw.
    """
    basis_a, basis_b, roots, rotated = _kronecker_sum_root_parts(a, b, noise)
    # Q_1 (s ⊙ Z̃) Q_2ᵀ
    return _from_eigenbasis(basis_a, basis_b, roots * rotated)


@_kronecker_sum_root_action.defjvp
def _kronecker_sum_root_action_jvp(primals, tangents):
    """Sylvester derivative of S vec(Z) for S² = A_1 ⊕ A_2, kept structured.

        d(S vec(Z)) = vec(Q_1 (s ⊙ dZ̃ + dS̃ Z̃) Q_2ᵀ)

    See `_kronecker_sum_root_tangent` (shared with `KroneckerSumSqrt`): it
    divides by *sums* of root eigenvalues rather than eigenvalue gaps, so
    repeated eigenvalues are harmless, and never forms A_1 ⊕ A_2.

    Args:
        primals: Tuple ``(a, b, noise)`` holding A_1, A_2 and Z.
        tangents: Tuple of their tangents dA_1, dA_2 and dZ.

    Returns:
        Tuple ``(primal_out, tangent_out)`` holding S vec(Z) and its
        derivative, as slabs.
    """
    a, b, noise = primals
    tangent_a, tangent_b, tangent_noise = tangents
    basis_a, basis_b, roots, rotated = _kronecker_sum_root_parts(a, b, noise)
    # Q_1 (s ⊙ Z̃) Q_2ᵀ
    primal_out = _from_eigenbasis(basis_a, basis_b, roots * rotated)
    # dS̃ Z̃
    term = _kronecker_sum_root_tangent(
        basis_a, basis_b, roots, tangent_a, tangent_b, rotated
    )
    # dZ̃ = Q_1ᵀ dZ Q_2
    rotated_tangent_noise = _to_eigenbasis(basis_a, basis_b, tangent_noise)
    # Q_1 (s ⊙ dZ̃ + dS̃ Z̃) Q_2ᵀ
    tangent_out = _from_eigenbasis(
        basis_a, basis_b, roots * rotated_tangent_noise + term
    )
    return primal_out, tangent_out


def _block_tridiag_draws(
    covariance: BlockTriDiag,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "S N"]:
    """Banded-Cholesky draws, or the dense root when the covariance is singular.

    The block Cholesky needs every pivot positive and returns non-finite
    values at a zero one, so a singular (semi-definite) covariance switches to
    the dense symmetric square root. ``lax.cond`` evaluates only the branch it
    takes, so the dense root costs nothing for a definite covariance.

    Args:
        covariance: Block-tridiagonal covariance Σ.
        key: PRNG key.
        num_draws: Number of draws.

    Returns:
        The draws L z, one row per draw.
    """
    # L Lᵀ = Σ, lower block-bidiagonal
    factor = cholesky(covariance)
    # L z
    draws = _factor_draws(factor, covariance, key, num_draws)
    # every pivot was positive
    all_finite = jnp.all(jnp.isfinite(draws))
    return jax.lax.cond(
        all_finite,
        lambda: draws,
        lambda: _dense_draws(covariance, key, num_draws),
    )


# Circulant embedding sizes tried for a Toeplitz covariance, as multiples of N.
_EMBEDDING_FACTORS = (2, 4, 8, 16)


def _toeplitz_draws(
    covariance: Toeplitz,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "S N"]:
    """Circulant-embedding draws, or the dense root when no embedding works.

    An embedding works when its spectrum γ = DFT(c) is strictly positive: a
    negative eigenvalue fails Wood--Chan, and a zero one leaves the draw
    correct but its pathwise derivative infinite. A concrete column takes
    the smallest embedding that works; a traced one checks the 2x embedding
    at run time.

    Args:
        covariance: Toeplitz covariance Σ = toeplitz(t).
        key: PRNG key.
        num_draws: Number of draws.

    Returns:
        The draws L z, one row per draw.
    """
    column = jnp.asarray(covariance.column)
    float_dtype = jnp.result_type(column.dtype, jnp.float32)
    # t, in a floating dtype
    column = column.astype(float_dtype)
    for embedding_factor in _EMBEDDING_FACTORS:
        # c, the first column of the circulant embedding
        embedding = _circulant_embedding(column, embedding_factor=embedding_factor)
        # γ = DFT(c), real for a symmetric embedding
        spectrum = jnp.fft.rfft(embedding).real
        # |γ|
        spectrum_magnitude = jnp.abs(spectrum)
        # max |γ|, a full reduction
        largest = jnp.max(spectrum_magnitude)
        scale = jnp.maximum(1.0, largest)
        # Round-off headroom as in `toeplitz_sample`, but on the positive side.
        threshold = 100 * jnp.finfo(column.dtype).eps * scale
        # γ ≻ 0, with headroom
        positive = jnp.all(spectrum > threshold)

        def circulant(embedding_factor: int = embedding_factor) -> Float[Array, "S N"]:
            return toeplitz_sample(
                column,
                key=key,
                num_samples=num_draws,
                embedding_factor=embedding_factor,
            )

        try:
            if bool(positive):
                return circulant()
        except jax.errors.TracerBoolConversionError:
            return _structured_or_dense(positive, circulant, covariance, key, num_draws)
    return _dense_draws(covariance, key, num_draws)


def _low_rank_is_sampleable(covariance: LowRankUpdate):
    """Whether A₀ + U D Uᵀ samples as L_A₀ z₁ + U D^½ z₂ with finite gradients.

    V = U is checked by the caller. Needs strictly positive weights and a
    base A₀ that is a covariance itself: tagged positive semi-definite, or a
    strictly positive diagonal.

    Args:
        covariance: Low-rank update Σ = A₀ + U D Uᵀ.

    Returns:
        A Python bool when that is known, and a JAX boolean when only the
        values' signs are left to check.
    """
    # every weight in D positive
    valid = jnp.all(covariance.d > 0)
    diagonal = _base_diagonal(covariance.base)
    if diagonal is not None:
        # a strictly positive diagonal A₀
        return valid & jnp.all(diagonal > 0)
    if not _safe_query(lx.is_positive_semidefinite, covariance.base):
        return False
    return valid


def _base_diagonal(base: lx.AbstractLinearOperator) -> Array | None:
    """The diagonal of a (tagged) diagonal operator, else ``None``.

    Args:
        base: Operator to inspect, looking through tags.

    Returns:
        Its diagonal, or ``None`` when it is not diagonal.
    """
    while isinstance(base, lx.TaggedLinearOperator):
        base = base.operator
    if isinstance(base, lx.DiagonalLinearOperator):
        return lx.diagonal(base)
    return None


def _low_rank_draws(
    covariance: LowRankUpdate,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "S N"]:
    """Sample Σ = A₀ + U D Uᵀ as a sum of two independent draws.

        L_A₀ z₁ + U D^½ z₂ ~ 𝒩(0, A₀ + U D Uᵀ)

    The base A₀ is sampled through `_zero_mean_draws`, so its own structure
    is kept.

    Args:
        covariance: Low-rank update Σ = A₀ + U D Uᵀ.
        key: PRNG key.
        num_draws: Number of draws.

    Returns:
        The draws, one row per draw.
    """
    base_key, update_key = jr.split(key)
    # L_A₀ z₁
    base = _zero_mean_draws(covariance.base, base_key, num_draws)
    # z₂ ~ 𝒩(0, I_k), one row per draw
    noise = jr.normal(
        update_key, (num_draws, covariance.rank), dtype=covariance.U.dtype
    )
    # D^½
    root_weights = jnp.sqrt(covariance.d)
    # D^½ z₂
    scaled = einx.multiply("draw rank, rank -> draw rank", noise, root_weights)
    # U D^½ z₂
    update = einsum(scaled, covariance.U, "draw rank, row rank -> draw row")
    # L_A₀ z₁ + U D^½ z₂
    return base + update


def sample_joint_conditional(
    joint_mean: tuple[Float[Array, " N"], Float[Array, " M"]],
    joint_covariance: dict[str, lx.AbstractLinearOperator],
    *,
    key: jax.Array,
    observed_index: int = 1,
    observed_value: Float[Array, " K"] | None = None,
    num_samples: int = 1,
    solver: AbstractSolveStrategy | None = None,
) -> dict[str, Any]:
    """Joint draws of a partitioned Gaussian, and conditional draws from them.

        [x_a; x_b] ~ 𝒩([μ_a; μ_b], [[Σ_aa, Σ_ab], [Σ_ba, Σ_bb]]),   Σ_ba = Σ_abᵀ

    As in `gaussx.conditional`, B indexes the observed block and A the free
    one: with ``observed_index=1`` (the default) A = a and B = b, and with
    ``observed_index=0`` A = b and B = a. The joint draw uses the block
    factorisation

        x_B = μ_B + L_B z₁,   L_B L_Bᵀ = Σ_BB
        x_A = μ_A + Σ_AB Σ_BB⁻¹ (x_B − μ_B) + Σ_A|B^½ z₂
        Σ_A|B = Σ_AA − Σ_AB Σ_BB⁻¹ Σ_BA

    with x_B drawn by `gaussx.sample_mvn`, so the observed block keeps its
    structure, and Σ_A|B, the Schur complement of Σ_BB, the conditional
    covariance. The joint covariance, of size (N + M) × (N + M), is never
    formed.

    When ``observed_value`` β is given, the conditional draws of
    x_A ∣ x_B = β reuse the same z₂:

        x_A|β = x_A + Σ_AB Σ_BB⁻¹ (β − x_B)
              = μ_A + Σ_AB Σ_BB⁻¹ (β − μ_B) + Σ_A|B^½ z₂

    which is exactly `gaussx.matheron_update` applied to the joint draws.

    The Schur complement is dense in general: Σ_AB Σ_BB⁻¹ Σ_BA is dense
    whatever the structure of Σ_AA, so Σ_A|B is factorised at O(N_A³)
    plus N_A solves against Σ_BB, with N_A = |A|. Its square root is the
    symmetric one from ``dense_symmetric_sqrt``, with round-off negative
    eigenvalues clipped. That tolerates the rank deficiency of free points
    that coincide with observed ones (a Cholesky would return NaNs there),
    and its custom JVP keeps gradients finite when Σ_A|B has repeated
    eigenvalues.

    Algorithm:
        1. Draw the observed deviations x_B − μ_B = L_B z₁ with
           `gaussx.sample_mvn`.
        2. Form Σ_A|B and its symmetric square root Σ_A|B^½.
        3. Apply the gain G = Σ_AB Σ_BB⁻¹ to the deviations, through the
           pseudo-inverse Σ_BB⁺ when Σ_BB is singular, and add Σ_A|B^½ z₂.
        4. When β is given, apply Matheron's update to the joint draws.

    Args:
        joint_mean: Block means ``(μ_a, μ_b)``.
        joint_covariance: Blocks ``{"aa": Σ_aa, "ab": Σ_ab, "bb": Σ_bb}``;
            Σ_ba is ``Σ_ab.T``.
        key: PRNG key.
        observed_index: Which block is observed: ``1`` for x_b (sample
            x_a ∣ x_b), ``0`` for x_a.
        observed_value: Optional value β of the observed block x_B. When
            given, the result also holds conditional draws.
        num_samples: Number of independent draws.
        solver: Optional solver strategy for the solves against Σ_BB
            (e.g. `gaussx.CGSolver`). When ``None``, routes through
            structural dispatch, as `gaussx.matheron_update` does.

    Returns:
        ``{"joint": (samples_a, samples_b)}`` with shapes
        ``(num_samples, N)`` and ``(num_samples, M)``, plus
        ``"conditional"``, one draw of x_A ∣ x_B = β per row, when
        ``observed_value`` is given.

    Raises:
        ValueError: On a missing covariance block, mismatched shapes,
            ``observed_index`` outside ``{0, 1}`` or ``num_samples`` below
            one.

    Examples:
        Condition two entries on a third:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> cross = jnp.array([[1.0], [0.5]])
        >>> blocks = {
        ...     "aa": lx.MatrixLinearOperator(2.0 * jnp.eye(2), psd),
        ...     "ab": lx.MatrixLinearOperator(cross),
        ...     "bb": lx.MatrixLinearOperator(jnp.eye(1), psd),
        ... }
        >>> means = (jnp.zeros(2), jnp.zeros(1))
        >>> result = gaussx.sample_joint_conditional(
        ...     means,
        ...     blocks,
        ...     key=jax.random.key(0),
        ...     observed_value=jnp.array([0.5]),
        ...     num_samples=3,
        ... )
        >>> samples_a, samples_b = result["joint"]
        >>> samples_a.shape, samples_b.shape, result["conditional"].shape
        ((3, 2), (3, 1), (3, 2))

        The conditional draws are Matheron's update of the joint ones (here
        Σ_bb = 1, so the gain is Σ_ab):

        >>> expected = samples_a + (0.5 - samples_b) * cross[:, 0]
        >>> bool(jnp.allclose(result["conditional"], expected, atol=1e-5))
        True

        Without an observed value there are only joint draws:

        >>> flipped = gaussx.sample_joint_conditional(
        ...     means, blocks, key=jax.random.key(1), observed_index=0
        ... )
        >>> sorted(flipped)
        ['joint']
    """
    mean_a, mean_b = (jnp.asarray(mean) for mean in joint_mean)
    covariance_aa, covariance_ab, covariance_bb = _covariance_blocks(
        joint_covariance, mean_a.shape, mean_b.shape
    )
    if observed_index not in (0, 1):
        raise ValueError(f"observed_index must be 0 or 1, got {observed_index}.")
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}.")

    # --- index sets ---
    if observed_index == 1:
        # B = b, A = a
        observed_mean, free_mean = mean_b, mean_a
        observed_covariance, free_covariance = covariance_bb, covariance_aa
        # Σ_AB = Σ_ab
        cross_covariance = covariance_ab
    else:
        # B = a, A = b
        observed_mean, free_mean = mean_a, mean_b
        observed_covariance, free_covariance = covariance_aa, covariance_bb
        # Σ_AB = Σ_ba = Σ_abᵀ, a lazy operator transpose
        cross_covariance = covariance_ab.T

    if observed_value is not None:
        observed_value = jnp.asarray(observed_value)
        if observed_value.shape != observed_mean.shape:
            raise ValueError(
                f"observed_value must have shape {observed_mean.shape}, got "
                f"{observed_value.shape}."
            )

    if solver is None:
        solve_observed = lambda vector: solve(observed_covariance, vector)
    else:
        solve_observed = lambda vector: dispatch_solve(
            observed_covariance, vector, solver
        )

    def gain(deviations: Float[Array, "S K"]) -> Float[Array, "S T"]:
        """Apply the gain G = Σ_AB Σ_BB⁺ to each row.

        The structured solve is used whenever it succeeds. A singular (only
        semi-definite) Σ_BB makes it return non-finite values, and then the
        batch is redone with the pseudo-inverse Σ_BB⁺: for a valid joint
        covariance both the deviations and the rows of Σ_AB lie in the
        range of Σ_BB, where the pseudo-inverse is exact. ``lax.cond``
        evaluates only the branch it takes.

        Args:
            deviations: Rows d to map, each in the observed space.

        Returns:
            The rows G d, each in the free space.
        """
        # Σ_BB⁻¹ d, one row at a time
        solved = jax.vmap(solve_observed)(deviations)
        all_finite = jnp.all(jnp.isfinite(solved))
        # Σ_BB⁺ d when the structured solve failed
        solved = jax.lax.cond(
            all_finite,
            lambda: solved,
            lambda: _pseudo_inverse_solve(observed_covariance, deviations),
        )
        # G d = Σ_AB Σ_BB⁺ d
        return jax.vmap(cross_covariance.mv)(solved)

    # --- joint draws ---
    observed_key, schur_key = jr.split(key)
    # x_B − μ_B = L_B z₁
    observed_deviations = sample_mvn(
        jnp.zeros_like(observed_mean),
        observed_covariance,
        key=observed_key,
        num_samples=num_samples,
    )
    dtype = observed_deviations.dtype
    # Σ_A|B^½
    schur_root = _schur_root(free_covariance, cross_covariance, gain)
    # z₂ ~ 𝒩(0, I), one row per draw
    noise = jr.normal(schur_key, (num_samples, free_mean.shape[0]), dtype=dtype)
    # Σ_A|B^½ z₂
    conditional_deviations = einsum(noise, schur_root, "draw col, row col -> draw row")

    # x_B = μ_B + L_B z₁
    observed_samples = observed_mean + observed_deviations
    # Σ_AB Σ_BB⁻¹ (x_B − μ_B)
    gained_deviations = gain(observed_deviations)
    # x_A = μ_A + Σ_AB Σ_BB⁻¹ (x_B − μ_B) + Σ_A|B^½ z₂
    free_samples = free_mean + gained_deviations + conditional_deviations
    if observed_index == 1:
        joint = (free_samples, observed_samples)
    else:
        joint = (observed_samples, free_samples)
    result: dict[str, Any] = {"joint": joint}

    # --- conditional draws ---
    if observed_value is not None:
        # Matheron's update of the returned joint draws, literally: with a
        # finite-tolerance solver the gain is not additive across right-hand
        # sides, so shifting from the mean instead would drift from it.
        # β − x_B
        observed_residual = observed_value - observed_samples
        # x_A|β = x_A + Σ_AB Σ_BB⁻¹ (β − x_B)
        result["conditional"] = free_samples + gain(observed_residual)
    return result


def _pseudo_inverse_solve(
    covariance: lx.AbstractLinearOperator,
    rows: Float[Array, "S K"],
) -> Float[Array, "S K"]:
    """Σ⁺ r for each row r, through the Hermitian pseudo-inverse of Σ.

    Args:
        covariance: Symmetric positive semi-definite Σ, possibly singular.
        rows: Rows r to solve for.

    Returns:
        The rows Σ⁺ r.
    """
    matrix = covariance.as_matrix()
    # Σ⁺
    inverse = jnp.linalg.pinv(matrix, hermitian=True)
    # Σ⁺ r, one row at a time
    return einsum(rows, inverse, "draw col, row col -> draw row")


def _covariance_blocks(
    joint_covariance: dict[str, lx.AbstractLinearOperator],
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
) -> tuple[
    lx.AbstractLinearOperator, lx.AbstractLinearOperator, lx.AbstractLinearOperator
]:
    """Validate and unpack the ``{"aa", "ab", "bb"}`` covariance blocks.

    Args:
        joint_covariance: Blocks ``{"aa": Σ_aa, "ab": Σ_ab, "bb": Σ_bb}``.
        shape_a: Shape of μ_a.
        shape_b: Shape of μ_b.

    Returns:
        Tuple ``(Σ_aa, Σ_ab, Σ_bb)``.

    Raises:
        ValueError: On a missing block, a non-vector mean or a block of the
            wrong shape.
    """
    missing = {"aa", "ab", "bb"} - set(joint_covariance)
    if missing:
        raise ValueError(
            f"joint_covariance is missing block(s) {sorted(missing)}; expected "
            '"aa", "ab" and "bb".'
        )
    if len(shape_a) != 1 or len(shape_b) != 1:
        raise ValueError("joint_mean entries must both be vectors.")
    (n,), (m,) = shape_a, shape_b
    blocks = joint_covariance["aa"], joint_covariance["ab"], joint_covariance["bb"]
    expected = ((n, n), (n, m), (m, m))
    for name, block, (rows, columns) in zip(
        ("aa", "ab", "bb"), blocks, expected, strict=True
    ):
        if (block.out_size(), block.in_size()) != (rows, columns):
            raise ValueError(
                f'joint_covariance["{name}"] must have shape ({rows}, {columns}), '
                f"got ({block.out_size()}, {block.in_size()})."
            )
    return blocks


def _schur_root(
    free_covariance: lx.AbstractLinearOperator,
    cross_covariance: lx.AbstractLinearOperator,
    gain,
) -> Float[Array, "T T"]:
    """The symmetric square root Σ_A|B^½ of the Schur complement of Σ_BB.

        Σ_A|B = Σ_AA − Σ_AB Σ_BB⁻¹ Σ_BA

    Uses ``dense_symmetric_sqrt`` rather than a bare ``eigh``: its custom
    JVP stays finite when Σ_A|B has repeated eigenvalues (an isotropic
    conditional covariance is nothing else), where differentiating through
    ``eigh``'s eigenvectors gives NaN.

    Args:
        free_covariance: Free block Σ_AA.
        cross_covariance: Cross block Σ_AB.
        gain: Callable applying G = Σ_AB Σ_BB⁻¹ to each row.

    Returns:
        The dense symmetric root Σ_A|B^½.
    """
    # Row i of Σ_AB is column i of Σ_BA, so the gain of the rows of Σ_AB
    # gives the rows of Σ_AB Σ_BB⁻¹ Σ_BA (a symmetric matrix).
    cross_matrix = cross_covariance.as_matrix()
    # Σ_AB Σ_BB⁻¹ Σ_BA
    correction = gain(cross_matrix)
    # Σ_A|B = Σ_AA − Σ_AB Σ_BB⁻¹ Σ_BA
    schur = free_covariance.as_matrix() - correction
    # ½ (Σ_A|B + Σ_A|Bᵀ), removing round-off asymmetry
    symmetric_schur = symmetrize(schur)
    # Σ_A|B^½
    return dense_symmetric_sqrt(symmetric_schur)
