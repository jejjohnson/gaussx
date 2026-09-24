r"""Structure-aware sampling from multivariate normal distributions.

`sample_mvn` draws $x = \mu + L\varepsilon$ with $LL^{\top} = K$, choosing the
factor $L$ from the structure of the covariance operator so a sample never
costs more than the structure demands.
"""

from __future__ import annotations

import functools as ft
import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum, _eigh_factor
from gaussx._operators._low_rank_update import (
    LowRankUpdate,
    _arrays_match,
    _is_nonnegative,
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
    r"""Draw samples from $\mathcal{N}(\mu, K)$, dispatching on $K$'s structure.

    Every branch draws $x = \mu + L\varepsilon$ with $\varepsilon \sim
    \mathcal{N}(0, I)$ and $LL^{\top} = K$ *exactly* -- no branch
    approximates the covariance -- and only the factor $L$ changes:

    | Covariance $K$ | Factor | Cost |
    |---|---|---|
    | `lineax.DiagonalLinearOperator` | $\sqrt{d} \odot z$ | $O(N)$ |
    | `gaussx.BlockDiag` | each block by its own rule | per block |
    | `gaussx.Kronecker` | per-factor eigendecomposition | $O(\sum_i N_i^3)$ |
    | `gaussx.KroneckerSum` | per-factor eigendecomposition | $O(n_A^3 + n_B^3)$ |
    | `gaussx.BlockTriDiag` | block-banded Cholesky | $O(T d^3)$ |
    | `gaussx.Toeplitz` | FFT circulant embedding | $O(N \log N)$ |
    | `gaussx.LowRankUpdate` | $L_B z_1 + U\sqrt{D} z_2$ | $B$'s cost $+ O(Nr)$ |
    | anything else | dense symmetric square root | $O(N^3)$ |

    `lineax.TaggedLinearOperator` wrappers are looked through, and a scalar
    multiple $cK$ (with $c > 0$) samples $K$ and rescales by $\sqrt{c}$, so
    wrapping never costs the structure underneath.

    The Kronecker branches work in the factors' eigenbases, where the
    eigenvalues of the whole operator are products (or sums) of the factors'
    ones. That keeps them exact when a factor is indefinite but the product
    is not -- ``(-A) \otimes (-B)`` -- and when the covariance is only
    semi-definite. The dense fallback is the symmetric square root for the
    same reason: a Cholesky factor of a singular covariance is ``NaN``.

    A `gaussx.SumOfKroneckers` takes the dense fallback: its matrix-free
    square root, `gaussx.sumkronecker_sample`, is a truncated Lanczos
    approximation, so it stays an explicit opt-in rather than a default.

    A `gaussx.Toeplitz` uses the smallest circulant embedding (2, 4, 8 or 16
    times ``N``) that passes the Wood--Chan condition, and the dense fallback
    when none does. With a traced column the choice cannot be made up front,
    so it uses a 2x embedding, which raises if the condition fails.

    A `gaussx.LowRankUpdate` $B + UDV^{\top}$ takes the low-rank branch only
    when $V = U$ is established (the same array, or equal values), the base
    $B$ is a covariance in its own right -- tagged positive semi-definite, or
    diagonal -- and $D \ge 0$. Anything known to fail those, such as a
    Woodbury downdate, falls back to the dense square root. Weights or a
    diagonal base that are only known at run time (under `jax.jit`) take the
    low-rank branch and raise if any entry is negative.

    Args:
        mean: Mean $\mu$ with shape ``(*batch, N)``. Each batch element gets
            independent noise; the covariance is shared.
        covariance: Positive-definite covariance operator $K$ of size
            ``(N, N)``.
        key: PRNG key.
        num_samples: Number of independent draws.

    Returns:
        Samples of shape ``(num_samples, *batch, N)``.

    Raises:
        ValueError: If ``covariance`` is not square, does not match the last
            axis of ``mean``, or ``num_samples`` is below one.
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
    draws = _zero_mean_draws(covariance, key, num_draws)

    axes = [f"b{index}" for index in range(len(batch_shape))]
    pattern = f"(s {' '.join(axes)}) n -> s {' '.join(axes)} n"
    draws = rearrange(draws, pattern, **dict(zip(axes, batch_shape, strict=True)))
    dtype = jnp.result_type(mean, draws)
    return mean.astype(dtype) + draws.astype(dtype)


def _zero_mean_draws(
    covariance: lx.AbstractLinearOperator,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "S N"]:
    """``num_draws`` exact samples from ``N(0, covariance)``, stacked row-wise."""
    if isinstance(covariance, lx.TaggedLinearOperator):
        return _zero_mean_draws(covariance.operator, key, num_draws)
    if isinstance(covariance, lx.MulLinearOperator):
        inner = _zero_mean_draws(covariance.operator, key, num_draws)
        return jnp.sqrt(covariance.scalar) * inner
    if isinstance(covariance, lx.DivLinearOperator):
        inner = _zero_mean_draws(covariance.operator, key, num_draws)
        return inner / jnp.sqrt(covariance.scalar)
    if isinstance(covariance, lx.IdentityLinearOperator | lx.DiagonalLinearOperator):
        return _factor_draws(cholesky(covariance), covariance, key, num_draws)
    if isinstance(covariance, BlockDiag):
        keys = jr.split(key, len(covariance.operators))
        return jnp.concatenate(
            [
                _zero_mean_draws(block, block_key, num_draws)
                for block, block_key in zip(covariance.operators, keys, strict=True)
            ],
            axis=1,
        )
    if isinstance(covariance, Kronecker):
        return _kronecker_draws(covariance.operators, key, num_draws, _product)
    if isinstance(covariance, KroneckerSum):
        return _kronecker_draws(
            (covariance.A, covariance.B), key, num_draws, _kronecker_sum
        )
    if isinstance(covariance, BlockTriDiag):
        return _factor_draws(cholesky(covariance), covariance, key, num_draws)
    if isinstance(covariance, Toeplitz):
        factor = _toeplitz_embedding_factor(covariance.column)
        if factor is not None:
            return toeplitz_sample(
                covariance.column,
                key=key,
                num_samples=num_draws,
                embedding_factor=factor,
            )
    if isinstance(covariance, LowRankUpdate) and _has_sampleable_update(covariance):
        return _low_rank_draws(covariance, key, num_draws)

    root = lx.MatrixLinearOperator(dense_symmetric_sqrt(covariance.as_matrix()))
    return _factor_draws(root, covariance, key, num_draws)


def _factor_draws(
    factor: lx.AbstractLinearOperator,
    covariance: lx.AbstractLinearOperator,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "S N"]:
    """``L z`` for ``num_draws`` standard-normal ``z``."""
    dtype = covariance.in_structure().dtype
    noise = jr.normal(key, (num_draws, covariance.in_size()), dtype=dtype)
    return jax.vmap(factor.mv)(noise)


def _product(left: Float[Array, " a"], right: Float[Array, " b"]) -> Array:
    return rearrange(einsum(left, right, "a, b -> a b"), "a b -> (a b)")


def _kronecker_sum(left: Float[Array, " a"], right: Float[Array, " b"]) -> Array:
    return rearrange(left[:, None] + right[None, :], "a b -> (a b)")


def _kronecker_draws(
    factors: tuple[lx.AbstractLinearOperator, ...],
    key: jax.Array,
    num_draws: int,
    combine,
) -> Float[Array, "S N"]:
    r"""Draws for ``⊗`` or ``⊕`` of symmetric factors, in their eigenbases.

    With ``A_i = Q_i Λ_i Q_iᵀ``, the Kronecker product (or sum) is
    ``(⊗ Q_i) Λ (⊗ Q_i)ᵀ`` with ``Λ`` the products (or sums) of the factors'
    eigenvalues, so ``(⊗ Q_i) Λ^{1/2} z`` is an exact draw. Round-off negatives
    in ``Λ`` are clipped, which also covers a semi-definite covariance.
    """
    decompositions = [_eigh_factor(factor) for factor in factors]
    eigenvalues = ft.reduce(combine, [values for values, _ in decompositions])
    basis = Kronecker(*(lx.MatrixLinearOperator(q) for _, q in decompositions))
    dtype = eigenvalues.dtype
    noise = jr.normal(key, (num_draws, eigenvalues.shape[0]), dtype=dtype)
    scaled = noise * jnp.sqrt(jnp.clip(eigenvalues, 0.0, None))
    return jax.vmap(basis.mv)(scaled)


# Circulant embedding sizes tried for a Toeplitz covariance, as multiples of N.
_EMBEDDING_FACTORS = (2, 4, 8, 16)


def _toeplitz_embedding_factor(column: Float[Array, " n"]) -> int | None:
    """Smallest embedding passing Wood--Chan, ``2`` if traced, else ``None``."""
    column = jnp.asarray(column)
    dtype = jnp.result_type(column.dtype, jnp.float32)
    column = column.astype(dtype)
    for factor in _EMBEDDING_FACTORS:
        spectrum = jnp.fft.rfft(
            _circulant_embedding(column, embedding_factor=factor)
        ).real
        # Same round-off allowance as `toeplitz_sample`'s own check.
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(spectrum)))
        tolerance = 100 * jnp.finfo(dtype).eps * scale
        try:
            if bool(jnp.all(spectrum >= -tolerance)):
                return factor
        except jax.errors.TracerBoolConversionError:
            return _EMBEDDING_FACTORS[0]
    return None


def _has_sampleable_update(covariance: LowRankUpdate) -> bool:
    """Whether ``B + U D Vᵀ`` can be sampled as ``L_B z₁ + U √D z₂``.

    Needs ``V = U`` -- established, not inferred from a symmetric tag, since
    ``V = 2U`` is symmetric too -- a base that is a covariance itself, and
    weights not *known* to be negative. Traced weights and traced diagonal
    bases are checked at run time in `_low_rank_draws`.
    """
    if not _arrays_match(covariance.U, covariance.V):
        return False
    if _known_negative(covariance.d):
        return False
    diagonal = _base_diagonal(covariance.base)
    if diagonal is not None:
        return not _known_negative(diagonal)
    return _safe_query(lx.is_positive_semidefinite, covariance.base)


def _known_negative(values: Array) -> bool:
    """True only when some entry is concretely negative."""
    try:
        return not bool(jnp.all(values >= 0))
    except jax.errors.TracerBoolConversionError:
        return False


def _base_diagonal(base: lx.AbstractLinearOperator) -> Array | None:
    """The diagonal of a (tagged) diagonal base, else ``None``."""
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
    r"""Sample ``B + U D Uᵀ`` as a sum of two independent draws.

    $L_B z_1 + U\sqrt{D} z_2$ has covariance $B + UDU^{\top}$, and the base
    $B$ is sampled through `_zero_mean_draws` so its own structure is kept.
    """
    base_key, update_key = jr.split(key)
    base = _zero_mean_draws(covariance.base, base_key, num_draws)
    diagonal = _base_diagonal(covariance.base)
    if diagonal is not None and not _is_nonnegative(diagonal):
        base = eqx.error_if(
            base,
            jnp.any(diagonal < 0),
            "sample_mvn: the diagonal base of a LowRankUpdate must be "
            "non-negative for low-rank sampling; build the covariance as a "
            "dense operator instead.",
        )
    weights = eqx.error_if(
        covariance.d,
        jnp.any(covariance.d < 0),
        "sample_mvn: LowRankUpdate weights must be non-negative for low-rank "
        "sampling; build the covariance as a dense operator instead.",
    )
    noise = jr.normal(
        update_key, (num_draws, covariance.rank), dtype=covariance.U.dtype
    )
    return base + einsum(noise * jnp.sqrt(weights), covariance.U, "s k, n k -> s n")


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
    r"""Joint draws of a partitioned Gaussian, and conditional draws from them.

    For

    $$
    \begin{pmatrix} x_a \\ x_b \end{pmatrix} \sim \mathcal{N}\!\left(
    \begin{pmatrix} m_a \\ m_b \end{pmatrix},
    \begin{pmatrix} K_{aa} & K_{ab} \\ K_{ba} & K_{bb} \end{pmatrix}
    \right),
    $$

    write $o$ for the observed block and $t$ for the other (with
    ``observed_index=1``, $o = b$ and $t = a$). The joint draw uses the block
    factorisation

    $$
    x_o = m_o + L_o z_1, \qquad
    x_t = m_t + K_{to} K_{oo}^{-1} (x_o - m_o) + L_S z_2,
    $$

    with $L_o L_o^{\top} = K_{oo}$ drawn by `gaussx.sample_mvn` -- so the
    observed block keeps its structure -- and $L_S L_S^{\top} = S = K_{tt} -
    K_{to} K_{oo}^{-1} K_{ot}$ the Schur complement. The joint covariance of
    size $(N + M)^2$ is never formed.

    When ``observed_value`` $\beta$ is given, the conditional draws of
    $x_t \mid x_o = \beta$ reuse the same $z_2$:

    $$
    x_t \mid \beta = m_t + K_{to} K_{oo}^{-1} (\beta - m_o) + L_S z_2,
    $$

    which is exactly `gaussx.matheron_update` applied to the joint draws.

    The Schur complement is dense in general: $K_{to} K_{oo}^{-1} K_{ot}$ is
    dense whatever the structure of $K_{tt}$, so $S$ is factorised at
    $O(N_t^3)$ plus $N_t$ solves against $K_{oo}$. Its square root is the
    symmetric one from ``dense_symmetric_sqrt``, with round-off negative
    eigenvalues clipped. That tolerates the rank deficiency of target points
    that coincide with observed ones -- a Cholesky would return NaNs there --
    and its custom JVP keeps gradients finite when $S$ has repeated
    eigenvalues.

    Args:
        joint_mean: ``(m_a, m_b)`` with shapes ``(N,)`` and ``(M,)``.
        joint_covariance: Operators ``{"aa": K_aa, "ab": K_ab, "bb": K_bb}``
            of shapes ``(N, N)``, ``(N, M)`` and ``(M, M)``; ``K_ba`` is
            ``K_ab.T``.
        key: PRNG key.
        observed_index: Which block is conditioned on: ``1`` for $x_b$
            (sample $x_a \mid x_b$), ``0`` for $x_a$.
        observed_value: Optional value $\beta$ of the observed block. When
            given, the result also holds conditional draws.
        num_samples: Number of independent draws.
        solver: Optional solver strategy for the solves against $K_{oo}$
            (e.g. `gaussx.CGSolver`). When ``None``, routes through
            structural dispatch, as `gaussx.matheron_update` does.

    Returns:
        ``{"joint": (samples_a, samples_b)}`` with shapes
        ``(num_samples, N)`` and ``(num_samples, M)``, plus
        ``"conditional"`` of shape ``(num_samples, N_t)`` when
        ``observed_value`` is given.

    Raises:
        ValueError: On a missing covariance block, mismatched shapes,
            ``observed_index`` outside ``{0, 1}`` or ``num_samples`` below
            one.
    """
    mean_a, mean_b = (jnp.asarray(mean) for mean in joint_mean)
    covariance_aa, covariance_ab, covariance_bb = _covariance_blocks(
        joint_covariance, mean_a.shape, mean_b.shape
    )
    if observed_index not in (0, 1):
        raise ValueError(f"observed_index must be 0 or 1, got {observed_index}.")
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}.")

    if observed_index == 1:
        mean_o, mean_t = mean_b, mean_a
        covariance_oo, covariance_tt = covariance_bb, covariance_aa
        cross = covariance_ab
    else:
        mean_o, mean_t = mean_a, mean_b
        covariance_oo, covariance_tt = covariance_aa, covariance_bb
        cross = covariance_ab.T

    if observed_value is not None:
        observed_value = jnp.asarray(observed_value)
        if observed_value.shape != mean_o.shape:
            raise ValueError(
                f"observed_value must have shape {mean_o.shape}, got "
                f"{observed_value.shape}."
            )

    if solver is None:
        solve_observed = lambda vector: solve(covariance_oo, vector)
    else:
        solve_observed = lambda vector: dispatch_solve(covariance_oo, vector, solver)

    def gain(deviations: Float[Array, "S K"]) -> Float[Array, "S T"]:
        """Apply ``K_to K_oo⁻¹`` to each row."""
        return jax.vmap(cross.mv)(jax.vmap(solve_observed)(deviations))

    observed_key, schur_key = jr.split(key)
    deviations_o = sample_mvn(
        jnp.zeros_like(mean_o), covariance_oo, key=observed_key, num_samples=num_samples
    )
    dtype = deviations_o.dtype
    schur_root = _schur_root(covariance_tt, cross, gain)
    noise = jr.normal(schur_key, (num_samples, mean_t.shape[0]), dtype=dtype)
    residual = einsum(noise, schur_root, "s k, t k -> s t")

    samples_o = mean_o + deviations_o
    samples_t = mean_t + gain(deviations_o) + residual
    joint = (samples_t, samples_o) if observed_index == 1 else (samples_o, samples_t)
    result: dict[str, Any] = {"joint": joint}
    if observed_value is not None:
        shift = gain((observed_value - mean_o)[None, :])
        result["conditional"] = mean_t + shift + residual
    return result


def _covariance_blocks(
    joint_covariance: dict[str, lx.AbstractLinearOperator],
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
) -> tuple[
    lx.AbstractLinearOperator, lx.AbstractLinearOperator, lx.AbstractLinearOperator
]:
    """Validate and unpack the ``{"aa", "ab", "bb"}`` covariance blocks."""
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
    covariance_tt: lx.AbstractLinearOperator,
    cross: lx.AbstractLinearOperator,
    gain,
) -> Float[Array, "T T"]:
    """The symmetric square root of ``S = K_tt - K_to K_oo⁻¹ K_ot``.

    Uses ``dense_symmetric_sqrt`` rather than a bare ``eigh``: its custom
    JVP stays finite when ``S`` has repeated eigenvalues -- an isotropic
    conditional covariance is nothing else -- where differentiating through
    ``eigh``'s eigenvectors gives NaN.
    """
    # Row i of K_to is column i of K_ot, so ``gain`` of the rows of K_to gives
    # the rows of K_to K_oo⁻¹ K_ot (a symmetric matrix).
    correction = gain(cross.as_matrix())
    schur = covariance_tt.as_matrix() - correction
    return dense_symmetric_sqrt(0.5 * (schur + schur.T))
