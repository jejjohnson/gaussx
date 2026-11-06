"""Variance-reduced stochastic trace estimators of a square N × N operator A.

XTrace is matfree's ``stochtrace.leave_one_out_xtrace`` (wired up in `gaussx.trace`);
Hutch++ is not in matfree, so it lives here. It spends N_probe matvecs on the probes
S (k × N) and G (s × N), with k = ⌊N_probe / 3⌋ and s = N_probe − 2k, and an
orthonormal basis Q of A's dominant range (N × k).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._primitives._samplers import SamplerName, resolve_sampler


def hutchpp_trace(
    operator: lx.AbstractLinearOperator,
    num_probes: int,
    key: jax.Array,
    sampler: SamplerName = "signs",
) -> Float[Array, ""]:
    """Hutch++ trace estimate (Meyer, Musco, Musco & Woodruff, 2021, Alg. 1).

        tr(A) ≈ tr(Qᴴ A Q) + (1 / s) ∑_i g_iᴴ (I − Q Qᴴ) A (I − Q Qᴴ) g_i

    For a complex operator every transpose is a conjugate transpose.

    Unbiased for any A; for PSD A it reaches relative error ε with O(1/ε) matvecs,
    against Hutchinson's O(1/ε²).

    Algorithm:
        1. Draw the range probes S (k rows) and the remainder probes G (s rows).
        2. Sketch the range, Q = orth(A Sᵀ) (k matvecs).
        3. Take the trace of the projection exactly, tr(Qᴴ A Q) (k matvecs).
        4. Deflate G⊥ = G − G Q̄ Qᵀ and average g⊥ᴴ A g⊥ over G⊥ (s matvecs).

    Args:
        operator: Square operator A.
        num_probes: Total matvec budget N_probe ≥ 3.
        key: PRNG key.
        sampler: Probe distribution for S and G (``"signs"``, ``"normal"`` or
            ``"sphere"``).

    Returns:
        The estimate of tr(A).

    Raises:
        ValueError: If ``num_probes < 3``, or ``sampler`` is not a known sampler.

    Examples:
        A low-rank plus small diagonal matrix, whose range the sketch captures:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> from gaussx._randomized._trace import hutchpp_trace
        >>> left = jr.normal(jr.key(0), (40, 3))
        >>> matrix = left @ left.T + 0.01 * jnp.eye(40)
        >>> estimate = hutchpp_trace(lx.MatrixLinearOperator(matrix), 30, jr.key(1))
        >>> bool(jnp.abs(estimate - jnp.trace(matrix)) / jnp.trace(matrix) < 0.05)
        True

        The estimate keeps the operator's dtype:

        >>> estimate.dtype == matrix.dtype
        True

        Fewer than three matvecs are rejected:

        >>> hutchpp_trace(lx.MatrixLinearOperator(matrix), 2, jr.key(1))
        Traceback (most recent call last):
        ...
        ValueError: Hutch++ needs num_probes >= 3 matvecs, got 2.

    References:
        Meyer, R. A., Musco, C., Musco, C. & Woodruff, D. P. (2021).
    """
    if num_probes < 3:
        raise ValueError(f"Hutch++ needs num_probes >= 3 matvecs, got {num_probes}.")
    size = operator.in_size()
    dtype = operator.in_structure().dtype
    # k = ⌊N_probe / 3⌋
    range_size = num_probes // 3
    # s = N_probe − 2k
    remainder_size = num_probes - 2 * range_size

    # --- probes ---
    key_range, key_rest = jax.random.split(key)
    range_sampler = resolve_sampler(sampler, size, range_size, dtype=dtype)
    # S, k rows
    range_probes = range_sampler(key_range)
    remainder_sampler = resolve_sampler(sampler, size, remainder_size, dtype=dtype)
    # G, s rows
    remainder_probes = remainder_sampler(key_rest)
    # rows in, rows out
    matmat = jax.vmap(operator.mv)

    # --- exact trace of the projection ---
    # Hermitian products throughout (Qᴴ Q = I for complex QR), so Q Qᴴ is an
    # orthogonal projector; conj is a no-op for real operators.
    # (A Sᵀ)ᵀ, one row A s_i per probe
    range_image = matmat(range_probes)
    # A Sᵀ
    range_image = rearrange(range_image, "k n -> n k")
    # Q = orth(A Sᵀ)
    basis, _ = jnp.linalg.qr(range_image)
    # Qᵀ
    basis_rows = rearrange(basis, "n k -> k n")
    # Q̄ᵀ = Qᴴ
    conjugate_rows = jnp.conj(basis_rows)
    # (A Q)ᵀ
    basis_image = matmat(basis_rows)
    # tr(Qᴴ A Q)
    low_rank = einsum(conjugate_rows, basis_image, "k n, k n ->")

    # --- Hutchinson on the remainder ---
    # Q̄
    conjugate_basis = jnp.conj(basis)
    # G Q̄
    coefficients = einsum(remainder_probes, conjugate_basis, "s n, n k -> s k")
    # G Q̄ Qᵀ
    projected = einsum(coefficients, basis_rows, "s k, k n -> s n")
    # G⊥ = G − G Q̄ Qᵀ, rows of (I − Q Qᴴ) g
    deflated_probes = remainder_probes - projected
    # conj(G⊥)
    conjugate_deflated = jnp.conj(deflated_probes)
    # (A G⊥ᵀ)ᵀ
    deflated_image = matmat(deflated_probes)
    # ∑_i g⊥_iᴴ A g⊥_i
    remainder_total = einsum(conjugate_deflated, deflated_image, "s n, s n ->")
    # (1 / s) ∑_i g⊥_iᴴ A g⊥_i
    remainder = remainder_total / remainder_size
    return low_rank + remainder
