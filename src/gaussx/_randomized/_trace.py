"""Variance-reduced stochastic trace estimators (G17).

XTrace is matfree's ``stochtrace.leave_one_out_xtrace`` (wired up in
`gaussx.trace`); Hutch++ is not in matfree, so it lives here.
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
    r"""Hutch++ trace estimate (Meyer, Musco, Musco & Woodruff, 2021, Alg. 1).

    Spends the ``num_probes`` = $m$ matvecs in three blocks: $k = \lfloor
    m/3 \rfloor$ to sketch the range, $Q = \operatorname{orth}(AS)$, $k$ to
    take the trace of the projection exactly, and the remaining $s = m - 2k$
    on Hutchinson for what is left:

    $$
    \widehat{\operatorname{tr}}(A) = \operatorname{tr}(Q^\top A Q)
    + \frac1s \sum_{i=1}^{s} g_i^\top (I - QQ^\top) A (I - QQ^\top) g_i .
    $$

    Unbiased for any $A$; for PSD $A$ it reaches relative error
    $\varepsilon$ with $O(1/\varepsilon)$ matvecs, against Hutchinson's
    $O(1/\varepsilon^2)$.

    ```text
    S = probes(k), G = probes(s)
    Q = qr(A S)                       # k matvecs
    t₁ = tr(Qᵀ (A Q))                 # k matvecs
    G⊥ = G − Q (Qᵀ G)
    t₂ = mean_i  g⊥ᵢᵀ A g⊥ᵢ           # s matvecs
    return t₁ + t₂
    ```

    Args:
        operator: Square operator $A$.
        num_probes: Total matvec budget $m \ge 3$.
        key: PRNG key.
        sampler: Probe distribution for $S$ and $G$ (``"signs"``,
            ``"normal"`` or ``"sphere"``).

    Returns:
        The estimate of $\operatorname{tr}(A)$.

    Raises:
        ValueError: If ``num_probes < 3``.
    """
    if num_probes < 3:
        raise ValueError(f"Hutch++ needs num_probes >= 3 matvecs, got {num_probes}.")
    n = operator.in_size()
    dtype = operator.in_structure().dtype
    k = num_probes // 3
    s = num_probes - 2 * k
    key_range, key_rest = jax.random.split(key)
    S = resolve_sampler(sampler, n, k, dtype=dtype)(key_range)
    G = resolve_sampler(sampler, n, s, dtype=dtype)(key_rest)
    matmat = jax.vmap(operator.mv)  # rows in, rows out

    Q, _ = jnp.linalg.qr(rearrange(matmat(S), "k n -> n k"))
    Qt = rearrange(Q, "n k -> k n")
    low_rank = einsum(Qt, matmat(Qt), "k n, k n ->")
    G_perp = G - einsum(einsum(G, Q, "s n, n k -> s k"), Qt, "s k, k n -> s n")
    remainder = einsum(G_perp, matmat(G_perp), "s n, s n ->") / s
    return low_rank + remainder
