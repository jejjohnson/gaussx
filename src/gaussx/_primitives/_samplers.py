"""Probe samplers z^(i) for the stochastic estimators (matfree backend).

The stochastic ``trace``, ``diag`` and ``frobenius_norm`` primitives and the
SLQ log-determinant strategies average a statistic of A z^(i) over probes
z^(i) ∈ ℝᴺ, i = 1, …, N_probe, with 𝔼[z zᵀ] = I. This leaf module builds
the probe distribution and splits keys between the parts of a structured
operator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import matfree.stochtrace
from jax.typing import DTypeLike


SamplerName = Literal["signs", "normal", "sphere"]

_SAMPLER_FACTORIES: dict[str, Callable] = {
    "signs": matfree.stochtrace.sampler_signs,
    "normal": matfree.stochtrace.sampler_normal,
    "sphere": matfree.stochtrace.sampler_sphere,
}


def resolve_sampler(
    name: SamplerName, n: int, num_probes: int, dtype: DTypeLike | None = None
) -> Callable:
    """Matfree probe sampler for N_probe probes z^(i) ∈ ℝᴺ, by name.

        z^(i) ~ p(z),   𝔼[z zᵀ] = I

    Args:
        name: Probe distribution: ``"signs"`` (Rademacher, entries ±1),
            ``"normal"`` (standard normal) or ``"sphere"`` (uniform on the
            sphere of radius √N).
        n: Probe length N.
        num_probes: Number of probes N_probe.
        dtype: Probe dtype; pass the operator's, so its matvecs are not
            promoted. ``None`` means JAX's default float.

    Returns:
        A matfree sampler that maps a PRNG key to the N_probe × N probes.

    Raises:
        ValueError: If ``name`` is not a known sampler.

    Examples:
        Rademacher probes have unit-magnitude entries:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gaussx._primitives._samplers import resolve_sampler
        >>> sample = resolve_sampler("signs", 4, 3)
        >>> probes = sample(jr.key(0))
        >>> probes.shape, bool(jnp.all(jnp.abs(probes) == 1.0))
        ((3, 4), True)
    """
    try:
        factory = _SAMPLER_FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown sampler {name!r}; expected one of {sorted(_SAMPLER_FACTORIES)}."
        ) from None
    return factory(jnp.zeros(n, dtype=dtype), num=num_probes)


def split_keys(key: jax.Array | None, num: int) -> list[jax.Array | None]:
    """Independent probe keys for ``num`` sub-estimates (``None`` stays ``None``).

    A stochastic primitive that recurses into several children (a sum, a
    block diagonal, a Kronecker product) gives each child its own probes
    (gh-320).

    Args:
        key: PRNG key to split, or ``None``.
        num: Number of children.

    Returns:
        ``num`` keys from one ``jax.random.split``, or ``num`` copies of
        ``None``.

    Examples:
        >>> import jax.random as jr
        >>> from gaussx._primitives._samplers import split_keys
        >>> len(split_keys(jr.key(0), 3)), split_keys(None, 2)
        (3, [None, None])
    """
    if key is None:
        return [None] * num
    return list(jax.random.split(key, num))
