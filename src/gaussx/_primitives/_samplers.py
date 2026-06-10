"""Probe-vector samplers for stochastic estimators (matfree backend).

Leaf module shared by the stochastic ``trace``/``diag``/``frobenius_norm``
primitives and the SLQ logdet strategies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import matfree.stochtrace


SamplerName = Literal["signs", "normal", "sphere"]

_SAMPLER_FACTORIES: dict[str, Callable] = {
    "signs": matfree.stochtrace.sampler_signs,
    "normal": matfree.stochtrace.sampler_normal,
    "sphere": matfree.stochtrace.sampler_sphere,
}


def resolve_sampler(name: SamplerName, n: int, num_probes: int) -> Callable:
    """Build a matfree probe sampler by name.

    Args:
        name: ``"signs"`` (Rademacher-style), ``"normal"``, or ``"sphere"``.
        n: Problem dimension.
        num_probes: Number of probe vectors.

    Returns:
        A matfree sampler callable.
    """
    try:
        factory = _SAMPLER_FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown sampler {name!r}; expected one of {sorted(_SAMPLER_FACTORIES)}."
        ) from None
    return factory(jnp.zeros(n), num=num_probes)
