"""Private utilities shared by operator implementations."""

from __future__ import annotations

from collections.abc import Callable

import jax


def vmap_over_batch_dims(fn: Callable, num_batch_dims: int) -> Callable:
    """Apply ``jax.vmap`` repeatedly over the leading batch dimensions."""
    for _ in range(num_batch_dims):
        fn = jax.vmap(fn)
    return fn
