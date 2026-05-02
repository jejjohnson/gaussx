"""Shared custom-JVP helpers for kernel matvec operators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from jaxtyping import Array


def _kernel_mv_with_jvp(
    impl: Callable[..., Array],
    jvp_rule: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Array, Array]],
) -> Callable[..., Array]:
    """Wrap a kernel matvec implementation with a custom JVP rule."""

    @jax.custom_jvp
    def kernel_mv(*args: Any) -> Array:
        return impl(*args)

    @kernel_mv.defjvp
    def kernel_mv_jvp(
        primals: tuple[Any, ...],
        tangents: tuple[Any, ...],
    ) -> tuple[Array, Array]:
        return jvp_rule(primals, tangents)

    return kernel_mv
