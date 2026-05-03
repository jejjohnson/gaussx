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
    """Wrap a kernel matvec implementation with a custom JVP rule.

    Args:
        impl: Forward kernel matvec implementation closed over any static data.
        jvp_rule: Custom JVP rule for ``impl`` with the standard
            ``jax.custom_jvp`` ``(primals, tangents)`` signature.

    Returns:
        A positional ``(*args)`` callable that forwards to ``impl`` and
        dispatches through ``jax.custom_jvp``. Note that the returned
        wrapper accepts only positional arguments — keyword-style calls
        to ``impl`` are not preserved by this helper. Callers should
        invoke the wrapper positionally (matching the ``primals`` tuple
        the JVP rule receives).

    Example:
        ```python
        def _make_kernel_mv(kernel_fn: Callable) -> Callable:
            def _impl(params, X1, X2, v):
                ...

            def _jvp(primals, tangents):
                ...

            return _kernel_mv_with_jvp(_impl, _jvp)
        ```
    """

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
