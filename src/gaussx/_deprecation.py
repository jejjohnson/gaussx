"""gaussx's own deprecation warning category and helper (gh-332)."""

from __future__ import annotations

import functools
import os
import warnings
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import equinox


P = ParamSpec("P")
R = TypeVar("R")


class GaussxDeprecationWarning(DeprecationWarning):
    """A gaussx API is deprecated.

    A subclass of `DeprecationWarning`, so ``pytest.warns(DeprecationWarning)``
    and ``-W ignore::DeprecationWarning`` still match it, while filters (such
    as the test suite's ``error::`` entry) can single out gaussx's own.
    """


# Frames inside these are skipped, so a warning lands on the caller's line:
# gaussx itself (the deprecated function may be reached through gaussx
# helpers) and equinox (an eqx.Module.__init__ is called from equinox's
# metaclass, so a fixed stacklevel would point into equinox).
_SKIP = (
    os.path.dirname(__file__),
    os.path.dirname(equinox.__file__),
)


# Public names renamed under the naming conventions in docs/api/index.md
# (gh-315), mapped to their replacements. ``gaussx.<old>`` still resolves,
# through the package ``__getattr__``, to the *same* object as the new name
# (so ``isinstance`` and ``is`` checks keep working) and warns on access.
RENAMED: dict[str, str] = {
    "DiagonalisedOperator": "DiagonalizedOperator",
    "as_diagonalised": "as_diagonalized",
    "discretise_mfd": "discretize_mfd",
    "discretise_mfd_sequence": "discretize_mfd_sequence",
    "EigenFactorization": "EigenDecomposition",
    "Circulant": "circulant",
    "SumOperator": "sum_operator",
    "ScaledOperator": "scaled_operator",
    "ProductOperator": "product_operator",
    "solve_tridiagonal": "tridiagonal_solve",
    "solve_tridiagonal_batched": "tridiagonal_solve_batched",
    "dist_kl_divergence": "gaussian_kl",
    "meanvar_to_natural": "mean_chol_to_natural",
    "natural_to_meanvar": "natural_to_mean_chol",
    "meanvar_to_expectation": "mean_chol_to_expectation",
    "expectation_to_meanvar": "expectation_to_mean_chol",
}
RENAMED_REMOVAL = "0.7.0"


def renamed_attribute(old: str, namespace: dict[str, Any]) -> Any:
    """Resolve the renamed public name ``old`` in ``namespace``, with a warning."""
    new = RENAMED[old]
    warn_deprecated(
        f"gaussx.{old} is deprecated and will be removed in gaussx "
        f"{RENAMED_REMOVAL}; use gaussx.{new} (gh-315)."
    )
    return namespace[new]


def warn_deprecated(message: str) -> None:
    """Emit a `GaussxDeprecationWarning` attributed to the caller's code."""
    warnings.warn(
        message, GaussxDeprecationWarning, stacklevel=2, skip_file_prefixes=_SKIP
    )


def renamed_kwargs(**renames: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Accept deprecated keyword names until their removal release.

    ``@renamed_kwargs(K_diag="K_xx_diag")`` maps a call with ``K_diag=...``
    onto ``K_xx_diag=...`` and emits a `GaussxDeprecationWarning`; passing
    both names raises `TypeError`.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        name = getattr(fn, "__name__", "function")

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            mapped: dict[str, Any] = dict(kwargs)
            for old, new in renames.items():
                if old in mapped:
                    if new in mapped:
                        msg = f"{name}() got both {old}= (deprecated) and {new}=."
                        raise TypeError(msg)
                    warn_deprecated(
                        f"{name}({old}=...) is deprecated and will be removed "
                        f"in gaussx 0.7.0; use {new}=... instead."
                    )
                    mapped[new] = mapped.pop(old)
            return fn(*args, **mapped)

        return wrapper

    return decorator
