"""Deprecated keyword aliases for renamed strategy options (gh-405).

Strategies with a renamed option (``maxiter`` → ``max_steps``,
``cg_tolerance`` → ``rtol``, …) take both keywords with the `UNSET` sentinel
as default, and resolve them here.
"""

from __future__ import annotations

from typing import Any

from gaussx._deprecation import warn_deprecated


class _Unset:
    """Sentinel type for a keyword the caller did not pass."""

    def __repr__(self) -> str:
        """``<unset>``, so signatures read cleanly in the docs."""
        return "<unset>"


UNSET: Any = _Unset()


def renamed(owner: str, old: str, old_value: Any, new: str, new_value: Any) -> Any:
    """The value for ``new``, taking a deprecated ``old`` keyword into account.

    Args:
        owner: The class name, for the messages.
        old: The deprecated keyword.
        old_value: Its value, or `UNSET`.
        new: The canonical keyword.
        new_value: Its value, or `UNSET`.

    Returns:
        ``old_value`` when only the old keyword was given (with a
        deprecation warning), else ``new_value`` (possibly `UNSET`).

    Raises:
        TypeError: If both were given.

    Examples:
        >>> from gaussx._strategies._renamed import UNSET, renamed
        >>> renamed("LSMRSolver", "maxiter", UNSET, "max_steps", 50)
        50
        >>> renamed("Solver", "old", 1, "new", 2)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        TypeError: Solver: pass new= only; old= is its deprecated alias.
    """
    if old_value is UNSET:
        return new_value
    if new_value is not UNSET:
        raise TypeError(f"{owner}: pass {new}= only; {old}= is its deprecated alias.")
    warn_deprecated(
        f"{owner}({old}=...) is deprecated; use {new}= instead (gh-405). The "
        "alias will be removed in gaussx 0.7.0."
    )
    return old_value


def default(value: Any, fallback: Any) -> Any:
    """``value``, or ``fallback`` when it is `UNSET`.

    Args:
        value: A keyword's value, or `UNSET`.
        fallback: The default to use when ``value`` is `UNSET`.

    Returns:
        ``value`` if it was passed, else ``fallback``.

    Examples:
        >>> from gaussx._strategies._renamed import UNSET, default
        >>> default(UNSET, 1000), default(20, 1000)
        (1000, 20)
    """
    return fallback if value is UNSET else value
