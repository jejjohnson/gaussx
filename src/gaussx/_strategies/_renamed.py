"""Deprecated keyword aliases for renamed strategy options (gh-405)."""

from __future__ import annotations

from typing import Any

from gaussx._deprecation import warn_deprecated


class _Unset:
    def __repr__(self) -> str:
        return "<unset>"


UNSET: Any = _Unset()


def renamed(owner: str, old: str, old_value: Any, new: str, new_value: Any) -> Any:
    """The value for *new*, taking a deprecated *old* keyword into account.

    Args:
        owner: The class name, for the messages.
        old: The deprecated keyword.
        old_value: Its value, or `UNSET`.
        new: The canonical keyword.
        new_value: Its value, or `UNSET`.

    Returns:
        *old_value* when only the old keyword was given (with a deprecation
        warning), else *new_value* (possibly `UNSET`).

    Raises:
        TypeError: If both were given.
    """
    if old_value is UNSET:
        return new_value
    if new_value is not UNSET:
        raise TypeError(f"{owner}: pass {new}= only; {old}= is its deprecated alias.")
    warn_deprecated(
        f"{owner}({old}=...) is deprecated; use {new}= instead (gh-405). The "
        "alias will be removed in a future release."
    )
    return old_value


def default(value: Any, fallback: Any) -> Any:
    """*value*, or *fallback* when it is `UNSET`."""
    return fallback if value is UNSET else value
