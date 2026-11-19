"""Mechanical checks of the naming conventions in docs/api/index.md (gh-315)."""

from __future__ import annotations

import inspect
import re

import pytest

import gaussx


# Words that end in "ise" but are not the en-GB spelling of an "-ize" verb.
_ISE_WORDS = frozenset(
    {"noise", "pairwise", "wise", "precise", "otherwise", "raise", "promise"}
    | {"rise", "concise", "exercise", "likewise", "elementwise", "pointwise"}
    | {"blockwise", "stepwise", "piecewise", "clockwise"}
)
_EN_GB = re.compile(r"(isation|isations|ised|iser|isers|ising)$")


def _words(name: str) -> list[str]:
    """``snake_case`` and ``CamelCase`` components of ``name``, lower-cased."""
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+", name)
    return [p.lower() for p in parts]


def _is_en_gb(word: str) -> bool:
    if _EN_GB.search(word):
        return True
    return word.endswith("ise") and word not in _ISE_WORDS


def test_camelcase_names_are_classes() -> None:
    """A CamelCase public callable must be a class, so isinstance works on it."""
    factories = [
        name
        for name in gaussx.__all__
        if name[0].isupper()
        and callable(getattr(gaussx, name))
        and not inspect.isclass(getattr(gaussx, name))
    ]
    assert not factories, f"CamelCase factory functions (use snake_case): {factories}"


def test_public_names_use_us_spelling() -> None:
    """Public identifiers use -ize / -ization, not -ise / -isation."""
    offenders = [n for n in gaussx.__all__ if any(map(_is_en_gb, _words(n)))]
    assert not offenders, f"en-GB spellings in public names: {offenders}"


@pytest.mark.parametrize("word", ["discretise", "diagonalised", "normalisation"])
def test_the_spelling_check_catches_en_gb(word: str) -> None:
    assert _is_en_gb(word)
