"""Docstring ``Args:`` / ``Returns:`` sections must match the signatures (gh-393).

The most common way a docstring rots is a renamed or added parameter whose
``Args:`` entry is not updated. For every public function in
``gaussx.__all__`` this checks that the ``Args:`` names equal the signature's
parameters and that a ``Returns:`` (or ``Yields:``) section exists unless the
function returns ``None``. For public classes it checks the weaker direction,
``Args:`` names ⊆ ``__init__`` parameters, since many document their fields
under ``Attributes:`` instead. Parameters named ``_private``
(positional-compat shims) and deprecated aliases (docstring summary starting
"Deprecated") are exempt.

Objects re-exported from other packages (lineax's ``is_*`` predicates and
tags) are skipped by their ``__module__``. The Google-style sections are
parsed with a small regex parser rather than griffe, which is only in the
docs dependency group.
"""

from __future__ import annotations

import inspect
import re

import pytest

import gaussx


_SECTION = re.compile(r"^(\w[\w ]*):\s*$")
_ARGS_SECTIONS = {"Args", "Arguments", "Parameters"}
_RETURNS_SECTIONS = {"Returns", "Return", "Yields", "Yield"}
_ENTRY = re.compile(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:")


def _sections(doc: str) -> dict[str, list[str]]:
    """Top-level Google-style sections of a cleaned docstring."""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in inspect.cleandoc(doc).splitlines():
        match = _SECTION.match(line)
        if match:
            current = sections.setdefault(match.group(1), [])
        elif current is not None and (not line or line[0].isspace()):
            current.append(line)
        elif line:
            current = None
    return sections


def _documented_args(doc: str) -> list[str]:
    names: list[str] = []
    for title in _ARGS_SECTIONS:
        lines = [line for line in _sections(doc).get(title, []) if line.strip()]
        if not lines:
            continue
        indent = len(lines[0]) - len(lines[0].lstrip())
        for line in lines:
            if len(line) - len(line.lstrip()) != indent:
                continue  # continuation of the previous entry
            match = _ENTRY.match(line.strip())
            if match:
                names.append(match.group(1).lstrip("*"))
    return names


def _parameters(obj) -> list[str]:
    """Public parameters: ``self``/``cls`` and ``_private`` shims excluded."""
    params = inspect.signature(obj).parameters.values()
    return [
        p.name for p in params if p.name not in ("self", "cls") and p.name[0] != "_"
    ]


def _public_objects() -> list[tuple[str, object]]:
    objects = []
    for name in sorted(gaussx.__all__):
        try:
            obj = getattr(gaussx, name)
        except AttributeError:  # a numpyro-backed name without numpyro
            continue
        module = getattr(obj, "__module__", None) or ""
        if not module.startswith("gaussx"):
            continue
        summary = (obj.__doc__ or "").strip().partition("\n")[0]
        if summary.lower().startswith("deprecated"):
            continue  # a deprecated alias points at its replacement's docs
        if inspect.isclass(obj) or inspect.isfunction(obj):
            objects.append((name, obj))
    return objects


_OBJECTS = _public_objects()
_FUNCTIONS = [(n, o) for n, o in _OBJECTS if inspect.isfunction(o)]
_CLASSES = [(n, o) for n, o in _OBJECTS if inspect.isclass(o)]


def _returns_none(func) -> bool:
    return inspect.signature(func).return_annotation in (None, "None")


@pytest.mark.parametrize(("name", "func"), _FUNCTIONS, ids=[n for n, _ in _FUNCTIONS])
def test_function_args_match_signature(name, func):
    doc = func.__doc__ or ""
    documented = _documented_args(doc)
    parameters = _parameters(func)
    assert sorted(documented) == sorted(parameters), (
        f"gaussx.{name}: Args: {sorted(documented)} != signature "
        f"{sorted(parameters)}; undocumented "
        f"{sorted(set(parameters) - set(documented))}, stale "
        f"{sorted(set(documented) - set(parameters))}"
    )


@pytest.mark.parametrize(("name", "func"), _FUNCTIONS, ids=[n for n, _ in _FUNCTIONS])
def test_function_has_returns(name, func):
    if _returns_none(func):
        return
    sections = _sections(func.__doc__ or "")
    assert _RETURNS_SECTIONS & set(sections), f"gaussx.{name} has no Returns: section"


@pytest.mark.parametrize(("name", "cls"), _CLASSES, ids=[n for n, _ in _CLASSES])
def test_class_args_are_init_parameters(name, cls):
    documented = set(_documented_args(cls.__doc__ or ""))
    try:
        parameters = set(_parameters(cls))
    except (TypeError, ValueError):  # no introspectable signature
        parameters = set()
    stale = documented - parameters
    assert not stale, (
        f"gaussx.{name}: Args: names {sorted(stale)} are not parameters of "
        f"its constructor {sorted(parameters)}"
    )
