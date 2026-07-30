"""Keep the API reference in sync with the public API.

``docs/api/*.md`` lists symbols explicitly via mkdocstrings ``members:`` blocks
rather than auto-generating pages, which keeps the reference organised by layer
but lets it drift: a newly exported name silently gets no page, and a renamed
one leaves a dangling entry that only shows up as a ``mkdocs build --strict``
failure in CI. These tests close both gaps at unit-test speed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import gaussx


DOCS_API_DIR = Path(__file__).resolve().parents[1] / "docs" / "api"

# ``members: [a, b, c]`` (inline flow sequence).
_INLINE_MEMBERS = re.compile(r"members:\s*\[([^\]]*)\]")
# ``members:`` followed by a ``- name`` block sequence.
_BLOCK_MEMBERS = re.compile(r"members:\s*\n((?:[ \t]*-[ \t]*\w+[ \t]*\n)+)")
_BLOCK_ITEM = re.compile(r"-[ \t]*(\w+)")


def _reference_pages() -> list[Path]:
    """The reference pages, i.e. every page except the prose overview."""
    return sorted(p for p in DOCS_API_DIR.glob("*.md") if p.name != "index.md")


def _documented_members() -> dict[str, set[str]]:
    """Map each reference page to the symbols it documents."""
    pages: dict[str, set[str]] = {}
    for path in _reference_pages():
        text = path.read_text()
        names: set[str] = set()
        for group in _INLINE_MEMBERS.findall(text):
            names |= {n.strip() for n in group.split(",") if n.strip()}
        for group in _BLOCK_MEMBERS.findall(text):
            names |= set(_BLOCK_ITEM.findall(group))
        pages[path.name] = names
    return pages


def _public_api() -> set[str]:
    """Every public name re-exported from ``gaussx/__init__.py``."""
    return {name for name in dir(gaussx) if not name.startswith("_")}


def test_docs_api_dir_is_discovered() -> None:
    """Guard against the glob silently matching nothing."""
    pages = _documented_members()
    assert pages, f"no reference pages found under {DOCS_API_DIR}"
    assert all(pages.values()), (
        f"pages with no documented members: "
        f"{sorted(name for name, members in pages.items() if not members)}"
    )


def test_every_public_symbol_is_documented() -> None:
    """No public export may be missing from the API reference."""
    undocumented = _public_api() - set().union(*_documented_members().values())
    assert not undocumented, (
        "public symbols missing from docs/api/*.md: "
        f"{sorted(undocumented)}. Add each to the `members:` list of the page "
        "matching its layer, or drop it from gaussx/__init__.py."
    )


def test_no_documented_symbol_is_stale() -> None:
    """Every documented symbol must still exist on the public API."""
    stale = set().union(*_documented_members().values()) - _public_api()
    assert not stale, (
        f"docs/api/*.md reference symbols that gaussx no longer exports: "
        f"{sorted(stale)}. These break `mkdocs build --strict`."
    )


def test_no_symbol_is_documented_twice() -> None:
    """Duplicate entries produce duplicate anchors and ambiguous cross-refs."""
    pages = _documented_members()
    duplicates = {
        name: sorted(page for page, members in pages.items() if name in members)
        for name in set().union(*pages.values())
        if sum(name in members for members in pages.values()) > 1
    }
    assert not duplicates, f"symbols documented on more than one page: {duplicates}"


@pytest.mark.parametrize("page", [p.name for p in _reference_pages()])
def test_page_is_linked_from_api_index(page: str) -> None:
    """Each reference page must be reachable from the API overview."""
    index = (DOCS_API_DIR / "index.md").read_text()
    assert f"({page})" in index, f"docs/api/{page} is not linked from docs/api/index.md"
