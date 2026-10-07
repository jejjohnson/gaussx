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
    """The declared public API, ``gaussx.__all__``."""
    return set(gaussx.__all__)


# The page each layer is documented on, keyed by the defining module's
# package (``obj.__module__``, truncated to two components).
_PAGE_OF_PACKAGE = {
    "gaussx._distributions": "distributions.md",
    "gaussx._expfam": "distributions.md",
    "gaussx._gmrf": "gmrf.md",
    "gaussx._gp": "gp.md",
    "gaussx._inference": "inference.md",
    "gaussx._linalg": "linalg.md",
    "gaussx._operators": "operators.md",
    "gaussx._preconditioners": "solvers.md",
    "gaussx._primitives": "primitives.md",
    "gaussx._quadrature": "quadrature.md",
    "gaussx._randomized": "randomized.md",
    "gaussx._sketching": "sketching.md",
    "gaussx._solve_frontend": "solvers.md",
    "gaussx._sparse": "sparse.md",
    "gaussx._ssm": "ssm.md",
    "gaussx._strategies": "solvers.md",
    "gaussx._tags": "operators.md",
    # The lineax tags and predicates gaussx re-exports.
    "lineax._operator": "operators.md",
    "lineax._tags": "operators.md",
}

# Intentional exceptions: documented with what they belong to rather than
# where they are defined.
_PAGE_EXCEPTIONS = {
    # The sparse operator lives with the sparse Cholesky machinery.
    "SparseOperator": "sparse.md",
    "SparsityPattern": "sparse.md",
    # Defined next to SqrtOperator in _primitives (which imports _operators),
    # documented next to KroneckerSumSqrt and SumOfKroneckers (gh-297).
    "SumOfKroneckersSqrt": "operators.md",
    "SumKroneckerSqrt": "operators.md",
}


def _expected_page(name: str) -> str | None:
    """The page ``name`` belongs on, from its defining package."""
    if name in _PAGE_EXCEPTIONS:
        return _PAGE_EXCEPTIONS[name]
    module = getattr(getattr(gaussx, name), "__module__", None) or ""
    return _PAGE_OF_PACKAGE.get(".".join(module.split(".")[:2]))


def test_all_matches_the_module_namespace() -> None:
    """``__all__`` and the non-underscore module attributes cannot drift."""
    public = {name for name in dir(gaussx) if not name.startswith("_")}
    assert len(gaussx.__all__) == len(set(gaussx.__all__)), "duplicates in __all__"
    assert set(gaussx.__all__) == public, (
        f"only in __all__: {sorted(set(gaussx.__all__) - public)}; "
        f"public but missing from __all__: {sorted(public - set(gaussx.__all__))}"
    )


def test_every_symbol_is_on_its_layer_page() -> None:
    """Each symbol sits on the page of the package that defines it (gh-322)."""
    where = {
        name: page
        for page, members in _documented_members().items()
        for name in members
    }
    unmapped = sorted(n for n in _public_api() if _expected_page(n) is None)
    assert not unmapped, (
        f"no page mapped for the defining package of {unmapped}; extend "
        "_PAGE_OF_PACKAGE (or _PAGE_EXCEPTIONS) in this test."
    )
    misplaced = {
        name: (where.get(name), _expected_page(name))
        for name in _public_api()
        if name in where and where[name] != _expected_page(name)
    }
    assert not misplaced, (
        f"symbols documented on the wrong page (found, expected): {misplaced}"
    )


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
