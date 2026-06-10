"""Guard against Sphinx/reStructuredText markup in docstrings.

The docs are built with MkDocs + mkdocstrings (MathJax for math,
``[`x`][path]`` for cross-references), which do **not** render Sphinx/RST
constructs such as ``:math:`...``` or ``.. math::`` — they leak into the
rendered page as literal text. Doctests cannot catch this because they only
execute ``>>>`` examples and ignore the surrounding prose, so this test scans
the source for the offending markup directly.
"""

import re
from pathlib import Path

import pytest

import gaussx


# Patterns that MkDocs/mkdocstrings will not render (use $...$/$$...$$ for math,
# `name` / [`name`][path] for cross-references, and fenced code blocks instead).
RST_PATTERNS = {
    ":math: role (use $...$)": re.compile(r":math:`"),
    ".. math:: directive (use $$...$$)": re.compile(r"\.\. math::"),
    ":class:/:func:/:meth:/:mod:/:data: role (use `name`)": re.compile(
        r":(?:class|func|meth|mod|data|attr|obj|exc|ref|term):`"
    ),
    ".. <directive>:: (use an admonition / fenced code block)": re.compile(
        r"\.\. (?:note|warning|code|code-block|seealso|admonition|versionadded"
        r"|versionchanged|deprecated|rubric)::"
    ),
    "backslash-escaped suffix (e.g. `Block`\\ s — drop the backslash)": re.compile(
        r"`+\\+ [A-Za-z]"
    ),
}

PKG_DIR = Path(gaussx.__file__).parent
SRC_FILES = sorted(PKG_DIR.rglob("*.py"))


@pytest.mark.parametrize(
    "path", SRC_FILES, ids=lambda p: str(p.relative_to(PKG_DIR.parent))
)
def test_no_rst_markup_in_source(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    offenders = [label for label, pat in RST_PATTERNS.items() if pat.search(text)]
    assert not offenders, (
        f"{path.name} contains Sphinx/RST markup that MkDocs won't render: "
        f"{offenders}. Convert it to MkDocs syntax."
    )
