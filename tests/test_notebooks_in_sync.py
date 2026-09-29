"""Keep each example notebook's executed ``.ipynb`` in sync with its ``.py``.

The docs site renders the committed ``.ipynb`` (``execute: false``), but the
jupytext ``.py`` is what gets edited. Editing the ``.py`` without re-running
``jupytext --to notebook --execute foo.py -o foo.ipynb`` publishes a page
that no longer matches its source (gh-395). These tests compare cell sources
only, so they need no kernel and run at unit-test speed.
"""

from __future__ import annotations

from pathlib import Path

import pytest


jupytext = pytest.importorskip("jupytext")

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "docs" / "notebooks"
SOURCES = sorted(NOTEBOOK_DIR.glob("*.py"))

# Committed outputs must not show a warning or a crash on the published page.
_BAD_OUTPUT = ("DeprecationWarning", "Traceback (most recent call last)")


def _cells(path: Path) -> list[tuple[str, str]]:
    return [(c.cell_type, c.source.strip()) for c in jupytext.read(path).cells]


def test_every_source_has_a_notebook():
    assert SOURCES, f"no notebook sources found in {NOTEBOOK_DIR}"
    missing = [p.name for p in SOURCES if not p.with_suffix(".ipynb").exists()]
    assert not missing, f"no executed .ipynb for: {missing}"


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_notebook_matches_source(source: Path):
    notebook = source.with_suffix(".ipynb")
    assert _cells(notebook) == _cells(source), (
        f"{notebook.name} is out of sync with {source.name}; re-run "
        f"`jupytext --to notebook --execute {source.name} -o {notebook.name}` "
        "in docs/notebooks/ and commit both files"
    )


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_notebook_outputs_have_no_warnings_or_tracebacks(source: Path):
    nb = jupytext.read(source.with_suffix(".ipynb"))
    texts = [
        "".join(out.get("text", "")) + "".join(out.get("traceback", []))
        for cell in nb.cells
        for out in cell.get("outputs", [])
    ]
    found = sorted({bad for bad in _BAD_OUTPUT for text in texts if bad in text})
    assert not found, f"{source.with_suffix('.ipynb').name} outputs contain {found}"
