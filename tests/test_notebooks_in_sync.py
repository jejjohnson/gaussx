"""Keep each example notebook's executed ``.ipynb`` in sync with its ``.py``.

The docs site renders the committed ``.ipynb`` (``execute: false``), but the
jupytext ``.py`` is what gets edited. Editing the ``.py`` without re-running
``jupytext --to notebook --execute foo.py -o foo.ipynb`` publishes a page
that no longer matches its source (gh-395). These tests compare cell sources
only, so they need no kernel and run at unit-test speed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


jupytext = pytest.importorskip("jupytext")

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "docs" / "notebooks"
if not NOTEBOOK_DIR.is_dir():
    # The sdist ships tests/ but not the notebooks.
    pytest.skip("docs/notebooks/ is not present", allow_module_level=True)
SOURCES = sorted(NOTEBOOK_DIR.glob("*.py"))

# Committed outputs must not show a warning (any category: "TqdmWarning: ...",
# "DeprecationWarning: ...") or a crash on the published page.
_WARNING = re.compile(r"\b\w*Warning: ")
_TRACEBACK = "Traceback (most recent call last)"


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
def test_notebook_was_executed(source: Path):
    """Matching sources alone pass for `jupytext --to notebook` without
    `--execute`, which publishes a page with no outputs or figures."""
    nb = jupytext.read(source.with_suffix(".ipynb"))
    unexecuted = [
        i
        for i, cell in enumerate(nb.cells)
        if cell.cell_type == "code"
        and cell.source.strip()
        and cell.get("execution_count") is None
    ]
    assert not unexecuted, (
        f"{source.with_suffix('.ipynb').name} has unexecuted code cells "
        f"{unexecuted}; re-run it with `jupytext --execute`"
    )


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_notebook_outputs_have_no_warnings_or_errors(source: Path):
    nb = jupytext.read(source.with_suffix(".ipynb"))
    found = []
    for i, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                found.append(f"cell {i}: {out.get('ename')}")
                continue
            text = "".join(out.get("text", ""))
            found += [f"cell {i}: {m}" for m in _WARNING.findall(text)]
            if _TRACEBACK in text:
                found.append(f"cell {i}: traceback")
    assert not found, f"{source.with_suffix('.ipynb').name} outputs contain {found}"
