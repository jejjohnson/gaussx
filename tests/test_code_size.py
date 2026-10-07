"""Size limits for functions and modules in ``src/gaussx`` (gh-415).

A function over 150 lines (docstring included) or a module over 800 lines
is hard to review and leaves its logic untestable except through the whole
body. Existing exceptions are listed below with their current size as a
ceiling: they may shrink, not grow, and an entry is deleted once it is
under the limit.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src" / "gaussx"
if not SRC.is_dir():
    pytest.skip("src/ is not present", allow_module_level=True)

MAX_FUNCTION_LINES = 150
MAX_MODULE_LINES = 800

# "<path relative to src/gaussx>::<function>": ceiling. Reasons:
FUNCTION_ALLOWLIST = {
    # Ensemble recipes; their split is the ensemble epic's (#385) business.
    "_inference/_eki.py::eki_step": 238,
    "_inference/_enkf.py::enkf_analysis": 217,
    # Docstring-dominated: 40, 46 and 48 lines of code under long docstrings.
    "_inference/_laplace.py::laplace_mode": 175,
    "_linalg/_diag_inv.py::diag_inv": 201,
    "_ssm/_nonlinear_filter.py::nonlinear_kalman_filter": 161,
    # Deferred: the in-flight square-root parallel filter work edits these
    # files, and #364 consolidates the Kalman signatures first.
    "_ssm/_kalman.py::kalman_filter": 243,
    "_ssm/_parallel_kalman.py::parallel_kalman_filter": 240,
    "_ssm/_utils.py::_normalise_tv_inputs": 159,
}

# "<path relative to src/gaussx>": ceiling.
MODULE_ALLOWLIST = {
    "__init__.py": 846,  # the public re-export list (grows with the API)
    "_distributions/_gmrf.py": 1213,
    "_operators/_sum_kronecker.py": 903,
}


def _line_count(path: Path) -> int:
    return len(path.read_text().splitlines())


def _modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _functions() -> list[tuple[str, int]]:
    found = []
    for path in _modules():
        rel = path.relative_to(SRC).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                found.append((f"{rel}::{node.name}", node.end_lineno - node.lineno + 1))
    return found


def test_function_sizes():
    too_long = {
        key: lines
        for key, lines in _functions()
        if lines > FUNCTION_ALLOWLIST.get(key, MAX_FUNCTION_LINES)
    }
    assert not too_long, (
        f"functions over {MAX_FUNCTION_LINES} lines (or over their allow-list "
        f"ceiling): {too_long}. Extract helpers."
    )


def test_module_sizes():
    too_long = {
        path.relative_to(SRC).as_posix(): _line_count(path)
        for path in _modules()
        if _line_count(path)
        > MODULE_ALLOWLIST.get(path.relative_to(SRC).as_posix(), MAX_MODULE_LINES)
    }
    assert not too_long, (
        f"modules over {MAX_MODULE_LINES} lines (or over their allow-list "
        f"ceiling): {too_long}. Split the module."
    )


def test_allowlists_are_current():
    """Entries that no longer exist or are back under the limit are removed."""
    sizes = dict(_functions())
    stale = [
        key for key in FUNCTION_ALLOWLIST if sizes.get(key, 0) <= MAX_FUNCTION_LINES
    ]
    stale += [
        rel
        for rel in MODULE_ALLOWLIST
        if not (SRC / rel).is_file() or _line_count(SRC / rel) <= MAX_MODULE_LINES
    ]
    assert not stale, f"remove these allow-list entries: {stale}"
