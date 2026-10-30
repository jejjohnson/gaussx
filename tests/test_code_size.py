"""Size limits for functions and modules in ``src/gaussx`` (gh-415).

A function over 150 lines or a module over 800 lines is hard to review and
leaves its logic untestable except through the whole body. Only code lines
count: docstrings, comment-only lines and blank lines are excluded, so the
equation comments and docstrings the house style asks for never push a
function over the limit. Existing exceptions are listed below with their
current size as a ceiling: they may shrink, not grow, and an entry is
deleted once it is under the limit.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src" / "gaussx"
if not SRC.is_dir():
    pytest.skip("src/ is not present", allow_module_level=True)

MAX_FUNCTION_LINES = 150
MAX_MODULE_LINES = 800

# "<path relative to src/gaussx>::<function>": ceiling. Reasons:
FUNCTION_ALLOWLIST: dict[str, int] = {}

# "<path relative to src/gaussx>": ceiling.
MODULE_ALLOWLIST = {
    "__init__.py": 823,  # the public re-export list (grows with the API)
}


def _non_code_lines(source: str) -> set[int]:
    """Line numbers holding only a docstring, a comment or whitespace."""
    lines = source.splitlines()
    skip = {i for i, line in enumerate(lines, start=1) if not line.strip()}
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    skip |= {
        tok.start[0]
        for tok in tokens
        if tok.type == tokenize.COMMENT
        and lines[tok.start[0] - 1].lstrip().startswith("#")
    }
    for node in ast.walk(ast.parse(source)):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        doc = node.body[0] if node.body else None
        if (
            isinstance(doc, ast.Expr)
            and isinstance(doc.value, ast.Constant)
            and isinstance(doc.value.value, str)
        ):
            skip |= set(range(doc.lineno, doc.end_lineno + 1))
    return skip


def _code_lines(source: str, start: int, end: int) -> int:
    skip = _non_code_lines(source)
    return sum(1 for line in range(start, end + 1) if line not in skip)


def _line_count(path: Path) -> int:
    source = path.read_text()
    return _code_lines(source, 1, len(source.splitlines()))


def _modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _functions() -> list[tuple[str, int]]:
    found = []
    for path in _modules():
        rel = path.relative_to(SRC).as_posix()
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                size = _code_lines(source, node.lineno, node.end_lineno)
                found.append((f"{rel}::{node.name}", size))
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
