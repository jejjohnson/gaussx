"""Keep the operator × primitive table in ``docs/architecture.md`` honest.

The table is hand-written (it carries complexity notes the code cannot
generate), so it drifts whenever a primitive gains or loses an ``isinstance``
branch. These tests read the ``isinstance(operator, ...)`` chain in the body
of each primitive with `ast` and check the table against it:

- every class a primitive branches on has a row;
- a **dense** / **lazy** cell (the fallback) really has no branch;
- any other cell has a branch in at least one of its column's primitives.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import lineax as lx
import pytest

import gaussx


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "docs" / "architecture.md"
PRIMITIVES = ROOT / "src" / "gaussx" / "_primitives"
if not (ARCHITECTURE.is_file() and PRIMITIVES.is_dir()):
    # The sdist ships tests/ but not docs/.
    pytest.skip("docs/ or src/ is not present", allow_module_level=True)

# Cells that mean "no branch: the primitive takes its fallback".
FALLBACK_CELLS = {"dense", "lazy"}

_BACKTICKED = re.compile(r"`([^`]+)`")


def _isinstance_classes(primitive: str) -> set[str]:
    """Class names in the ``isinstance(operator, ...)`` tests of a primitive.

    Only the primitive's own function body is read (not its helpers), which
    is where the structural dispatch chain lives.
    """
    tree = ast.parse((PRIMITIVES / f"_{primitive}.py").read_text())
    # The last definition is the implementation (earlier ones are overloads).
    func = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == primitive
    ][-1]
    names: set[str] = set()

    def collect(node: ast.expr) -> None:
        # ``lx.Foo`` -> ``Foo``; ``A | B`` and ``(A, B)`` -> both.
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.BinOp):
            collect(node.left)
            collect(node.right)
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                collect(elt)

    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "operator"
        ):
            collect(node.args[1])
    return names


def _table() -> tuple[list[list[str]], list[list[str]]]:
    """The table's column primitives and its rows (cells as stripped text)."""
    lines = ARCHITECTURE.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Operator |"))
    header = [c.strip() for c in lines[start].strip("|").split("|")]
    columns = [_BACKTICKED.findall(cell) for cell in header[1:]]
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return columns, rows


def _matches(doc_name: str, class_name: str) -> bool:
    """``Tagged`` in the table stands for ``lx.TaggedLinearOperator``."""
    return class_name in (doc_name, f"{doc_name}LinearOperator")


def _row_names(row: list[str]) -> list[str]:
    """The operator names a row covers (its first cell's leading names)."""
    # ``InverseOperator`` (from `inv`): only the names before any prose.
    head = row[0].split("(")[0]
    return _BACKTICKED.findall(head)


COLUMNS, ROWS = _table()
PRIMITIVE_NAMES = sorted({p for column in COLUMNS for p in column})
BRANCHES = {p: _isinstance_classes(p) for p in PRIMITIVE_NAMES}
NAMED_ROWS = [row for row in ROWS if _row_names(row)]


def test_table_shape():
    expected = ["solve", "logdet", "cholesky", "diag", "trace", "sqrt", "inv"]
    assert set(PRIMITIVE_NAMES) == set(expected)
    assert all(len(row) == len(COLUMNS) + 1 for row in ROWS)
    assert ROWS[-1][0] == "Everything else"


def _resolve(name: str) -> object | None:
    """The gaussx / lineax object a table name stands for, if any."""
    namespaces = [gaussx, lx] + [
        importlib.import_module(f"gaussx._primitives._{p}") for p in PRIMITIVE_NAMES
    ]
    for ns in namespaces:
        for attr in (name, f"{name}LinearOperator"):
            if hasattr(ns, attr):
                return getattr(ns, attr)
    return None


@pytest.mark.parametrize("row", NAMED_ROWS, ids=lambda row: row[0])
def test_row_names_resolve(row):
    """Every name in the operator column is a real gaussx or lineax object."""
    for name in _row_names(row):
        assert _resolve(name) is not None, (
            f"{name!r} in docs/architecture.md is not a gaussx or lineax name"
        )


@pytest.mark.parametrize("primitive", PRIMITIVE_NAMES)
def test_every_branch_has_a_row(primitive):
    documented = [name for row in NAMED_ROWS for name in _row_names(row)]
    missing = sorted(
        cls
        for cls in BRANCHES[primitive]
        if not any(_matches(name, cls) for name in documented)
    )
    assert not missing, (
        f"`{primitive}` branches on {missing}, which have no row in the "
        "operator × primitive table in docs/architecture.md"
    )


@pytest.mark.parametrize("row", NAMED_ROWS, ids=lambda row: row[0])
def test_cells_match_branches(row):
    # A factory function such as ``circulant`` is covered by the class it
    # returns, which shares its row.
    names = [name for name in _row_names(row) if isinstance(_resolve(name), type)]
    for primitives, cell in zip(COLUMNS, row[1:], strict=True):
        for name in names:
            has_branch = [
                any(_matches(name, cls) for cls in BRANCHES[p]) for p in primitives
            ]
            where = f"{name} × {'/'.join(primitives)}"
            if cell in FALLBACK_CELLS:
                assert not any(has_branch), (
                    f"{where} is documented as '{cell}' but the primitive "
                    "branches on it"
                )
            else:
                assert any(has_branch), (
                    f"{where} is documented as '{cell}' but no primitive in "
                    "the column branches on it"
                )
