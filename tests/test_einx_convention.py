"""Ratchet for the einx convention (AGENTS.md, gh-404).

ruff's ``TID251`` bans ``jnp.einsum`` / ``jnp.transpose`` / ``jnp.moveaxis`` /
``jnp.reshape`` outright. The other constructs the convention replaces with
einx cannot be banned by ruff: a method ``.reshape(`` / ``.ravel()``, an
``axis=`` keyword, a ``[:, None]`` broadcast, ``jnp.swapaxes`` and
``jnp.expand_dims``. Older code still has them, so this test caps each count
at its current value: new code may not add any, and converting old sites
lowers the count. Lower the ceiling in the same PR when you convert some.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

# Ceilings, as counted on the PR that introduced this test. Only ever lower them.
CEILINGS = {
    "src/gaussx": {
        "method .reshape/.ravel": 32,
        "axis= keyword": 92,
        "None-index broadcast": 127,
        "jnp.swapaxes/expand_dims": 22,
    },
    "tests": {
        "method .reshape/.ravel": 56,
        "axis= keyword": 66,
        "None-index broadcast": 135,
        "jnp.swapaxes/expand_dims": 10,
    },
}


def _has_none(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Tuple):
        return any(_has_none(elt) for elt in node.elts)
    return False


def _count(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for file in path.rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if func.attr in ("reshape", "ravel") and not (
                        isinstance(func.value, ast.Name) and func.value.id == "jnp"
                    ):
                        counts["method .reshape/.ravel"] += 1
                    if (
                        func.attr in ("swapaxes", "expand_dims")
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "jnp"
                    ):
                        counts["jnp.swapaxes/expand_dims"] += 1
                counts["axis= keyword"] += sum(kw.arg == "axis" for kw in node.keywords)
            elif isinstance(node, ast.Subscript) and _has_none(node.slice):
                counts["None-index broadcast"] += 1
    return counts


@pytest.mark.parametrize("tree", sorted(CEILINGS))
def test_no_new_non_einx_array_ops(tree):
    path = ROOT / tree
    if not path.is_dir():
        pytest.skip(f"{tree} is not present")
    counts = _count(path)
    over = {
        kind: (counts[kind], ceiling)
        for kind, ceiling in CEILINGS[tree].items()
        if counts[kind] > ceiling
    }
    assert not over, (
        f"{tree} adds non-einx array operations (count, ceiling): {over}. Use "
        "the gaussx._einx wrappers (rearrange / reduce / repeat / einsum) or "
        "einx.add / subtract / multiply / divide instead; see AGENTS.md."
    )
