"""In-function imports of gaussx modules are import-cycle edges only (gh-407).

An import inside a function hides a dependency from readers and tools, so
gaussx imports its own modules at module scope wherever that does not
create an import cycle. The remaining in-function imports each close a
cycle (mostly the dispatch cycle between ``_primitives``, ``_operators``,
``_strategies`` and ``_sparse``; see ``docs/architecture.md``). Each one is
listed here and carries a ``# lazy import, cycle: ...`` comment naming the
module chain that imports back. Lazy ``__getattr__`` hooks (deprecated
aliases, numpyro-backed names) are exempt.

To add an edge: check that the import really fails at module scope
(``python -c "import gaussx"``), add the comment and the entry below.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src" / "gaussx"
if not SRC.is_dir():
    pytest.skip("src/ is not present", allow_module_level=True)

COMMENT = "# lazy import, cycle:"

# (importing module, imported module), both relative to ``gaussx``.
ALLOWED_LAZY_IMPORTS = {
    ("_distributions._mvn_base", "_distributions._numpyro_kl"),
    ("_linalg._schur", "_linalg._linalg"),
    ("_operators._factored_eigen", "_operators._spectral_function"),
    ("_operators._masked", "_primitives._solve"),
    ("_operators._sum_kronecker", "_primitives._cholesky"),
    ("_operators._sum_kronecker", "_primitives._solve"),
    ("_operators._sum_kronecker", "_primitives._sqrt"),
    ("_operators._utils", "_primitives._diag"),
    ("_primitives._inv", "_primitives._solve"),
    ("_primitives._logdet", "_strategies._auto"),
    ("_primitives._logdet", "_strategies._slq_logdet"),
    ("_primitives._logdet", "_strategies._sparse_cholesky"),
    ("_primitives._solve", "_strategies._auto"),
    ("_primitives._solve", "_strategies._base"),
    ("_primitives._solve", "_strategies._cg"),
    ("_quadrature._integrator", "_quadrature._unscented"),
    ("_sparse._numeric", "_primitives._cholesky"),
    ("_sparse._numeric", "_primitives._solve"),
    ("_sparse._symbolic", "_sparse._cholmod"),
    ("_sparse._takahashi", "_linalg._selected_inverse"),
    ("_ssm._parallel_kalman", "_ssm._parallel_kalman_sqrt"),
}


def _module(path: Path) -> str:
    return ".".join(path.relative_to(SRC).with_suffix("").parts)


def _lazy_imports() -> list[tuple[str, str, int, bool]]:
    """``(module, target, line, has_comment)`` for each in-function import."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        lines = path.read_text().splitlines()
        tree = ast.parse("\n".join(lines))
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if func.name == "__getattr__":
                continue
            for node in ast.walk(func):
                if not (
                    isinstance(node, ast.ImportFrom)
                    and (node.module or "").startswith("gaussx")
                ):
                    continue
                # The comment block directly above the import.
                i = node.lineno - 2
                block = []
                while i >= 0 and lines[i].strip().startswith("#"):
                    block.append(lines[i].strip())
                    i -= 1
                has_comment = any(line.startswith(COMMENT) for line in block)
                target = node.module.removeprefix("gaussx.")
                found.append((_module(path), target, node.lineno, has_comment))
    return found


def test_lazy_imports_are_documented_cycle_edges():
    unexpected = [
        f"{module}:{line} -> {target}"
        for module, target, line, _ in _lazy_imports()
        if (module, target) not in ALLOWED_LAZY_IMPORTS
    ]
    assert not unexpected, (
        f"in-function gaussx imports outside the allow-list: {unexpected}. "
        "Import at module scope, or (if that creates an import cycle) add the "
        "edge to ALLOWED_LAZY_IMPORTS with a cycle comment."
    )
    uncommented = [
        f"{module}:{line} -> {target}"
        for module, target, line, has_comment in _lazy_imports()
        if not has_comment
    ]
    assert not uncommented, f"lazy imports without {COMMENT!r}: {uncommented}"


def test_allow_list_has_no_stale_entries():
    present = {(module, target) for module, target, _, _ in _lazy_imports()}
    stale = sorted(ALLOWED_LAZY_IMPORTS - present)
    assert not stale, f"remove these entries from ALLOWED_LAZY_IMPORTS: {stale}"


def test_import_without_optional_dependencies():
    """Module-scope imports must not pull numpyro into ``import gaussx``."""
    blocker = (
        "import importlib.abc, sys\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name.partition('.')[0] == 'numpyro':\n"
        "            raise ModuleNotFoundError(name, name=name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import gaussx\n"
        "assert 'numpyro' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", blocker], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
