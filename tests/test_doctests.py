"""Run the ``>>>`` examples in gaussx's docstrings (gh-389).

Examples are part of the API reference, so they must run. Each module's
docstring examples run as one test, with ``ELLIPSIS`` and
``NORMALIZE_WHITESPACE``. This file is in the Makefile's ``NO_X64_TESTS``, so
the examples run both with x64 on (the suite's default) and off (a user's
default): print rounded floats, shapes or type names rather than raw arrays,
whose repr carries the dtype.
"""

from __future__ import annotations

import doctest
import importlib
import importlib.util
import pkgutil
from pathlib import Path

import pytest

import gaussx


_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

# The examples policy (CODE_REVIEW.md): the Layer-0 primitives and the
# headline operators each carry at least one runnable example.
_MUST_HAVE_EXAMPLES = [
    "solve",
    "logdet",
    "cholesky",
    "diag",
    "trace",
    "sqrt",
    "inv",
    "Kronecker",
    "BlockDiag",
    "LowRankUpdate",
    "KroneckerSum",
    "SumOfKroneckers",
    "BlockTriDiag",
    "Toeplitz",
]


def _module_names() -> list[str]:
    """Modules whose source has a ``>>>`` (read as text, not imported)."""
    names = []
    for info in pkgutil.walk_packages(gaussx.__path__, "gaussx."):
        spec = importlib.util.find_spec(info.name)
        origin = spec.origin if spec else None
        if origin and origin.endswith(".py") and ">>>" in Path(origin).read_text():
            names.append(info.name)
    return sorted(names)


@pytest.mark.parametrize("module_name", _module_names())
def test_docstring_examples(module_name):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        if err.name and err.name.split(".")[0] == "numpyro":
            pytest.skip("numpyro is not installed")
        raise
    tests = [
        test
        for test in doctest.DocTestFinder().find(module, module_name)
        if test.examples and test.name.startswith(module_name)
    ]
    report: list[str] = []
    runner = doctest.DocTestRunner(optionflags=_FLAGS)
    for test in tests:
        runner.run(test, out=report.append)
    assert runner.failures == 0, "".join(report)


@pytest.mark.parametrize("name", _MUST_HAVE_EXAMPLES)
def test_headline_api_has_an_example(name):
    assert ">>>" in (getattr(gaussx, name).__doc__ or "")
