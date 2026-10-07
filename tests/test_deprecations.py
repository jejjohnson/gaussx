"""Every deprecation names its removal version, and is removed on time (gh-300).

The deprecation policy (docs/api/index.md) promises a removal version in each
warning. These tests read every ``warn_deprecated(...)`` call in the package:
one without a version fails, and once ``gaussx.__version__`` reaches a
promised version the guard fails, so CI goes red on the release that is due
to remove the code.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from packaging.version import Version

import gaussx
from gaussx import _deprecation


SRC = Path(gaussx.__file__).resolve().parent
_VERSION = r"gaussx (\d+\.\d+\.\d+)"


def _deprecation_calls() -> list[tuple[str, str]]:
    """``(location, message source)`` of every ``warn_deprecated`` call."""
    calls = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "warn_deprecated"
                and node.args
            ):
                where = f"{path.relative_to(SRC.parent)}:{node.lineno}"
                calls.append((where, ast.unparse(node.args[0])))
    return calls


def _promised_versions(message: str) -> set[str]:
    message = message.replace("{RENAMED_REMOVAL}", _deprecation.RENAMED_REMOVAL)
    return set(re.findall(_VERSION, message))


_CALLS = _deprecation_calls()


def test_deprecation_calls_are_found() -> None:
    assert len(_CALLS) > 20, "the AST scan found too few warn_deprecated calls"


@pytest.mark.parametrize(("where", "message"), _CALLS, ids=[w for w, _ in _CALLS])
def test_every_deprecation_names_its_removal_version(where: str, message: str) -> None:
    assert _promised_versions(message), (
        f"{where}: the deprecation message names no removal version "
        f"('... in gaussx X.Y.Z'): {message}"
    )


_PROMISED = sorted(
    {v for _, message in _CALLS for v in _promised_versions(message)}, key=Version
)


@pytest.mark.parametrize("version", _PROMISED)
def test_deprecations_are_removed_on_schedule(version: str) -> None:
    due = [w for w, message in _CALLS if version in _promised_versions(message)]
    assert Version(gaussx.__version__) < Version(version), (
        f"gaussx {gaussx.__version__} has shipped the removal version {version}; "
        f"remove the deprecated code at: {due}"
    )
