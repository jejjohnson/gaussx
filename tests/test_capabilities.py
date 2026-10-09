"""The capability index (docs/capabilities.md) is current.

Agents and contributors search the index before writing a helper ("Reuse
before you write" in AGENTS.md), so a stale index sends them to re-implement
something that exists. ``scripts/capabilities.py`` regenerates it; this
runs its ``--check`` in the fast lane. Upstream sections are compared only
at the versions the index records, so the weekly latest-deps run does not
fail on an upstream docstring edit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "capabilities.py"
if not SCRIPT.is_file() or not (ROOT / "docs" / "capabilities.md").is_file():
    # The sdist ships tests/ but not scripts/ or the index.
    pytest.skip("scripts/capabilities.py is not present", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("capabilities", SCRIPT)
assert _spec is not None and _spec.loader is not None
capabilities = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capabilities)


def test_capability_index_is_current() -> None:
    current = capabilities.INDEX.read_text(encoding="utf-8")
    assert not capabilities.stale(current, capabilities.render()), (
        "docs/capabilities.md is stale: run `make capabilities` and commit it."
    )


def test_no_name_shadows_upstream() -> None:
    assert not capabilities.shadows(), (
        "a gaussx name shadows a lineax / optimistix name with another object; "
        "rename it, or add it to ALLOWED_SHADOWS in scripts/capabilities.py "
        "with a reason."
    )
