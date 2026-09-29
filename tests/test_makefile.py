"""Every Makefile target must be declared ``.PHONY``.

A target missing from ``.PHONY`` silently becomes a no-op once a file or
directory with its name exists (gh-398: ``test-fast`` and ``init`` were
missing). Pattern rules cannot be phony, so they must depend on one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


MAKEFILE = Path(__file__).resolve().parents[1] / "Makefile"
if not MAKEFILE.is_file():
    # The sdist ships tests/ but not the Makefile.
    pytest.skip("Makefile is not present", allow_module_level=True)


def _phony(text: str) -> set[str]:
    # Join backslash-continued lines so a multi-line .PHONY reads as one.
    joined = text.replace("\\\n", " ")
    return {
        name
        for line in joined.splitlines()
        if line.startswith(".PHONY:")
        for name in line.removeprefix(".PHONY:").split()
    }


def test_every_target_is_phony():
    text = MAKEFILE.read_text()
    # Rule lines: `name: ...` at column 0 (not `X := ...` assignments).
    targets = set(re.findall(r"^([A-Za-z][\w-]*):(?!=)", text, flags=re.MULTILINE))
    missing = sorted(targets - _phony(text))
    assert not missing, f"Makefile targets missing from .PHONY: {missing}"


def test_pattern_rules_always_run():
    """Make reads .PHONY names literally, so a pattern rule such as
    ``check-env-%`` cannot be declared phony; it needs a phony FORCE
    prerequisite, or a file named ``check-env-FOO`` skips the check."""
    text = MAKEFILE.read_text()
    phony = _phony(text)
    rules = re.findall(r"^(\S*%\S*):(?!=)(.*)$", text, flags=re.MULTILINE)
    assert rules, "expected at least the check-env-% guard"
    unforced = [target for target, prereqs in rules if not set(prereqs.split()) & phony]
    assert not unforced, f"pattern rules without a phony prerequisite: {unforced}"
