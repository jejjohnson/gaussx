"""Every Makefile target must be declared ``.PHONY``.

A target missing from ``.PHONY`` silently becomes a no-op once a file or
directory with its name exists (gh-398: ``test-fast`` and ``init`` were
missing).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


MAKEFILE = Path(__file__).resolve().parents[1] / "Makefile"
if not MAKEFILE.is_file():
    # The sdist ships tests/ but not the Makefile.
    pytest.skip("Makefile is not present", allow_module_level=True)


def test_every_target_is_phony():
    text = MAKEFILE.read_text()
    # Join backslash-continued lines so a multi-line .PHONY reads as one.
    joined = text.replace("\\\n", " ")
    phony = {
        name
        for line in joined.splitlines()
        if line.startswith(".PHONY:")
        for name in line.removeprefix(".PHONY:").split()
    }
    # Rule lines: `name: ...` at column 0. Pattern rules (`check-env-%`) and
    # variable assignments (`X := ...`) are not targets to declare.
    targets = set(re.findall(r"^([a-z][a-z0-9-]*):(?!=)", text, flags=re.MULTILINE))
    missing = sorted(targets - phony)
    assert not missing, f"Makefile targets missing from .PHONY: {missing}"
