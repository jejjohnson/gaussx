"""List test functions that take the ``getkey`` fixture (gh-311).

For each test function under ``tests/`` that takes ``getkey``, report
whether it calls ``getkey()``, whether it references ``getkey`` at all, and
whether it bounds a statistic (``assert_sample_moments``, a KS test or an
SEM-style bound). With ``--dead`` it lists only the functions that take
``getkey`` and never reference it, which should not exist.

Usage:
    uv run python scripts/find_getkey_tests.py          # summary counts
    uv run python scripts/find_getkey_tests.py --dead   # dead parameters
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


STATISTICAL = ("assert_sample_moments", "kstest", "ks_2samp", "sem", "logdet_and_error")


def scan(root: Path = Path("tests")):
    rows = []
    for path in sorted(root.rglob("test_*.py")):
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("test"):
                continue
            if "getkey" not in [a.arg for a in fn.args.args]:
                continue
            body = [n for stmt in fn.body for n in ast.walk(stmt)]
            calls = any(
                isinstance(n, ast.Call) and getattr(n.func, "id", None) == "getkey"
                for n in body
            )
            refs = any(isinstance(n, ast.Name) and n.id == "getkey" for n in body)
            source = ast.get_source_segment(path.read_text(), fn) or ""
            stat = any(word in source for word in STATISTICAL)
            rows.append((f"{path}::{fn.name}", fn.lineno, calls, refs, stat))
    return rows


def main(argv: list[str]) -> int:
    rows = scan()
    if "--dead" in argv:
        for name, line, _, refs, _ in rows:
            if not refs:
                print(f"{name} (line {line})")
        return 0
    print(f"take getkey:            {len(rows)}")
    print(f"  call getkey():        {sum(r[2] for r in rows)}")
    print(f"  never reference it:   {sum(not r[3] for r in rows)}")
    print(f"  bound a statistic:    {sum(r[4] for r in rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
