"""Keep the Claude Code plugin's guidance runnable and current.

``plugins/gaussx/`` is what agents in downstream projects read before writing
covariance code, so a stale name or a broken example there teaches them the
wrong API. The plugin is a shelf (``tree/shelf.toml``): three listed skills,
each with a ``shelf/`` folder of entries, and one contract card per unit.

The fast tests run ``scripts/skill_shelf.py``'s check (the generated parts of
each ``SKILL.md`` are current, every public name belongs to exactly one entry
whose file names it, the line budgets hold) and check that every ``gaussx.X``
the plugin names is a current public name. The slow tests run every ``python``
block of the plugin, and the front door's worked example against a dense
reference.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import gaussx


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "gaussx"
SKILL = PLUGIN / "skills" / "structured-gaussian-linalg" / "SKILL.md"
if not SKILL.is_file():
    # The sdist ships tests/ but not plugins/.
    pytest.skip("the plugin is not present", allow_module_level=True)

_spec = importlib.util.spec_from_file_location(
    "skill_shelf", ROOT / "scripts" / "skill_shelf.py"
)
assert _spec is not None and _spec.loader is not None
ss = importlib.util.module_from_spec(_spec)
sys.modules["skill_shelf"] = ss  # dataclasses look their module up
_spec.loader.exec_module(ss)

_PAGES = sorted(PLUGIN.rglob("*.md"))
_EXAMPLES = ss.examples()
_NAMES = re.compile(r"\bgaussx\.([A-Za-z_][A-Za-z0-9_]*)")


def _run(block: str) -> dict:
    namespace: dict = {"__name__": "__plugin_example__"}
    exec(block, namespace)
    return namespace


def test_shelf_is_current_and_covers_the_public_api():
    assert ss.violations(ss.load(), ss.public_names()) == [], "run make skill-shelf"


def test_public_names_read_from_source_match_the_package():
    # The script reads __all__ from the source; numpyro-only names are there
    # whether or not the extra is installed.
    assert set(gaussx.__all__) <= ss.public_names()
    if "MultivariateNormal" in gaussx.__all__:
        assert set(gaussx.__all__) == ss.public_names()


@pytest.mark.parametrize("path", _PAGES, ids=lambda p: str(p.relative_to(PLUGIN)))
def test_named_api_is_current(path: Path):
    names = set(_NAMES.findall(path.read_text())) - {"__all__", "__version__"}
    stale = sorted(names - ss.public_names())
    assert not stale, f"{path.name} names gaussx objects that do not exist: {stale}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "block",
    [
        pytest.param(block, id=f"{path.relative_to(PLUGIN)}:{i}")
        for i, (path, block) in enumerate(_EXAMPLES)
    ],
)
def test_example_runs(block: str):
    _run(block)


@pytest.mark.slow
@pytest.mark.x64_only(reason="compares two solves at condition number 1e8")
def test_worked_example_matches_dense():
    (block,) = [b for p, b in _EXAMPLES if p == SKILL and "quad_and_logdet" in b]
    ns = _run(block)
    # yᵀK⁻¹y + log|K| on the Kronecker operator equals the dense one.
    rbf, x1, x2, y = ns["rbf"], ns["x1"], ns["x2"], ns["y"]
    K = jnp.kron(rbf(x1, 0.3), rbf(x2, 0.3))
    dense = y @ jnp.linalg.solve(K, y) + jnp.linalg.slogdet(K)[1]
    # K has condition number ~1e8 from its 1e-4 jitter, which bounds the
    # agreement of two x64 solves at about 1e-8 relative; 1e-6 leaves margin.
    np.testing.assert_allclose(ns["value"], dense, rtol=1e-6)
    A = ns["A"]
    np.testing.assert_allclose(
        ns["x"], jnp.linalg.solve(A.as_matrix(), jnp.ones(200)), rtol=1e-6
    )
    assert isinstance(A, lx.AbstractLinearOperator)
