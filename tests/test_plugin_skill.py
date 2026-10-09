"""Keep the Claude Code plugin's guidance runnable and current.

``plugins/gaussx/skills/structured-gaussian-linalg/SKILL.md`` is what agents
in downstream projects read before writing covariance code, so a stale name
or a broken example there teaches them the wrong API. These tests run its
worked example against a dense reference and check that every ``gaussx.X``
it names is a current public name.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import gaussx


ROOT = Path(__file__).resolve().parents[1]
SKILL = (
    ROOT / "plugins" / "gaussx" / "skills" / "structured-gaussian-linalg" / "SKILL.md"
)
AGENT = ROOT / "plugins" / "gaussx" / "agents" / "gaussx-reuse-reviewer.md"
if not SKILL.is_file():
    # The sdist ships tests/ but not plugins/.
    pytest.skip("the plugin is not present", allow_module_level=True)

TEXT = SKILL.read_text()
_BLOCKS = re.findall(r"```python\n(.*?)```", TEXT, re.S)
_NAMES = re.compile(r"\bgaussx\.([A-Za-z_][A-Za-z0-9_]*)")


def _example() -> dict:
    """Run the "Worked example" block and return its namespace."""
    (block,) = [b for b in _BLOCKS if "neg_mll" in b]
    namespace: dict = {}
    exec(block, namespace)
    return namespace


@pytest.mark.slow
def test_worked_example_matches_dense():
    ns = _example()
    # The Kronecker marginal likelihood equals the dense one: K_y = K₁ ⊗ K₂ + σ²I.
    rbf, x1, x2, y = ns["rbf"], ns["x1"], ns["x2"], ns["y"]
    K = jnp.kron(rbf(x1, 0.3), rbf(x2, 0.3)) + 0.1 * jnp.eye(y.shape[0])
    dense = -gaussx.gaussian_log_prob(
        jnp.zeros_like(y), lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag), y
    )
    # Round-off of two 1200-dim log-densities (x64), not a tuned tolerance.
    np.testing.assert_allclose(ns["value"], dense, rtol=1e-8)
    assert np.isfinite(ns["grad"])
    A = ns["A"]
    np.testing.assert_allclose(
        ns["x"], jnp.linalg.solve(A.as_matrix(), jnp.ones(1000)), rtol=1e-8
    )
    assert ns["means"].shape == (50, 2)
    assert ns["covs"].shape == (50, 2, 2)


@pytest.mark.parametrize("path", [SKILL, AGENT], ids=lambda p: p.name)
def test_named_api_is_current(path: Path):
    names = set(_NAMES.findall(path.read_text())) - {"__all__", "__version__"}
    stale = sorted(n for n in names if n not in gaussx.__all__)
    assert not stale, f"{path.name} names gaussx objects that do not exist: {stale}"
