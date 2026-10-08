"""Keep the README (and the docs home page that includes it) honest.

``docs/index.md`` includes the README's sections with ``pymdownx.snippets``
(gh-399), so the README is the single copy. These tests run its quick start
and check that every name in "What's Inside" is a current public name listed
under the layer its subpackage belongs to in ``docs/architecture.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import gaussx
from gaussx._deprecation import RENAMED
from gaussx._testing import default_tolerances


README = Path(__file__).resolve().parents[1] / "README.md"
if not README.is_file():
    pytest.skip("README.md is not present", allow_module_level=True)

TEXT = README.read_text()

# The layer of each subpackage, as in docs/architecture.md's package layout.
# ``None``: standalone tools outside the stack (docs/api/index.md lists "—").
LAYER_OF_PACKAGE = {
    "gaussx._primitives": "0",
    "gaussx._linalg": "0",
    "gaussx._operators": "1",
    "gaussx._sparse": "1",
    "gaussx._gmrf": "1",
    "gaussx._strategies": "1.5",
    "gaussx._preconditioners": "1.5",
    "gaussx._solve_frontend": "1.5",
    "gaussx._distributions": "2",
    "gaussx._expfam": "2",
    "gaussx._gp": "3",
    "gaussx._ssm": "3",
    "gaussx._quadrature": "3",
    "gaussx._inference": "3",
    "gaussx._sketching": None,
    "gaussx._randomized": None,
}


def _section(name: str) -> str:
    start = TEXT.index(f"<!-- --8<-- [start:{name}] -->")
    end = TEXT.index(f"<!-- --8<-- [end:{name}] -->")
    return TEXT[start:end]


def _names_by_layer() -> dict[str | None, set[str]]:
    """Backticked identifiers in "What's Inside", keyed by their layer heading."""
    layers: dict[str | None, set[str]] = {}
    layer: str | None = None
    for line in _section("inside").splitlines():
        heading = re.match(r"### Layer ([\d.]+)", line)
        if heading:
            layer = heading.group(1)
        elif line.startswith("### Outside the stack"):
            layer = None
        for name in re.findall(r"`([A-Za-z_]\w*)`", line):
            layers.setdefault(layer, set()).add(name)
    return layers


def _gaussx_names() -> list[tuple[str | None, str]]:
    return [
        (layer, name)
        for layer, names in _names_by_layer().items()
        for name in sorted(names)
        if name in gaussx.__all__ or name in RENAMED
    ]


def test_inside_names_exist():
    """Every gaussx name in "What's Inside" is public and not a renamed alias."""
    unknown = {
        name
        for names in _names_by_layer().values()
        for name in names
        if name not in gaussx.__all__
    }
    # Non-gaussx identifiers the prose mentions (Python / equinox / jax).
    unknown -= {"isinstance", "jit", "grad", "vmap", "orthonormal", "solver"}
    assert not unknown, f"README names that are not in gaussx.__all__: {unknown}"
    renamed = {name for _, name in _gaussx_names() if name in RENAMED}
    assert not renamed, f"README uses deprecated names: {renamed}"


@pytest.mark.parametrize(("layer", "name"), _gaussx_names())
def test_inside_layer_matches_architecture(layer, name):
    module = getattr(gaussx, name).__module__
    package = ".".join(module.split(".")[:2])
    assert LAYER_OF_PACKAGE[package] == layer, (
        f"README lists {name} ({package}) under layer {layer}, but "
        f"docs/architecture.md puts {package} in layer {LAYER_OF_PACKAGE[package]}"
    )


def test_quickstart_runs():
    pytest.importorskip("numpyro")  # MultivariateNormal needs numpyro.
    block = re.search(r"```python\n(.*?)```", _section("quickstart"), re.DOTALL)
    assert block is not None
    exec(compile(block.group(1), "README.md:quickstart", "exec"), {})


@pytest.mark.slow
def test_gp_example_runs():
    """The GP example runs, and its exact and Kronecker likelihoods agree."""
    block = re.search(r"```python\n(.*?)```", _section("gp-example"), re.DOTALL)
    assert block is not None
    ns: dict = {}
    exec(compile(block.group(1), "README.md:gp-example", "exec"), ns)
    rtol, _ = default_tolerances(ns["mll_exact"])
    assert abs(ns["mll_grid"] - ns["mll_exact"]) <= 1e2 * rtol * abs(ns["mll_exact"])
    assert ns["elbo"] <= ns["mll_exact"]
