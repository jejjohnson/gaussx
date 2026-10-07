"""lineax's own structural functions and solvers on gaussx operators (gh-410).

The inputs are pinned (`tests/operators/_zoo.py`) and the properties are
structural, so the tolerances are round-off bounds from
`default_tolerances`.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._testing import default_tolerances

from ._zoo import ZOO, exported_operator_classes


def _probe(operator: lx.AbstractLinearOperator):
    struct = operator.in_structure()
    return jr.normal(jr.key(1), struct.shape, dtype=struct.dtype)


@pytest.fixture(params=sorted(ZOO), ids=str)
def operator(request):
    return ZOO[request.param](jr.key(0))


def test_zoo_covers_every_exported_operator():
    built = {type(build(jr.key(0))) for build in ZOO.values()}
    missing = {
        cls
        for cls in exported_operator_classes()
        if not cls.__module__.startswith("lineax")
        and not any(issubclass(cls, b) or issubclass(b, cls) for b in built)
    }
    assert not missing, sorted(cls.__name__ for cls in missing)


def test_diagonal_matches_dense(operator):
    matrix = operator.as_matrix()
    rtol, atol = default_tolerances(matrix)
    assert jnp.allclose(lx.diagonal(operator), jnp.diag(matrix), rtol=rtol, atol=atol)


def test_materialise_and_linearise_preserve_matvec(operator):
    v = _probe(operator)
    expected = operator.mv(v)
    rtol, atol = default_tolerances(expected)
    for fn in (lx.materialise, lx.linearise):
        assert jnp.allclose(fn(operator).mv(v), expected, rtol=rtol, atol=atol)


def test_conj_matches_dense(operator):
    matrix = operator.as_matrix()
    rtol, atol = default_tolerances(matrix)
    conj = lx.conj(operator).as_matrix()
    assert jnp.allclose(conj, jnp.conj(matrix), rtol=rtol, atol=atol)


def test_conj_of_real_operator_is_identity():
    operator = ZOO["kronecker"](jr.key(0))
    assert lx.conj(operator) is operator


@pytest.mark.parametrize("cls", [gaussx.Kronecker, gaussx.BlockDiag])
def test_auto_linear_solve_on_diagonal_factors(cls):
    """``is_diagonal`` makes AutoLinearSolver call ``lx.diagonal`` (gh-410)."""
    D1 = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
    D2 = lx.DiagonalLinearOperator(jnp.array([3.0, 4.0, 5.0]))
    operator = cls(D1, D2)
    assert lx.is_diagonal(operator)
    b = jnp.ones(operator.in_size())
    x = lx.linear_solve(operator, b, lx.AutoLinearSolver(well_posed=True)).value
    expected = b / jnp.diag(operator.as_matrix())
    rtol, atol = default_tolerances(x)
    assert jnp.allclose(x, expected, rtol=rtol, atol=atol)


@pytest.mark.x64_only(reason="LSMR to rtol=1e-8 against lstsq needs float64")
def test_lsmr_on_toeplitz_cholesky():
    operator = ZOO["toeplitz_cholesky"](jr.key(0))
    b = jnp.ones(operator.out_size())
    solver = lx.LSMR(rtol=1e-12, atol=1e-12)
    x = lx.linear_solve(operator, b, solver).value
    expected = jnp.linalg.lstsq(operator.as_matrix(), b)[0]
    assert jnp.allclose(x, expected, rtol=1e-8, atol=1e-8)
