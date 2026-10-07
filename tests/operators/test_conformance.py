"""Operator-zoo conformance suite: every operator x every primitive (gh-417).

One registry (`ZOO`) holds a small instance of every exported operator
class, wrappers of a representative subset (``Tagged``, ``Mul``, ``Div``,
``Neg``, ``Add``) and the degenerate shapes of epic #276 (N = 1
``BlockTriDiag``, rank-0 and zero-weight ``LowRankUpdate``, 1 x 1, a
three-term ``SumOfKroneckers``). For each case the suite checks

- ``mv`` and the transpose against ``as_matrix``;
- every applicable primitive against a dense reference;
- for the primitives `docs/architecture.md` promises are structured, that
  the primitive never calls ``as_matrix`` on the case's structured class
  (a spy that raises; results are densified only after the call);
- (slow) that ``solve`` + ``logdet`` are finite under ``jit`` and ``grad``.

Cells that are known to fail are ``xfail(strict=True)`` with the issue that
tracks them, so a fix that lands without updating the registry turns the
suite red: the registry doubles as the live coverage matrix.

Inputs are pinned (``jr.key(0)``) and the properties are structural, so the
dense comparisons are round-off bounds (x64; the float32 lane runs the
matvec/transpose checks).
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx
from gaussx._einx import einsum, rearrange
from gaussx._primitives._inv import InverseOperator
from gaussx._testing import (
    default_tolerances,
    random_kronecker_pd,
    random_pd_operator,
    random_spd_block_tridiag,
    random_sum_of_kroneckers_pd,
)

from ._zoo import ZOO as BASE_ZOO, exported_operator_classes


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ALL = frozenset(
    {
        "solve",
        "logdet",
        "diag",
        "trace",
        "inv",
        "cholesky",
        "sqrt",
        "eigvals",
        "frobenius_norm",
        "submatrix",
    }
)
_SQUARE = frozenset({"solve", "logdet", "diag", "trace", "inv", "frobenius_norm"})
_KRON_LIKE = frozenset(
    {"solve", "logdet", "cholesky", "diag", "trace", "sqrt", "inv", "eigvals"}
)


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    build: Callable[[], lx.AbstractLinearOperator]
    prims: frozenset[str] = _ALL
    structured: frozenset[str] = frozenset()
    spy: type | None = None
    xfail: dict[str, str] = dataclasses.field(default_factory=dict)


def _base(name: str) -> Callable[[], lx.AbstractLinearOperator]:
    return lambda: BASE_ZOO[name](jr.key(0))


def _kron() -> gaussx.Kronecker:
    return random_kronecker_pd(jr.key(0), (2, 3))


def _psd_matrix(n: int) -> lx.MatrixLinearOperator:
    return random_pd_operator(jr.key(0), n)


def _block_tridiag_n1() -> gaussx.BlockTriDiag:
    return random_spd_block_tridiag(jr.key(0), 1, 2)


def _low_rank(d) -> gaussx.LowRankUpdate:
    d = jnp.asarray(d, dtype=jnp.result_type(float))
    U = 0.3 * jr.normal(jr.key(1), (4, d.shape[0]), dtype=d.dtype)
    base = lx.DiagonalLinearOperator(jnp.linspace(1.0, 2.0, 4, dtype=d.dtype))
    return gaussx.LowRankUpdate(base, U, d)


def _kronecker_sum(*tags) -> gaussx.KroneckerSum:
    return gaussx.KroneckerSum(
        random_pd_operator(jr.key(0), 2, tags=tags),
        random_pd_operator(jr.key(1), 3, tags=tags),
    )


def _circulant_pd() -> gaussx.DiagonalisedOperator:
    return gaussx.Circulant(
        jnp.array([3.0, 1.0, 0.5, 0.5, 1.0]),
        symmetric=True,
        tags=lx.positive_semidefinite_tag,
    )


def _three_term() -> gaussx.SumOfKroneckers:
    return random_sum_of_kroneckers_pd(jr.key(0), (2, 3), num_terms=3)


ZOO: list[Case] = [
    # Dense and degenerate.
    Case("dense", lambda: _psd_matrix(4)),
    Case("one_by_one", lambda: _psd_matrix(1)),
    # Kronecker and wrappers of it.
    Case("kronecker", _kron, structured=_KRON_LIKE, spy=gaussx.Kronecker),
    Case(
        "tagged_kronecker",
        lambda: lx.TaggedLinearOperator(_kron(), lx.positive_semidefinite_tag),
        structured=_KRON_LIKE,
        spy=gaussx.Kronecker,
    ),
    Case(
        "scaled_kronecker",
        lambda: 2.0 * _kron(),
        structured=_KRON_LIKE - {"eigvals"},
        spy=gaussx.Kronecker,
    ),
    Case(
        "div_kronecker",
        lambda: _kron() / 2.0,
        structured=_KRON_LIKE - {"eigvals"},
        spy=gaussx.Kronecker,
    ),
    Case(
        "neg_kronecker",
        lambda: -_kron(),
        prims=_SQUARE,
        structured=frozenset({"solve", "logdet", "diag", "trace", "inv"}),
        spy=gaussx.Kronecker,
    ),
    Case(
        "kronecker_plus_identity",
        lambda: _kron() + 0.5 * lx.IdentityLinearOperator(_kron().in_structure()),
        structured=frozenset({"diag", "trace", "submatrix"}),
        spy=gaussx.Kronecker,
    ),
    Case(
        "block_diag",
        _base("block_diag"),
        structured=_KRON_LIKE | {"submatrix", "frobenius_norm"},
        spy=gaussx.BlockDiag,
    ),
    Case(
        "low_rank_update",
        _base("low_rank_update"),
        structured=frozenset({"solve", "logdet", "diag", "trace", "inv", "submatrix"}),
        spy=gaussx.LowRankUpdate,
    ),
    Case(
        "low_rank_rank0",
        lambda: _low_rank(np.zeros(0)),
        structured=frozenset({"solve", "logdet", "diag", "trace", "inv"}),
        spy=gaussx.LowRankUpdate,
    ),
    Case(
        "low_rank_zero_weight",
        lambda: _low_rank([1.0, 0.0]),
        structured=frozenset({"solve", "logdet", "diag", "trace", "inv", "submatrix"}),
        spy=gaussx.LowRankUpdate,
    ),
    Case(
        "kronecker_sum",
        lambda: _kronecker_sum(lx.positive_semidefinite_tag),
        structured=frozenset({"solve", "logdet", "diag", "trace", "sqrt", "inv"}),
        spy=gaussx.KroneckerSum,
    ),
    Case(
        "kronecker_sum_untagged",
        _kronecker_sum,
        # sqrt rejects factors not tagged symmetric by design (ValueError).
        prims=_ALL - {"sqrt"},
        structured=frozenset({"logdet", "diag", "trace"}),
        spy=gaussx.KroneckerSum,
    ),
    Case("kronecker_sum_sqrt", _base("kronecker_sum_sqrt")),
    Case(
        "sum_of_kroneckers",
        _base("sum_of_kroneckers"),
        structured=frozenset({"solve", "logdet", "diag", "trace"}),
        spy=gaussx.SumOfKroneckers,
    ),
    Case(
        "sum_of_kroneckers_3",
        _three_term,
        structured=frozenset({"diag", "trace"}),
        spy=gaussx.SumOfKroneckers,
    ),
    Case("sum_kronecker_sqrt", _base("sum_kronecker_sqrt")),
    Case(
        "block_tridiag",
        _base("block_tridiag"),
        structured=frozenset({"solve", "logdet", "cholesky", "diag", "trace"}),
        spy=gaussx.BlockTriDiag,
    ),
    Case(
        "block_tridiag_n1",
        _block_tridiag_n1,
        structured=frozenset({"solve", "logdet", "cholesky", "diag", "trace"}),
        spy=gaussx.BlockTriDiag,
    ),
    Case(
        "lower_block_tridiag",
        _base("lower_block_tridiag"),
        prims=_SQUARE,
        structured=frozenset({"solve", "logdet", "diag", "trace"}),
        spy=gaussx.LowerBlockTriDiag,
    ),
    Case(
        "upper_block_tridiag",
        _base("upper_block_tridiag"),
        prims=_SQUARE,
        structured=frozenset({"solve", "logdet", "diag", "trace"}),
        spy=gaussx.UpperBlockTriDiag,
    ),
    Case(
        "toeplitz",
        _base("toeplitz"),
        structured=frozenset({"diag", "trace"}),
        spy=gaussx.Toeplitz,
    ),
    Case(
        "circulant_pd",
        _circulant_pd,
        structured=frozenset({"solve", "logdet", "diag", "trace", "sqrt", "inv"}),
        spy=gaussx.DiagonalisedOperator,
    ),
    Case(
        "circulant",
        _base("circulant"),
        prims=_SQUARE,
        structured=frozenset({"solve", "logdet", "diag", "trace", "inv"}),
        spy=gaussx.DiagonalisedOperator,
    ),
    Case(
        "circulant_complex",
        _base("circulant_complex"),
        prims=_SQUARE,
        structured=frozenset({"solve", "logdet", "diag", "trace", "inv"}),
        spy=gaussx.DiagonalisedOperator,
    ),
    Case("interpolated", _base("interpolated")),
    Case("masked", _base("masked"), prims=_SQUARE),
    Case(
        "sparse",
        _base("sparse"),
        prims=_ALL - {"cholesky"},  # a permuted SparseCholeskyFactor
        structured=frozenset({"diag"}),
        spy=gaussx.SparseOperator,
    ),
    Case(
        "spectral_function",
        _base("spectral_function"),
        structured=frozenset({"solve", "diag"}),
        spy=gaussx.SpectralFunction,
    ),
    Case(
        "inverse_operator",
        lambda: gaussx.inv(_psd_matrix(4)),
        structured=frozenset({"solve", "inv"}),
        spy=InverseOperator,
    ),
    # Rectangular: matvec / transpose only.
    Case("toeplitz_cholesky", _base("toeplitz_cholesky"), prims=frozenset()),
]

_BY_NAME = {case.name: case for case in ZOO}


def _cases():
    return [pytest.param(case, id=case.name) for case in ZOO]


# ---------------------------------------------------------------------------
# Primitive table: (call under the spy, densify afterwards, dense reference)
# ---------------------------------------------------------------------------

_ROWS = jnp.array([0, 2, -1])
_COLS = jnp.array([1, 2])


def _dense(result):
    if isinstance(result, lx.AbstractLinearOperator):
        return result.as_matrix()
    return result


def _gram(L):
    L = _dense(L)
    return einsum(L, jnp.conj(L), "i k, j k -> i j")


def _square(S):
    S = _dense(S)
    return einsum(S, S, "i k, k j -> i j")


PRIMITIVES = {
    "solve": (
        lambda op, b: gaussx.solve(op, b),
        _dense,
        lambda M, b: jnp.linalg.solve(M, b),
    ),
    "logdet": (
        lambda op, _: gaussx.logdet(op),
        _dense,
        lambda M, _: jnp.linalg.slogdet(M)[1],
    ),
    "diag": (lambda op, _: gaussx.diag(op), _dense, lambda M, _: jnp.diag(M)),
    "trace": (lambda op, _: gaussx.trace(op), _dense, lambda M, _: jnp.trace(M)),
    "inv": (lambda op, _: gaussx.inv(op), _dense, lambda M, _: jnp.linalg.inv(M)),
    "cholesky": (lambda op, _: gaussx.cholesky(op), _gram, lambda M, _: M),
    "sqrt": (lambda op, _: gaussx.sqrt(op), _square, lambda M, _: M),
    "eigvals": (
        lambda op, _: gaussx.eigvals(op),
        lambda e: jnp.sort(jnp.real(e)),
        lambda M, _: jnp.linalg.eigvalsh(M),
    ),
    "frobenius_norm": (
        lambda op, _: gaussx.frobenius_norm(op),
        _dense,
        lambda M, _: jnp.linalg.norm(M),
    ),
    "submatrix": (
        lambda op, _: gaussx.submatrix(op, _ROWS, _COLS),
        _dense,
        lambda M, _: M[jnp.ix_(_ROWS, _COLS)],
    ),
}


@contextlib.contextmanager
def forbid_as_matrix(cls: type | None):
    """Fail if any instance of ``cls`` is materialised inside the block."""
    if cls is None:
        yield
        return
    original = cls.as_matrix

    def _forbidden(self):
        raise AssertionError(f"{cls.__name__}.as_matrix called")

    cls.as_matrix = _forbidden
    try:
        yield
    finally:
        cls.as_matrix = original


def _probe(operator: lx.AbstractLinearOperator, key: int = 1):
    struct = operator.in_structure()
    return jr.normal(jr.key(key), struct.shape, dtype=struct.dtype)


def _cells():
    params = []
    for case in ZOO:
        for prim in sorted(case.prims):
            marks = []
            if prim in case.xfail:
                marks.append(pytest.mark.xfail(strict=True, reason=case.xfail[prim]))
            params.append(
                pytest.param(case, prim, id=f"{case.name}-{prim}", marks=marks)
            )
    return params


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_zoo_covers_every_exported_operator():
    built = {type(case.build()) for case in ZOO}
    missing = {
        cls
        for cls in exported_operator_classes()
        if not cls.__module__.startswith("lineax")
        and not any(issubclass(cls, b) or issubclass(b, cls) for b in built)
    }
    assert not missing, sorted(cls.__name__ for cls in missing)


def test_xfail_entries_name_known_primitives():
    for case in ZOO:
        assert set(case.xfail) <= case.prims, case.name
        assert case.structured <= case.prims | {"eigvals"}, case.name


@pytest.mark.parametrize("case", _cases())
def test_matvec_and_transpose(case):
    op = case.build()
    M = op.as_matrix()
    v = _probe(op)
    rtol, atol = default_tolerances(M)
    assert jnp.allclose(op.mv(v), einsum(M, v, "i j, j -> i"), rtol=rtol, atol=atol)
    assert jnp.allclose(
        op.T.as_matrix(), rearrange(M, "i j -> j i"), rtol=rtol, atol=atol
    )


@pytest.mark.x64_only(reason="dense references compared at round-off (x64)")
@pytest.mark.parametrize(("case", "prim"), _cells())
def test_primitive_matches_dense(case, prim):
    op = case.build()
    M = op.as_matrix()
    b = _probe(op)
    call, post, ref = PRIMITIVES[prim]
    spy = case.spy if prim in case.structured else None
    with forbid_as_matrix(spy):
        raw = call(op, b)  # only the primitive itself runs under the spy
    result, expected = post(raw), ref(M, b)
    assert jnp.allclose(result, expected, rtol=1e-7, atol=1e-9), jnp.max(
        jnp.abs(result - expected)
    )


_TRANSFORM_CASES = [
    "dense",
    "kronecker",
    "scaled_kronecker",
    "block_diag",
    "low_rank_update",
    "kronecker_sum",
    "sum_of_kroneckers",
    "block_tridiag",
    "circulant_pd",
]


@pytest.mark.slow
@pytest.mark.parametrize("name", _TRANSFORM_CASES)
def test_jit_and_grad_are_finite(name):
    op = _BY_NAME[name].build()
    b = jnp.ones(op.in_size(), dtype=op.in_structure().dtype)

    def f(o, b):
        return jnp.sum(gaussx.solve(o, b)) + gaussx.logdet(o)

    value = eqx.filter_jit(f)(op, b)
    assert jnp.isfinite(value)
    grads = eqx.filter_grad(f)(op, b)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
