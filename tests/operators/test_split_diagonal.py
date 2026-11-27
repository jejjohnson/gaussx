"""Tests for split_diagonal: A = K + diag(d) read back into K and d (gh-877)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._testing import random_pd_matrix, random_pd_operator, tree_allclose


_N = 5

_DTYPES = [
    jnp.float32,
    pytest.param(jnp.float64, marks=pytest.mark.x64_only(reason="float64 case")),
]


class _NoDensify(lx.MatrixLinearOperator):
    """A kernel operator that fails the test if anything densifies it."""

    def as_matrix(self):
        raise AssertionError("as_matrix called on the kernel part")


def _parts(key, dtype):
    """A random PD kernel K, positive variances d and a positive scalar s."""
    kernel_key, diagonal_key = jr.split(key)
    kernel = random_pd_operator(kernel_key, _N, dtype=dtype)
    variance = jr.uniform(diagonal_key, (_N,), dtype=dtype, minval=0.1, maxval=1.0)
    scale = jnp.asarray(0.3, dtype=dtype)
    return kernel, variance, scale


def _diagonal_summand(kind, variance, scale, structure):
    """The diagonal summand of one row of the issue's table, and its d."""
    diagonal = lx.DiagonalLinearOperator(variance)
    identity = lx.IdentityLinearOperator(structure)
    if kind == "diagonal":
        return diagonal, variance
    if kind == "scaled_identity":
        return scale * identity, scale * jnp.ones_like(variance)
    if kind == "scaled_diagonal":
        return scale * diagonal, scale * variance
    if kind == "divided_diagonal":
        return diagonal / scale, variance / scale
    raise ValueError(kind)


_KINDS = ["diagonal", "scaled_identity", "scaled_diagonal", "divided_diagonal"]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("tagged", [False, True])
@pytest.mark.parametrize("kernel_first", [True, False])
@pytest.mark.parametrize("kind", _KINDS)
def test_each_form_splits_exactly(getkey, kind, kernel_first, tagged, dtype):
    kernel, variance, scale = _parts(getkey(), dtype)
    summand, expected = _diagonal_summand(kind, variance, scale, kernel.in_structure())
    system = kernel + summand if kernel_first else summand + kernel
    if tagged:
        system = lx.TaggedLinearOperator(system, lx.positive_semidefinite_tag)

    split = gaussx.split_diagonal(system)

    assert split is not None
    assert split.diagonal.dtype == dtype
    # same floating-point products as the operator's own as_matrix, so exact
    assert jnp.array_equal(split.diagonal, expected)
    assert jnp.array_equal(split.base.as_matrix(), kernel.as_matrix())
    # round trip: base + diag(diagonal) densifies to A exactly
    rebuilt = split.base.as_matrix() + jnp.diag(split.diagonal)
    assert jnp.array_equal(rebuilt, system.as_matrix())


@pytest.mark.parametrize("dtype", _DTYPES)
def test_nested_sums_sum_their_diagonals(getkey, dtype):
    kernel, variance, scale = _parts(getkey(), dtype)
    other_kernel = random_pd_operator(getkey(), _N, dtype=dtype)
    identity = lx.IdentityLinearOperator(kernel.in_structure())
    diagonal = lx.DiagonalLinearOperator(variance)
    # ((D + K₁) + s I) + K₂
    system = ((diagonal + kernel) + scale * identity) + other_kernel

    split = gaussx.split_diagonal(system)

    assert split is not None
    # the kernel summands stay in their order: K₁ + K₂
    assert isinstance(split.base, lx.AddLinearOperator)
    assert split.base.operator1 is kernel
    assert split.base.operator2 is other_kernel
    # d = d₁ + s; the sum is re-associated, so equal to rounding
    assert tree_allclose(split.diagonal, variance + scale)
    rebuilt = split.base.as_matrix() + jnp.diag(split.diagonal)
    assert tree_allclose(rebuilt, system.as_matrix())


def test_tagged_inner_sum_is_one_summand(getkey):
    kernel, variance, _ = _parts(getkey(), None)
    inner = lx.TaggedLinearOperator(
        kernel + lx.DiagonalLinearOperator(variance), lx.symmetric_tag
    )
    extra = lx.DiagonalLinearOperator(2.0 * variance)

    split = gaussx.split_diagonal(inner + extra)

    assert split is not None
    assert split.base is inner
    assert jnp.array_equal(split.diagonal, 2.0 * variance)


def test_low_rank_update_with_diagonal_base_is_none(getkey):
    variance = jr.uniform(getkey(), (_N,), minval=0.1, maxval=1.0)
    factor = jr.normal(getkey(), (_N, 2))
    system = gaussx.low_rank_plus_diag(variance, factor)
    assert gaussx.split_diagonal(system) is None
    tagged = lx.TaggedLinearOperator(system, lx.positive_semidefinite_tag)
    assert gaussx.split_diagonal(tagged) is None


def test_dense_matrix_is_none(getkey):
    assert gaussx.split_diagonal(random_pd_operator(getkey(), _N)) is None


def test_no_kernel_part_or_no_diagonal_is_none(getkey):
    kernel, variance, scale = _parts(getkey(), None)
    diagonal = lx.DiagonalLinearOperator(variance)
    identity = lx.IdentityLinearOperator(kernel.in_structure())
    # a diagonal alone, a sum of diagonals only, a sum of kernels only
    assert gaussx.split_diagonal(diagonal) is None
    assert gaussx.split_diagonal(diagonal + scale * identity) is None
    assert gaussx.split_diagonal(kernel + kernel) is None


def test_psd_tag_of_the_system_passes_to_base(getkey):
    kernel, variance, _ = _parts(getkey(), None)
    # an untagged dense kernel part, so lineax cannot infer PSD on its own
    untagged = lx.MatrixLinearOperator(kernel.as_matrix())
    system = untagged + lx.DiagonalLinearOperator(variance)

    plain = gaussx.split_diagonal(system)
    tagged = gaussx.split_diagonal(
        lx.TaggedLinearOperator(system, lx.positive_semidefinite_tag)
    )

    assert plain is not None and tagged is not None
    assert not lx.is_positive_semidefinite(plain.base)
    assert lx.is_positive_semidefinite(tagged.base)
    # a base that is PSD by its own tags keeps them, unwrapped
    own = gaussx.split_diagonal(
        lx.TaggedLinearOperator(
            kernel + lx.DiagonalLinearOperator(variance),
            lx.positive_semidefinite_tag,
        )
    )
    assert own is not None
    assert own.base is kernel


def test_same_under_jit(getkey):
    kernel, variance, scale = _parts(getkey(), None)
    identity = lx.IdentityLinearOperator(kernel.in_structure())
    system = lx.TaggedLinearOperator(
        kernel + lx.DiagonalLinearOperator(variance) + scale * identity,
        lx.positive_semidefinite_tag,
    )

    @eqx.filter_jit
    def parts(operator):
        split = gaussx.split_diagonal(operator)
        return split.base.as_matrix(), split.diagonal

    eager = gaussx.split_diagonal(system)
    base_matrix, diagonal = parts(system)
    assert eager is not None
    assert tree_allclose(base_matrix, eager.base.as_matrix())
    assert tree_allclose(diagonal, eager.diagonal)


def test_grad_through_diagonal_and_base_leaves(getkey):
    kernel_matrix = random_pd_matrix(getkey(), _N)
    variance = jr.uniform(getkey(), (_N,), minval=0.1, maxval=1.0)
    scale = jnp.asarray(0.3)
    probe = jr.normal(getkey(), (_N,))
    weights = jr.normal(getkey(), (_N,))

    def split_loss(kernel_matrix, variance, scale):
        kernel = lx.MatrixLinearOperator(kernel_matrix)
        system = kernel + scale * lx.DiagonalLinearOperator(variance)
        split = gaussx.split_diagonal(system)
        # wᵀ K x + wᵀ d
        return weights @ split.base.mv(probe) + weights @ split.diagonal

    def dense_loss(kernel_matrix, variance, scale):
        # wᵀ K x + wᵀ (s d)
        return weights @ (kernel_matrix @ probe) + weights @ (scale * variance)

    arguments = (kernel_matrix, variance, scale)
    split_grads = jax.grad(split_loss, argnums=(0, 1, 2))(*arguments)
    dense_grads = jax.grad(dense_loss, argnums=(0, 1, 2))(*arguments)
    assert tree_allclose(split_grads, dense_grads)


def test_never_densifies(getkey):
    kernel = _NoDensify(random_pd_matrix(getkey(), _N), lx.positive_semidefinite_tag)
    variance = jr.uniform(getkey(), (_N,), minval=0.1, maxval=1.0)
    identity = lx.IdentityLinearOperator(kernel.in_structure())
    system = lx.TaggedLinearOperator(
        kernel + lx.DiagonalLinearOperator(variance) + 0.5 * identity,
        lx.positive_semidefinite_tag,
    )

    split = gaussx.split_diagonal(system)

    assert split is not None
    assert split.base is kernel
    assert tree_allclose(split.diagonal, variance + 0.5)


@pytest.mark.x64_only(reason="a float64 scalar must not promote a float32 system")
def test_float32_system_keeps_float32_under_x64(getkey):
    kernel = random_pd_operator(getkey(), _N, dtype=jnp.float32)
    identity = lx.IdentityLinearOperator(kernel.in_structure())
    system = kernel + jnp.asarray(0.25, dtype=jnp.float64) * identity

    split = gaussx.split_diagonal(system)

    assert split is not None
    assert split.diagonal.dtype == jnp.float32
