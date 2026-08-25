"""Structural ``solve`` / ``logdet`` dispatch for sums of Kronecker products.

Every exact path is pinned against the dense reference from
``as_matrix()``; every non-reducible shape is pinned as *still correct* on
the dense fallback, since the dispatch may only add speed, never change an
answer.

Keys are pinned rather than drawn from ``getkey``: the assertions are about
the factorization identity, not about sampling, so a fixed model makes the
1e-10 tolerances mean something (see CLAUDE.md).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import Kronecker, SumOfKroneckers, SumOperator
from gaussx._operators._sum_kronecker import (
    _is_eigen_reducible,
    _kronecker_terms,
    _sum_of_kroneckers_eigen,
)
from gaussx._primitives._cholesky import DenseFallbackWarning
from gaussx._primitives._inv import inv
from gaussx._primitives._logdet import logdet
from gaussx._primitives._solve import solve
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._cg import CGSolver
from gaussx._strategies._dense import DenseSolver
from gaussx._testing import dense_logdet, dense_solve, tree_allclose


N_A = 3
N_B = 4


def _psd_matrix(key, n):
    m = jr.normal(key, (n, n))
    return m @ m.T + n * jnp.eye(n)


def _psd_operator(key, n):
    return lx.MatrixLinearOperator(
        _psd_matrix(key, n),
        (lx.symmetric_tag, lx.positive_semidefinite_tag),
    )


def _symmetric_operator(key, n):
    return lx.MatrixLinearOperator(_psd_matrix(key, n), lx.symmetric_tag)


def _psd_kronecker(key):
    k_a, k_b = jr.split(key)
    return Kronecker(_psd_operator(k_a, N_A), _psd_operator(k_b, N_B))


def _identity(n):
    return lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n,), jnp.float64))


def _rhs(key):
    return jr.normal(key, (N_A * N_B,))


def _assert_matches_dense(operator, vector, *, atol=1e-10):
    """Both primitives agree with the dense reference for ``operator``."""
    assert tree_allclose(
        solve(operator, vector), dense_solve(operator, vector), atol=atol
    )
    assert tree_allclose(logdet(operator), dense_logdet(operator), atol=atol)


class TestExactTwoTermPaths:
    """Shapes the simultaneous diagonalization handles in closed form."""

    def test_scalar_shift(self):
        """Case 1: ``B ⊗ C + σ² I``, the classical Kronecker-exact GP."""
        operator = SumOfKroneckers(
            _psd_kronecker(jr.key(0)),
            Kronecker(_identity(N_A), 0.7 * _identity(N_B)),
            tags=lx.positive_semidefinite_tag,
        )
        _assert_matches_dense(operator, _rhs(jr.key(1)))

    def test_negative_scalar_shift(self):
        """An identity anchor folds in as a shift, so its sign is free.

        The whitening route would need a Cholesky of the anchor and return
        ``NaN`` here; the shift route stays exact, which is why the two are
        separate branches.
        """
        operator = SumOfKroneckers(
            _psd_kronecker(jr.key(0)),
            Kronecker(_identity(N_A), -0.5 * _identity(N_B)),
        )
        _assert_matches_dense(operator, _rhs(jr.key(1)))

    def test_per_output_noise(self):
        """Case 2 with a diagonal anchor: ``B ⊗ C + diag(s) ⊗ I``."""
        noise = jnp.array([0.3, 0.9, 1.4])
        operator = SumOfKroneckers(
            _psd_kronecker(jr.key(0)),
            Kronecker(lx.DiagonalLinearOperator(noise), _identity(N_B)),
            tags=lx.positive_semidefinite_tag,
        )
        _assert_matches_dense(operator, _rhs(jr.key(1)))

    def test_general_two_term(self):
        """Case 2: ``B₁ ⊗ C₁ + B₂ ⊗ C₂`` with the second term SPD."""
        operator = SumOfKroneckers(
            _psd_kronecker(jr.key(0)),
            _psd_kronecker(jr.key(1)),
            tags=lx.positive_semidefinite_tag,
        )
        _assert_matches_dense(operator, _rhs(jr.key(2)))

    def test_anchor_falls_back_to_first_term(self):
        """Only the *first* term is PSD-tagged, so that one is the anchor."""
        key_a, key_b = jr.split(jr.key(0))
        operator = SumOfKroneckers(
            _psd_kronecker(jr.key(1)),
            Kronecker(_symmetric_operator(key_a, N_A), _symmetric_operator(key_b, N_B)),
        )
        assert _is_eigen_reducible(operator)
        _assert_matches_dense(operator, _rhs(jr.key(2)))

    def test_no_dense_fallback_warning(self):
        """The exact paths never route through ``cholesky(SumOfKroneckers)``."""
        operator = SumOfKroneckers(_psd_kronecker(jr.key(0)), _psd_kronecker(jr.key(1)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", DenseFallbackWarning)
            solve(operator, _rhs(jr.key(2)))
            logdet(operator)


class TestAddLinearOperatorForms:
    """``SumOperator`` builds ``AddLinearOperator`` chains, not the operator."""

    def test_kronecker_plus_scalar_identity(self):
        operator = _psd_kronecker(jr.key(0)) + 0.7 * _identity(N_A * N_B)
        assert isinstance(operator, lx.AddLinearOperator)
        _assert_matches_dense(operator, _rhs(jr.key(1)))

    def test_sum_operator_with_tags(self):
        """A tagged sum takes the structured path without losing its tags."""
        operator = SumOperator(
            _psd_kronecker(jr.key(0)),
            0.7 * _identity(N_A * N_B),
            tags=lx.positive_semidefinite_tag,
        )
        assert isinstance(operator, lx.TaggedLinearOperator)
        _assert_matches_dense(operator, _rhs(jr.key(1)))

    def test_two_kroneckers(self):
        operator = _psd_kronecker(jr.key(0)) + _psd_kronecker(jr.key(1))
        _assert_matches_dense(operator, _rhs(jr.key(2)))

    def test_sum_of_kroneckers_plus_identity(self):
        """Identity leaves are summed and folded into one ``I ⊗ cI`` term."""
        operator = (
            _psd_kronecker(jr.key(0))
            + 0.25 * _identity(N_A * N_B)
            + 0.5 * _identity(N_A * N_B)
        )
        terms = _kronecker_terms(operator)
        assert terms is not None and len(terms) == 2
        _assert_matches_dense(operator, _rhs(jr.key(1)))


class TestFallbacksStayCorrect:
    """Shapes with no closed form keep the dense path — and its answer."""

    def test_three_terms(self):
        operator = SumOfKroneckers(
            _psd_kronecker(jr.key(0)),
            _psd_kronecker(jr.key(1)),
            _psd_kronecker(jr.key(2)),
        )
        assert _sum_of_kroneckers_eigen(operator) is None
        _assert_matches_dense(operator, _rhs(jr.key(3)))

    def test_untagged_factors(self):
        """Without symmetry/PSD tags we cannot justify the reduction."""
        key_a, key_b = jr.split(jr.key(0))
        untagged = Kronecker(
            lx.MatrixLinearOperator(_psd_matrix(key_a, N_A)),
            lx.MatrixLinearOperator(_psd_matrix(key_b, N_B)),
        )
        operator = SumOfKroneckers(untagged, _psd_kronecker(jr.key(1)))
        assert _sum_of_kroneckers_eigen(operator) is None
        _assert_matches_dense(operator, _rhs(jr.key(2)))

    def test_unfactorizable_diagonal_shift(self):
        """A general diagonal is not statically a Kronecker product."""
        shift = lx.DiagonalLinearOperator(
            jnp.abs(jr.normal(jr.key(0), (N_A * N_B,))) + 1.0
        )
        operator = _psd_kronecker(jr.key(1)) + shift
        assert _kronecker_terms(operator) is None
        _assert_matches_dense(operator, _rhs(jr.key(2)))

    def test_mismatched_factor_sizes(self):
        """Terms whose factors split the size differently share no basis."""
        operator = SumOfKroneckers(
            Kronecker(_psd_operator(jr.key(0), 2), _psd_operator(jr.key(1), 6)),
            Kronecker(_psd_operator(jr.key(2), 3), _psd_operator(jr.key(3), 4)),
        )
        assert _kronecker_terms(operator) is None
        _assert_matches_dense(operator, _rhs(jr.key(4)))

    def test_three_terms_via_iterative_solvers(self):
        """The documented escape hatch for Q ≥ 3: CG over the structured mv."""
        operator = SumOfKroneckers(
            _psd_kronecker(jr.key(0)),
            _psd_kronecker(jr.key(1)),
            _psd_kronecker(jr.key(2)),
            tags=lx.positive_semidefinite_tag,
        )
        vector = _rhs(jr.key(3))
        iterative = CGSolver(rtol=1e-10, atol=1e-10).solve(operator, vector)
        assert tree_allclose(iterative, dense_solve(operator, vector), atol=1e-6)


class TestTransformsAndConsumers:
    def test_logdet_gradient_matches_dense(self):
        """Gradients w.r.t. factor entries agree with dense autodiff."""
        base = _psd_matrix(jr.key(0), N_A)
        kron_b = _psd_operator(jr.key(1), N_B)
        anchor = Kronecker(_identity(N_A), 0.7 * _identity(N_B))

        def build(scale):
            factor = lx.MatrixLinearOperator(
                scale * base, (lx.symmetric_tag, lx.positive_semidefinite_tag)
            )
            return SumOfKroneckers(Kronecker(factor, kron_b), anchor)

        structured = jax.jit(jax.grad(lambda s: logdet(build(s))))(1.3)
        reference = jax.jit(jax.grad(lambda s: dense_logdet(build(s))))(1.3)
        assert tree_allclose(structured, reference, atol=1e-10)

    def test_solve_under_jit_and_vmap(self):
        operator = SumOfKroneckers(
            _psd_kronecker(jr.key(0)),
            Kronecker(_identity(N_A), 0.7 * _identity(N_B)),
        )
        vectors = jr.normal(jr.key(1), (5, N_A * N_B))
        batched = jax.jit(jax.vmap(lambda v: solve(operator, v)))(vectors)
        expected = jax.vmap(lambda v: dense_solve(operator, v))(vectors)
        assert tree_allclose(batched, expected, atol=1e-10)

    def test_inv_routes_through_structured_solve(self):
        operator = SumOfKroneckers(
            _psd_kronecker(jr.key(0)),
            _psd_kronecker(jr.key(1)),
            tags=lx.positive_semidefinite_tag,
        )
        vector = _rhs(jr.key(2))
        assert tree_allclose(
            inv(operator).mv(vector), dense_solve(operator, vector), atol=1e-10
        )


class TestAutoSolverClassification:
    @pytest.mark.parametrize("size_threshold", [1, 1000])
    def test_reducible_operator_picks_dense_strategy(self, size_threshold):
        """Structure beats size: the exact path is cheap at any dimension."""
        operator = SumOfKroneckers(
            _psd_kronecker(jr.key(0)),
            _psd_kronecker(jr.key(1)),
            tags=lx.positive_semidefinite_tag,
        )
        strategy = AutoSolver(size_threshold=size_threshold)._get_strategy(operator)
        assert isinstance(strategy, DenseSolver)

    def test_three_term_operator_keeps_size_rules(self):
        operator = SumOfKroneckers(
            _psd_kronecker(jr.key(0)),
            _psd_kronecker(jr.key(1)),
            _psd_kronecker(jr.key(2)),
            tags=lx.positive_semidefinite_tag,
        )
        strategy = AutoSolver(size_threshold=1)._get_strategy(operator)
        assert isinstance(strategy, CGSolver)


class TestAnchorEligibility:
    """Which factor may be whitened by, and which of two candidates wins."""

    def test_tags_on_a_wrapper_are_honoured(self):
        """Symmetry/PSD may live on a native ``TaggedLinearOperator``.

        Classifying after unwrapping would discard the only structural
        evidence the caller gave and drop the whole sum to the dense path.
        """
        wrapped = lx.TaggedLinearOperator(
            lx.MatrixLinearOperator(_psd_matrix(jr.key(0), N_B)),
            (lx.symmetric_tag, lx.positive_semidefinite_tag),
        )
        anchor = Kronecker(_psd_operator(jr.key(1), N_A), wrapped)
        operator = SumOfKroneckers(_psd_kronecker(jr.key(2)), anchor)
        assert _is_eigen_reducible(operator)
        _assert_matches_dense(operator, _rhs(jr.key(3)))

    def test_prefers_the_anchor_it_need_not_factorize(self):
        """A PSD *tag* permits a singular factor; a scalar shift cannot be.

        Whitening by the singular term would return ``NaN`` — the dense
        fallback answers this system fine — so the identity term has to win
        the anchor selection even though the other term is tried first.
        """
        singular = lx.MatrixLinearOperator(
            jnp.diag(jnp.array([0.0, 1.0, 1.0])),
            (lx.symmetric_tag, lx.positive_semidefinite_tag),
        )
        operator = SumOfKroneckers(
            Kronecker(singular, _psd_operator(jr.key(0), N_B)),
            Kronecker(_identity(N_A), 0.7 * _identity(N_B)),
        )
        solution = solve(operator, _rhs(jr.key(1)))
        assert jnp.all(jnp.isfinite(solution))
        _assert_matches_dense(operator, _rhs(jr.key(1)))
