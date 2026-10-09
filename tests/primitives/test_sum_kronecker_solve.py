"""Structural ``solve`` / ``logdet`` dispatch for sums of Kronecker products.

Every exact path is pinned against the dense reference from
``as_matrix()``; every non-reducible shape is pinned as *still correct* on
the dense fallback, since the dispatch may only add speed, never change an
answer.

Keys are pinned rather than drawn from ``getkey``: the assertions are about
the factorization identity, not about sampling, so a fixed model makes the
1e-10 tolerances mean something (see AGENTS.md).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import Kronecker, KroneckerSum, SumOfKroneckers, sum_operator
from gaussx._operators._sum_kronecker import (
    _DiagonalWhitener,
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
from gaussx._testing import (
    dense_logdet,
    dense_solve,
    random_kronecker_pd,
    random_pd_matrix,
    random_pd_operator,
    tree_allclose,
)


_SYM_PSD = (lx.symmetric_tag, lx.positive_semidefinite_tag)


N_A = 3
N_B = 4


def _symmetric_operator(key, n):
    return lx.MatrixLinearOperator(random_pd_matrix(key, n, jitter=n), lx.symmetric_tag)


def _kronecker_of_psd(key):
    k_a, k_b = jr.split(key)
    return Kronecker(
        random_pd_operator(k_a, N_A, jitter=N_A, tags=_SYM_PSD),
        random_pd_operator(k_b, N_B, jitter=N_B, tags=_SYM_PSD),
    )


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

    @pytest.mark.slow
    def test_scalar_shift(self):
        """Case 1: ``B ⊗ C + σ² I``, the classical Kronecker-exact GP."""
        operator = SumOfKroneckers(
            _kronecker_of_psd(jr.key(0)),
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
            _kronecker_of_psd(jr.key(0)),
            Kronecker(_identity(N_A), -0.5 * _identity(N_B)),
        )
        _assert_matches_dense(operator, _rhs(jr.key(1)))

    def test_per_output_noise(self):
        """Case 2 with a diagonal anchor: ``B ⊗ C + diag(s) ⊗ I``."""
        noise = jnp.array([0.3, 0.9, 1.4])
        operator = SumOfKroneckers(
            _kronecker_of_psd(jr.key(0)),
            Kronecker(lx.DiagonalLinearOperator(noise), _identity(N_B)),
            tags=lx.positive_semidefinite_tag,
        )
        _assert_matches_dense(operator, _rhs(jr.key(1)))

    def test_general_two_term(self):
        """Case 2: ``B₁ ⊗ C₁ + B₂ ⊗ C₂`` with the second term SPD."""
        operator = SumOfKroneckers(
            _kronecker_of_psd(jr.key(0)),
            _kronecker_of_psd(jr.key(1)),
            tags=lx.positive_semidefinite_tag,
        )
        _assert_matches_dense(operator, _rhs(jr.key(2)))

    def test_anchor_falls_back_to_first_term(self):
        """Only the *first* term is PSD-tagged, so that one is the anchor."""
        key_a, key_b = jr.split(jr.key(0))
        operator = SumOfKroneckers(
            _kronecker_of_psd(jr.key(1)),
            Kronecker(_symmetric_operator(key_a, N_A), _symmetric_operator(key_b, N_B)),
        )
        assert _is_eigen_reducible(operator)
        _assert_matches_dense(operator, _rhs(jr.key(2)))

    def test_no_dense_fallback_warning(self):
        """The exact paths never route through ``cholesky(SumOfKroneckers)``."""
        operator = SumOfKroneckers(
            _kronecker_of_psd(jr.key(0)), _kronecker_of_psd(jr.key(1))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", DenseFallbackWarning)
            solve(operator, _rhs(jr.key(2)))
            logdet(operator)


class TestAddLinearOperatorForms:
    """``sum_operator`` builds ``AddLinearOperator`` chains, not the operator."""

    def test_kronecker_plus_scalar_identity(self):
        operator = _kronecker_of_psd(jr.key(0)) + 0.7 * _identity(N_A * N_B)
        assert isinstance(operator, lx.AddLinearOperator)
        _assert_matches_dense(operator, _rhs(jr.key(1)))

    def test_sum_operator_with_tags(self):
        """A tagged sum takes the structured path without losing its tags."""
        operator = sum_operator(
            _kronecker_of_psd(jr.key(0)),
            0.7 * _identity(N_A * N_B),
            tags=lx.positive_semidefinite_tag,
        )
        assert isinstance(operator, lx.TaggedLinearOperator)
        _assert_matches_dense(operator, _rhs(jr.key(1)))

    def test_two_kroneckers(self):
        operator = _kronecker_of_psd(jr.key(0)) + _kronecker_of_psd(jr.key(1))
        _assert_matches_dense(operator, _rhs(jr.key(2)))

    def test_sum_of_kroneckers_plus_identity(self):
        """Identity leaves are summed and folded into one ``I ⊗ cI`` term."""
        operator = (
            _kronecker_of_psd(jr.key(0))
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
            _kronecker_of_psd(jr.key(0)),
            _kronecker_of_psd(jr.key(1)),
            _kronecker_of_psd(jr.key(2)),
        )
        assert _sum_of_kroneckers_eigen(operator) is None
        _assert_matches_dense(operator, _rhs(jr.key(3)))

    def test_untagged_factors(self):
        """Without symmetry/PSD tags we cannot justify the reduction."""
        key_a, key_b = jr.split(jr.key(0))
        untagged = Kronecker(
            lx.MatrixLinearOperator(random_pd_matrix(key_a, N_A, jitter=N_A)),
            lx.MatrixLinearOperator(random_pd_matrix(key_b, N_B, jitter=N_B)),
        )
        operator = SumOfKroneckers(untagged, _kronecker_of_psd(jr.key(1)))
        assert _sum_of_kroneckers_eigen(operator) is None
        _assert_matches_dense(operator, _rhs(jr.key(2)))

    def test_unfactorizable_diagonal_shift(self):
        """A general diagonal is not statically a Kronecker product."""
        shift = lx.DiagonalLinearOperator(
            jnp.abs(jr.normal(jr.key(0), (N_A * N_B,))) + 1.0
        )
        operator = _kronecker_of_psd(jr.key(1)) + shift
        assert _kronecker_terms(operator) is None
        _assert_matches_dense(operator, _rhs(jr.key(2)))

    def test_mismatched_factor_sizes(self):
        """Terms whose factors split the size differently share no basis."""
        operator = SumOfKroneckers(
            Kronecker(
                random_pd_operator(jr.key(0), 2, jitter=2, tags=_SYM_PSD),
                random_pd_operator(jr.key(1), 6, jitter=6, tags=_SYM_PSD),
            ),
            Kronecker(
                random_pd_operator(jr.key(2), 3, jitter=3, tags=_SYM_PSD),
                random_pd_operator(jr.key(3), 4, jitter=4, tags=_SYM_PSD),
            ),
        )
        assert _kronecker_terms(operator) is None
        _assert_matches_dense(operator, _rhs(jr.key(4)))

    def test_three_terms_via_iterative_solvers(self):
        """The documented escape hatch for Q ≥ 3: CG over the structured mv."""
        operator = SumOfKroneckers(
            _kronecker_of_psd(jr.key(0)),
            _kronecker_of_psd(jr.key(1)),
            _kronecker_of_psd(jr.key(2)),
            tags=lx.positive_semidefinite_tag,
        )
        vector = _rhs(jr.key(3))
        iterative = CGSolver(rtol=1e-10, atol=1e-10).solve(operator, vector)
        assert tree_allclose(iterative, dense_solve(operator, vector), atol=1e-6)


class TestTransformsAndConsumers:
    def test_logdet_gradient_matches_dense(self):
        """Gradients w.r.t. factor entries agree with dense autodiff."""
        base = random_pd_matrix(jr.key(0), N_A, jitter=N_A)
        kron_b = random_pd_operator(jr.key(1), N_B, jitter=N_B, tags=_SYM_PSD)
        anchor = Kronecker(_identity(N_A), 0.7 * _identity(N_B))

        def build(scale):
            factor = lx.MatrixLinearOperator(
                scale * base, (lx.symmetric_tag, lx.positive_semidefinite_tag)
            )
            return SumOfKroneckers(Kronecker(factor, kron_b), anchor)

        structured = jax.jit(jax.grad(lambda s: logdet(build(s))))(1.3)
        reference = jax.jit(jax.grad(lambda s: dense_logdet(build(s))))(1.3)
        assert tree_allclose(structured, reference, atol=1e-10)

    @pytest.mark.parametrize("anchor_kind", ["shift", "diagonal"])
    @pytest.mark.x64_only(reason="dense-reference tolerance below float32 round-off")
    def test_solve_gradient_matches_dense_with_degenerate_factors(self, anchor_kind):
        """Solve gradients stay exact on repeated / clustered eigenvalues.

        The multi-output GP shape: a rank-one coregionalization ``w wᵀ``
        (``N_A - 1`` repeated zero eigenvalues) against a smooth RBF Gram
        (a tail of eigenvalues clustered near zero). Differentiating through
        the factor ``eigh`` here gave gradients off by orders of magnitude.
        """
        x = jnp.linspace(0.0, 1.0, 12)
        y = jr.normal(jr.key(0), (N_A * x.size,))
        noise = jnp.array([0.05, 0.1, 0.2])

        def build(w, lengthscale, noise):
            gram = jnp.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / lengthscale**2)
            main = Kronecker(
                lx.MatrixLinearOperator(jnp.outer(w, w), lx.positive_semidefinite_tag),
                lx.MatrixLinearOperator(gram, lx.positive_semidefinite_tag),
            )
            identity_b = lx.IdentityLinearOperator(
                jax.ShapeDtypeStruct((x.size,), jnp.float64)
            )
            if anchor_kind == "shift":
                anchor = Kronecker(_identity(N_A), noise[0] * identity_b)
            else:
                anchor = Kronecker(lx.DiagonalLinearOperator(noise), identity_b)
            return SumOfKroneckers(main, anchor)

        def quad(solve_fn, w, lengthscale, noise, rhs):
            return rhs @ solve_fn(build(w, lengthscale, noise), rhs)

        args = (jnp.array([1.0, 0.5, -0.3]), 0.4, noise, y)
        assert _is_eigen_reducible(build(*args[:3]))
        grad = jax.grad(quad, argnums=(1, 2, 3, 4))
        structured = jax.jit(grad, static_argnums=0)(solve, *args)
        reference = grad(dense_solve, *args)
        assert tree_allclose(structured, reference, rtol=1e-6, atol=1e-8)

    def test_solve_under_jit_and_vmap(self):
        operator = SumOfKroneckers(
            _kronecker_of_psd(jr.key(0)),
            Kronecker(_identity(N_A), 0.7 * _identity(N_B)),
        )
        vectors = jr.normal(jr.key(1), (5, N_A * N_B))
        batched = jax.jit(jax.vmap(lambda v: solve(operator, v)))(vectors)
        expected = jax.vmap(lambda v: dense_solve(operator, v))(vectors)
        assert tree_allclose(batched, expected, atol=1e-10)

    def test_inv_routes_through_structured_solve(self):
        operator = SumOfKroneckers(
            _kronecker_of_psd(jr.key(0)),
            _kronecker_of_psd(jr.key(1)),
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
            _kronecker_of_psd(jr.key(0)),
            _kronecker_of_psd(jr.key(1)),
            tags=lx.positive_semidefinite_tag,
        )
        strategy = AutoSolver(size_threshold=size_threshold)._get_strategy(operator)
        assert isinstance(strategy, DenseSolver)

    def test_three_term_operator_keeps_size_rules(self):
        operator = SumOfKroneckers(
            _kronecker_of_psd(jr.key(0)),
            _kronecker_of_psd(jr.key(1)),
            _kronecker_of_psd(jr.key(2)),
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
            lx.MatrixLinearOperator(random_pd_matrix(jr.key(0), N_B, jitter=N_B)),
            (lx.symmetric_tag, lx.positive_semidefinite_tag),
        )
        anchor = Kronecker(
            random_pd_operator(jr.key(1), N_A, jitter=N_A, tags=_SYM_PSD), wrapped
        )
        operator = SumOfKroneckers(_kronecker_of_psd(jr.key(2)), anchor)
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
            Kronecker(
                singular, random_pd_operator(jr.key(0), N_B, jitter=N_B, tags=_SYM_PSD)
            ),
            Kronecker(_identity(N_A), 0.7 * _identity(N_B)),
        )
        solution = solve(operator, _rhs(jr.key(1)))
        assert jnp.all(jnp.isfinite(solution))
        _assert_matches_dense(operator, _rhs(jr.key(1)))


def _signed_diagonal_sum(diagonal, scale=1.0):
    """``A ⊗ B + diag(diagonal) ⊗ (scale I)`` with a PD first term (gh-317)."""
    return SumOfKroneckers(
        _kronecker_of_psd(jr.key(0)),
        Kronecker(lx.DiagonalLinearOperator(diagonal), scale * _identity(N_B)),
    )


_NON_POSITIVE_ANCHORS = {
    "negative_entry": ([1.0, -1.0, 2.0], 1.0),
    "zero_entry": ([0.0, 1.0, 1.0], 1.0),
    "negative_identity": ([1.0, 2.0, 3.0], -0.5),
}


class TestNonPositiveDiagonalAnchor:
    """gh-317: a diagonal anchor is whitened by its square root."""

    @pytest.mark.parametrize(
        ("diagonal", "scale"),
        list(_NON_POSITIVE_ANCHORS.values()),
        ids=list(_NON_POSITIVE_ANCHORS),
    )
    def test_concrete_falls_back_to_the_pd_anchor(self, diagonal, scale):
        operator = _signed_diagonal_sum(jnp.array(diagonal), scale)
        # The full matrix is PD; only the diagonal whitening is invalid.
        assert jnp.linalg.eigvalsh(operator.as_matrix()).min() > 0
        factorization = _sum_of_kroneckers_eigen(operator)
        assert factorization is not None
        assert not isinstance(factorization.wa, _DiagonalWhitener)
        _assert_matches_dense(operator, _rhs(jr.key(1)))

    def test_positive_diagonal_keeps_the_diagonal_whitener(self):
        operator = _signed_diagonal_sum(jnp.array([1.0, 3.0, 2.0]))
        factorization = _sum_of_kroneckers_eigen(operator)
        assert factorization is not None
        assert isinstance(factorization.wa, _DiagonalWhitener)
        _assert_matches_dense(operator, _rhs(jr.key(1)))

    def test_only_anchor_non_positive_uses_the_dense_fallback(self):
        # The other term is merely symmetric, so no Cholesky plan exists.
        operator = SumOfKroneckers(
            Kronecker(
                _symmetric_operator(jr.key(0), N_A),
                _symmetric_operator(jr.key(1), N_B),
            ),
            Kronecker(
                lx.DiagonalLinearOperator(jnp.array([1.0, -1.0, 2.0])),
                _identity(N_B),
            ),
        )
        assert _is_eigen_reducible(operator)
        assert _sum_of_kroneckers_eigen(operator) is None
        _assert_matches_dense(operator, _rhs(jr.key(2)))

    @pytest.mark.parametrize("primitive", ["solve", "logdet"])
    def test_traced_non_positive_raises_instead_of_nan(self, primitive):
        b = _rhs(jr.key(1))

        def run(diagonal):
            operator = _signed_diagonal_sum(diagonal)
            return solve(operator, b) if primitive == "solve" else logdet(operator)

        with pytest.raises(Exception, match="must be strictly positive"):
            jax.block_until_ready(jax.jit(run)(jnp.array([1.0, -1.0, 2.0])))
        # A valid traced diagonal is unaffected.
        valid = jnp.array([1.0, 3.0, 2.0])
        assert tree_allclose(jax.jit(run)(valid), run(valid), atol=1e-12)


# ---------------------------------------------------------------------------
# Silent densification is flagged, at the caller's line (gh-406)
# ---------------------------------------------------------------------------


def _three_term():
    keys = jr.split(jr.key(0), 3)
    return SumOfKroneckers(*(random_kronecker_pd(k, (2, 3)) for k in keys))


def _untagged_kronecker_sum():
    return KroneckerSum(
        lx.MatrixLinearOperator(random_pd_matrix(jr.key(0), 2)),
        lx.MatrixLinearOperator(random_pd_matrix(jr.key(1), 3)),
    )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        pytest.param(
            lambda: solve(_three_term(), jnp.ones(6)), "solve", id="solve_sok"
        ),
        pytest.param(lambda: logdet(_three_term()), "logdet", id="logdet_sok"),
        pytest.param(
            lambda: solve(_untagged_kronecker_sum(), jnp.ones(6)),
            "symmetric_tag",
            id="solve_kronecker_sum",
        ),
        pytest.param(
            lambda: solve(
                2.0 * lx.TaggedLinearOperator(_three_term(), ()), jnp.ones(6)
            ),
            "solve",
            id="solve_wrapped_sok",
        ),
    ],
)
def test_dense_fallback_warns_at_caller(call, match):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call()
    hits = [w for w in record if issubclass(w.category, DenseFallbackWarning)]
    assert len(hits) == 1
    assert match in str(hits[0].message)
    assert hits[0].filename == __file__


def test_structured_paths_do_not_warn():
    two_term = SumOfKroneckers(
        random_kronecker_pd(jr.key(0), (2, 3)), random_kronecker_pd(jr.key(1), (2, 3))
    )
    tagged = KroneckerSum(
        lx.MatrixLinearOperator(random_pd_matrix(jr.key(0), 2), lx.symmetric_tag),
        lx.MatrixLinearOperator(random_pd_matrix(jr.key(1), 3), lx.symmetric_tag),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DenseFallbackWarning)
        solve(two_term, jnp.ones(6))
        logdet(two_term)
        solve(tagged, jnp.ones(6))
