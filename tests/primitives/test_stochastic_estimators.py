"""Tests for the matfree-0.6-backed stochastic estimators and new primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx import diag as gaussx_diag, frobenius_norm, trace, trace_and_diag
from gaussx._operators import BlockDiag, Kronecker, LowRankUpdate
from gaussx._primitives._root import root_decomposition
from gaussx._strategies._slq_logdet import IndefiniteSLQLogdet, SLQLogdet
from gaussx._testing import default_tolerances, random_pd_matrix, tree_allclose


@pytest.fixture
def psd_op(getkey):
    mat = random_pd_matrix(getkey(), 40)
    return lx.MatrixLinearOperator(
        mat, (lx.symmetric_tag, lx.positive_semidefinite_tag)
    )


def _matvec_only(op):
    """Wrap as a FunctionLinearOperator so structured paths can't trigger."""
    return lx.FunctionLinearOperator(op.mv, op.in_structure())


class TestStochasticTrace:
    @pytest.mark.slow
    def test_xtrace_beats_hutchinson_budget(self, psd_op):
        true = jnp.trace(psd_op.as_matrix())
        est = trace(
            _matvec_only(psd_op),
            stochastic=True,
            num_probes=25,
            algorithm="xtrace",
        )
        assert jnp.abs(est - true) / jnp.abs(true) < 1e-6

    def test_sampler_option(self, psd_op):
        true = jnp.trace(psd_op.as_matrix())
        for sampler in ("signs", "normal", "sphere"):
            est = trace(
                _matvec_only(psd_op),
                stochastic=True,
                num_probes=400,
                sampler=sampler,
            )
            assert jnp.abs(est - true) / jnp.abs(true) < 0.25

    def test_xtrace_rejects_signs_sampler(self, psd_op):
        with pytest.raises(ValueError, match="rotationally invariant"):
            trace(
                _matvec_only(psd_op),
                stochastic=True,
                algorithm="xtrace",
                sampler="signs",
            )

    def test_unknown_algorithm(self, psd_op):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            trace(_matvec_only(psd_op), stochastic=True, algorithm="bogus")


class TestTraceAndDiag:
    def test_joint_estimates(self, psd_op):
        mat = psd_op.as_matrix()
        tr, dg = trace_and_diag(_matvec_only(psd_op), num_probes=600)
        assert jnp.abs(tr - jnp.trace(mat)) / jnp.trace(mat) < 0.1
        rel = jnp.linalg.norm(dg - jnp.diag(mat)) / jnp.linalg.norm(jnp.diag(mat))
        assert rel < 0.2

    def test_key_changes_estimate(self, psd_op):
        tr1, _ = trace_and_diag(_matvec_only(psd_op), num_probes=5, key=jr.PRNGKey(1))
        tr2, _ = trace_and_diag(_matvec_only(psd_op), num_probes=5, key=jr.PRNGKey(2))
        assert not tree_allclose(tr1, tr2)


class TestFrobeniusNorm:
    def test_diagonal(self, getkey):
        d = jr.normal(getkey(), (7,))
        op = lx.DiagonalLinearOperator(d)
        assert tree_allclose(frobenius_norm(op), jnp.linalg.norm(d))

    def test_block_diag_and_kronecker(self, getkey):
        a = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        b = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4)))
        for op in (BlockDiag(a, b), Kronecker(a, b)):
            assert tree_allclose(frobenius_norm(op), jnp.linalg.norm(op.as_matrix()))

    def test_scaled_and_negated(self, getkey):
        a = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4)))
        for op in (2.5 * a, -a, a / 2.0):
            assert tree_allclose(frobenius_norm(op), jnp.linalg.norm(op.as_matrix()))

    def test_stochastic(self, psd_op):
        true = jnp.linalg.norm(psd_op.as_matrix())
        est = frobenius_norm(_matvec_only(psd_op), stochastic=True, num_probes=400)
        assert jnp.abs(est - true) / true < 0.1

    def test_dense_fallback(self, getkey):
        mat = jr.normal(getkey(), (5, 5))
        op = lx.MatrixLinearOperator(mat)
        assert tree_allclose(frobenius_norm(op), jnp.linalg.norm(mat))


class TestFrobeniusLowRank:
    """‖A₀ + U D Vᵀ‖_F against the dense norm, by the k × k Gram route (gh-694)."""

    @staticmethod
    def _bases(key, size=6):
        return {
            "zero": lx.DiagonalLinearOperator(jnp.zeros(size)),
            "diagonal": lx.DiagonalLinearOperator(jr.normal(key, (size,))),
            "identity": lx.IdentityLinearOperator(
                jax.eval_shape(lambda: jnp.zeros(size))
            ),
            "scaled_identity": 2.5
            * lx.IdentityLinearOperator(jax.eval_shape(lambda: jnp.zeros(size))),
            "dense": lx.MatrixLinearOperator(jr.normal(key, (size, size))),
        }

    @pytest.mark.parametrize(
        "base_name", ["zero", "diagonal", "identity", "scaled_identity", "dense"]
    )
    @pytest.mark.parametrize("update", ["symmetric", "weighted", "non_symmetric"])
    def test_matches_the_dense_norm(self, getkey, base_name, update):
        base = self._bases(getkey())[base_name]
        left = jr.normal(getkey(), (6, 3))
        weights = jr.normal(getkey(), (3,)) if update != "symmetric" else None
        right = jr.normal(getkey(), (6, 3)) if update == "non_symmetric" else None
        op = LowRankUpdate(base, left, weights, right)
        rtol, atol = default_tolerances(left)
        assert tree_allclose(
            frobenius_norm(op),
            jnp.linalg.norm(op.as_matrix()),
            rtol=100 * rtol,
            atol=100 * atol,
        )

    def test_rectangular_update(self, getkey):
        base = lx.MatrixLinearOperator(jr.normal(getkey(), (6, 4)))
        op = LowRankUpdate(
            base, jr.normal(getkey(), (6, 2)), V=jr.normal(getkey(), (4, 2))
        )
        assert tree_allclose(frobenius_norm(op), jnp.linalg.norm(op.as_matrix()))

    def test_rank_zero_is_the_base(self, getkey):
        base = lx.DiagonalLinearOperator(jr.normal(getkey(), (5,)))
        op = LowRankUpdate(base, jnp.zeros((5, 0)))
        assert tree_allclose(frobenius_norm(op), frobenius_norm(base))

    def test_an_ensemble_covariance_forms_no_state_sized_matrix(self):
        """No N × N array in the jaxpr, value or gradient, for N = 1e4."""
        size = 10_000
        particles = jnp.zeros((5, size))

        def norm(state):
            return frobenius_norm(gaussx.ensemble_covariance(state, bessel=True))

        for form in (norm, jax.grad(norm)):
            assert f"[{size},{size}]" not in str(jax.make_jaxpr(form)(particles))

    def test_structured_base_is_not_materialised(self, getkey, monkeypatch):
        base = BlockDiag(
            lx.DiagonalLinearOperator(jr.normal(getkey(), (2,))),
            lx.DiagonalLinearOperator(jr.normal(getkey(), (2,))),
        )
        op = LowRankUpdate(base, jr.normal(getkey(), (4, 2)))
        expected = jnp.linalg.norm(op.as_matrix())

        def forbidden(self):
            raise AssertionError("the operator was materialised")

        monkeypatch.setattr(BlockDiag, "as_matrix", forbidden)
        monkeypatch.setattr(LowRankUpdate, "as_matrix", forbidden)
        assert tree_allclose(frobenius_norm(op), expected)

    def test_an_empty_operator_has_zero_norm(self):
        update = LowRankUpdate(
            lx.MatrixLinearOperator(jnp.zeros((0, 0))), jnp.zeros((0, 1))
        )
        assert frobenius_norm(update) == 0.0

    def test_stochastic_base_is_estimated_not_materialised(self, psd_op, getkey):
        """A matrix-free base keeps its probes; the update stays exact."""
        left = jr.normal(getkey(), (psd_op.in_size(), 2))
        matrix_free = LowRankUpdate(_matvec_only(psd_op), left)
        exact = jnp.linalg.norm(LowRankUpdate(psd_op, left).as_matrix())
        estimate = frobenius_norm(matrix_free, stochastic=True, num_probes=400)
        assert jnp.abs(estimate - exact) / exact < 0.1

    def test_gradient_matches_the_dense_norm(self, getkey):
        particles = jr.normal(getkey(), (6, 4))

        def structured(state):
            return frobenius_norm(gaussx.ensemble_covariance(state, bessel=True))

        def dense(state):
            matrix = gaussx.ensemble_covariance(state, bessel=True).as_matrix()
            return jnp.linalg.norm(matrix)

        rtol, atol = default_tolerances(particles)
        assert tree_allclose(
            jax.grad(structured)(particles),
            jax.grad(dense)(particles),
            rtol=1000 * rtol,
            atol=1000 * atol,
        )

    @pytest.mark.parametrize(
        "make_base",
        [
            pytest.param(lx.DiagonalLinearOperator, id="diagonal"),
            pytest.param(
                lambda x: lx.MatrixLinearOperator(jnp.diag(x)),
                id="dense",
            ),
            pytest.param(lambda x: 2.0 * lx.DiagonalLinearOperator(x), id="scaled"),
            pytest.param(lambda x: lx.DiagonalLinearOperator(x) / 2.0, id="divided"),
            pytest.param(
                lambda x: lx.TaggedLinearOperator(
                    lx.DiagonalLinearOperator(x), lx.diagonal_tag
                ),
                id="tagged",
            ),
        ],
    )
    def test_gradient_through_a_zero_base_matches_the_dense_norm(self, make_base):
        """The update's norm is nonzero, so a zero base must not give NaN."""
        factor = jnp.ones((2, 1))

        def structured(values):
            return frobenius_norm(LowRankUpdate(make_base(values), factor))

        def dense(values):
            return jnp.linalg.norm(LowRankUpdate(make_base(values), factor).as_matrix())

        zeros = jnp.zeros(2)
        gradient = jax.grad(structured)(zeros)
        assert jnp.all(jnp.isfinite(gradient))
        assert tree_allclose(gradient, jax.grad(dense)(zeros))

    @pytest.mark.parametrize("scale", [1e30, 1e20, 1e10])
    def test_opposite_factor_scales_stay_in_range(self, scale):
        # U and V have squares out of float32 range, while U Vᵀ is near one.
        update = LowRankUpdate(
            lx.DiagonalLinearOperator(jnp.zeros(1, dtype=jnp.float32)),
            jnp.array([[scale]], dtype=jnp.float32),
            jnp.ones(1, dtype=jnp.float32),
            jnp.array([[1.0 / scale]], dtype=jnp.float32),
        )
        assert jnp.allclose(frobenius_norm(update), 1.0, rtol=1e-5)

    @pytest.mark.parametrize(
        ("scale", "weight"), [(1e20, 1e-30), (1e-20, 1e30), (1e30, 1e-30)]
    )
    def test_a_scale_product_out_of_range_can_be_offset_by_the_weight(
        self, scale, weight
    ):
        # s t or d s leaves float32's range, while d s t does not.
        update = LowRankUpdate(
            lx.DiagonalLinearOperator(jnp.zeros(1, dtype=jnp.float32)),
            jnp.array([[scale]], dtype=jnp.float32),
            jnp.array([weight], dtype=jnp.float32),
            jnp.array([[1.0]], dtype=jnp.float32),
        )
        balanced = LowRankUpdate(
            lx.DiagonalLinearOperator(jnp.zeros(1, dtype=jnp.float32)),
            jnp.array([[jnp.sqrt(jnp.float32(scale))]], dtype=jnp.float32),
            jnp.array([weight], dtype=jnp.float32),
            jnp.array([[jnp.sqrt(jnp.float32(scale))]], dtype=jnp.float32),
        )
        expected = scale * weight
        assert jnp.allclose(frobenius_norm(update), expected, rtol=1e-5)
        assert jnp.allclose(frobenius_norm(balanced), expected, rtol=1e-5)

    def test_complex_update_matches_the_dense_norm(self):
        keys = jr.split(jr.key(0), 4)
        left = jr.normal(keys[0], (4, 2)) + 1j * jr.normal(keys[1], (4, 2))
        right = jr.normal(keys[2], (4, 2)) + 1j * jr.normal(keys[3], (4, 2))
        weights = jnp.array([1.0 + 2.0j, 0.5 - 1.0j])
        diagonal = jnp.array([1.0 + 1.0j, 2.0, 0.0, 1.0j])
        update = LowRankUpdate(
            lx.DiagonalLinearOperator(diagonal), left, weights, right
        )
        dense = jnp.sqrt(jnp.sum(jnp.abs(update.as_matrix()) ** 2))
        result = frobenius_norm(update)
        assert jnp.isrealobj(result)
        rtol, atol = default_tolerances(result)
        assert tree_allclose(result, dense, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "make_base",
        [
            lambda zeros: BlockDiag(
                lx.MatrixLinearOperator(zeros), lx.MatrixLinearOperator(zeros)
            ),
            lambda zeros: Kronecker(
                lx.MatrixLinearOperator(zeros),
                lx.MatrixLinearOperator(jnp.eye(2)),
            ),
            lambda zeros: LowRankUpdate(
                BlockDiag(
                    lx.MatrixLinearOperator(zeros), lx.MatrixLinearOperator(zeros)
                ),
                jnp.zeros((4, 1)),
                jnp.ones(1),
                jnp.zeros((4, 1)),
            ),
        ],
        ids=["block-diagonal", "kronecker", "nested-update"],
    )
    def test_gradient_through_a_zero_structured_base_is_finite(self, make_base):
        left = jr.normal(jr.key(1), (4, 2))
        right = jr.normal(jr.key(2), (4, 2))
        weights = jnp.array([1.0, -0.5])

        def norm(zeros):
            return frobenius_norm(LowRankUpdate(make_base(zeros), left, weights, right))

        zeros = jnp.zeros((2, 2))
        gradient = jax.grad(norm)(zeros)
        assert jnp.isfinite(gradient).all()
        tangent = jr.normal(jr.key(3), (2, 2))
        step = 1e-3
        difference = (norm(step * tangent) - norm(-step * tangent)) / (2.0 * step)
        # Central difference of a smooth norm at a nonzero value: O(step²), and
        # float32 round-off of about ε / step in the difference.
        assert jnp.allclose(jnp.vdot(gradient, tangent), difference, atol=1e-2)

    def test_each_mode_is_balanced_separately(self):
        scales = jnp.array([1e25, 1.0, 1e-25])
        left = jr.normal(jr.key(0), (6, 3)) * scales
        right = jr.normal(jr.key(1), (6, 3)) / scales
        weights = jr.normal(jr.key(2), (3,))
        expected = jnp.array(0.0)
        for mode in range(3):
            for other in range(3):
                # Each pair of modes in units where its entries are near one
                left_gram = jnp.sum(left[:, mode] / scales[mode] * left[:, other])
                right_gram = jnp.sum(right[:, mode] * scales[mode] * right[:, other])
                expected += weights[mode] * weights[other] * left_gram * right_gram
        update = LowRankUpdate(
            lx.DiagonalLinearOperator(jnp.zeros(6)), left, weights, right
        )
        assert tree_allclose(frobenius_norm(update), jnp.sqrt(expected), rtol=1e-4)

    def test_jit_and_vmap_match_a_loop(self, getkey):
        ensembles = jr.normal(getkey(), (3, 6, 4))

        def norm(state):
            return frobenius_norm(gaussx.ensemble_covariance(state, bessel=True))

        looped = jnp.stack([norm(state) for state in ensembles])
        assert tree_allclose(jax.vmap(norm)(ensembles), looped)
        assert tree_allclose(jax.jit(norm)(ensembles[0]), looped[0])


class TestSLQErrorBars:
    def test_logdet_and_error(self, psd_op):
        slq = SLQLogdet(num_probes=30, lanczos_order=25)
        est, sem = slq.logdet_and_error(psd_op)
        true = jnp.linalg.slogdet(psd_op.as_matrix())[1]
        assert sem > 0.0
        assert jnp.abs(est - true) < 6.0 * sem + 1.0

    @pytest.mark.slow
    def test_point_estimate_matches_mean(self, psd_op):
        slq = SLQLogdet(num_probes=20, lanczos_order=20, seed=3)
        est, _ = slq.logdet_and_error(psd_op)
        assert tree_allclose(slq.logdet(psd_op), est)

    def test_indefinite_variant(self, psd_op):
        islq = IndefiniteSLQLogdet(num_probes=30, lanczos_order=25)
        est, sem = islq.logdet_and_error(psd_op)
        true = jnp.linalg.slogdet(psd_op.as_matrix())[1]
        assert sem > 0.0
        assert jnp.abs(est - true) < 6.0 * sem + 1.0


class TestPivotedCholeskyViaMatfree:
    def test_exact_rank(self, getkey):
        b = jr.normal(getkey(), (10, 4))
        op = lx.MatrixLinearOperator(
            b @ b.T, (lx.symmetric_tag, lx.positive_semidefinite_tag)
        )
        rd = root_decomposition(op, method="pivoted_cholesky", rank=4)
        assert tree_allclose(rd.root @ rd.root.T, op.as_matrix(), atol=1e-6)

    # Keys pinned (randomness is incidental — any rank-3 draw would do):
    # 27 is a regression key for gh-236, where the unguarded matfree pivot
    # division produced inf (not NaN) surplus columns; 0 is a clean draw.
    @pytest.mark.parametrize("seed", [0, 27])
    def test_rank_deficient_request_stays_finite(self, seed):
        b = jr.normal(jr.key(seed), (10, 3))
        op = lx.MatrixLinearOperator(
            b @ b.T, (lx.symmetric_tag, lx.positive_semidefinite_tag)
        )
        rd = root_decomposition(op, method="pivoted_cholesky", rank=6)
        assert bool(jnp.all(jnp.isfinite(rd.root)))
        assert tree_allclose(rd.root @ rd.root.T, op.as_matrix(), atol=1e-6)

    def test_jit(self, getkey):
        mat = random_pd_matrix(getkey(), 8)
        op = lx.MatrixLinearOperator(
            mat, (lx.symmetric_tag, lx.positive_semidefinite_tag)
        )

        @jax.jit
        def f(m):
            inner = lx.MatrixLinearOperator(
                m, (lx.symmetric_tag, lx.positive_semidefinite_tag)
            )
            return root_decomposition(inner, method="pivoted_cholesky", rank=8).root

        root = f(mat)
        assert tree_allclose(root @ root.T, op.as_matrix(), atol=1e-6)


# ---------------------------------------------------------------------------
# stochastic=True is forwarded through every wrapper and structure (gh-320)
# ---------------------------------------------------------------------------


class _NoMatrix(lx.FunctionLinearOperator):
    """Matvec-only operator whose ``as_matrix`` is forbidden."""

    def as_matrix(self):
        raise AssertionError("as_matrix called")


def _spy():
    M = random_pd_matrix(jr.key(0), 4)
    return _NoMatrix(lambda v: M @ v, jax.ShapeDtypeStruct((4,), M.dtype))


_WRAPPERS = [
    pytest.param(lambda op: 2.0 * op, id="mul"),
    pytest.param(lambda op: op / 2.0, id="div"),
    pytest.param(lambda op: -op, id="neg"),
    pytest.param(lambda op: op + op, id="add"),
    pytest.param(lambda op: BlockDiag(op, op), id="block_diag"),
    pytest.param(lambda op: Kronecker(op, op), id="kronecker"),
]
_PRIMITIVES = [
    pytest.param(trace, id="trace"),
    pytest.param(gaussx_diag, id="diag"),
    pytest.param(frobenius_norm, id="frobenius_norm"),
]


@pytest.mark.parametrize("primitive", _PRIMITIVES)
@pytest.mark.parametrize("wrap", _WRAPPERS)
def test_stochastic_is_forwarded(primitive, wrap):
    result = primitive(wrap(_spy()), stochastic=True, num_probes=4, key=jr.key(0))
    assert jnp.all(jnp.isfinite(result))


@pytest.mark.parametrize("primitive", _PRIMITIVES)
def test_stochastic_scalar_wrapper_is_exact_rescaling(primitive):
    """Same key, same probes: the wrapper's estimate is the rescaled one."""
    op = _spy()
    kw = dict(stochastic=True, num_probes=4, key=jr.key(0))
    inner = primitive(op, **kw)
    scale = 1.0 if primitive is frobenius_norm else -1.0
    assert jnp.array_equal(primitive(2.0 * op, **kw), 2.0 * inner)
    assert jnp.array_equal(primitive(op / 2.0, **kw), inner / 2.0)
    assert jnp.array_equal(primitive(-op, **kw), scale * inner)
