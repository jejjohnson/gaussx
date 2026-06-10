"""Tests for the matfree-0.6-backed stochastic estimators and new primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import frobenius_norm, trace, trace_and_diag
from gaussx._operators import BlockDiag, Kronecker
from gaussx._primitives._root import root_decomposition
from gaussx._strategies._slq_logdet import IndefiniteSLQLogdet, SLQLogdet
from gaussx._testing import random_pd_matrix, tree_allclose


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


class TestSLQErrorBars:
    def test_logdet_and_error(self, psd_op):
        slq = SLQLogdet(num_probes=30, lanczos_order=25)
        est, sem = slq.logdet_and_error(psd_op)
        true = jnp.linalg.slogdet(psd_op.as_matrix())[1]
        assert sem > 0.0
        assert jnp.abs(est - true) < 6.0 * sem + 1.0

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

    def test_rank_deficient_request_is_nan_free(self, getkey):
        b = jr.normal(getkey(), (10, 3))
        op = lx.MatrixLinearOperator(
            b @ b.T, (lx.symmetric_tag, lx.positive_semidefinite_tag)
        )
        rd = root_decomposition(op, method="pivoted_cholesky", rank=6)
        assert not bool(jnp.any(jnp.isnan(rd.root)))
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
