"""Tests for distribution-level KL divergence."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import gaussx
from gaussx._testing import random_pd_operator


class TestDistKLDivergence:
    @pytest.mark.slow
    def test_kl_same_distribution(self):
        """KL(p || p) = 0."""
        n = 4
        key = jax.random.PRNGKey(0)
        mu = jax.random.normal(key, (n,))
        cov = random_pd_operator(jax.random.PRNGKey(1), n, jitter=1.0)
        kl = gaussx.gaussian_kl(mu, cov, mu, cov)
        assert jnp.allclose(kl, 0.0, atol=1e-4)

    @pytest.mark.slow
    def test_kl_positive(self):
        """KL is non-negative."""
        n = 5
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
        mu_p = jax.random.normal(k1, (n,))
        mu_q = jax.random.normal(k2, (n,))
        cov_p = random_pd_operator(k3, n, jitter=1.0)
        cov_q = random_pd_operator(k4, n, jitter=1.0)
        kl = gaussx.gaussian_kl(mu_p, cov_p, mu_q, cov_q)
        assert kl >= -1e-6

    def test_kl_known_value(self):
        """KL between two known Gaussians matches closed-form."""
        n = 3
        mu_p = jnp.zeros(n)
        mu_q = jnp.ones(n)
        cov_p = lx.MatrixLinearOperator(jnp.eye(n), lx.positive_semidefinite_tag)
        cov_q = lx.MatrixLinearOperator(2.0 * jnp.eye(n), lx.positive_semidefinite_tag)
        kl = gaussx.gaussian_kl(mu_p, cov_p, mu_q, cov_q)
        # KL(N(0,I) || N(1, 2I)) = 0.5 * (tr(0.5 I) + 0.5*||1||^2 - 3 + 3*ln2)
        # = 0.5 * (1.5 + 1.5 - 3 + 3*ln2) = 0.5 * 3 * ln2 = 1.5 * ln2 ≈ 1.0397
        expected = 0.5 * (n / 2.0 + 0.5 * n - n + n * jnp.log(2.0))
        assert jnp.allclose(kl, expected, atol=1e-4)

    def test_kl_diagonal_operators(self):
        """KL with diagonal covariance operators."""
        n = 4
        mu_p = jnp.zeros(n)
        mu_q = jnp.zeros(n)
        diag_p = jnp.array([1.0, 2.0, 3.0, 4.0])
        diag_q = jnp.array([2.0, 3.0, 4.0, 5.0])
        cov_p = lx.DiagonalLinearOperator(diag_p)
        cov_q = lx.DiagonalLinearOperator(diag_q)
        kl = gaussx.gaussian_kl(mu_p, cov_p, mu_q, cov_q)
        # Closed-form for diagonal
        expected = 0.5 * (
            jnp.sum(diag_p / diag_q) - n + jnp.sum(jnp.log(diag_q / diag_p))
        )
        assert jnp.allclose(kl, expected, atol=1e-5)

    def test_kl_asymmetric(self):
        """KL(p||q) != KL(q||p) in general."""
        n = 3
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
        mu_p = jax.random.normal(k1, (n,))
        mu_q = jax.random.normal(k2, (n,))
        cov_p = random_pd_operator(k3, n, jitter=1.0)
        cov_q = random_pd_operator(k4, n, jitter=1.0)
        kl_pq = gaussx.gaussian_kl(mu_p, cov_p, mu_q, cov_q)
        kl_qp = gaussx.gaussian_kl(mu_q, cov_q, mu_p, cov_p)
        assert not jnp.allclose(kl_pq, kl_qp, atol=1e-3)


class TestKLStandardNormalConsistency:
    """Verify kl_standard_normal is consistent with gaussian_kl."""

    def test_matches_dist_kl_diagonal(self):
        """kl_standard_normal matches gaussian_kl with identity prior."""
        n = 4
        key = jax.random.PRNGKey(7)
        m = jax.random.normal(key, (n,))
        diag = jnp.array([0.5, 1.0, 2.0, 3.0])
        S = lx.DiagonalLinearOperator(diag)

        result = gaussx.kl_standard_normal(m, S)

        # Equivalent gaussian_kl call: KL(N(m,S) || N(0, I))
        identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n,), m.dtype))
        zero = jnp.zeros(n)
        expected = gaussx.gaussian_kl(m, S, zero, identity)

        assert jnp.allclose(result, expected, atol=1e-5)

    def test_matches_dist_kl_dense(self):
        """kl_standard_normal matches gaussian_kl for dense covariance."""
        n = 3
        m = jax.random.normal(jax.random.PRNGKey(42), (n,))
        S_op = random_pd_operator(jax.random.PRNGKey(43), n, jitter=1.0)

        result = gaussx.kl_standard_normal(m, S_op)

        identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n,), m.dtype))
        zero = jnp.zeros(n)
        expected = gaussx.gaussian_kl(m, S_op, zero, identity)

        assert jnp.allclose(result, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# gaussian_kl(solver=...) (gh-884)
# ---------------------------------------------------------------------------

_N = 6
# CG to 1e-10 leaves the solves ~1e-9 from dense in float64; in float32 the
# strategies' own dtype-aware defaults apply, so the tolerances follow the
# dtype.
_TIGHT = (
    {"rtol": 1e-10, "atol": 1e-10, "max_steps": 200}
    if jax.config.read("jax_enable_x64")
    else {}
)


def _tolerance(dtype):
    return 1e-7 if dtype == jnp.float64 else 1e-3


def _pair(dtype=None):
    """Two random Gaussians p and q on ℝ⁶, in the active default float."""
    keys = jax.random.split(jax.random.PRNGKey(3), 4)
    p_cov = random_pd_operator(keys[0], _N, jitter=1.0, dtype=dtype)
    q_cov = random_pd_operator(keys[1], _N, jitter=1.0, dtype=dtype)
    dtype = p_cov.in_structure().dtype
    p_loc = jax.random.normal(keys[2], (_N,), dtype=dtype)
    q_loc = jax.random.normal(keys[3], (_N,), dtype=dtype)
    return p_loc, p_cov, q_loc, q_cov


_SOLVERS = {
    "dense": lambda: gaussx.DenseSolver(),
    "cg": lambda: gaussx.CGSolver(**_TIGHT),
    "cg_jacobi": lambda: gaussx.CGSolver(
        preconditioner=gaussx.JacobiPreconditioner(), **_TIGHT
    ),
    "bbmm": lambda: gaussx.BBMMSolver(**_TIGHT),
}


@pytest.mark.parametrize("name", list(_SOLVERS))
def test_solver_matches_dense_up_to_its_own_logdet(name):
    """Each solver's KL is the dense KL with log det Σ_q swapped for its own.

    The solves are exact to the solver's tolerance, so the only difference
    from the structural KL is the solver's own log det Σ_q (stochastic for
    CG and BBMM), which `strategy.logdet` gives with the same default seed.
    """
    p_loc, p_cov, q_loc, q_cov = _pair()
    solver = _SOLVERS[name]()
    dense = gaussx.gaussian_kl(p_loc, p_cov, q_loc, q_cov)
    got = gaussx.gaussian_kl(p_loc, p_cov, q_loc, q_cov, solver=solver)
    # ½ (log det Σ_q by the solver − exact log det Σ_q)
    logdet_shift = 0.5 * (solver.logdet(q_cov) - gaussx.logdet(q_cov))
    tolerance = _tolerance(p_loc.dtype)
    assert jnp.allclose(got, dense + logdet_shift, rtol=tolerance, atol=tolerance)


def test_exact_logdet_composition_matches_dense():
    """Preconditioned CG with an exact log-determinant gives the dense KL."""
    p_loc, p_cov, q_loc, q_cov = _pair()
    solver = gaussx.compose_solver(
        gaussx.CGSolver(**_TIGHT),
        preconditioner=gaussx.JacobiPreconditioner(),
        logdet=gaussx.DenseLogdet(),
    )
    dense = gaussx.gaussian_kl(p_loc, p_cov, q_loc, q_cov)
    got = gaussx.gaussian_kl(p_loc, p_cov, q_loc, q_cov, solver=solver)
    tolerance = _tolerance(p_loc.dtype)
    assert jnp.allclose(got, dense, rtol=tolerance, atol=tolerance)


class _RaisingSolver(gaussx.DenseSolver):
    """A solver that fails whenever it is called."""

    def solve(self, operator, vector):
        raise RuntimeError("solver reached")

    def logdet(self, operator, *, key=None):
        raise RuntimeError("solver reached")


class _RaisingLogdet(gaussx.DenseLogdet):
    """A log-determinant strategy that fails whenever it is called."""

    def logdet(self, operator, *, key=None):
        raise RuntimeError("p_logdet reached")


def test_raising_solver_is_reached():
    """The argument is used, not ignored (the pyrox guides' gh-884 symptom)."""
    p_loc, p_cov, q_loc, q_cov = _pair()
    with pytest.raises(RuntimeError, match="solver reached"):
        gaussx.gaussian_kl(p_loc, p_cov, q_loc, q_cov, solver=_RaisingSolver())


def test_p_logdet_takes_the_first_covariance():
    """``p_logdet`` serves log det Σ_p, and only with a solver."""
    p_loc, p_cov, q_loc, q_cov = _pair()
    with pytest.raises(RuntimeError, match="p_logdet reached"):
        gaussx.gaussian_kl(
            p_loc,
            p_cov,
            q_loc,
            q_cov,
            solver=gaussx.DenseSolver(),
            p_logdet=_RaisingLogdet(),
        )
    # with solver=None the structural path ignores it
    gaussx.gaussian_kl(p_loc, p_cov, q_loc, q_cov, p_logdet=_RaisingLogdet())


def test_solver_none_is_bit_identical():
    """``solver=None`` keeps the structural path, whatever the other options."""
    p_loc, p_cov, q_loc, q_cov = _pair()
    today = gaussx.gaussian_kl(p_loc, p_cov, q_loc, q_cov)
    explicit = gaussx.gaussian_kl(
        p_loc, p_cov, q_loc, q_cov, solver=None, key=jax.random.PRNGKey(0)
    )
    assert today == explicit
    grad_today = jax.grad(gaussx.gaussian_kl)(p_loc, p_cov, q_loc, q_cov)
    grad_explicit = jax.grad(
        lambda loc: gaussx.gaussian_kl(loc, p_cov, q_loc, q_cov, solver=None)
    )(p_loc)
    assert jnp.array_equal(grad_today, grad_explicit)


def test_self_kl_is_zero_with_an_exact_solver():
    """KL[q ‖ q] = 0: the trace is D_x and the log-determinants cancel."""
    _, _, q_loc, q_cov = _pair()
    kl = gaussx.gaussian_kl(q_loc, q_cov, q_loc, q_cov, solver=gaussx.DenseSolver())
    # 1e-8 is the gh-884 bound in float64; float32 round-off on six solves
    # of a matrix with κ ~ 10 is ~1e-6 relative to the D_x = 6 trace.
    atol = 1e-8 if kl.dtype == jnp.float64 else 1e-5
    assert jnp.abs(kl) < atol


def test_hutchinson_trace_within_its_sampling_error():
    """With a key, tr(Σ_q⁻¹ Σ_p) is a Rademacher estimate, bounded by its σ.

    Var(zᵀ B z) = ∑_{i≠j} (B_ij² + B_ij B_ji) for sign probes, so the
    estimate's error is below 7 σ / √N_probe (gh-220).
    """
    p_loc, p_cov, q_loc, q_cov = _pair()
    num_probes = 64
    key = jax.random.PRNGKey(5)
    solver = gaussx.CGSolver(**_TIGHT)
    exact = gaussx.gaussian_kl(p_loc, p_cov, q_loc, q_cov, solver=solver)
    estimate = gaussx.gaussian_kl(
        p_loc, p_cov, q_loc, q_cov, solver=solver, num_probes=num_probes, key=key
    )
    # The key also reseeds CG's SLQ log det Σ_q; remove both log-determinants.
    _, q_key, _ = jax.random.split(key, 3)
    exact_logdet = solver.logdet(q_cov)
    keyed_logdet = solver.logdet(q_cov, key=q_key)
    trace_error = 2.0 * (estimate - exact) - (keyed_logdet - exact_logdet)
    # B = Σ_q⁻¹ Σ_p
    product = jnp.linalg.solve(q_cov.as_matrix(), p_cov.as_matrix())
    off_diagonal = product - jnp.diag(jnp.diag(product))
    variance = jnp.sum(off_diagonal**2 + off_diagonal * off_diagonal.T)
    assert jnp.abs(trace_error) < 7.0 * jnp.sqrt(variance / num_probes)


def test_hutchinson_trace_is_exact_for_equal_covariances():
    """Σ_q⁻¹ Σ_p = I makes every sign probe exact: zᵀ z = D_x."""
    _, _, q_loc, q_cov = _pair()
    solver = gaussx.ComposedSolver(gaussx.CGSolver(**_TIGHT), gaussx.DenseLogdet())
    kl = gaussx.gaussian_kl(
        q_loc, q_cov, q_loc, q_cov, solver=solver, key=jax.random.PRNGKey(0)
    )
    assert jnp.abs(kl) < _tolerance(q_loc.dtype)


def test_jit_vmap_grad_with_a_solver():
    """jit, vmap over the means and grad in both means and both covariances."""
    p_loc, p_cov, q_loc, q_cov = _pair()
    p_matrix, q_matrix = p_cov.as_matrix(), q_cov.as_matrix()
    solver = gaussx.ComposedSolver(gaussx.CGSolver(**_TIGHT), gaussx.DenseLogdet())

    def kl(p_loc, p_matrix, q_loc, q_matrix, solver=solver):
        return gaussx.gaussian_kl(
            p_loc,
            lx.MatrixLinearOperator(p_matrix, lx.positive_semidefinite_tag),
            q_loc,
            lx.MatrixLinearOperator(q_matrix, lx.positive_semidefinite_tag),
            solver=solver,
        )

    def dense(p_loc, p_matrix, q_loc, q_matrix):
        return kl(p_loc, p_matrix, q_loc, q_matrix, solver=None)

    tolerance = _tolerance(p_loc.dtype)
    arguments = (p_loc, p_matrix, q_loc, q_matrix)
    assert jnp.allclose(jax.jit(kl)(*arguments), dense(*arguments), atol=tolerance)

    shifts = jax.random.normal(jax.random.PRNGKey(9), (3, _N), dtype=p_loc.dtype)
    batched = jax.vmap(lambda shift: kl(p_loc + shift, p_matrix, q_loc, q_matrix))
    looped = jnp.stack(
        [dense(p_loc + shift, p_matrix, q_loc, q_matrix) for shift in shifts]
    )
    assert jnp.allclose(batched(shifts), looped, atol=tolerance)

    got = jax.grad(kl, argnums=(0, 1, 2, 3))(*arguments)
    expected = jax.grad(dense, argnums=(0, 1, 2, 3))(*arguments)
    for got_part, expected_part in zip(got, expected, strict=True):
        assert jnp.all(jnp.isfinite(got_part))
        assert jnp.allclose(got_part, expected_part, rtol=tolerance, atol=tolerance)
