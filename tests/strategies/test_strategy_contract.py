"""Contract suite for the solve strategies (gh-409).

Every strategy promises the same things, whatever its algorithm:

1. **grad vs dense**: ``jax.grad`` of a solve, with respect to a parameter of
   the operator and with respect to the right-hand side, matches the dense
   ``jnp.linalg.solve`` gradient.
2. **vmap over rhs**: ``jax.vmap(strategy.solve, in_axes=(None, 0))`` matches
   a Python loop.
3. **float32**: a default-constructed strategy solves a float32 system of
   condition number 1e3 to working precision instead of raising.
4. **max_steps**: an iterative strategy that runs out of steps says so, by
   raising, instead of returning an unconverged iterate silently.

It also checks that each preconditioner, at a rank ``k < n``, at least
halves the number of CG steps; replacing ``as_operator`` with the identity
fails that test.

Every key is pinned (``jr.key(0)`` and friends) and each tolerance's
provenance is in a comment. Known failures are ``xfail(strict=True)`` and
name their issue, so the PR that fixes one has to remove its marker.
"""

from __future__ import annotations

from collections.abc import Callable

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._einx import einsum
from gaussx._linalg._symmetrize import symmetrize
from gaussx._testing import psd_operator, tree_allclose


# ---------------------------------------------------------------------------
# The strategies under contract
# ---------------------------------------------------------------------------

# A strategy factory takes the PSD part ``K`` of a system ``A = K + σ² I``
# (Nyström is built from ``K``, the lazy partial Cholesky is told σ²), σ²,
# and options in the canonical spelling (rtol/atol/max_steps), which it
# translates to the strategy's own keywords. ``None`` options keep the
# strategy's defaults.
Factory = Callable[..., gaussx.AbstractSolverStrategy]


def _cg_options(rtol=None, atol=None, max_steps=None):
    opts = {"rtol": rtol, "atol": atol, "max_steps": max_steps}
    return {k: v for k, v in opts.items() if v is not None}


def _bbmm(K, noise, rtol=None, atol=None, max_steps=None):
    opts = {"cg_tolerance": rtol, "cg_max_iter": max_steps}
    return gaussx.BBMMSolver(**{k: v for k, v in opts.items() if v is not None})


def _lsmr(damp):
    def make(K, noise, rtol=None, atol=None, max_steps=None):
        opts = {"btol": rtol, "atol": atol, "maxiter": max_steps}
        opts = {k: v for k, v in opts.items() if v is not None}
        return gaussx.LSMRSolver(damp=damp, **opts)

    return make


def _nystrom(K, noise, **opts):
    pre = gaussx.NystromPreconditioner.from_operator(
        psd_operator(K), rank=5, shift=noise, key=jr.key(0)
    )
    return gaussx.CGSolver(preconditioner=pre, **_cg_options(**opts))


STRATEGIES: dict[str, Factory] = {
    "dense": lambda K, noise, **opts: gaussx.DenseSolver(),
    "cg": lambda K, noise, **opts: gaussx.CGSolver(**_cg_options(**opts)),
    "cg_jacobi": lambda K, noise, **opts: gaussx.CGSolver(
        preconditioner=gaussx.JacobiPreconditioner(), **_cg_options(**opts)
    ),
    "cg_partial_cholesky": lambda K, noise, **opts: gaussx.CGSolver(
        preconditioner=gaussx.PartialCholeskyPreconditioner(rank=5, shift=noise),
        **_cg_options(**opts),
    ),
    "cg_nystrom": _nystrom,
    "preconditioned_cg": lambda K, noise, **opts: gaussx.PreconditionedCGSolver(
        preconditioner_rank=5, shift=noise, **_cg_options(**opts)
    ),
    "bbmm": _bbmm,
    "minres": lambda K, noise, **opts: gaussx.MINRESSolver(**_cg_options(**opts)),
    "lsmr": _lsmr(0.0),
    "lsmr_damped": _lsmr(0.5),
    # size_threshold below n, so a PSD system goes to CG.
    "auto": lambda K, noise, **opts: gaussx.AutoSolver(size_threshold=10),
}

ITERATIVE = [name for name in STRATEGIES if name not in ("dense", "auto")]


def _reference(name: str, A: jax.Array, b: jax.Array) -> jax.Array:
    """What the strategy should return: ``A⁻¹ b``, or the damped LSMR one."""
    normal, rhs = _normal_equations(name, A, b)
    return jnp.linalg.solve(normal, rhs)


def _normal_equations(name: str, A: jax.Array, b: jax.Array):
    """The square system the strategy solves: ``(A, b)``, or damped LSMR's."""
    if name == "lsmr_damped":
        # argmin ||A x − b||² + damp² ||x||² = (AᵀA + damp² I)⁻¹ Aᵀ b.
        normal = einsum(A, A, "k i, k j -> i j") + 0.25 * jnp.eye(A.shape[0])
        return normal, einsum(A, b, "k i, k -> i")
    return A, b


# ---------------------------------------------------------------------------
# 1. grad vs dense
# ---------------------------------------------------------------------------

# A 30-point RBF Gram plus 0.1 noise (κ ≈ 1e2), the gh-312 system.
_X = jnp.linspace(0.0, 1.0, 30)
_Y = jnp.sin(6.0 * _X)
_NOISE = 0.1
# Solver tolerance 1e-10 (x64) leaves the iterative gradients ~1e-8 from the
# dense one, so rtol=1e-6 (the gh-409 contract) has two digits of headroom.
_TIGHT = {"rtol": 1e-10, "atol": 1e-10, "max_steps": 500}
_GRAD_RTOL = 1e-6
# AutoSolver builds CGSolver() with its default 1e-5 tolerances, which it
# does not expose, so its gradient is only good to about that.
_GRAD_RTOL_AUTO = 1e-4


def _kernel(lengthscale):
    sq = einx.subtract("i, j -> i j", _X, _X) ** 2
    return jnp.exp(-0.5 * sq / lengthscale**2)


def _system(lengthscale):
    return _kernel(lengthscale) + _NOISE * jnp.eye(_X.shape[0])


def _grad_rtol(name):
    return _GRAD_RTOL_AUTO if name == "auto" else _GRAD_RTOL


# Grad cases that fail today, with the issue that fixes them.
_GRAD_XFAIL: dict[str, str] = {}


def _cases(names, known: dict[str, str], fast: tuple[str, ...]):
    """Parametrise over *names*: xfail the *known* failures, keep *fast* fast.

    Each case compiles its own solver programs (~0.5-3 s warm), so only a
    representative few run in the fast lane and the rest are ``slow``
    (gh-409: the fast lane may grow by under 15 s).
    """
    cases = []
    for name in names:
        marks = []
        if name in known:
            marks.append(pytest.mark.xfail(strict=True, reason=known[name]))
        if name not in fast:
            marks.append(pytest.mark.slow)
        cases.append(pytest.param(name, marks=marks))
    return cases


@pytest.mark.x64_only(reason="rtol=1e-6 gradient contract needs float64 solves")
@pytest.mark.parametrize("name", _cases(STRATEGIES, _GRAD_XFAIL, ("dense", "cg")))
def test_grad_wrt_operator_matches_dense(name):
    make = STRATEGIES[name]

    def loss(lengthscale):
        A = _system(lengthscale)
        strategy = make(_kernel(lengthscale), _NOISE, **_TIGHT)
        return _Y @ strategy.solve(psd_operator(A), _Y)

    def dense_loss(lengthscale):
        return _Y @ _reference(name, _system(lengthscale), _Y)

    got = jax.grad(loss)(0.3)
    expected = jax.grad(dense_loss)(0.3)
    assert jnp.isfinite(got)
    assert jnp.allclose(got, expected, rtol=_grad_rtol(name)), (got, expected)


@pytest.mark.x64_only(reason="rtol=1e-6 gradient contract needs float64 solves")
@pytest.mark.parametrize("name", _cases(STRATEGIES, _GRAD_XFAIL, ("dense", "cg")))
def test_grad_wrt_rhs_matches_dense(name):
    A = _system(0.3)
    op = psd_operator(A)
    strategy = STRATEGIES[name](_kernel(0.3), _NOISE, **_TIGHT)

    got = jax.grad(lambda b: jnp.sum(strategy.solve(op, b) ** 2))(_Y)
    expected = jax.grad(lambda b: jnp.sum(_reference(name, A, b) ** 2))(_Y)
    # atol: entries of the gradient near zero (|g| ~ 1e-3 of the largest).
    assert tree_allclose(got, expected, rtol=_grad_rtol(name), atol=1e-8)


# ---------------------------------------------------------------------------
# 2. vmap over the right-hand side
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _cases(STRATEGIES, {}, ("dense", "cg", "minres")))
def test_vmap_over_rhs_matches_loop(name):
    A = _system(0.3)
    op = psd_operator(A)
    strategy = STRATEGIES[name](_kernel(0.3), _NOISE, **_TIGHT)
    B = jr.normal(jr.key(0), (3, _X.shape[0]), dtype=A.dtype)

    batched = jax.vmap(strategy.solve, in_axes=(None, 0))(op, B)
    looped = jnp.stack([strategy.solve(op, b) for b in B])
    # A batched CG/MINRES run stops when every column has converged, so a
    # column may take a few more steps than alone: equal to the solver
    # tolerance (1e-10; AutoSolver's CG uses its default 1e-5), not bit for
    # bit.
    rtol, atol = (1e-4, 1e-6) if name == "auto" else (1e-7, 1e-9)
    assert tree_allclose(batched, looped, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# 3. float32 defaults
# ---------------------------------------------------------------------------

# The gh-327 system: n = 300, eigenvalues logspace(0, 3) (κ = 1e3), float32.
_F32_N = 300
# Under x64, lineax's LSMR builds float64 constants into a float32 iteration
# (a lax.select dtype error), and matfree's damped LSMR does the same in its
# Givens rotations. Both are upstream; float32 with x64 off is unaffected.
_F32_XFAIL = {
    "lsmr": "lineax LSMR mixes float64 into a float32 solve under x64 (upstream)",
    "lsmr_damped": "matfree LSMR mixes float64 into a float32 solve under x64 "
    "(upstream)",
}


def _f32_system(log_kappa):
    q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (_F32_N, _F32_N), dtype=jnp.float32))
    ev = jnp.logspace(0, log_kappa, _F32_N, dtype=jnp.float32)
    A = symmetrize(einsum(q * ev, q, "i k, j k -> i j"))
    b = jr.normal(jr.key(1), (_F32_N,), dtype=jnp.float32)
    return A, b


@pytest.mark.parametrize("name", _cases(STRATEGIES, _F32_XFAIL, ("cg", "bbmm")))
def test_float32_defaults_converge(name):
    A, b = _f32_system(3)
    # Split A = K + 0.5 I for the preconditioners that need the noise.
    K = A - 0.5 * jnp.eye(_F32_N, dtype=A.dtype)
    strategy = STRATEGIES[name](K, 0.5)
    x = strategy.solve(psd_operator(A), b)
    assert x.dtype == jnp.float32
    # gh-327 acceptance: relative residual ≤ 1e-3. Float32 CG at tolerance
    # 1e-3 reaches ~5e-4 on this system (the gh-327 table).
    M, rhs = _normal_equations(name, A, b)
    residual = jnp.linalg.norm(M @ x - rhs) / jnp.linalg.norm(rhs)
    assert residual <= 1e-3, residual


# ---------------------------------------------------------------------------
# 4. max_steps exhaustion
# ---------------------------------------------------------------------------

# MINRES runs a fixed-length scan and returns its last iterate whatever the
# residual; the damped LSMR path (matfree) likewise reports nothing.
_MAX_STEPS_XFAIL = {
    "minres": "gh-336: MINRESSolver returns unconverged iterates silently",
    "lsmr_damped": "gh-336: damped LSMR (matfree) returns unconverged iterates",
}


@pytest.mark.parametrize("name", _cases(ITERATIVE, _MAX_STEPS_XFAIL, ("cg", "minres")))
def test_exhausting_max_steps_raises(name):
    # κ = 1e4, n = 50: no Krylov method converges to 1e-8 in 3 steps.
    n = 50
    q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (n, n)))
    A = symmetrize(einsum(q * jnp.logspace(0, 4, n), q, "i k, j k -> i j"))
    b = jr.normal(jr.key(1), (n,), dtype=A.dtype)
    K = A - 0.5 * jnp.eye(n, dtype=A.dtype)
    strategy = STRATEGIES[name](K, 0.5, rtol=1e-8, atol=1e-8, max_steps=3)
    with pytest.raises(eqx.EquinoxRuntimeError):
        jax.block_until_ready(strategy.solve(psd_operator(A), b))


# ---------------------------------------------------------------------------
# Preconditioner quality at rank k < n
# ---------------------------------------------------------------------------

# The gh-409 spectrum: K = Q diag(1e3 · exp(−i/4)) Qᵀ, n = 200, σ² = 1e-2.
_PRE_N, _PRE_RANK, _PRE_NOISE = 200, 20, 1e-2


def _pre_kernel():
    q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (_PRE_N, _PRE_N)))
    ev = 1e3 * jnp.exp(-0.25 * jnp.arange(_PRE_N))
    return symmetrize(einsum(q * ev, q, "i k, j k -> i j"))


def _cg_steps(system, b, pre):
    options = {} if pre is None else {"preconditioner": pre}
    sol = lx.linear_solve(
        system,
        b,
        lx.CG(rtol=1e-8, atol=1e-8, max_steps=20000),
        options=options,
        throw=False,
    )
    return int(sol.stats["num_steps"])


def _badly_scaled_system():
    """``D (K + σ² I) D`` with ``D`` spanning four decades, for Jacobi."""
    scale = jnp.logspace(0, 2, _PRE_N)
    A = _pre_kernel() + _PRE_NOISE * jnp.eye(_PRE_N)
    return einx.multiply("i j, i, j -> i j", A, scale, scale)


def _preconditioned_system(name):
    """``(A, M⁻¹)`` for one preconditioner on its system."""
    psd = lx.positive_semidefinite_tag
    if name == "jacobi":
        A = _badly_scaled_system()
        pre = gaussx.JacobiPreconditioner().as_operator(lx.MatrixLinearOperator(A, psd))
        return A, pre
    K = _pre_kernel()
    A = K + _PRE_NOISE * jnp.eye(_PRE_N)
    if name == "partial_cholesky":
        pre = gaussx.PartialCholeskyPreconditioner.from_operator(
            lx.MatrixLinearOperator(K, psd), rank=_PRE_RANK, shift=_PRE_NOISE
        )
    elif name == "partial_cholesky_lazy":
        pre = gaussx.PartialCholeskyPreconditioner(rank=_PRE_RANK, shift=_PRE_NOISE)
    else:
        pre = gaussx.NystromPreconditioner.from_operator(
            lx.MatrixLinearOperator(K, psd),
            rank=_PRE_RANK,
            shift=_PRE_NOISE,
            key=jr.key(0),
        )
    return A, pre.as_operator(lx.MatrixLinearOperator(A, psd))


@pytest.mark.parametrize(
    "name",
    _cases(
        ["jacobi", "partial_cholesky", "partial_cholesky_lazy", "nystrom"],
        {},
        ("partial_cholesky", "nystrom"),
    ),
)
def test_preconditioner_halves_cg_steps_below_full_rank(name):
    A, pre = _preconditioned_system(name)
    system = psd_operator(A)
    b = jr.normal(jr.key(1), (_PRE_N,))
    plain = _cg_steps(system, b, None)
    # gh-409 acceptance: at most half the unpreconditioned count. The
    # identity preconditioner takes exactly `plain`, so it fails this.
    assert _cg_steps(system, b, pre) <= plain // 2
