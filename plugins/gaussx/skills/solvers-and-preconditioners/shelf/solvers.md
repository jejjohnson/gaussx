# Choosing a solver

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: strategies.

Open this when a solve or a log-determinant is too big or too slow for the default, or when you must choose how it is computed.
The verbs `solve` and `logdet` and the operators live in the skill `gaussx:structured-gaussian-linalg`.

## Names

Every strategy has `solve(operator, vector)` and `logdet(operator, *, key=None)`, except where noted.

| Name | What it is |
|---|---|
| `AutoSolver(size_threshold=1000, throw=True, preconditioner=None)` | Picks an exact structural or dense solve, or CG for a large PSD operator. |
| `DenseSolver()` | Calls `gaussx.solve` and `gaussx.logdet`, so structured operators stay exact and cheap. |
| `CGSolver(rtol, atol, max_steps, preconditioner, throw, ...)` | Lineax CG for the solve and a stochastic Lanczos quadrature logdet. |
| `PreconditionedCGSolver(preconditioner_rank=50, shift=1.0, preconditioner=None, ...)` | CG with a partial Cholesky preconditioner, built per solve or passed in. |
| `BBMMSolver(num_probes=10, lanczos_order=100, ...)` | CG solve and stochastic logdet with the defaults of Gardner et al. 2018. |
| `MINRESSolver(shift=0.0, ...)` | MINRES for symmetric, possibly indefinite or singular systems. |
| `LSMRSolver(damp=0.0, ...)` | LSMR least squares for rectangular or ridge-damped systems. |
| `SketchAndPrecondLSMR(sketch="sparse_sign", sampling_factor=4.0, damp=0.0, ...)` | LSMR preconditioned by a sketch, for tall least squares with `n` up to about 10⁴. |
| `sketch_and_solve(operator, vector, *, sketch, damp=0.0)` | Approximate least squares from one sketch, as a function rather than a strategy. |
| `SparseCholeskySolver(ordering="rcm", backend="jax")` | Exact solve, logdet and `diag_inv` for a `SparseOperator` precision. |
| `LineaxSolver(solver)` | Wraps a `lineax.AbstractLinearSolver` as a solve-only strategy. |
| `ComposedSolver(solve_strategy, logdet_strategy)` | Takes the solve from one strategy and the logdet from another. |
| `compose_solver(solver=None, *, preconditioner=None, logdet=None)` | One strategy from three separate choices: attaches the preconditioner to the solve (and to a BBMM logdet), keeps the solver's own logdet when `logdet` is `None`, and raises naming the classes for a combination that would ignore a preconditioner or give two. |
| `KeyedSolver(strategy, key)` | Binds a PRNG key to a strategy's stochastic logdet. |

## Use it

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import gaussx

PSD = lx.positive_semidefinite_tag
n = 100
t = jnp.linspace(0.0, 1.0, n)
K = jnp.exp(-0.5 * ((t[:, None] - t[None, :]) / 0.2) ** 2)  # (n, n)
A = lx.MatrixLinearOperator(K + jnp.eye(n), PSD)  # K + sigma^2 I, sigma^2 = 1
b = jr.normal(jr.key(0), (n,))
ref = jnp.linalg.solve(A.as_matrix(), b)
ref_ld = jnp.linalg.slogdet(A.as_matrix())[1]

x_auto = gaussx.AutoSolver().solve(A, b)  # n <= 1000: DenseSolver
P = gaussx.PartialCholeskyPreconditioner.from_operator(
    lx.MatrixLinearOperator(K, PSD), rank=20, shift=1.0
)
x_cg = gaussx.CGSolver(preconditioner=P).solve(A, b)
assert jnp.allclose(x_auto, ref, atol=1e-3)
assert jnp.allclose(x_cg, ref, atol=1e-2)

exact_ld = gaussx.ComposedSolver(gaussx.CGSolver(), gaussx.DenseLogdet())
assert jnp.allclose(exact_ld.logdet(A), ref_ld, rtol=1e-4)
keyed = gaussx.KeyedSolver(gaussx.CGSolver(), jr.key(1))  # fresh probes per key
assert jnp.allclose(keyed.logdet(A), ref_ld, rtol=0.1)
```

## Rules

- `solver=None` means structural dispatch on the operator, not a default strategy; it is exact for structured operators and for dense ones.
- `AutoSolver` takes DenseSolver for any operator with an exact structural solve and logdet, and for any operator of size up to `size_threshold`.
- Above the threshold `AutoSolver` takes CG for a PSD operator and DenseSolver for anything else, and nothing picks `SparseCholeskySolver` by size.
- A stochastic logdet called without a `key` draws its probes from `PRNGKey(seed)`, so every call sees the same probes.
- Strategies that give that stochastic logdet: `AutoSolver` (large PSD), `CGSolver`, `PreconditionedCGSolver`, `BBMMSolver`, `MINRESSolver`, `LSMRSolver` and `SketchAndPrecondLSMR`.
- Pass `key=` to `logdet`, or wrap the strategy in `KeyedSolver`, when estimates are averaged or used under MCMC. A key passed to `logdet` overrides the bound one.
- `ComposedSolver(CGSolver(), DenseLogdet())` is the exact-logdet version of CG.

| Situation | Choose |
|---|---|
| Structured operator, or n up to 1000 | `None` or `DenseSolver`, which `AutoSolver` also picks |
| Large PSD operator, unknown conditioning | `AutoSolver`, or `CGSolver` directly |
| Large `K + σ² I` kernel system | `CGSolver(preconditioner=...)` or `PreconditionedCGSolver`, see [preconditioners](preconditioners.md) |
| Need the logdet exact with an iterative solve | `ComposedSolver(CGSolver(), DenseLogdet())` |
| Symmetric but indefinite or singular | `MINRESSolver` |
| Rectangular, tall or ridge least squares | `LSMRSolver`; `SketchAndPrecondLSMR` when `m` is much larger than `n` |
| GMRF or Laplace precision as a `SparseOperator` | `SparseCholeskySolver`, passed explicitly |
| A lineax solver you already trust | `LineaxSolver(lx.CG(...))`, solve only |
| Stochastic logdet with fresh probes | `KeyedSolver(strategy, key)` |

## Goes wrong

- `CGSolver` raises when it does not converge within `max_steps`; `throw=False` returns the last iterate unchecked, which can be far worse than zero.
- Left unset, `rtol` is 1e-3 in float32 and 1e-5 in float64, because a float32 CG cannot reach 1e-5 once the condition number passes about 10³.
- `KeyedSolver(strategy, key).solve` raises `TypeError` when the wrapped strategy is logdet-only.
- `SketchAndPrecondLSMR` with `damp=0` needs full column rank, otherwise `throw=True` raises.
- `LSMRSolver.logdet` raises `ValueError` on a non-square operator.

## Next

- Logdet-only strategies to put in a `ComposedSolver`: [logdet](logdet.md).
- Preconditioners for CG: [preconditioners](preconditioners.md).
- Accepting `solver=` in your own library: [pass-through](pass-through.md).
- Sketches and the factorisations behind them: [randomized](randomized.md).
