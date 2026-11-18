# Passing a strategy through

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: strategies.

Open this when you write a library on top of gaussx and want its users to choose the solver, or when all you have is a matvec and need a solve.
The verb `solve` and the operators live in the skill `gaussx:structured-gaussian-linalg`.

## Names

| Name | What it is |
|---|---|
| `AbstractSolveStrategy` | The base class of anything with `solve(operator, vector)`. Annotate a solve-only `solver=` with it. |
| `AbstractSolverStrategy` | A strategy with both `solve` and `logdet`, the type that gaussx distributions take as `solver=`. |
| `linear_solve(operator, vector, *, solver=None, preconditioner=None)` | The front door: `operator` is a lineax operator or a `(matvec, shape)` pair, and the result is `x` for `A x = b`. |
| `as_linear_operator(matvec, *, shape=None, in_structure=None, dtype=float, symmetric=False, positive_semidefinite=False, negative_definite=False)` | Wraps a bare `v -> A @ v` as a tagged `lineax.FunctionLinearOperator`. |

## Use it

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import gaussx


def fit(A, b, *, solver: gaussx.AbstractSolverStrategy | None = None):
    x = gaussx.solve(A, b, solver=solver)  # None keeps structural dispatch
    return x, gaussx.logdet(A) if solver is None else solver.logdet(A)


n = 50
M = jr.normal(jr.key(0), (n, n))
dense = M @ M.T / n + jnp.eye(n)  # (n, n)
A = lx.MatrixLinearOperator(dense, lx.positive_semidefinite_tag)
b = jr.normal(jr.key(1), (n,))
ref = jnp.linalg.solve(dense, b)
x0, _ = fit(A, b)
x1, ld1 = fit(
    A, b, solver=gaussx.ComposedSolver(gaussx.CGSolver(), gaussx.DenseLogdet())
)
assert jnp.allclose(x0, ref, atol=1e-3) and jnp.allclose(x1, ref, atol=1e-2)
assert jnp.allclose(ld1, jnp.linalg.slogdet(dense)[1], rtol=1e-4)
matvec = lambda v: dense @ v  # all the caller has
x2 = gaussx.linear_solve(
    (matvec, (n, n)), b, solver=lx.LU()
)  # untagged: needs a solver
psd = gaussx.as_linear_operator(matvec, shape=(n, n), positive_semidefinite=True)
x3 = gaussx.linear_solve(psd, b, solver=gaussx.CGSolver())
neg = gaussx.as_linear_operator(
    lambda v: -matvec(v), shape=(n, n), negative_definite=True
)
x4 = gaussx.linear_solve(neg, -b)  # solved as the positive system (-A) x = b
assert all(jnp.allclose(x, ref, atol=1e-2) for x in (x2, x3, x4))
```

## Rules

- Type a `solver=` argument that needs both `solve` and `logdet` as `gaussx.AbstractSolverStrategy | None`, as gaussx's own distributions do.
- Type it as `gaussx.AbstractSolveStrategy | None` when your code only solves; `AbstractLogdetStrategy` is the type for a logdet-only argument.
- Default it to `None` and pass it straight down, so that `None` keeps structural dispatch and a strategy takes over the whole solve.
- Do not choose a solver inside your library; take it from the caller.
- `gaussx.solve(A, b, solver=...)` takes a gaussx strategy or a lineax solver, and `linear_solve` also wraps a lineax solver in `LineaxSolver`.
- With `solver=None`, `linear_solve` uses `AutoSolver` for a PSD operator, `MINRESSolver` for a symmetric one, and raises `ValueError` otherwise.
- Tag the operator you build from a matvec with `symmetric=True` or `positive_semidefinite=True`, or a non-symmetric one needs an explicit `solver=`.
- A negative semidefinite operator that is not also PSD is solved through the equivalent system `(-A) x = -b`, so `x` is still the solution of `A x = b`.
- `linear_solve(..., preconditioner=...)` accepts an `AbstractPreconditioner`, a lineax operator or a callable, and attaches it to the CG solve only.

## Goes wrong

- A `(matvec, shape)` pair carries no tags, so `linear_solve` raises `ValueError` for it unless you pass `solver=`; build it with `as_linear_operator` and tags instead.
- A lineax solver, wrapped in `LineaxSolver`, is solve-only, so it cannot be used where a distribution also needs a `logdet`.
- Passing anything that is neither a strategy nor a lineax solver raises `TypeError`.
- `linear_solve(..., preconditioner=...)` raises `ValueError` for a strategy with no CG solve inside, and for `PreconditionedCGSolver`, which owns its preconditioner.
- `as_linear_operator` raises `ValueError` when given neither `shape` nor `in_structure`.

## Next

- The strategies you can pass: [solvers](solvers.md), [logdet](logdet.md).
- The preconditioners `linear_solve` accepts: [preconditioners](preconditioners.md).
- Strategies that need a sketch: [randomized](randomized.md).
