---
name: solvers-and-preconditioners
description: Choose how a linear solve or a log-determinant is computed in JAX with gaussx — AutoSolver, dense, conjugate gradients, preconditioned CG, BBMM, MINRES, LSMR, sparse Cholesky; stochastic Lanczos and Nyström log-determinants; Jacobi, Nyström and partial-Cholesky preconditioners; randomized SVD, Nyström and sketches; linear_solve for a bare matvec. Use when a solve does not converge or is too slow, when a system is too large to factorise, when tuning a preconditioner, or when a library built on gaussx needs to accept and pass through a `solver=` argument.
---

# Solvers and preconditioners with gaussx

A gaussx verb computes exactly, by the structure of its operator.
A strategy changes how: iteratively, stochastically, or with a factorisation you choose.
You change the numerics by passing a strategy, never by rewriting the mathematics.

This skill covers rung ② of gaussx, the strategies, in the vocabulary of the `role-ladder` skill.
The operators and the verbs are in `gaussx:structured-gaussian-linalg`.

## The contract

- A strategy is an object with `solve(operator, vector)` and `logdet(operator, *, key=None)`.
- `gaussx.solve(op, b, solver=strategy)` and the functions above it (`gaussian_log_prob`, `kalman_filter`, `build_prediction_cache`, …) take one as `solver=`.
- `solver=None` means structural dispatch on the operator. It is not a default strategy.
- `gaussx.logdet(op)` takes no `solver=`; call `strategy.logdet(op)` for a strategy's log-determinant.
- A stochastic log-determinant takes a `key=`. Without one it draws the same probes on every call.

## Which strategy

| Situation | Choose |
|---|---|
| A structured operator, or a dense one up to about a thousand rows | `None`, or `gaussx.DenseSolver()` |
| You do not want to choose | `gaussx.AutoSolver()`: exact when the structure allows or the size is under `size_threshold`, CG for a large PSD operator |
| A large PSD system | `gaussx.CGSolver()` |
| A large `K + σ² I` kernel system | `gaussx.PreconditionedCGSolver()`, or `gaussx.CGSolver(preconditioner=...)` |
| A solve and a log-determinant for GP training at scale | `gaussx.BBMMSolver()` |
| Symmetric but indefinite or singular | `gaussx.MINRESSolver()` |
| Rectangular or ridge-damped least squares | `gaussx.LSMRSolver()` |
| A sparse precision as a `SparseOperator` | `gaussx.SparseCholeskySolver()`, passed explicitly |
| An iterative solve with an exact log-determinant | `gaussx.ComposedSolver(gaussx.CGSolver(), gaussx.DenseLogdet())` |
| A bare `v ↦ A v` | `gaussx.linear_solve((matvec, shape), b, solver=...)` |

## Worked example

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import gaussx

PSD = lx.positive_semidefinite_tag
n = 100
t = jnp.linspace(0.0, 1.0, n)
K = jnp.exp(-0.5 * ((t[:, None] - t[None, :]) / 0.2) ** 2)  # (n, n)
A = lx.MatrixLinearOperator(K + jnp.eye(n), PSD)  # K + σ² I, σ² = 1
b = jr.normal(jr.key(0), (n,))
reference = jnp.linalg.solve(A.as_matrix(), b)

x = gaussx.solve(A, b, solver=gaussx.CGSolver())  # the adverb on the verb
assert jnp.allclose(x, reference, atol=1e-2)


def fit(operator, y, *, solver: gaussx.AbstractSolverStrategy | None = None):
    """A function above gaussx: it takes the strategy and hands it down."""
    return gaussx.gaussian_log_prob(jnp.zeros_like(y), operator, y, solver=solver)


exact = fit(A, b)
iterative = fit(
    A, b, solver=gaussx.ComposedSolver(gaussx.CGSolver(), gaussx.DenseLogdet())
)
assert jnp.allclose(exact, iterative, rtol=1e-2)
```

## Don't write it

| Don't write | Use |
|---|---|
| A CG loop, a Lanczos loop, Hutchinson probes | `solver=gaussx.CGSolver()` or `BBMMSolver()`, `gaussx.SLQLogdet`, `gaussx.trace_and_diag` |

## The shelf

Each entry is one file with the names it covers, a worked example, the rules and what goes wrong.
Open the one the task needs; do not read them all.

<!-- shelf:begin -->
| Role | Entry | Covers | Names |
|---|---|---|---|
| ② strategy | [Choosing a solver](shelf/solvers.md) | dense, CG, preconditioned CG, BBMM, MINRES, LSMR, sparse Cholesky, and composing them | 13 |
| ② strategy | [Multigrid](shelf/multigrid.md) | V-cycles over a caller-built hierarchy, as a CG preconditioner or a solve strategy | 6 |
| ② strategy | [Log-determinant strategies](shelf/logdet.md) | dense, stochastic Lanczos quadrature, Nyström | 5 |
| ② strategy | [Preconditioners](shelf/preconditioners.md) | Jacobi, Nyström, partial Cholesky, and wrapping your own | 5 |
| ② strategy | [Passing a strategy through](shelf/pass-through.md) | how a library above gaussx exposes `solver=`; `linear_solve` for a bare matvec | 4 |
| setup | [Randomized factorisations](shelf/randomized.md) | randomized SVD and eigh, Nyström, RP-Cholesky, CUR, sketches | 17 |
<!-- shelf:end -->

## Before you finish

- No solver is fixed inside a function that could take one as `solver=`.
- Every stochastic log-determinant that is averaged or differentiated receives a `key`.
- An iterative solve was checked against a dense reference on a small case, with a tolerance that suits float32.

<!-- shelf:below:begin -->
## The units below this skill

A card is enough to call a unit.
Load its skill only when the card's last line applies.

### Structures and primitives · ⓪ ① · skill `gaussx:structured-gaussian-linalg`

- **Provides.** Covariance and precision operators that keep their structure, and the verbs that use it.
- **Nouns.** `gaussx.Kronecker`, `gaussx.SumOfKroneckers`, `gaussx.BlockDiag`, `gaussx.BlockTriDiag`, `gaussx.Toeplitz`, `gaussx.SparseOperator`, `gaussx.low_rank_plus_diag`; any `lineax.AbstractLinearOperator`, tagged.
- **Verbs.** `gaussx.solve`, `gaussx.logdet`, `gaussx.cholesky`, `gaussx.sqrt`, `gaussx.inv`, `gaussx.diag`, `gaussx.trace`; `gaussx.solve_matrix`, `gaussx.woodbury_solve`, `gaussx.safe_cholesky`.
- **Adverbs.** `solver=` on `gaussx.solve`; left out, the verb dispatches on the structure.
- **Keep.** A dense matrix enters as `lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)`; factors are operators, not arrays.
- **Never.** Call `as_matrix()` and then a dense `jnp.linalg` solve, or build `jnp.kron(A, B)`.
- **One rung down.** Open `gaussx:structured-gaussian-linalg` to build a new structure or to see which verb has a fast path for which noun.
<!-- shelf:below:end -->

<!-- skill-tree:begin · generated by geostack scripts/skill_tree.py; do not edit by hand -->
## Builds on

Nothing else in the stack: this is a base skill.

Used by (pointers, not skills to load): argminax, filterax, finitevolX, gauss_flows, integrix, kernellib, manipy, plumax, pyrox, radtrax, spectraldiffx, vardax.

<!-- skill-tree:end -->
