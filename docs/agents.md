# Building with agents

gaussx exists so that code which touches a covariance matrix keeps its
structure. Coding agents tend to re-implement what they cannot see — a
`cho_solve` here, a hand-written Woodbury step there — and each one quietly
turns an O(n₁³ + n₂³) problem back into an O(N³) one. gaussx ships three
things that let them find it.

## The capability index

The [capability index](capabilities.md) lists every public gaussx name,
grouped like the API reference, with a one-line summary; then the public API
of lineax, matfree and optimistix, which gaussx builds on. It is regenerated
from the code and checked in the test suite, so it never drifts. Point an
agent at it before it writes a solve, a factorisation or a Gaussian helper.

## The Claude Code plugin

The repository is a Claude Code plugin marketplace. In any project:

```text
/plugin marketplace add jejjohnson/gaussx
/plugin install gaussx@gaussx
```

The plugin adds:

- **`structured-gaussian-linalg`** (skill) — loads whenever a task solves a
  linear system, takes a log-determinant, factorises or samples from a
  covariance, or builds a GP / state-space / Bayesian model in JAX: which
  layer to enter, the rules (operators not arrays, PSD tags, `solver=`, JAX
  purity), a worked example, and a "don't write it — use gaussx" table.
- **`gaussx-reuse-reviewer`** (subagent) — a read-only check of a diff for
  dense linear algebra and Gaussian code that gaussx already provides.

## llms.txt

For other agents and tools, the docs site serves
[`llms.txt`](https://jejjohnson.github.io/gaussx/llms.txt): a curated map of
gaussx and its key pages.

## Rules for your project's `AGENTS.md`

Paste this into the agent instructions of a project that builds on gaussx:

```markdown
## Linear algebra and Gaussians: build on gaussx

This project uses gaussx for covariance / precision algebra in JAX. Before
writing a solve, log-determinant, Cholesky, sampler, Gaussian density, KL,
Kalman step, quadrature rule or iterative solver, search the capability
index (https://jejjohnson.github.io/gaussx/capabilities/) or
`gaussx.__all__`, and compose what exists:

- Covariances and precisions are lineax operators:
  `lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)` for a dense PSD
  matrix, and the structured operator when there is structure
  (`gaussx.Kronecker`, `KroneckerSum`, `SumOfKroneckers`, `BlockDiag`,
  `BlockTriDiag`, `low_rank_plus_diag` / `LowRankUpdate`, `Toeplitz`,
  `SparseOperator`).
- Use `gaussx.solve` / `logdet` / `cholesky` / `sqrt` / `inv` on them, never
  `jnp.linalg` on `.as_matrix()`; heed `DenseFallbackWarning`.
- Gaussian densities, KLs, conditioning and sampling: `gaussian_log_prob`,
  `gaussian_kl`, `conditional`, `sample_mvn`, `MultivariateNormal`.
- Pick numerics with `solver=` (`CGSolver()`, `BBMMSolver()`, `SLQLogdet()`)
  instead of rewriting the math; pass PRNG keys explicitly.
- Check new code against a dense reference on a small case, under `jax.jit`
  and `jax.grad`.
```

## Working on gaussx itself

Contributors (and their agents) follow
[`AGENTS.md`](https://github.com/jejjohnson/gaussx/blob/main/AGENTS.md) in the
repository: the layer map, what gaussx is built on, "reuse before you write",
the three contracts and the tests that enforce them, and recipe skills for
adding operators, fast paths, solver strategies and recipes.
