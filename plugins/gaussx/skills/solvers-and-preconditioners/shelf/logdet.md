# Log-determinant strategies

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: strategies.

Open this when `log|A|` of a large operator is the expensive part, for example in a Gaussian log-likelihood, and you want to choose how it is computed.
The verb `logdet` and the operators live in the skill `gaussx:structured-gaussian-linalg`.

## Names

Each of these has `logdet(operator, *, key=None)` and no `solve`; pair one with a solve strategy through `ComposedSolver`.

| Name | What it is |
|---|---|
| `AbstractLogdetStrategy` | The base class: a strategy with `logdet(operator, *, key=None)`. Deterministic strategies ignore `key`. |
| `DenseLogdet()` | Exact, through `gaussx.logdet`, so structured operators keep their cheap exact rule. |
| `SLQLogdet(num_probes=20, lanczos_order=30, seed=0, sampler="signs", preconditioner=None)` | Stochastic Lanczos quadrature for a PSD operator; with a `preconditioner` that knows `P` it adds the exact `log det P` and runs SLQ on the whitened, near-identity operator. |
| `IndefiniteSLQLogdet(num_probes=20, lanczos_order=30, shift=0.0, seed=0, sampler="signs")` | Stochastic estimate of `log|det(A + shift I)|` for a symmetric, possibly indefinite operator. |
| `NystromLogdet(shift, rank=50, oversample=0, num_probes=10, lanczos_order=30, seed=0, sampler="signs")` | Nyström-preconditioned logdet of `A + μ I`, with the PSD part sketched and the remainder estimated by SLQ. |

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
A = lx.MatrixLinearOperator(K + 0.1 * jnp.eye(n), PSD)  # K + mu I, mu = 0.1
ref = jnp.linalg.slogdet(A.as_matrix())[1]

assert jnp.allclose(gaussx.DenseLogdet().logdet(A), ref, rtol=1e-4)

slq = gaussx.SLQLogdet(num_probes=50, lanczos_order=40)
est = slq.logdet(A, key=jr.key(0))  # a fresh key gives fresh probes
assert jnp.allclose(est, ref, atol=0.2 * jnp.abs(ref))

nys = gaussx.NystromLogdet(shift=0.1, rank=30)  # shift is the mu inside A
assert jnp.allclose(nys.logdet(A, key=jr.key(1)), ref, atol=0.2 * jnp.abs(ref))
```

## Rules

- A logdet strategy holds no solve; to use one in a distribution or with `solve`, put it in `ComposedSolver(solve_strategy, logdet_strategy)`.
- `SLQLogdet` and `IndefiniteSLQLogdet` are stochastic: with no `key` they draw probes from `PRNGKey(seed)`, so every call sees the same probes.
- Fixed probes suit stochastic-gradient training, but averaging such estimates reduces no variance and MCMC then targets a fixed pseudo-likelihood.
- To vary the probes, pass `key=` to `logdet` or bind one with `gaussx.KeyedSolver`, see [solvers](solvers.md).
- Use `SLQLogdet` for PSD operators and `IndefiniteSLQLogdet` when the operator may be indefinite or negative-definite.
- `NystromLogdet` takes the full system `A + μ I` as its operator, and `shift` must be the `μ` that the operator contains.
- `NystromLogdet.shift` is a PyTree leaf, so it can be a traced, learned noise variance.
- Choose `rank` for `NystromLogdet` at least as large as the effective dimension of `A` at `μ`; the preconditioned remainder then clusters near one and SLQ needs few steps.

## Goes wrong

- `NystromLogdet` is for covariance form `K + σ² I`. For a precision-form GMRF system the hard directions are the smallest eigenvalues, so use `SLQLogdet` or an exact sparse factorisation there.
- A `NystromLogdet.shift` that does not match the `μ` inside the operator gives a wrong estimate, because the sketched part is `operator − shift·I`.
- A stochastic estimate is not exact, so use `DenseLogdet` where the value must be exact.

## Next

- Pair a logdet with a solve, and the strategies that also solve: [solvers](solvers.md).
- The Nyström factorisation inside `NystromLogdet`: [randomized](randomized.md).
- Handing a strategy down through your own library: [pass-through](pass-through.md).
