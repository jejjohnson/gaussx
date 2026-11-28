# Preconditioners

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: strategies.

Open this when CG needs many iterations on a large system, typically `K + σ² I`, and you want to attach an approximate inverse.
The verbs `solve` and `logdet` and the operators live in the skill `gaussx:structured-gaussian-linalg`.

## Names

A preconditioner is an `equinox.Module` whose `as_operator(operator=None)` returns a PSD lineax operator that applies `M⁻¹`, or `None` for no preconditioning.

| Name | What it is |
|---|---|
| `AbstractPreconditioner` | The base class; subclass it and implement `as_operator`. Calling an instance applies `M⁻¹` to a vector. |
| `JacobiPreconditioner(diagonal=None)` | `diag(1 / abs(diag(A)))`, the cheapest choice; read from the operator when `diagonal` is not given. |
| `NystromPreconditioner(basis, eigenvalues, shift)` | Randomized Nyström preconditioner for `A + μ I`; build it with `NystromPreconditioner.from_operator(K, rank, *, shift, key=...)`. |
| `PartialCholeskyPreconditioner(rank=50, shift=1.0, pivoting="greedy", ...)` | Pivoted partial Cholesky of the PSD part, applied by the Woodbury identity; build it once with `from_operator(K, rank, *, shift, ...)`. |
| `FITCPreconditioner(U, prior_residual, diagonal_shift=None)` | Nyström plus its diagonal correction, `P = U Uᵀ + diag(r + λ)`, for `K_XX + Λ`; build it with `FITCPreconditioner.from_kernel(kernel, X, Z)`. With `diagonal_shift=None` it reads `λ` from each system, so one object serves every Laplace step. |
| `OperatorPreconditioner(approx_inverse, in_structure=None)` | Wraps an approximate inverse you already have, a lineax operator or a callable `v -> M⁻¹ v`. |

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
mu = 0.5
K_op = lx.MatrixLinearOperator(K, PSD)  # the PSD part, without the noise
A = lx.MatrixLinearOperator(K + mu * jnp.eye(n), PSD)  # K + mu I
b = jr.normal(jr.key(0), (n,))
ref = jnp.linalg.solve(A.as_matrix(), b)

chol = gaussx.PartialCholeskyPreconditioner.from_operator(K_op, rank=20, shift=mu)
nys = gaussx.NystromPreconditioner.from_operator(K_op, rank=20, shift=mu, key=jr.key(1))
jac = gaussx.JacobiPreconditioner()
own = gaussx.OperatorPreconditioner(lambda v: v / jnp.diag(A.as_matrix()))
for P in (chol, nys, jac, own):
    x = gaussx.CGSolver(preconditioner=P).solve(A, b)
    assert jnp.allclose(x, ref, atol=1e-2)

x = gaussx.PreconditionedCGSolver(preconditioner=chol).solve(A, b)
assert jnp.allclose(x, ref, atol=1e-2)
```

## Rules

- Build `PartialCholeskyPreconditioner` and `NystromPreconditioner` once with `from_operator` on the PSD part `K` and an explicit `shift=σ²`, then reuse them across solves.
- Pass `K`, not `K + σ² I`, to `from_operator`, or the noise is counted twice.
- Attach one with `CGSolver(preconditioner=P)`, `AutoSolver(preconditioner=P)` or `PreconditionedCGSolver(preconditioner=P)`.
- Through `linear_solve(..., preconditioner=...)` a preconditioner can also be a lineax operator or a callable, which is wrapped in `OperatorPreconditioner`.
- `PartialCholeskyPreconditioner(rank, shift)` built directly, without `from_operator`, factors `A − shift·I` from the system operator at every solve, and `shift` must not exceed the noise actually in `A`.
- Choose the Nyström `rank` of at least about `2⌈1.5 d_eff(μ)⌉ + 1`, where `d_eff(μ) = tr(A (A + μ I)⁻¹)`, for a condition number below 28 in expectation.
- Use `OperatorPreconditioner` to bring in an approximate inverse built elsewhere, such as a spectral solve or a multigrid cycle.

## Goes wrong

- Nyström and partial Cholesky capture the top of the spectrum, so they suit `K + σ² I` and not a precision-form GMRF system `Q + AᵀWA`; use Jacobi-preconditioned CG or an exact sparse factorisation there.
- `JacobiPreconditioner` without `diagonal` reads it from the operator at every solve, which for a matrix-free operator costs one matvec per row.
- `PreconditionedCGSolver` already owns a preconditioner, so `linear_solve(..., preconditioner=...)` with it raises `ValueError`.
- `from_operator` raises `ValueError` for `rank < 1`, and `NystromPreconditioner.from_operator` also for a non-positive concrete `shift`.
- A preconditioner changes the CG iteration count, not the solution CG converges to, so judge one by iterations and not by accuracy.

## Next

- The solvers that take a preconditioner: [solvers](solvers.md).
- The factorisations inside them: [randomized](randomized.md).
- Passing one through your own library: [pass-through](pass-through.md).
