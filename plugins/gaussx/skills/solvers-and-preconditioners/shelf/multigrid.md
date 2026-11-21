# Multigrid

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: strategies.

Open this when an elliptic or lattice system (a Poisson or Helmholtz operator, a GMRF precision) needs CG in tens of steps instead of hundreds, and you can describe a hierarchy of coarser levels.
gaussx owns the V-cycle; the caller builds the levels (a grid library coarsens its mask and writes the transfer stencils).

## Names

| Name | What it is |
|---|---|
| `MultigridLevel(operator, prolongation, restriction, smoother)` | One level: A_ℓ, P_ℓ (from the next coarser level), R_ℓ (`None` means Pᵀ) and a smoother. |
| `galerkin_levels(operator, prolongations, *, restriction_scale=1.0, smoother=None)` | A hierarchy with dense Galerkin coarse operators A_ℓ₊₁ = c P_ℓᵀ A_ℓ P_ℓ. |
| `WeightedJacobi(inv_diagonal, omega=2/3, sweeps=1)` | u ← u + ω D⁻¹ (b − A u); `from_operator` estimates ω from λ_max(D⁻¹ A). |
| `AbstractSmoother` | The smoother protocol, `smoother(operator, u, b) -> u`. |
| `MultigridPreconditioner(levels, n_pre=2, n_post=2, bottom_solve=None, null_space=None)` | One V-cycle as M⁻¹, for `CGSolver(preconditioner=...)`; negated for a negative-definite A. |
| `MultigridSolver(levels, ..., rtol, atol, max_steps, throw)` | CG on the given operator preconditioned by a V-cycle; `cycles(b, n)` runs plain V-cycles. |

## Use it

```python
import jax.numpy as jnp
import lineax as lx

import gaussx

# a 1-D Poisson problem on 63 points, two coarser levels of 31 and 15 points
n = 63
L = 2 * jnp.eye(n) - jnp.eye(n, k=1) - jnp.eye(n, k=-1)
A = lx.MatrixLinearOperator(L, lx.positive_semidefinite_tag)


def interpolation(n_coarse):
    """Linear interpolation from n_coarse to 2 n_coarse + 1 nested points."""
    p = jnp.zeros((2 * n_coarse + 1, n_coarse))
    j = jnp.arange(n_coarse)
    return p.at[2 * j + 1, j].set(1.0).at[2 * j, j].add(0.5).at[2 * j + 2, j].add(0.5)


transfers = [lx.MatrixLinearOperator(interpolation(m)) for m in (31, 15)]
levels = gaussx.galerkin_levels(A, transfers, restriction_scale=0.5)
solver = gaussx.MultigridSolver(levels, rtol=1e-8, atol=1e-8)
x = solver.solve(A, jnp.ones(n))
print(float(jnp.max(jnp.abs(A.mv(x) - 1.0))) < 1e-6)
```
