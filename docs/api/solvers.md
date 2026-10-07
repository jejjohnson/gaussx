# Solvers & Preconditioners

Layer 1.5: strategy objects that encapsulate *how* a solve or logdet is
computed, decoupled from *what* is being solved. Everything that accepts a
`solver=` keyword anywhere in gaussx takes one of these; `None` falls back to
structural dispatch on the operator.

The one exception is the primitive `gaussx.solve(A, b, solver=...)`, whose
`solver=` historically takes a *lineax* solver for its dense fallback (and per
structured factor). Both kinds are accepted in both places: `gaussx.solve`
hands the whole solve to a gaussx strategy, and `linear_solve` wraps a lineax
solver in [`LineaxSolver`](#gaussx.LineaxSolver) (solve only: a distribution
also needs a `logdet`). Anything else raises `TypeError`.

A strategy bundles a `solve` and a `logdet` algorithm. Mix and match with
[`ComposedSolver`](#gaussx.ComposedSolver) — e.g. CG for the solve, stochastic
Lanczos quadrature for the logdet — which is the standard recipe for large
kernel matrices.

## The front door

`linear_solve` is the high-level entry point: it accepts a lineax operator *or*
a bare `(matvec, shape)` pair, normalises negative-definite systems, picks a
sensible iterative solver from the operator's tags, and threads a
preconditioner through. `as_linear_operator` wraps a raw matvec callable into a
tagged `FunctionLinearOperator` for matrix-free workflows.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [linear_solve, as_linear_operator]

## Abstract interfaces

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [AbstractSolveStrategy, AbstractLogdetStrategy, AbstractSolverStrategy]

## Direct & iterative solvers

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [DenseSolver, AutoSolver, CGSolver, PreconditionedCGSolver, MINRESSolver, LSMRSolver, BBMMSolver, ComposedSolver, KeyedSolver, LineaxSolver, SparseCholeskySolver]

### Tolerances and float32

The iterative strategies (`CGSolver`, `PreconditionedCGSolver`, `BBMMSolver`,
`MINRESSolver`, `LSMRSolver`) leave their tolerances unset by default and
resolve them from the operator's dtype at solve time: the historical
constants in float64 (`1e-5`; `1e-4` for BBMM; `1e-6` for LSMR) and `1e-3`
in float32. A float32 CG solve cannot reach a relative residual of `1e-5` once
the condition number passes about $10^3$. It runs out of steps, or breaks
down, and returns an iterate worse than zero. Set `rtol`/`atol` explicitly to
override.

`CGSolver`, `PreconditionedCGSolver`, `BBMMSolver` and `AutoSolver` raise when
CG exhausts its step budget. With `throw=False` they return the last iterate
unchecked instead. Use that only where the caller checks the result, because
an unconverged CG iterate can be far from the solution.

### Probe keys and common random numbers

A stochastic log-determinant (SLQ, used by `CGSolver`, `PreconditionedCGSolver`,
`BBMMSolver`, `MINRESSolver`, `LSMRSolver` and a large PSD `AutoSolver`) called
without a `key` draws its probes from `PRNGKey(seed)`, so every call sees the
same probes. These are common random numbers: right for stochastic-gradient
training, where the objective becomes a fixed smooth function of the
parameters, but averaging such estimates reduces no variance, and MCMC then
targets a fixed pseudo-likelihood. Pass `key=` to `gaussian_log_prob`,
`gaussian_entropy` or `kl_standard_normal`. Distribution methods cannot take
one, so wrap their strategy in [`KeyedSolver`](#gaussx.KeyedSolver), whose key
is a PyTree leaf, or change `seed`.

## Sketch-and-precondition least squares

`SketchAndPrecondLSMR` solves tall (optionally ridge-damped) least squares
$\min_x \|Ax - b\|^2 + \delta^2\|x\|^2$, $m \gg n$, in a number of LSMR
steps that does not grow with $m$ or $\kappa(A)$. A sketch
$S \in \mathbb{R}^{d \times m}$ ([Sketching](sketching.md)) of $A$ gives
$[SA;\ \delta I] = QR$, and $M = R^{-1}$ is a right preconditioner with

$$
\kappa(\tilde A M) \le \frac{1+\varepsilon}{1-\varepsilon},
\qquad \tilde A = [A;\ \delta I],
$$

when $S$ is an $\varepsilon$-embedding for $\operatorname{range}(A)$. The
sketch-and-solve estimate $x_0 = R^{-1}Q_{1:d}^\top Sb$ is the warm start;
`sketch_and_solve` returns it on its own (an $O(\varepsilon)$ approximation).
The dense QR costs $O(dn^2)$, so this targets $n \lesssim 10^4$; for larger
$n$, solve the normal equations with `PreconditionedCGSolver` and a
`NystromPreconditioner` on $A^\top A$ with `shift=δ²`.

```python
# Gauss–Newton step on a (10⁶ pixels × 300 parameters) Jacobian J_op
step = gx.SketchAndPrecondLSMR(damp=1e-3).solve(J_op, -residuals)
```

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [SketchAndPrecondLSMR, sketch_and_solve]

## Logdet strategies

Dense eigendecomposition for exactness; stochastic Lanczos quadrature (SLQ) for
$O(n^2 \cdot \text{rank})$ estimates on large PSD (or symmetric-indefinite)
operators.

`NystromLogdet` (Wenger et al., 2022) is the variance-reduced SLQ for a
covariance-form $K + \sigma^2 I$. The log-determinant of the Nyström
preconditioner $P = \hat K + \sigma^2 I$ is exact, and SLQ only estimates
$\log|P^{-1/2}(K + \sigma^2 I)P^{-1/2}|$, whose operator is close to the
identity, so the probe variance collapses. Use plain `SLQLogdet` for
precision-form GMRF systems.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [DenseLogdet, SLQLogdet, IndefiniteSLQLogdet, NystromLogdet]

## Preconditioners

Approximate inverses $M^{-1} \approx A^{-1}$ that accelerate the iterative
solvers above. Pass them via the `preconditioner=` argument of
[`linear_solve`](#gaussx.linear_solve), [`CGSolver`](#gaussx.CGSolver), or
[`PreconditionedCGSolver`](#gaussx.PreconditionedCGSolver).

For a covariance-form system $K + \sigma^2 I$, build
[`NystromPreconditioner`](#gaussx.NystromPreconditioner) or
[`PartialCholeskyPreconditioner`](#gaussx.PartialCholeskyPreconditioner) once
with `from_operator(K, rank, shift=σ²)`: the operator is the PSD part $K$ only
and the noise is passed separately, so it is never counted twice. With a
Nyström rank $\ell \gtrsim 2\lceil 1.5\,d_{\text{eff}}(\sigma^2)\rceil + 1$,
$d_{\text{eff}}(\mu) = \operatorname{tr}\big(K(K + \mu I)^{-1}\big)$, CG
needs a number of iterations that does not grow with $n$ (Frangella, Tropp &
Udell, 2023).

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [AbstractPreconditioner, JacobiPreconditioner, OperatorPreconditioner, NystromPreconditioner, PartialCholeskyPreconditioner]
