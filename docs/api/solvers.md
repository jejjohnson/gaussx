# Solvers & Preconditioners

Layer 1.5: strategy objects that encapsulate *how* a solve or logdet is
computed, decoupled from *what* is being solved. Everything that accepts a
`solver=` keyword anywhere in gaussx takes one of these; `None` falls back to
structural dispatch on the operator.

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
      members: [DenseSolver, AutoSolver, CGSolver, PreconditionedCGSolver, MINRESSolver, LSMRSolver, BBMMSolver, ComposedSolver, KeyedSolver, SparseCholeskySolver]

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

## Logdet strategies

Dense eigendecomposition for exactness; stochastic Lanczos quadrature (SLQ) for
$O(n^2 \cdot \text{rank})$ estimates on large PSD (or symmetric-indefinite)
operators.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [DenseLogdet, SLQLogdet, IndefiniteSLQLogdet]

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
