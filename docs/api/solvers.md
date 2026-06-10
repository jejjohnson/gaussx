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
      members: [DenseSolver, AutoSolver, CGSolver, PreconditionedCGSolver, MINRESSolver, LSMRSolver, BBMMSolver, ComposedSolver]

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

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [AbstractPreconditioner, JacobiPreconditioner, OperatorPreconditioner, NystromPreconditioner, PartialCholeskyPreconditioner]
