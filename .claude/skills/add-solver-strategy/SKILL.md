---
name: add-solver-strategy
description: Add a solver strategy (how a solve or log-determinant is computed — CG variants, Krylov, stochastic logdet, sketch-and-solve) or a preconditioner to gaussx (Layer 1.5), keeping the strategy contract (grad, vmap, float32, max_steps) and the preconditioner-efficacy test. Use when asked to add, port or wrap an iterative solver, logdet estimator or preconditioner in src/gaussx/_strategies or src/gaussx/_preconditioners.
---

# Add a solver strategy or preconditioner

Read contract 2 ("Solver strategies") in `AGENTS.md`. A strategy changes
*how* a solve or logdet is computed, never *what*: any code that takes
`solver=` must give the same answer, to tolerance, with every strategy.

## 1. Make sure it does not exist yet

- `docs/capabilities.md`: the gaussx strategies (`DenseSolver`,
  `CGSolver`, `PreconditionedCGSolver`, `MINRESSolver`, `LSMRSolver`,
  `BBMMSolver`, `ComposedSolver`, `KeyedSolver`, `AutoSolver`,
  `SLQLogdet`, `NystromLogdet`, …), lineax's solvers (`CG`, `GMRES`,
  `BiCGStab`, `NormalCG`, `LSMR`, …) and matfree (`decomp`, `funm`,
  `stochtrace`).
- A lineax solver needs no new class: `LineaxSolver` / `as_solve_strategy`
  wraps it, and `gaussx.solve(A, b, solver=lx.GMRES(...))` already threads
  it through the structural rules.
- Mixing an existing solve with an existing logdet is `ComposedSolver`, not
  a new class.

## 2. A strategy (`src/gaussx/_strategies/_<name>.py`)

- Subclass `AbstractSolveStrategy` (solve only), `AbstractLogdetStrategy`
  (logdet only, taking an optional `key`) or `AbstractSolverStrategy`
  (both), from `_strategies/_base.py`. It is an `eqx.Module`, and **every
  scalar option** (tolerances, step counts, probe counts, thresholds, seeds,
  sampler names) is `eqx.field(static=True)`, so `jax.jit` can take the
  strategy, or a distribution holding it, as an argument (gh-301). The
  exceptions are deliberate leaves, such as `KeyedSolver`'s key (gh-384)
  and `NystromLogdet`'s shift (gh-486). Use the
  canonical option names the other strategies use (`rtol`, `atol`,
  `max_steps`, …; gh-405).
- Tolerances default to `None` and resolve per dtype with
  `_strategies/_tolerances.py` (`resolve_tolerance`, `rhs_scaling`), so a
  default-constructed strategy works in float32.
- Build on lineax (`lx.linear_solve` with an `lx.CG` / … solver) or matfree
  (Lanczos, SLQ); don't write a new Krylov loop unless neither has it.
- Out of iterations is an error: `throw=True` by default (lineax) or
  `eqx.error_if`; never return an unconverged iterate silently.
- Stochastic estimators take an explicit `key`; with `key=None` they either
  derive a documented default or raise.
- Gradients: lineax's implicit adjoint usually gives them for free; a custom
  rule needs a test against the dense gradient.
- Export from `_strategies/__init__.py` and `src/gaussx/__init__.py`, list
  it on `docs/api/solvers.md`, `make capabilities`.

## 3. A preconditioner (`src/gaussx/_preconditioners/_<name>.py`)

- Subclass `AbstractPreconditioner` and implement `as_operator(operator)`,
  returning a lineax operator that applies `M⁻¹` (PSD-tagged when it is).
- Offer `from_operator(...)` when the factorisation can be built once and
  reused, and a lazy form when it is built per solve (see
  `PartialCholeskyPreconditioner` and `NystromPreconditioner`).
- It plugs into `CGSolver(preconditioner=...)` and `linear_solve(...,
  preconditioner=...)`; don't add a preconditioned copy of a strategy.

## 4. Tests

- Unit tests in `tests/strategies/test_<name>.py` against `DenseSolver` /
  `gaussx._testing.dense_solve` / `dense_logdet`, on a dense PSD operator and
  a structured one; stochastic logdets bounded by their own variance, with
  the bound's provenance in a comment.
- **The contract suite** (`tests/strategies/test_strategy_contract.py`): add
  a factory to `STRATEGIES` (iterative ones join `ITERATIVE` automatically,
  so the `max_steps` test covers them). grad vs dense, vmap over rhs and the
  float32 defaults then run for it, but `_cases` marks every strategy
  outside each test's short `fast` tuple as `slow`, so the new cases do
  **not** run in `make test-fast` or PR CI: run
  `uv run pytest tests/strategies/test_strategy_contract.py -k <name>` and
  add the `run-slow` label to the PR. A known failure is
  `xfail(strict=True)` in the matching `_*_XFAIL` dict, naming its issue.
- A preconditioner joins `test_preconditioner_halves_cg_steps_below_full_rank`
  (`_preconditioned_system`): at rank `k < n` it must at least halve the CG
  steps.
- `tests/strategies/test_pytree_config.py` finds exported strategies
  automatically and checks their options are static treedef data; add a
  `_default` entry only if the class cannot be built with no arguments
  (or a `_LEAF_CARRYING` entry for a deliberate leaf).
  `test_option_names.py` uses hard-coded parametrize lists for the canonical
  option names and their deprecated aliases; add the new strategy to them
  if it takes those options.

## 5. Verify

`uv run pytest -q tests/strategies tests/test_docs_api_coverage.py tests/test_capabilities.py`,
then the `pre-pr-check` skill.
