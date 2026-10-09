# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the
single source of truth for every coding agent working here (the layer map,
what gaussx is built on, "reuse before you write", the three contracts, the
tests that enforce them, commands, the pre-commit checklist, git and PR
rules).

The essentials, in case you only read this file:

- One package, `src/gaussx/`: private subpackages by layer (`_primitives`,
  `_operators`, `_strategies`, `_distributions`, `_gp`, `_ssm`, …), public
  API in `gaussx.__all__`. Search [`docs/capabilities.md`](../docs/capabilities.md)
  before writing a helper.
- Keep the three contracts in `AGENTS.md`:
  - **operators** are `lineax.AbstractLinearOperator`s with their lineax and
    gaussx predicates registered, an `isinstance` fast path per primitive
    that never densifies, a row in the dispatch table and a case in the
    conformance zoo;
  - **solver strategies** keep grad / vmap / float32 / `max_steps` behaviour;
  - **JAX numerics**: pure functions, explicit PRNG keys, the input dtype
    preserved, no Python control flow on traced values, `eqx.Module` (never
    dataclasses), and the einx convention.
- Before committing, from the repo root: `make test-fast`, `make test-no-x64`,
  `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`,
  `make typecheck`; `make capabilities` after a public API change.
- Step-by-step recipes (add an operator, a fast path, a strategy, a recipe;
  deprecate; pre-PR check; review) are plain Markdown in
  `.claude/skills/<name>/SKILL.md`; the "Recipes" table in `AGENTS.md` lists
  them. Follow the matching one.
- Path-scoped standards live in `.github/instructions/`; code review follows
  [`CODE_REVIEW.md`](../CODE_REVIEW.md).
