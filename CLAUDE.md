# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GaussX: Structured linear algebra, Gaussian distributions, and exponential family primitives for JAX. Built on top of lineax, equinox, and matfree.

## Architecture

### Four-layer stack

| Layer | Name | Contents |
|-------|------|----------|
| 0 | Primitives | Pure functions: `solve`, `logdet`, `cholesky`, `diag`, `trace`, `sqrt`, `inv` |
| 1 | Operators | `Kronecker`, `BlockDiag`, `LowRankUpdate` (extend `lineax.AbstractLinearOperator`) |
| 1.5 | Strategies | `DenseSolver`, `CGSolver` (encapsulate solve + logdet) |
| 2 | Distributions | `MultivariateNormal` + sugar ops (v0.2+) |
| 3 | Recipes | Kalman filter, ensemble covariance, natural gradients (v0.3+) |

### Package structure

All implementation lives in `src/gaussx/`. The public API is re-exported through `src/gaussx/__init__.py`.

### Key directories

| Path | Purpose |
|------|---------|
| `src/gaussx/` | Main package source code |
| `src/gaussx/_primitives/` | Layer 0 — pure functions with structural dispatch |
| `src/gaussx/_operators/` | Layer 1 — lineax operator extensions |
| `src/gaussx/_strategies/` | Layer 1.5 — solver strategy objects |
| `src/gaussx/_tags.py` | Structural tags for dispatch |
| `src/gaussx/_testing.py` | Test utilities (random PD matrices, assertions) |
| `tests/` | Test suite |
| `docs/` | Documentation (MkDocs) |
| `notebooks/` | Jupyter notebooks |

### Key dependencies

| Package | Role |
|---------|------|
| `jax` / `jaxlib` | Array backend |
| `equinox` | Module system, PyTrees |
| `lineax` | Linear operators, solvers |
| `matfree` | Krylov methods, stochastic trace |
| `jaxtyping` | Array type annotations |
| `einops` | Tensor reshaping (D9: einops for all reshape/einsum) |

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Run tests: uv run pytest -v
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/gaussx
make precommit            # Run pre-commit on all files
make docs-serve           # Local docs server
```

### Running a single test

```bash
uv run pytest tests/test_example.py::TestClass::test_method -v
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -v                              # Tests
uv run --group lint ruff check .              # Lint — ENTIRE repo, not just src/gaussx/
uv run --group lint ruff format --check .     # Format — ENTIRE repo
uv run --group typecheck ty check src/gaussx  # Typecheck — package only
```

**Critical**: Always lint/format with `.` (repo root), not `src/gaussx/`. CI runs `ruff check .` which includes `tests/` and `scripts/`.

## Coding Conventions

- All operators are `equinox.Module` subclasses (immutable, PyTree-compatible)
- All primitives are pure functions with isinstance-based dispatch
- Use `jaxtyping` annotations for array shapes
- Use `einops` for all tensor reshaping — no raw `jnp.reshape`/`jnp.transpose`/`jnp.einsum`
- Google-style docstrings
- Type hints on all public functions and methods
- Pure functions where possible; side effects isolated and explicit
- Surgical changes only — don't refactor adjacent code or add docstrings to unchanged code

## Documentation Examples

Example notebooks live in `docs/notebooks/` as jupytext percent-format `.py` files. The workflow:

1. Write the `.py` source (jupytext percent format)
2. Execute locally via `jupytext --to notebook --execute foo.py -o foo.ipynb`
3. Commit both the `.py` source and the executed `.ipynb` (which contains inline figure outputs)
4. `mkdocs-jupyter` renders the pre-executed `.ipynb` with `execute: false`

Figures render inline via `plt.show()` — do **not** use `savefig` or commit separate PNG files. The `.ipynb` cell outputs are the single source of rendered figures.

See `.github/instructions/docs-examples.instructions.md` for full standards.

## Plans

Plans and design documents go in `.plans/` (gitignored, never committed). Track work via GitHub issues instead.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. To obtain the required `threadId`, first list the pull request's review threads via the GitHub GraphQL API (see the "Pull Request Review Comments" section in `AGENTS.md` for a minimal query and end-to-end workflow).

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
