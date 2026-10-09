# AGENTS.md

Standing instructions for **every** coding agent working in this repository
(Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth:
`CLAUDE.md` and `.github/copilot-instructions.md` point here.

## What this repo is

`gaussx` is structured linear algebra, Gaussian distributions and
exponential-family primitives for JAX, built on lineax, equinox and matfree.
Its thesis: a covariance or precision matrix has structure (Kronecker, block,
low-rank, banded, Toeplitz, sparse), and every solve, log-determinant,
Cholesky and sample should exploit it instead of calling dense LAPACK.

pyrox (`pyrox-gp`, `pyrox-lgm`), finitevolX and spectraldiffx build on it, so
**gaussx is a library of primitives**: new code composes what is here, and
anything genuinely new lands where the next person will find and reuse it.

It is one package (`src/gaussx/`). Every subpackage is private; the public
API is `gaussx.__all__`, re-exported from `src/gaussx/__init__.py`. The
layers ([`docs/architecture.md`](docs/architecture.md) has the long form):

| Layer | Subpackages | Owns |
|---|---|---|
| 0 · Primitives | `_primitives/`, `_linalg/` | `solve`, `logdet`, `cholesky`, `diag`, `trace`, `sqrt`, `inv`, `eig`, `svd`, root decompositions; Woodbury, Schur, `safe_cholesky`, tridiagonal and Lyapunov solves |
| 1 · Operators | `_operators/`, `_tags.py`, `_sparse/`, `_gmrf/` | `Kronecker`, `BlockDiag`, `LowRankUpdate`, `KroneckerSum`, `SumOfKroneckers`, `BlockTriDiag`, `Toeplitz`, interpolated / masked operators, structural tags; `SparseOperator` + sparse Cholesky; GMRF precision builders |
| 1.5 · Strategies | `_strategies/`, `_preconditioners/`, `_solve_frontend.py` | *How* a solve / logdet is computed: `DenseSolver`, `CGSolver`, `BBMMSolver`, `MINRESSolver`, SLQ logdets, preconditioners, the `linear_solve` front door |
| 2 · Distributions | `_distributions/`, `_expfam/` | `MultivariateNormal(Precision)`, GMRFs, `LGSSM`; log-prob / entropy / KL sugar; natural ↔ expectation parameters |
| 3 · Recipes | `_gp/`, `_ssm/`, `_quadrature/`, `_inference/` | GP conditioning, ELBOs, LOVE; SDE kernels, Kalman / RTS (sequential, parallel, infinite-horizon); quadrature and moment matching; BLR, natural gradients, ensemble Kalman |
| Outside the stack | `_sketching/`, `_randomized/` | Subspace embeddings, randomized SVD / eigh / Nyström, RP-Cholesky |

Layers 2 and 3 import only from below. Layers 0, 1 and 1.5 form one
dispatch core with import cycles; an in-function import is allowed only on
an edge that closes one of them (see "What enforces them").

## What gaussx is built on

Each foundation brings a rule. Breaking one runs fine eagerly and fails
under `jit`, `grad` or `vmap`, or in someone else's pipeline.

| Library | gaussx uses it for | The rule it brings |
|---|---|---|
| **lineax** | `AbstractLinearOperator`, property tags, `is_*` predicates, solvers | Every structured matrix is a lineax operator. Register lineax's predicates and structure functions for it. Tag PSD operators (an untagged PSD matrix gets LU, a tagged one Cholesky). Don't re-wrap a lineax operator or solver that already does the job. |
| **equinox** | Immutable `eqx.Module` pytrees, `eqx.error_if`, `filter_*` transforms | Operators, strategies, states and results are `eqx.Module`s, never dataclasses (a dataclass is not a pytree). Hashable metadata is a static field. Runtime failures under `jit` go through `eqx.error_if`. |
| **matfree** | Lanczos, Arnoldi, SLQ, Hutchinson, matrix functions | Never hand-roll a Krylov decomposition or stochastic trace / diagonal estimator. |
| **optimistix** | Fixed points and roots with implicit-differentiation adjoints (`dare`) | Use it instead of a hand-written `while_loop` when gradients must flow through an iteration. |
| **einx** | Every array operation that names axes | Contractions, transposes, reshapes, axis reductions and inserted-axis broadcasts go through `gaussx._einx` (rules below). |
| **jaxtyping** | Shape annotations (`Float[Array, "n n"]`) | Annotate every public array argument and return. |
| **jax** | `jit`, `grad`, `vmap`, x64, PRNG | Pure functions, explicit keys, the input dtype preserved, no Python control flow on traced values. |
| **numpyro** (extra `gaussx[numpyro]`) | `MultivariateNormal` as a numpyro distribution | Optional. Names that need it are appended to `__all__` only when it is installed; never import it at module scope elsewhere. |

## Reuse before you write

Before writing a helper, a factorisation, an operator or a solver, find out
whether it exists:

1. **Search the capability index.** [`docs/capabilities.md`](docs/capabilities.md)
   lists every public gaussx name, grouped like the API reference, with a
   one-line summary; then the public API of lineax, matfree and optimistix;
   then the private toolkits below. It is generated (`make capabilities`) and
   checked in the fast lane, so it is current.
2. **Search the private toolkits** for plumbing gaussx's own modules share.
3. **If it is missing, add it at its layer**, in the subpackage that owns
   the concept, and export it; a helper two modules need goes into the
   owning toolkit, not inline in the one caller you have today.
4. **One object, one name.** Renames go through the deprecation policy
   (an alias that warns, a removal version), never a silent second name.

| You are about to write… | Use instead |
|---|---|
| `jnp.linalg.solve` / `cho_solve` / `solve_triangular` on a covariance | `gaussx.solve(A, b)` on a PSD-tagged operator (`lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)`) |
| `jnp.linalg.slogdet`, `2 * sum(log(diag(L)))` | `gaussx.logdet(A)`; from an existing factor, `cholesky_logdet` |
| `jnp.linalg.cholesky` + jitter retries | `gaussx.cholesky` (lazy, structure-preserving); `safe_cholesky` / `add_jitter` for ill-conditioned dense input |
| `jnp.linalg.inv(A) @ B`, explicit inverses | `gaussx.inv` (lazy), `solve_matrix` / `solve_columns` / `solve_rows` |
| Woodbury / matrix-determinant-lemma code, `K + U Uᵀ` | `LowRankUpdate`, `low_rank_plus_diag`, `woodbury_solve` |
| `jnp.kron(A, B)` then a solve | `Kronecker`; `A ⊗ I + I ⊗ B` is `KroneckerSum`; `Σᵢ Aᵢ ⊗ Bᵢ` is `SumOfKroneckers` |
| A block-diagonal or block-tridiagonal (state-space) solve | `BlockDiag`, `BlockTriDiag`, `tridiagonal_solve` |
| A Toeplitz / circulant matvec with FFTs | `Toeplitz`, `circulant`, `DiagonalizedOperator` |
| A CG loop, a preconditioner, a matrix-free solve | `CGSolver`, `PreconditionedCGSolver`, `*Preconditioner`, `linear_solve` / `as_linear_operator` for a bare `matvec` |
| Stochastic logdet / trace / diagonal | `SLQLogdet`, `BBMMSolver`, `inv_quad_logdet`, `trace_and_diag`; below them, `matfree.stochtrace` / `matfree.funm` |
| Lanczos, Arnoldi, bidiagonalisation | `matfree.decomp` (`tridiag_sym`, `hessenberg`, `bidiag`) |
| Gaussian log-density, entropy, KL, conditioning, sampling | `gaussian_log_prob`, `gaussian_entropy`, `gaussian_kl`, `conditional`, `sample_mvn`; as objects, `MultivariateNormal(Precision)` |
| Natural ↔ mean / expectation parameter conversions | `gaussx._expfam` conversions (`mean_cov_to_natural`, `natural_to_mean_cov`, …) |
| A Kalman filter / RTS smoother, SDE → state space | `kalman_filter`, `rts_smoother`, `parallel_kalman_filter`, `MaternSDE` and the other `SDEKernel`s, `discretize_mfd` |
| Gauss-Hermite / unscented / cubature expectations | `GaussHermiteIntegrator`, `UnscentedIntegrator`, `CubatureIntegrator`, `expected_log_likelihood` |
| GP predictive mean / variance, ELBOs, whitening | `predict_mean`, `predict_variance`, `collapsed_elbo`, `variational_elbo_gaussian`, `whiten_covariance`, `love_cache` |
| EnKF / ETKF / localisation / inflation | `enkf_analysis`, `etkf_transform`, `localization_matrix`, `inflate_*` |
| A Newton / fixed-point iteration you need gradients through | `optimistix.root_find` / `fixed_point` |
| `jnp.einsum`, `.reshape`, `.T` on arrays, `x[:, None]`, `axis=` reductions | `gaussx._einx` (`einsum`, `rearrange`, `reduce`, `repeat`) and `einx.add` / `einx.multiply` / … |
| Random PD test matrices, dense references, moment checks (tests) | `gaussx._testing` (`random_pd_operator`, `random_kronecker_pd`, `dense_solve`, `assert_sample_moments`, …) |
| A deprecation warning | `gaussx._deprecation` (`warn_deprecated`, `renamed_kwargs`, the `RENAMED` table) |
| Default iterative tolerances | `gaussx._strategies._tolerances` (`resolve_tolerance`) |
| Routing a `solver=` argument | `gaussx._strategies._dispatch` (`dispatch_solve`, `dispatch_logdet`) |
| Registering a new operator with lineax | `gaussx._operators._utils.register_lineax_structure_functions` |

A direct `jnp.linalg` call is fine on a small dense matrix that has no
structure to exploit (a dense fallback, a per-block factor); it is a defect
on an operator that has structure.

## The three contracts

Everything composes because it keeps three contracts. The long form is
[`docs/architecture.md`](docs/architecture.md) and the conventions in
[`docs/api/index.md`](docs/api/index.md); each rule below names the test
that enforces it.

### 1. Operators: lineax operators that dispatch

- **Subclass `lx.AbstractLinearOperator`** (an `eqx.Module`) and implement
  `mv`, `as_matrix`, `transpose`, `in_structure`, `out_structure`. Hold
  arrays and child operators as fields; hashable metadata (shapes, sparsity
  patterns) as static fields. Never mutate.
- **Register its properties.** Every lineax predicate (`is_symmetric`,
  `is_diagonal`, `is_positive_semidefinite`, `is_negative_semidefinite`,
  `is_tridiagonal`, `is_lower_triangular`, `is_upper_triangular`,
  `has_unit_diagonal`), the matching gaussx `is_*` predicate for its
  structure tag (`gaussx._tags`), and the structure functions via
  `register_lineax_structure_functions` (gh-410), all in
  `_operators/__init__.py`. A missing registration surfaces as
  `NotImplementedError` deep inside a lineax solver.
- **Dispatch, don't densify.** A fast path is an `isinstance` branch in the
  primitive (`_primitives/_solve.py`, `_logdet.py`, …), before the dense
  fallback. A structured path never calls `as_matrix()` on the structured
  operator. When a structured operator must be densified anyway, warn with
  `DenseFallbackWarning` and name the matrix-free alternative.
- **Lazy over dense.** `cholesky`, `sqrt` and `inv` return operators where
  structure allows.
- **Document the dispatch.** Every branch has its row and cell in the
  dispatch table in `docs/architecture.md`
  (`tests/test_docs_dispatch_table.py`).
- **Join the zoo.** Every exported operator class has a case in
  `tests/operators/_zoo.py` and `tests/operators/test_conformance.py`
  (`ZOO`), which checks `mv` / transpose against `as_matrix`, every primitive
  against a dense reference, and, for promised fast paths, that `as_matrix`
  is never called. Known gaps are `xfail(strict=True)` with their issue.

### 2. Solver strategies: interchangeable numerics

- **Subclass `AbstractSolveStrategy`, `AbstractLogdetStrategy` or
  `AbstractSolverStrategy`** (`_strategies/_base.py`). A strategy changes
  *how* a solve or logdet is computed, never *what*; math code takes
  `solver: AbstractSolverStrategy | None = None`, where `None` means
  structural dispatch, and routes it with `dispatch_solve` /
  `dispatch_logdet`.
- **Keep the strategy contract** (`tests/strategies/test_strategy_contract.py`):
  `jax.grad` with respect to the operator and the right-hand side matches the
  dense gradient; `vmap` over right-hand sides matches a loop; a
  default-constructed strategy solves a float32 system of condition number
  1e3; an iterative strategy that runs out of `max_steps` **raises**
  (lineax's `throw=True`, or `eqx.error_if`) instead of returning an
  unconverged iterate.
- **Tolerances default to `None`** and resolve per dtype
  (`_strategies/_tolerances.py`).
- **A preconditioner earns its place**: at rank `k < n` it at least halves
  the CG steps (same suite).

### 3. JAX numerics: safe under every transform

- **Pure functions.** Arrays and operators in, arrays and operators out. No
  global state, no in-place updates, no printing.
- **Explicit randomness.** Every stochastic routine takes a `key`; split it,
  never reuse it.
- **Preserve the dtype.** float32 in gives float32 out, even with x64 on
  (`tests/test_dtype_preservation.py`, gh-408). Build constants with the
  input's dtype; don't let a Python float or `jnp.float64` promote.
- **Traceable control flow.** No Python `if` / `while` on traced values; use
  `jnp.where`, `lax.cond`, `lax.scan`, `lax.while_loop`. Shapes and
  structure are static; values are traced.
- **Differentiable.** A routine on the gradient path differentiates
  correctly through `jit` and `vmap`; a custom rule (`jax.custom_vjp`, as in
  `_sparse/_vjp.py`) is tested against finite differences or a dense
  gradient.
- **The einx convention.** Contractions and matrix products of arrays
  (`gaussx._einx.einsum`), transposes, permutations, flatten / unflatten
  (`rearrange`), axis reductions (`reduce`, not `jnp.sum(x, axis=...)`), and
  broadcasts that insert an axis (`einx.add` / `subtract` / `multiply` /
  `divide`, or `repeat`, not `x[:, None]`). Exempt: lineax operator methods
  (`op.T`, `op.mv`, `op @ other`), plain `L @ z` matrix–vector products, full
  reductions with no axis, elementwise ops on same-shaped arrays, and
  constructors (`jnp.diag`, `jnp.eye`, `jnp.kron`). ruff `TID251` bans
  `jnp.einsum` / `transpose` / `moveaxis` / `reshape`;
  `tests/test_einx_convention.py` caps the legacy constructs at their current
  counts, so new code adds none (lower a ceiling when you convert old sites).

### The public API

- **Export** a new public name from `src/gaussx/__init__.py` (`import X as X`
  and an entry in `__all__`), list it in the `members:` block of its layer's
  page in `docs/api/` (`tests/test_docs_api_coverage.py`), and run
  `make capabilities`.
- **Name it by the rules** in [`docs/api/index.md`](docs/api/index.md#naming):
  US spelling, CamelCase only for classes, `*Result` / `*State` / `*Cache` /
  `*Decomposition` / `*Params` containers, `solve_<rhs-shape>` vs
  `<structure>_solve`, KL(first ‖ second) (`tests/test_naming.py`).
- **Deprecate, don't break.** Renames and removals follow the
  [deprecation policy](docs/api/index.md#deprecation-policy): a
  `GaussxDeprecationWarning` naming the replacement and the removal version
  (`tests/test_deprecations.py`).
- **Docstrings.** Google style; jaxtyping shapes; the equation in Unicode or
  MathJax `$…$` (no Sphinx / RST markup such as `:math:` or `.. math::`,
  `tests/test_docstrings.py`); `Args:` matching the signature and a `Returns:`
  (`tests/test_docstring_signatures.py`); and a runnable `>>>` example under
  `Examples:` for every new public name. Doctests run in both the x64 and the
  float32 lane, so print rounded floats, shapes or type names, not raw arrays
  (`tests/test_doctests.py`; the headline API is in `_MUST_HAVE_EXAMPLES`).

## What enforces them

Most rules here are tests in the fast lane, so CI tells you when one breaks.
Read the test's docstring before changing what it checks.

| Test | Enforces |
|---|---|
| `tests/operators/test_conformance.py` (+ `_zoo.py`) | Every operator × every primitive vs a dense reference; promised fast paths never densify |
| `tests/operators/test_lineax_interop.py` | Every operator works with lineax's own solvers and predicates |
| `tests/strategies/test_strategy_contract.py` | grad / vmap / float32 / `max_steps` for every strategy; preconditioner efficacy |
| `tests/test_docs_dispatch_table.py` | The dispatch table in `docs/architecture.md` matches the `isinstance` chains |
| `tests/test_docs_api_coverage.py` | `__all__` ↔ `dir(gaussx)` ↔ `docs/api/*.md`, each name on its layer's page |
| `tests/test_capabilities.py` | `docs/capabilities.md` is current; no gaussx name shadows a lineax / optimistix one |
| `tests/test_docstring_signatures.py`, `tests/test_doctests.py`, `tests/test_docstrings.py` | Docstrings match signatures; examples run in both lanes; no Sphinx / RST markup |
| `tests/test_naming.py`, `tests/test_deprecations.py` | Naming rules; every deprecation names a removal version and is removed on time |
| `tests/test_dtype_preservation.py` | float32 stays float32 under x64 |
| `tests/test_einx_convention.py` + ruff `TID251` | The einx convention |
| `tests/test_lazy_imports.py` | In-function imports only on dispatch-core cycle edges, each with a `# lazy import, cycle: …` comment |
| `tests/test_code_size.py` | Functions ≤ 150 lines, modules ≤ 800 (listed exceptions may only shrink) |
| `tests/test_makefile.py`, `tests/test_readme.py`, `tests/test_notebooks_in_sync.py` | Every Make target is `.PHONY`; README examples run; notebook `.py` / `.ipynb` pairs agree |

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in
`.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any
agent can read and follow them):

| Job | Recipe |
|---|---|
| Add a structured operator (Layer 1) | `add-operator` |
| Add or fix a fast path in a primitive | `add-dispatch-path` |
| Add a solver strategy or preconditioner (Layer 1.5) | `add-solver-strategy` |
| Add a distribution, GP, SSM, quadrature or inference routine (Layers 2–3) | `add-recipe` |
| Rename, deprecate or remove a public name | `deprecate-or-rename` |
| Add or update an example notebook | `add-notebook` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `gaussx-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `numerics-reviewer.md`) |
| Triage a red scheduled run (`ci-failure` issue) | `triage-ci-failure` |
| Write a squash commit message | `squash-commit` |
| Open or link GitHub issues | `create-gh-issue`, `link-gh-issues` (templates in `.github/ISSUE_TEMPLATE/`) |

## Working in the repo

Always run Python tools through `uv run` (never the system Python); `git`,
`ls` and other non-Python commands need no `uv run`.

```bash
make install              # uv sync --all-groups + pre-commit hooks
make test-fast            # fast tier, x64 on (what PR CI runs, without coverage)
make test-no-x64          # float32 lane: the NO_X64_TESTS subset with x64 off (PR CI)
make test-slow            # slow + integration tiers
make test                 # everything, in parallel
make lint                 # ruff check .   (entire repo)
make format               # ruff format . && ruff check --fix .
make typecheck            # ty check src/gaussx
make capabilities         # regenerate docs/capabilities.md
make docs-serve           # local MkDocs preview
```

Run one test with `uv run pytest tests/operators/test_kronecker.py::test_name -v`.

### Test tiers

CI on every PR runs the fast tier (`-m "not slow and not integration"`, with
coverage gated by `fail_under` in `pyproject.toml`, a ratchet that is never
lowered to make a PR pass) on Python 3.12 and 3.13, plus the float32 lane.
The "Extended Tests" workflow (`tests-extended.yml`) runs the slow and
integration tiers weekly, on PRs labelled `run-slow` (add it to PRs that
touch the SSM filters, distributions or numpyro paths) and on demand. The
weekly "Latest Dependencies" workflow re-resolves at the newest versions and
at the declared floors; a failure opens a `ci-failure` issue.

- **Unmarked:** unit tests under ~1 s each.
- **`@pytest.mark.slow`:** over ~1.5 s (heavy numerics, `jit`+`grad`+`vmap`
  sweeps, long scans). For a parametrised test, mark only the expensive
  cases slow (`pytest.param(64, marks=pytest.mark.slow)`).
- **`@pytest.mark.integration`:** end-to-end workflows (MCMC / SVI through
  numpyro), usually also `slow`.

Most runtime is XLA compiling each eagerly run operation, so a test's cost is
the number of distinct programs it compiles. `tests/conftest.py` turns on
JAX's persistent compilation cache; judge a test's tier with a warm cache
(`--durations`). `pytest-timeout` fails any test after 120 s; raise it for
one test with `@pytest.mark.timeout(...)`.

### The float32 lane

`tests/conftest.py` enables x64 for the suite, so CI also runs the
`NO_X64_TESTS` subset (Makefile) with `GAUSSX_TEST_X64=0`, where float64 does
not exist.

- Build test inputs in the active default float: the `gaussx._testing`
  `random_*` helpers do when `dtype` is `None`.
- Parametrise dtypes as `[jnp.float32, pytest.param(jnp.float64,
  marks=pytest.mark.x64_only(reason="float64 case"))]`.
- A test that genuinely needs float64 takes
  `@pytest.mark.x64_only(reason="...")`; the reason is required.

### Tests that assert on random draws

The `getkey` fixture is an `equinox.internal.GetKey` seeded per test from a
CRC of its node id, so a failure reproduces by rerunning it;
`EQX_GETKEY_SEED=<n>` overrides the seed, and a failing test prints a
copy-pasteable rerun line. `scripts/find_getkey_tests.py --dead` lists tests
that take `getkey` without using it (keep it empty).

- **Incidental randomness** (any model would do): `getkey` or `jr.key(0)`;
  a tolerance tuned on one draw must survive a short `EQX_GETKEY_SEED` sweep.
- **Sampling behaviour under test:** bound the estimator by its own sampling
  distribution (`gaussx._testing.assert_sample_moments`, 7σ by default), not
  a flat `atol` (gh-220).

Either way, say in a comment where the bound came from.

### Before every commit

All of these must pass, from the repo root:

1. `make test-fast` and `make test-no-x64`: zero failures (plus
   `make test-slow` on what you touched if it has slow tests).
2. `uv run --group lint ruff check .`: the **entire** repo, which includes
   `tests/`, `scripts/` and `docs/notebooks/*.py`. Never lint a subdirectory.
3. `uv run --group lint ruff format --check .`
4. `make typecheck`
5. After changing a dependency or the version: `uv lock`, and commit
   `uv.lock` (CI runs `uv lock --check`; release PRs bump the lockfile's
   own gaussx version through release-please's `extra-files`). Runtime
   dependencies take `>=` floors and no upper bounds.
6. After changing a public API: `make capabilities`.
7. After changing docstrings, `docs/` or `mkdocs.yml`:
   `uv run --group docs mkdocs build --strict`.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several
   readings, name them instead of picking one silently; ask when unsure.
2. **Simplicity first.** The minimum code that solves the problem: no
   speculative features, no single-use abstractions, no configurability
   nobody asked for.
3. **Surgical changes.** Touch only what the task needs; match the existing
   style; don't refactor or add docstrings to code you didn't change; remove
   only what your change made unused.
4. **Goal-driven.** Turn the task into a check (a failing test, a reproduced
   bug, a dense reference to match) and loop until it passes.

Also: `from __future__ import annotations` in every module, type hints on
every public function, Google-style docstrings, pure functions with side
effects isolated and explicit.

## Git, commits and pull requests

- Never push to or merge into `main` unless explicitly told to ("push to
  main", "merge to main"). Work on a feature branch, commit locally, and push
  only when asked. "Merge the branch" means push the feature branch, not
  merge into `main`.
- Commit messages and PR titles follow
  [Conventional Commits](https://www.conventionalcommits.org/) with a
  lowercase subject (`feat(ssm): add …`); CI validates PR titles. Types:
  `feat`, `fix`, `deprecate`, `docs`, `style`, `refactor`, `perf`, `test`,
  `build`, `ci`, `chore`, `revert`. Breaking changes use `!` and a
  `BREAKING CHANGE:` footer.
- Releases are cut by release-please; don't bump versions by hand.
- **Never replace or remove an existing PR title or description.** Read the
  existing description first and only append checklist items or update
  their status. Change the title only to fix a Conventional Commits
  violation.
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md).

### Pull Request Review Comments

After fixing a review comment, resolve its thread. Don't resolve threads you
didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

When the `gh` CLI is unavailable, use the GitHub MCP tools for the same
operations.

## Documentation

MkDocs + Material + mkdocstrings + mkdocs-jupyter; `pages.yml` deploys on
every push to `main`, and CI builds with `--strict`.

- **API pages** (`docs/api/*.md`) list members explicitly, organised by
  layer; prose pages are `index.md`, `vision.md`, `architecture.md` and
  `docs/design/`.
- **Notebooks** live in `docs/notebooks/` as jupytext percent-format `.py`
  sources **and** their executed `.ipynb` (both committed;
  `jupytext --to notebook --execute foo.py -o foo.ipynb`). Figures render
  inline with `plt.show()`: no `savefig`, no committed PNGs. Full standards:
  `.github/instructions/docs-examples.instructions.md`.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never
committed); track work in GitHub issues. Committed design references live in
`docs/design/`.
