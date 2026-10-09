# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this
repository. gaussx is a JAX numerics library: most defects worth finding are
numerical (a densified fast path, a silent dtype promotion, Python control
flow on a traced value, an unconverged solve returned as an answer), not
stylistic. Read "Reuse before you write" and "The three contracts" in
[`AGENTS.md`](AGENTS.md) first; this file is the checklist and the report
format.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the fast-lane tests already enforce (formatting,
import order, the einx bans, CamelCase and spelling, docstring/signature
agreement); review
what they cannot see.

### 1. Reuse

- Every function, class or module the diff **adds** has been checked
  against [`docs/capabilities.md`](docs/capabilities.md) (gaussx, lineax,
  matfree, optimistix) and the private toolkits. A re-implemented solve,
  Cholesky, Woodbury identity, Kalman step, CG loop, Lanczos or stochastic
  estimator is a **High** finding, with the existing name to use.
- `jnp.linalg.solve` / `cholesky` / `inv` / `slogdet` or
  `jax.scipy.linalg.cho_solve` / `solve_triangular` on an operator that has
  structure; fine on a small dense block or a dense fallback.
- A helper two modules need lives in the owning toolkit, not inline.

### 2. The operator contract

- Subclasses `lx.AbstractLinearOperator`; implements `mv`, `as_matrix`,
  `transpose`, `in_structure`, `out_structure`; holds no mutable state.
- Every lineax predicate, the gaussx `is_*` predicate for its tag and
  `register_lineax_structure_functions` are registered in
  `_operators/__init__.py`.
- New fast paths are `isinstance` branches before the dense fallback, never
  call `as_matrix()` on the structured operator, and have their row/cell in
  the dispatch table in `docs/architecture.md`. A forced densification warns
  with `DenseFallbackWarning`.
- `cholesky` / `sqrt` / `inv` results stay lazy operators where structure
  allows.
- The class has a case in the conformance zoo (`tests/operators/_zoo.py`,
  `test_conformance.py`); known gaps are `xfail(strict=True)` with an issue.

### 3. The solver-strategy contract

- Subclasses the right `Abstract*Strategy`; changes *how*, never *what*.
- `solver=None` keeps structural dispatch; routing uses `dispatch_solve` /
  `dispatch_logdet`.
- grad vs dense, vmap over right-hand sides, float32 at condition number
  1e3, and raising at `max_steps` all hold
  (`tests/strategies/test_strategy_contract.py`).
- Tolerances default to `None` and resolve per dtype.

### 4. JAX numerics

- **Traceability:** no Python `if` / `while` / `bool()` / `float()` /
  `.item()` on traced values; `lax.cond` / `scan` / `while_loop` /
  `jnp.where` instead. Shapes and structure are static.
- **Dtypes:** float32 in, float32 out with x64 on. Watch arrays built
  without the input's dtype (`jnp.eye(n)`, `jnp.zeros(...)`,
  `jnp.array([...])`, `jnp.float64`), and a constant such as
  `jnp.asarray(0.5)` returned or stored on its own. Python scalars and a
  bare `jnp.asarray(0.5)` combined with an array are weakly typed and keep
  its dtype.
- **Randomness:** an explicit `key` argument, split before each use, never
  reused for two draws.
- **Gradients:** the routine differentiates through `jit` / `vmap`; a
  `custom_vjp` / `custom_jvp` is tested against a dense or finite-difference
  gradient; `jnp.where` branches don't produce NaN gradients in the unused
  branch (the double-`where` trick).
- **Pytrees:** containers are `eqx.Module`, never `dataclasses` or plain
  classes; hashable metadata is `eqx.field(static=True)`; no array in a
  static field.
- **Runtime checks:** shape and argument errors on static values raise
  eagerly (`ValueError` / `TypeError`); conditions on traced values use
  `eqx.error_if` (or lineax's `throw=True`), never a silent wrong answer.
- **The einx convention:** axis-naming array ops go through
  `gaussx._einx`; exempt operator methods and full reductions are fine.

### 5. Numerical correctness

- No explicit inverse where a solve will do; no `log(det(A))` where `logdet`
  will do.
- Symmetric results are symmetric (`symmetrize`), PSD factorisations of
  ill-conditioned input use `safe_cholesky` / `add_jitter`, log-densities
  stay in log space (`logsumexp`, `log1p`), and subtractions that cancel
  catastrophically (variance = E[x²] − E[x]²) are flagged.
- Iterative results are checked for convergence, not assumed.
- Every tolerance in a test states where it came from (round-off bound,
  sampling distribution, published value); a sampling test bounds the
  estimator by its own distribution (`assert_sample_moments`).
- The claimed complexity is the real one: a Kronecker path that
  materialises a factor product, or a per-step Python loop that should be a
  `scan`, defeats the point of the library.

### 6. Public API and documentation

- New public names: exported from `src/gaussx/__init__.py` and `__all__`,
  listed on their layer's `docs/api/*.md` page, `docs/capabilities.md`
  regenerated.
- Naming follows `docs/api/index.md` (US spelling, CamelCase only for
  classes, result-container suffixes, `solve_*` vs `*_solve`,
  KL(first ‖ second)).
- Renames and removals go through `gaussx._deprecation` with a removal
  version, and the PR uses the `deprecate:` type (or a `deprecate(...)` line
  in a `BEGIN_COMMIT_OVERRIDE` block).
- Docstrings: Google style, jaxtyping shapes, the equation (Unicode or
  MathJax, no Sphinx markup), `Args:` / `Returns:`, a reference for a
  published algorithm, and a runnable `>>>` example for every new public
  name, printing rounded values or shapes so it passes in both lanes.
- Inline comments explain *why*; complex algorithms get step-by-step
  comments with the equation (`# Σ⁻¹ = A⁻¹ − A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹`).

### 7. Tests

- New behaviour is tested against a dense reference, a hand-computed value
  or a published one, for both a structured and a plain dense input.
- Inputs come from `gaussx._testing` in the active default float; float64-only
  tests carry `x64_only(reason=...)`; the test file is covered by
  `NO_X64_TESTS` (Makefile; `tests/operators`, `tests/primitives` and
  `tests/linalg` are listed whole) when it should run in the float32 lane.
- Tier markers are right: unmarked under ~1 s with a warm compilation
  cache, `slow` above ~1.5 s, `integration` for end-to-end numpyro fits.
- `jit` / `grad` / `vmap` are exercised for code on the gradient path.

### 8. Modern Python (≥ 3.12)

- `from __future__ import annotations` in every module; type hints on every
  public function; `X | None`, built-in generics; f-strings; specific
  exceptions with `raise ... from ...`; guard clauses over deep nesting.
- `Enum` or `Literal` for fixed option sets (strategy names, methods).
- Functions ≤ 150 lines, modules ≤ 800 (`tests/test_code_size.py`).

### 9. Dependencies and security

- No new runtime dependency without discussion; optional ones go behind an
  extra and a lazy import, like numpyro. Runtime pins are `>=` floors only,
  and `uv.lock` is updated with them.
- No secrets, no network access at import or in fast tests, no
  `subprocess(shell=True)` in scripts.

---

## gaussx-Specific Checks

### Structural dispatch, not densification

```python
# ❌ Throws the Kronecker structure away: O((n₁n₂)³)
x = jnp.linalg.solve(K.as_matrix(), y)

# ✅ Per-factor solves: O(n₁³ + n₂³)
x = gaussx.solve(K, y)
```

### PSD tags pick the algorithm

```python
# ❌ Untagged: lineax falls back to LU
A = lx.MatrixLinearOperator(K + noise * jnp.eye(n, dtype=K.dtype))

# ✅ Tagged: Cholesky, and gaussx knows it may take the PSD paths
A = lx.MatrixLinearOperator(
    K + noise * jnp.eye(n, dtype=K.dtype), lx.positive_semidefinite_tag
)
```

### Control flow on traced values

```python
# ❌ Fails under jit (a traced bool), and silently specialises outside it
if jnp.any(d <= 0):
    d = d + jitter

# ✅ Traceable
d = jnp.where(d <= 0, d + jitter, d)
```

### Dtype preservation

```python
# ❌ float64 under x64 even for float32 input
eye = jnp.eye(n)
jitter = jnp.zeros(n) + 1e-6

# ✅ Follows the input
eye = jnp.eye(n, dtype=A.dtype)
jitter = jnp.full(n, 1e-6, dtype=A.dtype)
```

### Pytrees, not dataclasses

```python
# ❌ Not a pytree: breaks jit / vmap / grad
@dataclass
class KalmanState:
    mean: Array
    cov: Array


# ✅ An immutable pytree; static metadata marked static
class KalmanState(eqx.Module):
    mean: Float[Array, " n"]
    cov: Float[Array, "n n"]
    num_steps: int = eqx.field(static=True)
```

### Unconverged solves

```python
# ❌ Returns whatever CG reached after max_steps
solution = lx.linear_solve(A, b, lx.CG(rtol, atol, max_steps=50), throw=False)

# ✅ Fails loudly (or reports the result and handles it explicitly)
solution = lx.linear_solve(A, b, lx.CG(rtol, atol, max_steps=50))
```

### The einx convention

```python
# ❌ Banned / capped constructs
cov = jnp.einsum("ni,nj->ij", X, X)
y = x[:, None] * w

# ✅ The index pattern written out
cov = einsum(X, X, "n i, n j -> i j")  # gaussx._einx
y = einx.multiply("n, n k -> n k", x, w)
```

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
