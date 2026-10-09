---
name: add-recipe
description: Add a Layer 2 or Layer 3 function to gaussx — a Gaussian distribution helper, an exponential-family conversion, a GP routine (conditioning, ELBO, whitening, LOVE), a state-space / Kalman routine, a quadrature rule or likelihood, or a Bayesian-inference / ensemble-Kalman step — composed from the lower layers. Use when asked to add, port or implement such an algorithm in src/gaussx/_distributions, _expfam, _gp, _ssm, _quadrature or _inference.
---

# Add a recipe (Layers 2–3)

Recipes are the equations from a paper, written as pure functions over
operators. Their value is that they **compose the lower layers**: a recipe
that calls `jnp.linalg.solve` on a covariance has thrown away every structure
gaussx exists to exploit.

## 1. Make sure it does not exist yet

- Search `docs/capabilities.md` for the algorithm by name and synonyms
  (ETKF / ensemble square-root filter, LOVE / Lanczos variance, SpInGP, CVI,
  EP / ADF, …). Many pyrox algorithms were promoted here already (gh-130).
- If a close function exists, extend it (a parameter, a mode) rather than
  adding a sibling. If it belongs in a downstream package (a model, a
  training loop, a numpyro model), it does not belong here.
- Pick its subpackage by layer: `_distributions` / `_expfam` (Layer 2),
  `_gp`, `_ssm`, `_quadrature`, `_inference` (Layer 3). Layers 2 and 3
  import only from below.

## 2. Write it

- A pure function (or an `eqx.Module` for a stateful object such as a
  distribution, a filter state, a likelihood). Operators in, operators out:
  accept `lx.AbstractLinearOperator` for covariances / precisions, and keep
  the result lazy where structure allows.
- Every linear-algebra step goes through the primitives: `solve`, `logdet`,
  `cholesky`, `inv`, `sqrt`, `diag`, `trace`; the utilities `woodbury_solve`,
  `schur_complement`, `safe_cholesky`, `symmetrize`, `solve_columns`; and
  the existing Layer 2 helpers (`gaussian_log_prob`, `gaussian_kl`,
  `conditional`, `sample_mvn`, `joseph_update`). Search the "Reuse before you
  write" table in `AGENTS.md` before each one you are tempted to write.
- Take `solver: AbstractSolverStrategy | None = None` where the recipe
  solves or computes a logdet, and route it with `dispatch_solve` /
  `dispatch_logdet`, so users can swap in CG / BBMM.
- Sequential algorithms use `lax.scan` (and the parallel-scan form where
  one exists in `_ssm`); no Python loop over time steps. Carries are
  `*State` modules; one-shot outputs are `*Result` modules.
- Keys are explicit arguments; the input dtype is preserved; no Python
  control flow on traced values; the einx convention for axis-naming ops.
- Names follow `docs/api/index.md` (`*Result` / `*State` / `*Cache`,
  `gaussian_*` helpers, KL(first ‖ second) stated in the docstring's first
  line, `mean_cov` vs `mean_chol`).
- Docstring: the equations, the reference (paper, equation numbers), shapes
  in jaxtyping, `Args:` / `Returns:`, and a runnable `>>>` example that
  prints rounded values or shapes.

## 3. Export and document

`src/gaussx/__init__.py` (import + `__all__`), the `members:` block of the
layer's page (`docs/api/distributions.md`, `gp.md`, `ssm.md`,
`quadrature.md` or `inference.md`), `make capabilities`. A user-facing flow
may deserve a notebook (the `add-notebook` skill).

## 4. Tests

- Against an independent reference: a dense computation of the same
  equations (`gaussx._testing.dense_*`), a published value, or a special
  case with a closed form (a Kalman filter on a 1-D random walk, a GP with
  one point). Structured and dense inputs give the same answer.
- `jit`, `grad` (where it is on a gradient path) and `vmap` over a batch.
- Sampling behaviour bounded by `assert_sample_moments`, not a flat `atol`.
- float32: build inputs with `gaussx._testing` defaults; add the test file
  to `NO_X64_TESTS` when the routine should hold in float32; float64-only
  cases take `x64_only(reason=...)`.
- Tier: SSM filters, numpyro paths and long scans are often `slow`; add the
  `run-slow` label to the PR when you touch them.

## 5. Verify

Run the subpackage's tests, then the `pre-pr-check` skill.
