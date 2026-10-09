---
name: add-dispatch-path
description: Add or change a structural fast path in a gaussx primitive (solve, logdet, cholesky, diag, trace, sqrt, inv, eig, svd, …) for an operator class, keeping the dispatch table in docs/architecture.md and the conformance zoo in sync. Use when a primitive densifies an operator that has structure, or when asked to make solve/logdet/cholesky fast for some operator.
---

# Add a fast path to a primitive

The primitives in `src/gaussx/_primitives/_<primitive>.py` are one readable
chain of `isinstance` checks ending in a dense (or lineax-solver) fallback.
No registry, no plugin system: a fast path is one more branch.

## 1. Confirm the gap

- Reproduce the densification: wrap the operator class's `as_matrix` in a
  spy that raises (as `tests/operators/test_conformance.py` does) and call
  the primitive. If the conformance `Case` for the class already lists the
  primitive in `structured=`, the path exists.
- Write down the identity you will use and its cost (Roth's lemma, the
  matrix-determinant lemma, a joint eigenbasis, block substitution, …), and
  a reference.

## 2. The branch

- Add `if isinstance(operator, YourClass): return _<primitive>_<structure>(operator, ...)`
  in the primitive's own function body, **before** the dense fallback and
  before any wrapper that would swallow the class. The test reads the
  `isinstance(operator, ...)` calls in that function body only, so keep the
  check there (the helper does the work).
- The helper `_<primitive>_<structure>` lives in the same module, is pure,
  preserves the dtype, never calls `as_matrix()` on the structured operator
  (on a small child it may), and recurses through the public primitive for
  children so their own structure is used (`solve(child, ...)`, not
  `jnp.linalg.solve`).
- Respect a `solver=` argument where the primitive takes one: a lineax
  solver is threaded into the structural rule (per factor, per block); a
  gaussx strategy owns the whole solve (see `_solve.py`).
- A result that must be densified anyway warns with `DenseFallbackWarning`
  and names the matrix-free alternative.
- If the helper needs an operator module that imports the primitives back,
  import it inside the function with a `# lazy import, cycle: ...` comment
  and add the edge to `tests/test_lazy_imports.py`.

## 3. The dispatch table

Update the operator's row in the table in `docs/architecture.md` ("How
dispatch actually works"): replace **dense** / **lazy** with the method and
cost. `tests/test_docs_dispatch_table.py` fails if a branch has no row, or a
**dense** / **lazy** cell has a branch.

## 4. Tests

- In the conformance suite, add the primitive to the class's
  `structured=` set (and remove a matching `xfail`), so the spy proves the
  path never densifies and the dense reference proves it is right.
- A focused test in `tests/primitives/test_<primitive>.py` or
  `tests/operators/test_<name>.py` for the edge cases the identity has
  (singular blocks, rank 0, 1×1, non-PSD input, complex input if supported).
- `jit` and `grad` through the new path (slow if it compiles much).

## 5. Verify

`uv run pytest -q tests/primitives tests/operators/test_conformance.py tests/test_docs_dispatch_table.py tests/test_lazy_imports.py`,
then the `pre-pr-check` skill.
