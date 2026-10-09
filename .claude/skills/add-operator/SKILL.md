---
name: add-operator
description: Add a structured linear operator to gaussx (Layer 1) — a lineax AbstractLinearOperator such as a Kronecker / block / low-rank / banded / Toeplitz / sparse variant — with its predicate registrations, primitive fast paths, conformance-zoo case and docs. Use when asked to add, port or implement an operator, a structured matrix, or a new matrix structure in src/gaussx/_operators.
---

# Add a gaussx operator

Read "What gaussx is built on" and "The three contracts" in `AGENTS.md`
first; this skill is the step-by-step for contract 1. An operator touches
about eight places, and the fast-lane tests fail on each one you miss.

## 1. Make sure it does not exist yet

- Search `docs/capabilities.md` for the structure (and its synonyms:
  "Kronecker sum" vs "sum of Kroneckers" are different operators), including
  the lineax section: `lx.DiagonalLinearOperator`,
  `lx.TridiagonalLinearOperator`, `lx.FunctionLinearOperator`,
  `lx.TaggedLinearOperator` and lineax's operator algebra (`A + B`, `2 * A`,
  `A @ B`) may already be enough.
- If it is a special case of an existing operator (a new constructor, a new
  tag), add a factory (`snake_case`, returning the existing class) or a
  parameter instead of a sibling class. CamelCase is reserved for classes.

## 2. The class (`src/gaussx/_operators/_<name>.py`)

- Subclass `lx.AbstractLinearOperator`. Fields: arrays and child operators;
  shapes, flags and other hashable metadata as `eqx.field(static=True)`.
  Accept `tags` and store them as a static `frozenset` (see `Kronecker` in
  `_kronecker.py`, which also adds its own structure tag).
- Implement `mv`, `as_matrix`, `transpose`, `in_structure`,
  `out_structure`. `mv` must exploit the structure (that is the point); keep
  the input dtype; go through `gaussx._einx` for any axis-naming array op.
- Validate static shapes eagerly in `__init__` / `__check_init__` with a
  `ValueError` that names the operator.
- Docstring: the structure as an equation, the cost of `mv` / solve /
  logdet in O(·), `Args:`, and a runnable `>>>` example that prints a
  shape or rounded values (it runs in both lanes).
- Keep the module under 800 lines and every function under 150
  (`tests/test_code_size.py`).

## 3. Register it (`src/gaussx/_operators/__init__.py`)

- Every lineax predicate: `lx.is_symmetric`, `lx.is_diagonal`,
  `lx.is_positive_semidefinite`, `lx.is_negative_semidefinite`, and the
  defaults for `lx.is_tridiagonal`, `lx.is_lower_triangular`,
  `lx.is_upper_triangular`, `lx.has_unit_diagonal` (add the class to
  `_ALL_TRIDIAG_DEFAULTS` / `_TRI_DEFAULTS` unless it is genuinely
  triangular or element-tridiagonal). Derive properties from children or
  tags (`lx.positive_semidefinite_tag in operator.tags`).
- `register_lineax_structure_functions(YourClass)`, so lineax's own solvers
  can `linearise` / `materialise` / `diagonal` / `conj` it (gh-410).
- A new structure tag goes in `gaussx/_tags.py` (a `_Tag` singleton with an
  attribute docstring, plus an `is_*` singledispatch predicate registered
  for the class).

## 4. Fast paths

For each primitive that can exploit the structure (`solve`, `logdet`,
`cholesky`, `diag`, `trace`, `sqrt`, `inv`, and optionally `eig`, `svd`,
`frobenius_norm`, `submatrix`), follow the `add-dispatch-path` skill: an
`isinstance` branch before the dense fallback, never calling `as_matrix()`
on the structured operator, plus its cell in the dispatch table in
`docs/architecture.md`. Every class a primitive branches on needs a row
(`tests/test_docs_dispatch_table.py`); cells with no branch say **dense**
(or **lazy** for `inv`).

## 5. Export and document

- `src/gaussx/__init__.py`: `from gaussx._operators import YourClass as
  YourClass` and an entry in `__all__`.
- `docs/api/operators.md`: add it to the `members:` list of the right
  section (`tests/test_docs_api_coverage.py`).
- If it is a headline operator, add it to `_MUST_HAVE_EXAMPLES` in
  `tests/test_doctests.py`.
- `make capabilities`.

## 6. Tests

- `tests/operators/test_<name>.py`: `mv` and `as_matrix` against a hand-built
  dense matrix, the transpose, the fast paths against `gaussx._testing`'s
  `dense_solve` / `dense_logdet` / …, a float32 and a float64 case
  (`x64_only` on the latter), and `jit` / `grad` through `solve` + `logdet`
  (mark `slow` if it compiles much).
- Conformance: add a builder to `ZOO` in `tests/operators/_zoo.py` (small,
  PD, built from `gaussx._testing` in the active default float) and a `Case`
  in `tests/operators/test_conformance.py` with `structured=` naming the
  primitives you promise never densify and `spy=YourClass`. A primitive
  that is known wrong gets `xfail={"<primitive>": "gh-NNN"}` (strict), with
  an issue.
- `tests/operators/test_lineax_interop.py` fails until every exported
  operator class has a `ZOO` builder, then runs lineax's own structure
  functions and solvers on it.
- If the module belongs in the float32 lane, add its test file to
  `NO_X64_TESTS` in the Makefile.

## 7. Verify

Run the `pre-pr-check` skill. At minimum:
`uv run pytest -q tests/operators tests/primitives tests/test_docs_dispatch_table.py tests/test_docs_api_coverage.py tests/test_capabilities.py tests/test_doctests.py`.
