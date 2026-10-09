---
applyTo: "src/**/*.py,tests/**/*.py,docs/notebooks/**/*.py"
---

# Python Coding Standards

## Modern Python (3.12+)

- `from __future__ import annotations` at the top of every module
- Type hints on **all** public functions, methods, and module-level variables
- Modern union syntax: `X | None` not `Optional[X]`, `X | Y` not `Union[X, Y]`
- Built-in generics: `list[int]`, `dict[str, Any]` not `List[int]`, `Dict[str, Any]`
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- `equinox.Module` for data containers (operators, states, results): a dataclass is not a pytree; plain `dataclasses` only for host-side, never-traced records
- `Enum` or `Literal` for fixed sets of constants
- Context managers (`with` statements) for resource handling
- Specific exception types (never bare `except:`)
- Proper exception chaining (`raise ... from ...`)
- Early returns / guard clauses to reduce nesting

## Package Preferences

No new runtime dependency without discussion; build on what gaussx already
depends on (see "What gaussx is built on" in `AGENTS.md`).

| Purpose | Preferred Package |
|---------|-------------------|
| Data containers | `equinox.Module` (pytrees); `dataclasses` only for host-side records |
| Linear operators, tags, direct / Krylov solvers | `lineax` (and gaussx's own operators and strategies) |
| Lanczos, SLQ, stochastic trace / diagonal | `matfree` |
| Root finding, fixed points with implicit gradients | `optimistix` |
| Axis-naming array ops | `einx`, through `gaussx._einx` |
| Shape annotations | `jaxtyping` |
| Path handling | `pathlib` (stdlib) |
| Testing | `pytest` (+ `gaussx._testing`) |

## JAX

Keep the three contracts in `AGENTS.md`: pure functions, explicit PRNG keys,
the input dtype preserved, no Python control flow on traced values, runtime
checks on traced values through `eqx.error_if`, and lineax operators with
their predicates registered.

## Documentation

- Module-level docstrings explaining purpose
- Function/method docstrings for all public APIs (Google style)
- Inline comments explaining *why*, not *what*
- Scientific algorithms should include Unicode equations in docstrings (e.g. `# σ² = Σ(xᵢ − μ)² / N`)
- New public classes and functions include at least one runnable `>>>` example (doctested by `tests/test_doctests.py`; see CODE_REVIEW.md)
