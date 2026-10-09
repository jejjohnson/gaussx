---
name: Feature / Enhancement
about: One deliverable — a primitive, operator, strategy, recipe, notebook or docs page.
title: "<area>(<scope>): <short description>"
labels: ["type:feature"]
---

<!--
Title: the layer or subpackage, then the symbol, e.g.
  "operators(kronecker): …", "primitives(solve): …", "ssm(kalman): …".
Write the issue so another contributor (human or agent) can implement it
without opening other repos or chats: lead with the exact API and the
equations.
-->

## Problem / Request
<!-- What's needed? One or two sentences. -->

## Motivation
<!-- Why now; who needs it (a downstream package, a notebook, a paper); what breaks without it. -->

## Proposed API
```python
# Signatures with jaxtyping shapes, and a usage example.
```

## Mathematical Notes
<!--
Required for algorithmic issues; delete only when truly non-numerical.
Defining equations, parameterisation / sign conventions, the identity or
factorisation that gives the structured cost, the complexity (dense vs
structured), invariants the tests should pin down, stability notes and edge
cases. GitHub math ($…$, $$…$$) or Unicode (σ², Λ⁻¹, ⊗, O(n³)).
-->

## Reuse
<!--
What this composes (gaussx primitives / operators / strategies, lineax,
matfree, optimistix); see docs/capabilities.md. Name anything that looks
similar and why it is not enough.
-->

## References & Existing Code
- Paper / equations:
- Reference implementation:
- Related code: `src/gaussx/<path>`

## Implementation Steps
<!-- Concrete, file-level, checkable. The add-operator / add-dispatch-path / add-solver-strategy / add-recipe skills list them. -->
- [ ] Add `<symbol>` in `src/gaussx/_<subpackage>/_<module>.py`
- [ ] Export from `src/gaussx/__init__.py` and list on `docs/api/<page>.md`
- [ ] ...

## Definition of Done
- [ ] Tests against a dense / closed-form / published reference, in both lanes (or `x64_only` with a reason)
- [ ] `jit` / `grad` / `vmap` exercised where the code is on a gradient path
- [ ] Contract suites updated where they apply (conformance zoo, dispatch table, strategy contract)
- [ ] Docstring with equation, reference and a runnable example
- [ ] `make capabilities`; `make test-fast`, `make test-no-x64`, `make lint`, `make typecheck` green

## Relationships
<!-- Apply the native links after opening (the link-gh-issues skill). -->
- Parent: #
- Blocked by: #
- Blocks: #
- Related: #
