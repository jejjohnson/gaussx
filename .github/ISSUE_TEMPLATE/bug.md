---
name: Bug report
about: Something isn't working — a wrong result, a crash, a failure under jit / grad / vmap, a dtype or performance regression.
title: "<area>(<scope>): <what goes wrong>"
labels: ["bug"]
---

## Problem
<!-- What's broken? One or two sentences. -->

## Reproduction
```python
# Minimal, self-contained. Say whether it fails eagerly, under jax.jit,
# jax.grad or jax.vmap, and whether x64 is on.
```

## Expected Behavior
<!-- The right answer: a dense reference, a closed form, or the documented behaviour. -->

## Actual Behavior
<!-- What happens instead. Traceback, or the wrong numbers next to the reference. -->

## Environment
- gaussx:
- jax / jaxlib:
- equinox / lineax / matfree:
- numpyro (if involved):
- Python / platform / device (CPU, GPU):
- x64 enabled:

## Definition of Done
- [ ] A regression test reproduces it (both lanes, or `x64_only` with a reason)
- [ ] The fix lands; the test is green
- [ ] Contract suites updated if the bug was a gap there (zoo `xfail` removed, dispatch table, strategy contract)

## Relationships
- Related: #
