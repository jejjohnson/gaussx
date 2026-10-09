---
name: code-review
description: Review a change or pull request in gaussx against CODE_REVIEW.md, the three contracts in AGENTS.md (operators, solver strategies, JAX numerics) and reuse of gaussx / lineax / matfree primitives.
---

# Code review

Read the "Reuse before you write" table and "The three contracts" in
[`AGENTS.md`](../../../AGENTS.md); for every function, class or module the
diff adds, search [`docs/capabilities.md`](../../../docs/capabilities.md) for
an existing equivalent; then apply [`CODE_REVIEW.md`](../../../CODE_REVIEW.md)
and report in its format.
