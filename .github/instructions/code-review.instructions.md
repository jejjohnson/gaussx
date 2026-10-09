---
applyTo: "**"
---

# Code Review Instructions

When performing code review, use `/CODE_REVIEW.md` as the source of truth for:

- Review checklist (reuse, the operator and solver-strategy contracts, JAX numerics, numerical correctness, public API and docs, tests, idioms, dependencies)
- gaussx-specific checks (structural dispatch, PSD tags, traced control flow, dtype preservation, `eqx.Module` pytrees, unconverged solves, the einx convention)
- Output format and priority levels
- Suggestion type emojis and review tone

Key principles:
- Sacrifice *cleverness* for *clarity*. Sacrifice *brevity* for *explicitness*.
- Don't worry about formatting — CI (ruff format, pre-commit) handles that automatically.
- Be **constructive** and **specific**. Acknowledge good patterns with 👍.
- Every suggestion must include a concrete alternative, not just criticism.
