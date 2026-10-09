---
name: code-review
description: Review a change or pull request in gaussx against CODE_REVIEW.md, the three contracts in AGENTS.md (operators, solver strategies, JAX numerics) and reuse of gaussx / lineax / matfree primitives.
---

# Code review

Follow the same steps as the repo's review skill,
[`.claude/skills/gaussx-review/SKILL.md`](../../../.claude/skills/gaussx-review/SKILL.md):
check the diff for re-implemented functionality with the procedure in
[`.claude/agents/reuse-reviewer.md`](../../../.claude/agents/reuse-reviewer.md)
(search [`docs/capabilities.md`](../../../docs/capabilities.md) for every
helper the diff adds), check it for JAX numerical defects with the procedure
in [`.claude/agents/numerics-reviewer.md`](../../../.claude/agents/numerics-reviewer.md),
check "The three contracts" in [`AGENTS.md`](../../../AGENTS.md), then
apply [`CODE_REVIEW.md`](../../../CODE_REVIEW.md) and report in its format.
