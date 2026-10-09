---
name: gaussx-review
description: Review a change or pull request in gaussx against the repo's own rules — CODE_REVIEW.md, the three contracts in AGENTS.md (operators, solver strategies, JAX numerics), reuse of gaussx / lineax / matfree primitives, and numerical correctness. Use when asked to review a diff, branch or PR in this repo.
---

# Review a gaussx change

1. **Get the diff** as `CODE_REVIEW.md` describes ("How to Obtain the
   Diff"), or from the PR. Note which layers it touches.
2. **Reuse** — run the `reuse-reviewer` subagent on the diff. A recipe that
   hand-rolls a solve, a factorisation or a Krylov loop is the main way this
   library decays: it works, and it silently throws the structure away.
3. **Numerics** — run the `numerics-reviewer` subagent on the diff (the two
   subagents can run in parallel). It traces traced-value control flow,
   dtype promotion, densified fast paths, PRNG handling, gradients and
   tolerances.
4. **Contracts** — check "The three contracts" in `AGENTS.md` for every new
   or changed operator, strategy, primitive branch or recipe:
   - operator: lineax interface, every predicate and
     `register_lineax_structure_functions` registered, fast paths that never
     call `as_matrix()`, a dispatch-table row, a conformance-zoo `Case`;
   - strategy: the right abstract base, static options, `None` tolerances,
     raising at `max_steps`, a `STRATEGIES` entry in the contract suite;
   - public API: export, `docs/api` members, `docs/capabilities.md`
     regenerated, naming rules, deprecations through `gaussx._deprecation`,
     docstring with equation, reference and a doctest that passes in float32.
5. **Checklist** — the rest of `CODE_REVIEW.md`, skipping anything ruff, ty
   or the fast-lane convention tests already enforce.
6. **Verify claims** — for anything you would flag as a bug, trace a real
   input to the failure, or run it (`uv run python -c ...` against a dense
   reference, or under `jax.jit`), before reporting it.

Report in the format `CODE_REVIEW.md` gives (overview, then suggestions with
priority, file, lines and a concrete proposed change, then a summary).
