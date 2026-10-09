---
name: triage-ci-failure
description: Triage a failed scheduled CI run in gaussx — a ci-failure issue opened by the weekly "Latest Dependencies" (newest versions or declared floors) or "Extended Tests" (slow + integration) workflow — by reproducing it at the resolved versions, finding the upstream change or flaky test, and proposing the fix. Use when asked to look at a ci-failure issue, a red scheduled run, or a breakage after a jax / lineax / equinox / matfree / numpyro release.
---

# Triage a scheduled CI failure

Two scheduled workflows open (or comment on) a `ci-failure` issue when they
go red, through `.github/actions/report-scheduled-failure`:

| Workflow | What it runs | Usual cause |
|---|---|---|
| `latest-deps.yml`, `resolution: highest` | Fast lane at the newest jax, jaxlib, equinox, lineax, matfree, jaxtyping, einx, numpyro | An upstream release changed behaviour or removed an API |
| `latest-deps.yml`, `resolution: lowest-direct` | Fast lane at the `>=` floors in `pyproject.toml` | gaussx started using something newer than its floor |
| `tests-extended.yml` (weekly) | The entire suite, on 3.12 and 3.13 | A slow test regressed, a tolerance tuned on one draw, a timeout |

## 1. Read the run

From the issue's run link, find the failing job, the failing test ids and
the "Show resolved versions" step (latest-deps) to see which upstream
versions it ran with. Compare with `uv.lock` to see what moved.

## 2. Reproduce locally, in a scratch copy of the lock

```bash
cp uv.lock /tmp/uv.lock.bak
uv lock --upgrade --resolution highest          # or lowest-direct
uv sync --group dev
uv run pytest -n auto -m "not slow and not integration" <failing test ids>
cp /tmp/uv.lock.bak uv.lock && uv sync --all-groups   # restore
```

For an extended-tests failure, `uv run pytest -v <test id>` at the locked
versions; for a test that takes `getkey`, rerun with the seed it printed
(`EQX_GETKEY_SEED=<n>`) and sweep a few others.

If several packages moved, bisect: start from the committed lock and upgrade
one package at a time (`uv lock --upgrade-package <pkg>`) until one upgrade
reproduces it. Use the job's Python: latest-deps runs `highest` on 3.13 and
`lowest-direct` on 3.12 (`uv sync --python 3.13 ...`).

## 3. Classify and fix

- **Upstream behaviour change** (a new lineax predicate becoming required,
  an equinox field rule, a jax dtype or tracing change): fix gaussx to work
  on both old and new versions, or raise the floor in `pyproject.toml` if
  the old version cannot be supported (then `uv lock`, and say so in the
  PR). Link the upstream changelog or issue.
- **Floor too low**: raise the floor to the first version that has what
  gaussx uses; `uv lock`.
- **Upstream bug**: report it upstream with a minimal reproduction; in
  gaussx, a narrow workaround with a comment linking the upstream issue, or
  an `xfail(strict=True)` naming it — never a skip without an issue.
- **Flaky numerics** (a tolerance tuned on one draw): bound the estimator by
  its sampling distribution (`assert_sample_moments`) or derive the
  tolerance, with its provenance in a comment. Don't just widen `atol`.
- **Too slow / timeout**: mark it `slow`, shrink the problem, or raise its
  `@pytest.mark.timeout(...)` with a reason.

Never skip, disable or loosen a contract test to get green.

## 4. Close the loop

Open the fix as a PR referencing the `ci-failure` issue ("Fixes #N"),
re-run the failing workflow on the branch
(`gh workflow run latest-deps.yml --ref <branch>` or
`gh workflow run tests-extended.yml --ref <branch> -f suite=full`), and
comment on the issue with the cause and the PR.
