# Unified Solvers in gaussx — Design & Execution Plan

**Status:** gaussx side (Phases 0–3) implemented; PDE-package wrappers in progress
**Repos in scope:** `gaussx` (hub), `spectraldiffx`, `finitevolX`

---

## 1. Motivation

Three JAX libraries have grown overlapping numerical-solver machinery:

- **gaussx** — structured *probabilistic* linear algebra. Solves `A x = b` and
  `logdet(A)` for PSD covariance/precision operators (CG, MINRES, LSMR,
  preconditioned CG, BBMM, SLQ). Built on `lineax` + `matfree` + `equinox`.
- **finitevolX** — finite-volume operators on Arakawa C-grids. Re-implements its
  own CG loop (`solve_cg`), three preconditioners (spectral, Nyström,
  multigrid), tridiagonal solves, and a masked-Laplacian operator.
- **spectraldiffx** — pseudospectral PDE discretization. Owns spectral
  Helmholtz/Poisson solvers and a **capacitance-matrix** solver for masked
  irregular domains (a low-rank correction around a fast base solve).

The duplication is concentrated in the **raw linear-solver + preconditioner
layer**. The same CG loop, the same Nyström preconditioner idea, and the same
Sherman–Morrison/Woodbury correction (capacitance) exist in two or three places,
each separately tested and separately maintained.

There is also a deeper **mathematical** reason to unify (see §3): elliptic PDE
operators and Gaussian covariance/precision operators are, under the SPDE view,
*the same objects*. gaussx already models a chunk of this (`Toeplitz`,
`KroneckerSum`, `_ssm/_matern`, `_sde_kernel`, `spingp`).

### Goals (ranked, from the scoping discussion)

1. **Kill duplicated CG / preconditioner code** — one tested implementation.
2. **Easier maintenance / testing** — solver numerics tested once, in gaussx.
3. **Mathematical unification (SPDE / GP ↔ PDE)** — elliptic operators and
   covariance operators share the same solver substrate.

### Non-goals / explicit scope boundaries

gaussx must remain *structured ops + raw solvers*. It will **not** acquire:

- grids, coordinates, or grid spacing as first-class concepts;
- boundary-condition modelling (Dirichlet/Neumann/periodic dispatch);
- spectral transforms (FFT / DST / DCT / SHT);
- mask → boundary-index extraction (which cells are land/ocean/boundary);
- multigrid V-cycles or spectral Helmholtz field solvers.

Those stay in `finitevolX` / `spectraldiffx`. They cross the boundary only as
**passed-in callables/operators** (e.g. "here is an approximate inverse, use it
as a preconditioner").

---

## 2. User stories

- **As a finitevolX user**, I call `fvx.solve_cg(matvec, rhs, preconditioner=...)`
  exactly as before, but the CG loop and the Nyström preconditioner are now
  gaussx code — one implementation, fixed and tuned in one place.

- **As a spectraldiffx user**, I call `build_capacitance_solver(mask, dx, dy)`
  as before; internally it extracts the boundary ring (spectral concern) and
  hands gaussx the *base solve callable* + *boundary indices*, getting back a
  reusable capacitance operator.

- **As a gaussx user doing GP regression on a grid**, I can build a
  `NystromPreconditioner` or a `JacobiPreconditioner` and pass it straight into
  `CGSolver` / `PreconditionedCGSolver`, instead of preconditioning living only
  inside `PreconditionedCGSolver`.

- **As a maintainer**, when a CG edge case (e.g. zero rhs, NaN guard, max-steps
  behaviour) needs fixing, I fix it once in gaussx and all three libraries
  inherit it.

- **As a researcher bridging GP and PDE**, I can take an elliptic operator from a
  PDE package and a covariance operator from gaussx and solve both through the
  same `gaussx.linear_solve(A, b, solver=..., preconditioner=...)` front door.

---

## 3. Math background

### 3.1 The SPDE / GMRF bridge (why this is one problem, not two)

The Whittle–Matérn SPDE (Lindgren, Rue & Lindgren, 2011) states that a Gaussian
field `u` with Matérn covariance is the stationary solution of

```
(κ² − Δ)^{α/2} u(x) = 𝒲(x),     α = ν + d/2
```

where `𝒲` is white noise and `Δ` the Laplacian. Discretised, the **precision**
matrix `Q` of the Matérn field is a (power of a) **Helmholtz operator**
`κ²I − Δ`. Consequences relevant here:

- A **separable Laplacian** on a tensor grid is `Δ = Δ_x ⊕ Δ_y` — exactly
  gaussx's `KroneckerSum`, which it already solves by eigendecomposition.
- A **stationary kernel on a regular grid** is Toeplitz / BTTB = an FFT
  convolution — exactly gaussx's `Toeplitz` (circulant embedding) and exactly
  spectraldiffx's FFT-diagonalised Helmholtz inverse.
- A **Helmholtz / Laplacian operator** diagonalised in a spectral basis is a
  `diag-in-eigenbasis` operator: `solve` and `logdet` are elementwise on the
  eigenvalues — the same shape as gaussx's structured primitives.

So "the covariance operator" and "the elliptic operator" are the same algebraic
object viewed from two communities. Unifying the *solvers* is the engineering
expression of that fact.

### 3.2 Conjugate Gradient + preconditioning

For SPD `A`, CG solves `A x = b` in Krylov space. Convergence depends on
`κ(A) = λ_max/λ_min`. A preconditioner `M ≈ A` (cheap to invert) replaces the
spectrum of `A` with that of `M⁻¹A`. The four preconditioners in play:

| Preconditioner | `M⁻¹` is… | Generic? |
|---|---|---|
| Jacobi | `diag(A)⁻¹` | yes |
| Nyström / partial-Cholesky | low-rank approx inverse from a few matvecs | yes |
| Spectral | an FFT/DST/DCT Helmholtz solve used as approx inverse | **no** (needs transforms) |
| Multigrid | one V-cycle used as approx inverse | **no** (needs grid/mask) |

The generic two move into gaussx as concrete classes. The PDE-specific two are
*adapted* into gaussx's preconditioner protocol via a generic
`OperatorPreconditioner(approx_inverse)` wrapper — the concrete approximate
inverse is built in the PDE repo and passed in. This is what dodges the
dependency cycle (gaussx must never import spectraldiffx/finitevolX).

### 3.3 Capacitance matrix method (Sherman–Morrison / Woodbury)

To solve `A u = f` on an irregular domain Ω embedded in a regular domain R for
which a *fast* solver `B⁻¹` exists (FFT/DST/DCT):

1. Treat the `N_b` boundary unknowns as constraints. Define Green's functions
   `g_b = B⁻¹ e_b` for each boundary index `b` (one fast solve each, **offline**).
2. Form the **capacitance matrix** `C` = restriction of the Green's functions to
   the boundary rows; precompute `C⁻¹` (small, `N_b × N_b`).
3. **Online:** `u = B⁻¹f − G (C⁻¹ (R_b B⁻¹ f − target))`, i.e. one fast base
   solve plus an `N_b`-sized correction.

This is precisely a **Woodbury / low-rank update** around the base solve — and
gaussx *already* has `LowRankUpdate`, `SVDLowRankUpdate`, Woodbury and Schur
machinery. The generic algebra (given `B⁻¹` and the boundary index set) belongs
in gaussx; *which* cells are boundary and *which* `B⁻¹` to use stay in the PDE
repos.

---

## 4. Current state

### 4.1 gaussx (`src/gaussx/`)

- `_strategies/_base.py` — `AbstractSolveStrategy.solve(operator, vector)`,
  `AbstractLogdetStrategy.logdet(operator, *, key)`, `AbstractSolverStrategy`.
  All `equinox.Module`, operate on `lineax.AbstractLinearOperator`.
- `_strategies/{_cg,_minres,_lsmr,_dense,_bbmm,_precond_cg,_slq_logdet,_auto,
  _composed}.py` — concrete strategies. **PSD-assuming.**
- `_strategies/_dispatch.py` — `dispatch_solve` / `dispatch_logdet` (fall back to
  primitives when `solver is None`).
- `_precond_cg.py` — preconditioning is **buried inside** `PreconditionedCGSolver`
  (partial pivoted Cholesky via `matfree.low_rank`). Not reusable standalone.
- `_operators/` — `Kronecker`, `KroneckerSum`, `BlockDiag`, `BlockTriDiag`,
  `LowRankUpdate`, `SVDLowRankUpdate`, `Toeplitz`, kernel operators, lazy algebra.
- `_linalg/_woodbury.py`, `_schur.py` — low-rank correction algebra (capacitance
  building blocks already exist here).
- **No** raw-`matvec` front door, **no** standalone preconditioner objects,
  **no** capacitance operator, **no** indefinite/negative-definite front-door
  handling.

### 4.2 finitevolX (`finitevolx/_src/solvers/`)

- `iterative.py` — `solve_cg(matvec, rhs, x0, preconditioner, rtol, atol,
  max_steps) -> (x, CGInfo)`, `CGInfo(iterations, residual_norm, converged)`,
  `masked_laplacian(psi, mask, dx, dy, lambda_)`. Operates on **raw-array matvec**
  for **negative-definite** masked Laplacians.
- `preconditioners.py` — `make_preconditioner(kind, ...)`,
  `make_spectral_preconditioner(dx, dy, lambda_, bc)`,
  `make_nystrom_preconditioner(matvec, shape, rank, key)`,
  `make_multigrid_preconditioner(mg_solver)`.
- `tridiagonal.py` — `solve_tridiagonal`, `solve_tridiagonal_batched` (thin
  wrappers over `lineax.Tridiagonal`).
- `multigrid.py` — geometric V-cycle, Arakawa-C face staggering, mask
  restriction/prolongation. **Stays in finitevolX.**
- `spectral.py`, `spectral_transforms.py` — pure re-exports of spectraldiffx.
- `elliptic.py` — convenience wrappers (`streamfunction_from_vorticity`,
  `pressure_from_divergence`, `pv_inversion`) + `build_capacitance_solver`.
- `inhomogeneous.py` — lifting trick (mask/BC concern). **Stays.**
- Depends on `spectraldiffx`, `lineax`, `equinox`, `scipy`, `diffrax`.

### 4.3 spectraldiffx (`spectraldiffx/_src/`)

- `fourier/solvers.py` — `solve_helmholtz_*` / `solve_poisson_*` pure functions +
  `SpectralHelmholtzSolver{1,2,3}D` modules. **Stays.**
- `fourier/eigenvalues.py` — pure 1-D Laplacian eigenvalue functions. Stays
  (spectral-native), portable in principle.
- `fourier/capacitance.py` — `CapacitanceSolver(eqx.Module)` with `_C_inv`,
  `_green_flat`, `_mask`, `_j_b`, `_i_b`, `dx`, `dy`, `lambda_`, `base_bc`;
  `build_capacitance_solver(mask, dx, dy, lambda_, base_bc)`. **Tightly coupled to
  the base spectral solve + mask boundary extraction.** This is the split target.
- `chebyshev/solvers.py`, `spherical/solvers.py` — collocation / SHT mode
  inversion. **Stay** (transform/grid coupled).
- Depends on `jax`, `equinox`, `jaxtyping`, `finitediffx`, `kernex`.

---

## 5. Target state

### 5.1 gaussx — new/extended surface

```
src/gaussx/
  _operators/
    _capacitance.py          # NEW: CapacitanceSolver (Woodbury-family op)
  _preconditioners/          # NEW package
    __init__.py
    _base.py                 # AbstractPreconditioner protocol
    _jacobi.py               # JacobiPreconditioner
    _nystrom.py              # NystromPreconditioner
    _partial_cholesky.py     # PartialCholeskyPreconditioner (factored from _precond_cg)
    _operator.py             # OperatorPreconditioner  (the spectral/MG adapter slot)
  _strategies/
    _cg.py                   # EXTENDED: optional `preconditioner` field
    _minres.py               # EXTENDED: optional `preconditioner` field
    _precond_cg.py           # REFACTORED to delegate to PartialCholeskyPreconditioner
  _solve_frontend.py         # NEW: as_linear_operator(), linear_solve()
  _solvers_functional.py     # NEW: solve_tridiagonal(_batched) re-home (optional)
```

Public API additions (re-exported from `__init__.py`):

- `gaussx.as_linear_operator(matvec, *, in_structure | shape, symmetric=False,
  positive_semidefinite=False, negative_definite=False)`
- `gaussx.linear_solve(A, b, *, solver=None, preconditioner=None)` — accepts an
  operator **or** a raw `(matvec, shape)`; handles sign tags.
- `gaussx.AbstractPreconditioner`, `JacobiPreconditioner`, `NystromPreconditioner`,
  `PartialCholeskyPreconditioner`, `OperatorPreconditioner`
- `gaussx.CapacitanceSolver`
- `gaussx.solve_tridiagonal`, `gaussx.solve_tridiagonal_batched`

### 5.2 finitevolX — thin wrappers

- `iterative.solve_cg` → delegates to `gaussx.linear_solve` (keeps `CGInfo`
  return shape and the negative-definite convention).
- `preconditioners.make_nystrom_preconditioner` → `gaussx.NystromPreconditioner`.
- `make_spectral_preconditioner` → builds the spectral approx-inverse callable
  (spectraldiffx), wraps it in `gaussx.OperatorPreconditioner`.
- `make_multigrid_preconditioner(mg)` → `gaussx.OperatorPreconditioner(mg)`.
- `tridiagonal.*` → re-export `gaussx.solve_tridiagonal*`.
- `multigrid.py`, `masked_laplacian`, `inhomogeneous.py`, `elliptic.py`
  convenience wrappers — unchanged behaviour; only their solver internals are
  delegated.

### 5.3 spectraldiffx — thin wrapper

- `fourier/capacitance.build_capacitance_solver` → extracts boundary indices from
  the mask (spectral concern), defines `base_solve = λ rhs: solve_helmholtz_<bc>
  (rhs, dx, dy, lambda_)`, and constructs `gaussx.CapacitanceSolver(base_solve,
  boundary_indices, n)`. The returned object reshapes field ↔ flat.
- `solve_helmholtz_*`, eigenvalues, Chebyshev, spherical — unchanged.

### 5.4 Dependency direction (must stay acyclic)

```
finitevolX ──depends-on──▶ spectraldiffx ──depends-on──▶ gaussx
        └───────────────depends-on──────────────────────▶ gaussx
```

gaussx depends on **neither** of the other two. Spectral/MG specifics enter
gaussx solvers only as runtime callables/operators.

---

## 6. Demo API

### 6.1 Raw-matvec front door + sign handling (gaussx)

```python
import gaussx
import jax.numpy as jnp


# A negative-definite 5-point Laplacian as a raw matvec (finitevolX style)
def laplacian(x):  # x: (N,) flattened field
    ...
    return Lx


A = gaussx.as_linear_operator(
    laplacian, shape=(N, N), symmetric=True, negative_definite=True
)

x = gaussx.linear_solve(A, b, solver=gaussx.CGSolver())  # routes neg-def correctly
```

### 6.2 Standalone preconditioners (gaussx)

```python
# Generic, gaussx-native
P_jac = gaussx.JacobiPreconditioner(diagonal=A.diagonal())
P_nys = gaussx.NystromPreconditioner(matvec=A.mv, shape=(N, N), rank=50, key=key)

x = gaussx.linear_solve(A, b, solver=gaussx.CGSolver(), preconditioner=P_nys)

# Adapter slot: any approximate-inverse operator/callable becomes a preconditioner
P_spec = gaussx.OperatorPreconditioner(
    my_spectral_inverse
)  # callable from spectraldiffx
P_mg = gaussx.OperatorPreconditioner(my_multigrid_solver)  # Mg object from finitevolX
```

### 6.3 Capacitance operator (gaussx core, PDE repos supply the parts)

```python
# base_solve: a fast regular-domain inverse (e.g. FFT Helmholtz) — passed in.
# boundary_indices: flat indices of the constrained boundary ring — passed in.
cap = gaussx.CapacitanceSolver(
    base_solve=base_solve,  # Callable[[Array[N]], Array[N]]
    boundary_indices=boundary_idx,  # Int[Array, "Nb"]
    n=N,
)
u = cap(rhs_flat)  # one base solve + Nb-sized correction
```

---

## 7. API usage examples (end-to-end, post-refactor)

### 7.1 finitevolX — unchanged call site, gaussx internals

```python
import finitevolx as fvx

# Public API identical to today
x, info = fvx.solve_cg(
    matvec=lambda v: fvx.masked_laplacian(v.reshape(Ny, Nx), mask, dx, dy).ravel(),
    rhs=b,
    preconditioner=fvx.make_preconditioner("nystrom", matvec=mv, shape=(Ny, Nx)),
    rtol=1e-6,
)
# info: CGInfo(iterations=..., residual_norm=..., converged=...)
# Internally: fvx.solve_cg -> gaussx.linear_solve; nystrom -> gaussx.NystromPreconditioner
```

### 7.2 spectraldiffx — capacitance wrapper

```python
import spectraldiffx as sdx

cap = sdx.build_capacitance_solver(mask, dx=dx, dy=dy, lambda_=0.0, base_bc="fft")
psi = cap(rhs)  # field in, field out (wrapper reshapes; gaussx does the algebra)
```

### 7.3 gaussx — GP on a grid reuses the same preconditioner

```python
import gaussx

K = gaussx.Toeplitz(first_col)  # stationary kernel on 1-D grid
P = gaussx.NystromPreconditioner(matvec=K.mv, shape=K.shape, rank=64, key=key)
alpha = gaussx.linear_solve(K + noise, y, solver=gaussx.CGSolver(), preconditioner=P)
```

---

## 8. Steps to execute

Each phase is independently shippable, lands on
`claude/gaussx-solver-refactor-scope-3hTFq` in the relevant repo, and keeps all
existing public APIs green. **gaussx phases (0–3) must merge/publish before the
wrapper phases (4–5).**

### Phase 0 — gaussx: raw-matvec front door + sign handling
- Add `src/gaussx/_solve_frontend.py`:
  - `as_linear_operator(matvec, *, in_structure | shape, symmetric,
    positive_semidefinite, negative_definite)` → wraps in
    `lx.FunctionLinearOperator` with the right lineax tags.
  - `linear_solve(A, b, *, solver=None, preconditioner=None)` — normalise `A`
    (operator or `(matvec, shape)`), route negative-definite via MINRES or
    negation, default `solver=AutoSolver()`.
- Re-export both from `__init__.py`.
- Tests: `tests/test_solve_frontend.py` — PSD via CG, indefinite via MINRES,
  negative-definite Laplacian solve, `(matvec, shape)` input parity with operator
  input.
- Acceptance: `gaussx.linear_solve` solves a known neg-def Laplacian to `rtol`.

### Phase 1 — gaussx: preconditioner protocol + generic preconditioners
- Add `src/gaussx/_preconditioners/`:
  - `_base.py` — `AbstractPreconditioner(eqx.Module)` with
    `as_operator() -> lx.AbstractLinearOperator` (and/or `__call__(v)`).
  - `_jacobi.py`, `_nystrom.py`, `_partial_cholesky.py` (factor the partial
    pivoted Cholesky out of `_precond_cg.py`), `_operator.py`
    (`OperatorPreconditioner(approx_inverse: Operator | Callable)`).
- Refactor `_strategies/_cg.py` and `_minres.py` to accept an optional
  `preconditioner: AbstractPreconditioner | None` field; wire into
  `lx.linear_solve(..., options={"preconditioner": P.as_operator()})`.
- Refactor `_precond_cg.py` to delegate to `PartialCholeskyPreconditioner`
  (behaviour-preserving; keep the class + its defaults).
- Re-export new classes.
- Tests: each preconditioner reduces CG iteration count on an ill-conditioned
  SPD system; `PreconditionedCGSolver` numerics unchanged (regression test).
- Acceptance: `_precond_cg` regression test passes byte-for-byte on outputs
  within tolerance; Nyström preconditioner cuts iterations measurably.

### Phase 2 — gaussx: capacitance operator
- Add `src/gaussx/_operators/_capacitance.py`:
  - `CapacitanceSolver(eqx.Module)` holding precomputed `C_inv`, Green's columns,
    `boundary_indices`, `n`. Constructor takes `base_solve: Callable`,
    `boundary_indices: Int[Array, "Nb"]`, `n: int`. Built on existing
    `_linalg/_woodbury.py` where possible.
  - `__call__(rhs_flat) -> sol_flat`.
- Re-export `gaussx.CapacitanceSolver`.
- Tests: reproduce a known masked-domain Poisson solution; verify it equals a
  dense reference solve on a small irregular mask; verify it equals
  spectraldiffx's current `CapacitanceSolver` output on a fixed fixture
  (port one spectraldiffx test as a golden file).
- Acceptance: matches dense reference to `1e-8` on a 16×16 masked grid.

### Phase 3 — gaussx: tridiagonal re-home (optional, low value)
- Add `solve_tridiagonal`, `solve_tridiagonal_batched` (thin `lineax.Tridiagonal`
  wrappers) and export. Skip if deemed not worth the surface.

### Phase 4 — spectraldiffx: adopt gaussx capacitance
- Add `gaussx` to `pyproject.toml` deps.
- Rewrite `fourier/capacitance.build_capacitance_solver` to:
  (a) extract boundary indices from the mask (unchanged logic),
  (b) build `base_solve` from the chosen `solve_helmholtz_<bc>`,
  (c) return a thin field↔flat wrapper around `gaussx.CapacitanceSolver`.
- Keep `CapacitanceSolver` name as a deprecated alias if external code imports it
  directly (one release of overlap).
- Tests: existing capacitance tests must pass unchanged.
- Acceptance: spectraldiffx capacitance test suite green against gaussx core.

### Phase 5 — finitevolX: adopt gaussx solvers/preconditioners
- Add/raise `gaussx` dependency in `pyproject.toml` (via spectraldiffx or direct).
- `iterative.solve_cg` → wrap `gaussx.linear_solve` (preserve `CGInfo`, neg-def
  convention, `x0`, tolerances, `max_steps`).
- `preconditioners.make_nystrom_preconditioner` → `gaussx.NystromPreconditioner`.
- `make_spectral_preconditioner` / `make_multigrid_preconditioner` → wrap their
  approximate inverses in `gaussx.OperatorPreconditioner`.
- `tridiagonal.*` → re-export gaussx functions (or keep local if Phase 3 skipped).
- `masked_laplacian`, `multigrid.py`, `inhomogeneous.py`, `elliptic.py` — leave
  behaviour; only swap solver internals.
- Tests: finitevolX solver tests pass unchanged; add a parity test (old local CG
  vs gaussx-backed CG on a fixed masked Laplacian).
- Acceptance: full finitevolX test suite green; numerical parity within tol.

### Phase 6 — cleanup, deprecation, docs
- Remove dead local implementations after one release of deprecation shims.
- Docs:
  - gaussx: new "Preconditioners" + "Capacitance" + "Raw-matvec front door"
    sections; SPDE bridge note linking covariance ↔ elliptic operators.
  - finitevolX / spectraldiffx: note that solver internals now come from gaussx.
- Cross-repo CHANGELOG entries.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Dependency cycle** (gaussx importing PDE repos) | Spectral/MG specifics enter only as runtime callables via `OperatorPreconditioner`; CI guard: gaussx imports must not reference spectraldiffx/finitevolX. |
| **Sign-convention bugs** (PSD vs neg-def Laplacian) | Explicit `negative_definite` tag on the front door; parity tests against existing finitevolX outputs. |
| **operator vs raw-matvec impedance** | `as_linear_operator` adapter + `linear_solve` accepting both; covered by Phase 0 tests. |
| **Behavioural drift in `PreconditionedCGSolver`** | Refactor is delegation-only; golden regression test before/after. |
| **Capacitance numerics regression** | Golden-file test ported from spectraldiffx; dense reference check. |
| **Release ordering** | gaussx (0–3) published first; PDE repos pin the new gaussx version. |
| **Scope creep back into gaussx** | §1 non-goals are normative; reviewers reject grid/BC/transform/mask code in gaussx. |

---

## 10. Definition of done

- gaussx exposes: raw-matvec front door, preconditioner protocol + Jacobi /
  Nyström / partial-Cholesky / operator-adapter, capacitance operator — all
  tested, no import of the PDE repos.
- finitevolX `solve_cg`, Nyström/spectral/multigrid preconditioners, and
  tridiagonal route through gaussx with unchanged public APIs and numerical
  parity.
- spectraldiffx `build_capacitance_solver` routes through
  `gaussx.CapacitanceSolver` with unchanged public API.
- All three test suites green on `claude/gaussx-solver-refactor-scope-3hTFq`.
- Dependency graph acyclic: `finitevolX → spectraldiffx → gaussx`.
