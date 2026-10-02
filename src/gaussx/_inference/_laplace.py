r"""Precision-form Laplace approximation, the INLA inner loop: `laplace_mode`.

With $\eta = Ax + o$, $f(\eta) = \log p(y\mid\eta)$ (factorising over sites),
$g = f'$ and $W = -\operatorname{diag}(f'')$, the log posterior
$\ell(x) = f(Ax + o) - \tfrac12(x-\mu)^\top Q(x-\mu)$ has gradient
$A^\top g - Q(x-\mu)$ and negative Hessian $H = Q + A^\top WA$, so a Newton
step solves $H\,x_{t+1} = Q\mu + A^\top(g + WAx_t)$ with a matrix whose
pattern never changes. The mode is differentiated implicitly
(`jax.lax.custom_root`), and the Laplace log-marginal
$\log\tilde\pi(y\mid\theta) = f(A\hat x + o) + \log\pi(\hat x\mid\theta)
- \log\tilde\pi_G(\hat x\mid y,\theta)$ differentiates exactly through the
structured (or sparse, G4) log-determinants.
"""

from __future__ import annotations

import math
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from gaussx._distributions._gmrf import GaussianMRF, IntrinsicGMRF, _scaled, _unwrap
from gaussx._einx import einsum, rearrange
from gaussx._inference._inference import newton_update
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._sparse import SparseOperator
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._diag import diag
from gaussx._primitives._logdet import logdet
from gaussx._primitives._solve import solve
from gaussx._quadrature._likelihood import AbstractLikelihood
from gaussx._strategies._sparse_cholesky import SparseCholeskySolver


_LOG_2PI = math.log(2.0 * math.pi)
_PSD = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
# ``newton_update``'s default: keeps W positive for non-log-concave sites.
_PRECISION_FLOOR = 1e-6


class LaplaceResult(eqx.Module):
    r"""The Gaussian approximation $\mathcal N(\hat x, H^{-1})$ at the mode.

    Attributes:
        mode: $\hat x$, shape ``(N,)``. Implicitly differentiable in every
            array the prior and likelihood hold ($\theta$).
        hessian: $H = Q + A^\top WA$ at the mode: a `BlockTriDiag`,
            `SparseOperator`, diagonal or dense operator like $Q$ where the
            projector allows (see `laplace_mode`).
        factor: The factorisation of $H$, with ``solve(b)`` and ``logdet()``:
            a `SparseCholeskyFactor` (whose ``diag_inv`` gives the marginal
            variances by Takahashi) for a sparse $H$, else the structured
            Cholesky factor's solves.
        log_marginal: $\log\tilde\pi(y\mid\theta)$; see `laplace_mode` for
            the constants it includes.
        n_iter: Newton steps taken.
        converged: Whether the last step moved $x$ by less than the
            tolerance. Implicit gradients assume it; re-run or discard a
            $\theta$ where it is ``False``.
    """

    mode: Float[Array, " N"]
    hessian: lx.AbstractLinearOperator
    factor: Any
    log_marginal: Float[Array, ""]
    n_iter: Int[Array, ""]
    converged: Bool[Array, ""]


# ---------------------------------------------------------------------------
# Factorisations of H
# ---------------------------------------------------------------------------


class _CholeskyFactor(eqx.Module):
    """``H = L Lᵀ`` with a structured lower factor (block, diagonal or dense)."""

    lower: lx.AbstractLinearOperator

    def solve(self, b: Float[Array, " N"]) -> Float[Array, " N"]:
        return solve(self.lower.T, solve(self.lower, b))

    def logdet(self) -> Float[Array, ""]:
        return 2.0 * jnp.sum(jnp.log(diag(self.lower)))


class _OperatorFactor(eqx.Module):
    """No factor: ``gaussx.solve`` / ``gaussx.logdet`` dispatch on ``H`` itself."""

    operator: lx.AbstractLinearOperator

    def solve(self, b: Float[Array, " N"]) -> Float[Array, " N"]:
        return solve(self.operator, b)

    def logdet(self) -> Float[Array, ""]:
        return logdet(self.operator)


_CHOLESKY = (BlockTriDiag, lx.MatrixLinearOperator, lx.DiagonalLinearOperator)


def _factor(H: lx.AbstractLinearOperator) -> Any:
    if isinstance(H, SparseOperator):
        # The symbolic analysis is cached per pattern: every Newton step and
        # every θ reuses it, and only the numeric factorisation runs.
        return SparseCholeskySolver().factor(H)
    if isinstance(H, _CHOLESKY):
        return _CholeskyFactor(cholesky(H))
    return _OperatorFactor(H)


# ---------------------------------------------------------------------------
# H = Q + Aᵀ W A, values only on a fixed structure
# ---------------------------------------------------------------------------


def _is_selection(A: lx.AbstractLinearOperator | None) -> bool:
    """At most one stored entry per row of a sparse ``A``: ``AᵀWA`` is diagonal."""
    if not isinstance(A, SparseOperator):
        return False
    rows, _, _ = A.pattern._full
    return bool(np.all(np.bincount(rows, minlength=A.pattern.shape[0]) <= 1))


def _selection_diagonal(A: SparseOperator, w: Float[Array, " M"]) -> Array:
    """``diag(AᵀWA) = Σ_k w_k A_kj²`` for one entry per row."""
    rows, cols, _ = A.pattern._full
    values = A._full_values()
    return jax.ops.segment_sum(
        w[rows] * values**2, cols, num_segments=A.pattern.shape[1]
    )


def _add_diagonal(Q: lx.AbstractLinearOperator, d: Float[Array, " N"]):
    """``Q + diag(d)``, keeping a sparse, banded, diagonal or dense structure."""
    if isinstance(Q, SparseOperator):
        return Q.add_diagonal(d, tags=_PSD)
    if isinstance(Q, BlockTriDiag):
        size = Q._block_size
        blocks = rearrange(d, "(n a) -> n a", a=size)
        eye = jnp.eye(size, dtype=d.dtype)
        added = einx.multiply("n a, a b -> n a b", blocks, eye)
        return BlockTriDiag(
            Q.diagonal + added, Q.sub_diagonal, symmetric=Q.symmetric, tags=_PSD
        )
    if isinstance(Q, lx.DiagonalLinearOperator):
        return lx.DiagonalLinearOperator(Q.diagonal + d)
    if isinstance(Q, lx.MatrixLinearOperator):
        return lx.MatrixLinearOperator(Q.matrix + jnp.diag(d), _PSD)
    return lx.TaggedLinearOperator(
        lx.AddLinearOperator(Q, lx.DiagonalLinearOperator(d)), _PSD
    )


def _as_sparse(Q: lx.AbstractLinearOperator) -> SparseOperator | None:
    """A banded or diagonal ``Q`` as a symmetric `SparseOperator` (values traced)."""
    if isinstance(Q, SparseOperator):
        return Q
    if isinstance(Q, lx.DiagonalLinearOperator):
        n = Q.in_size()
        return SparseOperator.from_coo(
            np.arange(n), np.arange(n), Q.diagonal, (n, n), symmetric=True
        )
    if isinstance(Q, BlockTriDiag) and Q.symmetric:
        nb, d = Q._num_blocks, Q._block_size
        # Host index arrays: the lower triangle of each diagonal block, then
        # every entry of each sub-diagonal block.
        i, j = np.tril_indices(d)
        start = np.arange(nb) * d
        diag_rows = np.add.outer(start, i).ravel()
        diag_cols = np.add.outer(start, j).ravel()
        diag_values = rearrange(Q.diagonal[:, i, j], "n k -> (n k)")
        a, b = np.divmod(np.arange(d * d), d)
        sub_rows = np.add.outer(start[1:], a).ravel()
        sub_cols = np.add.outer(start[:-1], b).ravel()
        sub_values = rearrange(Q.sub_diagonal[:, a, b], "n k -> (n k)")
        n = nb * d
        return SparseOperator.from_coo(
            np.concatenate([diag_rows, sub_rows]),
            np.concatenate([diag_cols, sub_cols]),
            jnp.concatenate([diag_values, sub_values]),
            (n, n),
            symmetric=True,
        )
    return None


def _hessian(
    Q: lx.AbstractLinearOperator,
    A: lx.AbstractLinearOperator | None,
    w: Float[Array, " M"],
) -> lx.AbstractLinearOperator:
    """``H = Q + Aᵀ diag(w) A`` on a structure fixed by ``(Q, A)`` alone."""
    if A is None:
        return _add_diagonal(Q, w)
    if _is_selection(A):
        assert isinstance(A, SparseOperator)
        return _add_diagonal(Q, _selection_diagonal(A, w))
    sparse = _as_sparse(Q) if isinstance(A, SparseOperator) else None
    if sparse is not None:
        assert isinstance(A, SparseOperator)
        return sparse.union(sparse.congruence(A, w), tags=_PSD)
    if isinstance(Q, lx.MatrixLinearOperator) or not isinstance(A, SparseOperator):
        dense = A.as_matrix()
        update = einsum(
            einx.multiply("k i, k -> k i", dense, w), dense, "k i, k j -> i j"
        )
        return lx.MatrixLinearOperator(Q.as_matrix() + update, _PSD)
    root = lx.DiagonalLinearOperator(jnp.sqrt(w)) @ A
    return lx.TaggedLinearOperator(lx.AddLinearOperator(Q, root.T @ root), _PSD)


# ---------------------------------------------------------------------------
# The model pieces
# ---------------------------------------------------------------------------


class _Model(eqx.Module):
    """``(Q, μ, A, o, likelihood, V)`` with the maps the Newton loop needs."""

    precision: lx.AbstractLinearOperator
    loc: Float[Array, " N"]
    projector: lx.AbstractLinearOperator | None
    offset: Float[Array, " M"] | Float[Array, ""]
    likelihood: AbstractLikelihood
    null_space: Float[Array, "N c"] | None

    def project(self, x: Array) -> Array:
        return x if self.projector is None else self.projector.mv(x)

    def project_t(self, r: Array) -> Array:
        return r if self.projector is None else self.projector.transpose().mv(r)

    def grad(self, x: Float[Array, " N"]) -> Float[Array, " N"]:
        """``∇l(x) = Aᵀ f'(Ax + o) − Q(x − μ)``."""
        g = jax.grad(self.likelihood.log_prob)(self.project(x) + self.offset)
        return self.project_t(g) - self.precision.mv(x - self.loc)

    def hessian(self, x: Float[Array, " N"]) -> tuple[Array, lx.AbstractLinearOperator]:
        """``(nat1, H)`` at ``x``: ``nat1 = g + W A x`` and ``H = Q + AᵀWA``."""
        z = self.project(x)
        g, h = self.likelihood.site_derivatives(z + self.offset)
        nat1, w = newton_update(z, g, h, precision_floor=_PRECISION_FLOOR)
        return nat1, _hessian(self.precision, self.projector, w)

    # -- the hard constraint Vᵀx = 0 ----------------------------------------

    def null_component(self, x: Array) -> Array:
        """``V (VᵀV)⁻¹ Vᵀ x`` (zero without a constraint)."""
        if self.null_space is None:
            return jnp.zeros_like(x)
        V = self.null_space
        gram = einsum(V, V, "n c, n d -> c d")
        coeffs = jnp.linalg.solve(gram, einsum(V, x, "n c, n -> c"))
        return einsum(V, coeffs, "n c, c -> n")

    def krige(self, factor: Any, z: Array) -> tuple[Array, Array | None]:
        """``z − U S⁻¹ Vᵀz`` with ``U = H⁻¹V``, ``S = VᵀH⁻¹V`` (and ``S``)."""
        if self.null_space is None:
            return z, None
        Vt = rearrange(self.null_space, "n c -> c n")
        U = jax.vmap(factor.solve)(Vt)
        S = einsum(Vt, U, "c n, d n -> c d")
        coeffs = jnp.linalg.solve(S, einsum(Vt, z, "c n, n -> c"))
        return z - einsum(coeffs, U, "c, c n -> n"), S

    def newton(self, x: Float[Array, " N"]) -> Float[Array, " N"]:
        """The (constrained) Newton target ``krige(H⁻¹(Qμ + Aᵀ nat1))``."""
        nat1, H = self.hessian(x)
        factor = _factor(H)
        rhs = self.precision.mv(self.loc) + self.project_t(nat1)
        return self.krige(factor, factor.solve(rhs))[0]


def _stop_gradient(tree: Any) -> Any:
    dynamic, static = eqx.partition(tree, eqx.is_inexact_array)
    return eqx.combine(jax.lax.stop_gradient(dynamic), static)


@eqx.filter_jit
def _newton_loop(
    model: _Model,
    x0: Float[Array, " N"],
    max_iter: int,
    tol: float,
    damping: float,
) -> tuple[Array, Array, Array]:
    """Damped Newton to ``‖Δx‖∞ ≤ tol (1 + ‖x‖∞)``: ``(x, n_iter, converged)``."""
    eps = float(jnp.finfo(x0.dtype).eps)
    tol = max(tol, 64.0 * eps)

    def threshold(x: Array) -> Array:
        return tol * (1.0 + jnp.max(jnp.abs(x)))

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        x, i, delta = carry
        return (i < max_iter) & (delta > threshold(x))

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        x, i, _ = carry
        x_next = x + damping * (model.newton(x) - x)
        return x_next, i + 1, jnp.max(jnp.abs(x_next - x))

    init = (x0, jnp.asarray(0), jnp.asarray(jnp.inf, dtype=x0.dtype))
    x, n_iter, delta = jax.lax.while_loop(cond, body, init)
    return x, n_iter, delta <= threshold(x)


def _implicit_mode(model: _Model, x_star: Float[Array, " N"]) -> Float[Array, " N"]:
    r"""``x_star``, with tangents ``∂x̂ = H⁻¹ ∂_θ∇l`` by the implicit function theorem.

    The root function is ``∇l(x)``, or with the hard constraint
    ``P∇l(Px) − Π_V x`` (``P = I − Π_V``), whose roots are the constrained
    modes and whose Jacobian ``−(PHP + Π_V)`` is symmetric. Its tangent
    solve uses one factor of ``H`` at the mode (kriged for the constraint),
    wrapped in `jax.lax.custom_linear_solve` so that higher derivatives see
    ``H``'s dependence on θ through the matvec.
    """
    constrained = model.null_space is not None

    def root(x: Array) -> Array:
        if not constrained:
            return model.grad(x)
        inner = x - model.null_component(x)
        grad = model.grad(inner)
        return grad - model.null_component(grad) - model.null_component(x)

    fixed = _stop_gradient(model)
    factor = _factor(fixed.hessian(jax.lax.stop_gradient(x_star))[1])

    def neg_inverse(_: Any, b: Array) -> Array:
        null = fixed.null_component(b)
        inner = fixed.krige(factor, factor.solve(b - null))[0]
        return -(inner + null)

    def tangent_solve(matvec: Any, b: Array) -> Array:
        return jax.lax.custom_linear_solve(matvec, b, neg_inverse, symmetric=True)

    return jax.lax.custom_root(
        root, jax.lax.stop_gradient(x_star), lambda _, x0: x0, tangent_solve
    )


def _model(
    prior: GaussianMRF | IntrinsicGMRF,
    likelihood: AbstractLikelihood,
    projector: lx.AbstractLinearOperator | None,
    offset: ArrayLike | None,
) -> _Model:
    if isinstance(prior, GaussianMRF):
        precision = _unwrap(prior.precision)
        null_space = None
    elif isinstance(prior, IntrinsicGMRF):
        if prior.constraint == "soft":
            raise NotImplementedError(
                "laplace_mode supports IntrinsicGMRF with constraint='hard' or "
                "'none'; the soft constraint's V Vᵀ / s² term is dense."
            )
        precision = _unwrap(_scaled(prior.structure, prior.precision_scale))
        null_space = prior.null_space if prior.constraint == "hard" else None
    else:
        raise TypeError(
            "laplace_mode needs a GaussianMRF or IntrinsicGMRF prior, got "
            f"{type(prior).__name__}."
        )
    n = prior.event_shape[0]
    if projector is not None and projector.in_size() != n:
        raise ValueError(
            f"projector must have {n} columns, got shape "
            f"({projector.out_size()}, {projector.in_size()})."
        )
    dtype = prior.loc.dtype
    offset = jnp.zeros((), dtype=dtype) if offset is None else jnp.asarray(offset)
    return _Model(precision, prior.loc, projector, offset, likelihood, null_space)


def laplace_mode(
    prior: GaussianMRF | IntrinsicGMRF,
    likelihood: AbstractLikelihood,
    *,
    projector: lx.AbstractLinearOperator | None = None,
    offset: Float[ArrayLike, " M"] | Float[ArrayLike, ""] | None = None,
    init: Float[ArrayLike, " N"] | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    damping: float = 1.0,
) -> LaplaceResult:
    r"""Mode, Hessian and log-marginal of the Laplace approximation, in precision form.

    The latent Gaussian model is $x \sim$ ``prior`` (precision $Q$, mean
    $\mu$), $\eta = Ax + o$ and $y \mid \eta \sim$ ``likelihood``, which
    holds $y$ and must factorise over sites (its Hessian in $\eta$ is
    diagonal). Newton on $\ell(x) = \log p(y\mid Ax+o) + \log\pi(x)$:

    $$
    H_t = Q + A^\top W_tA,\qquad
    H_t\,\tilde x = Q\mu + A^\top(g_t + W_tAx_t),\qquad
    x_{t+1} = x_t + \text{damping}\,(\tilde x - x_t),
    $$

    with $g_t$, $W_t = \max(-f''_t, 10^{-6})$ from the likelihood's
    `site_derivatives` (`newton_update`'s diagonal path), until
    $\|x_{t+1}-x_t\|_\infty \le \text{tol}\,(1 + \|x_{t+1}\|_\infty)$
    (``tol`` is floored at 64 machine epsilons of the dtype). $H_t$ is
    rebuilt values-only on a structure fixed by $Q$ and $A$:

    - $A$ the identity (``None``) or one entry per row (a row selection,
      or a scaled one): $A^\top WA$ is diagonal, so a `BlockTriDiag` $Q$
      (RW1, RW2, AR(1)) stays `BlockTriDiag`, and a `SparseOperator`, diagonal
      or dense $Q$ keeps its class;
    - a general `SparseOperator` $A$ (a FEM projector, an intercept and
      covariates): `SparseOperator.congruence` on the union pattern, with a
      banded or diagonal $Q$ converted to sparse;
    - a dense $A$ or $Q$: dense; any other $Q$ (a grid `SpectralFunction` or
      `KroneckerSum` with a non-diagonal update): the matrix-free sum, solved
      and log-determined by `gaussx.solve` / `gaussx.logdet` dispatch.

    A sparse $H$ is factored by `SparseCholeskySolver`, whose symbolic
    analysis is cached per pattern and shared by every step and every
    $\theta$. A Gaussian likelihood converges in one step (a second confirms
    it).

    **Intrinsic priors.** An `IntrinsicGMRF` contributes $Q = \tau R$. With
    ``constraint="hard"`` the field lives on $V^\top x = 0$ ($V$ =
    ``null_space``): each Newton solve is kriged onto it (``c`` extra
    solves with the same factor), and the Gaussian approximation is the
    constrained density (Rue et al., 2009, eq. 3; Rue & Held, 2005,
    eq. 2.30). $H$ must be positive definite on the whole space, which holds
    when the observations see $\ker R$ (an RW2 observed at two or more
    times, a Besag field observed in every connected component).
    ``constraint="none"`` runs unconstrained; ``"soft"`` is not supported.

    **The log-marginal.** At the mode $\hat x$,
    $\log\tilde\pi(y\mid\theta) = \log p(y\mid A\hat x + o) + \log\pi(\hat x)
    - \log\tilde\pi_G(\hat x)$, where $\log\pi$ is the prior's ``log_prob``
    and

    $$
    \log\tilde\pi_G(\hat x) = \tfrac12\log|H| - \tfrac N2\log 2\pi
        \;\Big[+ \tfrac12\log|V^\top H^{-1}V| - \tfrac12\log|V^\top V|
        + \tfrac c2\log 2\pi\Big],
    $$

    the bracket for the hard constraint. For a `GaussianMRF` this is
    $f(A\hat x + o) - \tfrac12(\hat x-\mu)^\top Q(\hat x-\mu)
    + \tfrac12\log|Q| - \tfrac12\log|H|$, the exact log-evidence for a
    Gaussian likelihood. For an `IntrinsicGMRF` it is exact with
    ``include_normalizer=True`` and otherwise off by the θ-free constant
    $\tfrac12\log|R|_+ - \tfrac{N-c}2\log 2\pi$. The likelihood's own
    normalising terms are included.

    **Gradients.** Every array in ``prior``, ``likelihood``, ``projector``
    and ``offset`` is part of $\theta$. The mode is differentiated by the
    implicit function theorem through `jax.lax.custom_root`
    ($\partial_\theta\hat x = H^{-1}\partial_\theta\nabla\ell$, one solve
    with a factor of $H$ at the mode), and the log-determinants through
    their structured or sparse (Takahashi) VJPs. Reverse-over-reverse
    (``jax.jacrev(jax.jacrev(...))``, `theta_design`'s default Hessian)
    is exact too. The loop itself is not differentiated, so gradients assume
    ``converged``.

    Args:
        prior: The latent field, a `GaussianMRF` or `IntrinsicGMRF`.
        likelihood: A site-factorising `AbstractLikelihood` holding $y$
            (`PoissonLikelihood`, `BinomialLikelihood`,
            `NegativeBinomialLikelihood`, `BernoulliLikelihood`,
            `GaussianLikelihood`, ...).
        projector: $A$, shape ``(M, N)``; ``None`` is the identity.
        offset: $o$, a scalar or shape ``(M,)`` (e.g. log expected counts).
        init: Starting point, shape ``(N,)``; zeros by default.
        max_iter: Maximum Newton steps.
        tol: Relative tolerance on the step's sup-norm.
        damping: Step size in ``(0, 1]``; below 1 for hard, non-log-concave
            problems.

    Returns:
        The `LaplaceResult`.

    Raises:
        TypeError: For another prior class.
        NotImplementedError: For an `IntrinsicGMRF` with
            ``constraint="soft"``.
        ValueError: If ``projector`` does not have ``N`` columns.

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np
        import gaussx as gx

        # Poisson counts with an RW2 seasonal effect; θ = log τ. Even n keeps
        # rw2_structure(n) at size n (odd n adds a padding node: see below).
        n = 60
        t = jnp.arange(n, dtype=float)
        counts = jnp.asarray(np.random.default_rng(0).poisson(np.exp(np.sin(t / 9.0))))
        null = jnp.column_stack([jnp.ones(n), t - t.mean()])

        def log_marginal(log_tau):
            prior = gx.IntrinsicGMRF(
                jnp.zeros(n), jnp.exp(log_tau), gx.rw2_structure(n), null
            )
            result = gx.laplace_mode(prior, gx.PoissonLikelihood(counts))
            return result.log_marginal  # H stays BlockTriDiag

        value, grad = jax.value_and_grad(log_marginal)(2.0)  # exact

        # Odd n: rw2_structure(n) has n + 1 rows. Observe the first n nodes
        # through a row selection (H stays BlockTriDiag), with the null
        # space zero on the padding node.
        m = 61
        s = jnp.arange(m, dtype=float)
        V = jnp.zeros((m + 1, 2)).at[:m, 0].set(1.0).at[:m, 1].set(s - s.mean())
        select = gx.SparseOperator.from_coo(
            np.arange(m), np.arange(m), jnp.ones(m), (m, m + 1)
        )
        prior = gx.IntrinsicGMRF(jnp.zeros(m + 1), 10.0, gx.rw2_structure(m), V)
        y = jnp.ones(m)
        fit = gx.laplace_mode(prior, gx.PoissonLikelihood(y), projector=select)
        trend = fit.mode[:m]  # drop the padding node
        ```
    """
    model = _model(prior, likelihood, projector, offset)
    n = prior.event_shape[0]
    dtype = prior.loc.dtype
    x0 = jnp.zeros(n, dtype=dtype) if init is None else jnp.asarray(init, dtype=dtype)
    x0 = x0 - model.null_component(x0)

    x_star, n_iter, converged = _newton_loop(
        _stop_gradient(model), jax.lax.stop_gradient(x0), max_iter, tol, damping
    )
    mode = _implicit_mode(model, x_star)

    _, H = model.hessian(mode)
    factor = _factor(H)
    eta = model.project(mode) + model.offset
    log_gaussian = 0.5 * factor.logdet() - 0.5 * n * _LOG_2PI
    _, S = model.krige(factor, mode)
    if S is not None:
        V = model.null_space
        assert V is not None
        c = V.shape[1]
        gram = einsum(V, V, "n c, n d -> c d")
        log_gaussian = (
            log_gaussian
            + jnp.sum(jnp.log(jnp.diagonal(jnp.linalg.cholesky(S))))
            - 0.5 * jnp.linalg.slogdet(gram)[1]
            + 0.5 * c * _LOG_2PI
        )
    log_marginal = likelihood.log_prob(eta) + prior.log_prob(mode) - log_gaussian
    return LaplaceResult(mode, H, factor, log_marginal, n_iter, converged)
