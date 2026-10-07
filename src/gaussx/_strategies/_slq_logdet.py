"""Standalone logdet strategies: SLQ, indefinite SLQ, and dense."""

from __future__ import annotations

import functools as ft
from collections.abc import Callable

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import matfree.stochtrace
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._einx import einsum
from gaussx._primitives._samplers import SamplerName, resolve_sampler
from gaussx._strategies._base import AbstractLogdetStrategy


def _lanczos(
    matvec: Callable[..., Array],
    v0: Float[Array, " n"],
    order: int,
    *parameters,
) -> tuple[Float[Array, " k"], Float[Array, " k"]]:
    """Fully reorthogonalised Lanczos that stops cleanly at breakdown (gh-520).

    Once the Krylov space is exhausted (``β_j ≤ sqrt(eps) · ‖T‖``) the
    remaining steps are padded: ``β = 0`` and ``α = 1``, a decoupled
    identity block that carries no quadrature weight. Every division and
    square root is double-``where`` guarded, so reverse mode through the
    breakdown is finite instead of ``0 · ∞ = NaN``.

    Args:
        matvec: ``(v, *parameters) -> A v`` for a symmetric ``A``.
        v0: Unit-norm starting vector.
        order: Number of Lanczos steps ``k``.
        *parameters: Passed to ``matvec``.

    Returns:
        ``(alphas, betas)``, the diagonal and the off-diagonal (in
        ``betas[:-1]``) of the ``k × k`` tridiagonal.
    """
    n = v0.shape[0]
    dtype = v0.dtype
    sqrt_eps = jnp.sqrt(jnp.finfo(dtype).eps)
    zero = jnp.zeros((), dtype=dtype)

    def step(carry, j):
        basis, q, q_prev, beta_prev, broken, scale = carry
        basis = basis.at[j].set(q)
        w = matvec(q, *parameters) - beta_prev * q_prev
        alpha = jnp.dot(q, w)
        w = w - alpha * q
        # Full reorthogonalisation against every stored vector, twice.
        for _ in range(2):
            w = w - einsum(basis, einsum(basis, w, "k n, n -> k"), "k n, k -> n")
        squared = jnp.dot(w, w)
        scale = jnp.maximum(scale, jnp.abs(alpha) + beta_prev)
        broken_now = broken | (squared <= (sqrt_eps * scale) ** 2)
        safe = jnp.where(broken_now, jnp.ones((), dtype=dtype), squared)
        norm = jnp.sqrt(safe)
        beta = jnp.where(broken_now, zero, norm)
        q_next = jnp.where(broken_now, jnp.zeros_like(w), w / norm)
        alpha = jnp.where(broken, jnp.ones((), dtype=dtype), alpha)
        return (basis, q_next, q, beta, broken_now, scale), (alpha, beta)

    init = (
        jnp.zeros((order, n), dtype=dtype),
        v0,
        jnp.zeros_like(v0),
        zero,
        jnp.zeros((), dtype=bool),
        zero,
    )
    _, (alphas, betas) = jax.lax.scan(step, init, jnp.arange(order))
    return alphas, betas


@ft.partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _gauss_quadrature(
    fun: Callable[[Array], Array],
    dfun: Callable[[Array], Array],
    tridiagonal: Float[Array, "k k"],
) -> Float[Array, ""]:
    """``e₁ᵀ f(T) e₁`` for a symmetric tridiagonal ``T`` (Gauss quadrature)."""
    del dfun
    eigenvalues, eigenvectors = jnp.linalg.eigh(tridiagonal)
    return jnp.sum(eigenvectors[0] ** 2 * fun(eigenvalues))


@_gauss_quadrature.defjvp
def _gauss_quadrature_jvp(fun, dfun, primals, tangents):
    r"""Daleckii-Krein derivative of ``e₁ᵀ f(T) e₁``.

    $$
    d\,e_1^\top f(T) e_1 = \sum_{ij} c_i\, (V^\top dT\, V)_{ij}\,
    f[\lambda_i, \lambda_j]\, c_j, \qquad c = V^\top e_1,
    $$

    with the divided difference $f[\lambda_i, \lambda_j]$ (and
    $f'(\lambda_i)$ on ties). Unlike differentiating ``eigh``, which divides
    by eigenvalue gaps, this stays finite when eigenvalues coincide, as they
    do in the padded block after a Lanczos breakdown (gh-520).
    """
    (tridiagonal,) = primals
    (tangent,) = tangents
    eigenvalues, eigenvectors = jnp.linalg.eigh(tridiagonal)
    weights = eigenvectors[0]
    gaps = einx.subtract("i, j -> i j", eigenvalues, eigenvalues)
    rises = einx.subtract("i, j -> i j", fun(eigenvalues), fun(eigenvalues))
    mids = 0.5 * einx.add("i, j -> i j", eigenvalues, eigenvalues)
    sizes = einx.add("i, j -> i j", jnp.abs(eigenvalues), jnp.abs(eigenvalues))
    tie = jnp.abs(gaps) <= jnp.sqrt(jnp.finfo(gaps.dtype).eps) * sizes
    safe_gaps = jnp.where(tie, jnp.ones_like(gaps), gaps)
    divided = jnp.where(tie, dfun(mids), rises / safe_gaps)
    rotated = einsum(eigenvectors, tangent, eigenvectors, "a i, a b, b j -> i j")
    primal = jnp.sum(weights**2 * fun(eigenvalues))
    return primal, einsum(weights, rotated * divided, weights, "i, i j, j ->")


def _quadform_integrand(
    order: int, fun: Callable[[Array], Array], dfun: Callable[[Array], Array]
):
    """SLQ integrand ``‖v‖² e₁ᵀ f(T) e₁`` in matfree's integrand signature."""

    def quadform(matvec, v0, *parameters):
        length = jnp.linalg.norm(v0)
        alphas, betas = _lanczos(matvec, v0 / length, order, *parameters)
        off = betas[:-1]
        tridiagonal = jnp.diag(alphas) + jnp.diag(off, 1) + jnp.diag(off, -1)
        return length**2 * _gauss_quadrature(fun, dfun, tridiagonal)

    return quadform


def _reciprocal(x: Array) -> Array:
    return 1.0 / x


def _log_abs(x: Array) -> Array:
    return jnp.log(jnp.abs(x))


def _logdet_integrand(order: int):
    """SLQ integrand for ``log det(A)`` of a PSD operator."""
    return _quadform_integrand(order, jnp.log, _reciprocal)


def _logabsdet_integrand(order: int):
    """SLQ integrand for ``log|det(A)|`` of a symmetric operator."""
    return _quadform_integrand(order, _log_abs, _reciprocal)


def _slq_estimators(
    integrand,
    n: int,
    num_probes: int,
    sampler: SamplerName,
    dtype: DTypeLike,
) -> tuple[Callable, Callable]:
    """Build (point, mean-and-sem) SLQ estimators sharing one sampler."""
    probe_fn = resolve_sampler(sampler, n, num_probes, dtype)
    point_raw = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    with_sem_raw = matfree.stochtrace.estimator_monte_carlo_mean_and_sem(
        integrand, probe_fn
    )

    # The probes and matvecs run in the operator's dtype, but matfree's funm
    # integrand projects onto ``np.eye(k)[0]``, built in JAX's default float,
    # so the scalar comes back float64 under x64. Cast it back.
    def point(matvec, key):
        return jnp.asarray(point_raw(matvec, key), dtype=dtype)

    def with_sem(matvec, key):
        return jax.tree.map(
            lambda x: jnp.asarray(x, dtype=dtype), with_sem_raw(matvec, key)
        )

    return point, with_sem


def _check_symmetric(operator: lx.AbstractLinearOperator, name: str) -> None:
    """Warn (for now) when SLQ gets an operator not tagged symmetric (gh-402).

    Symmetric Lanczos on a non-symmetric operator returns a silently wrong
    log-determinant. The check is on the tags, so it is free under ``jit``.
    """
    if not lx.is_symmetric(operator):
        warn_deprecated(
            f"{name} got an operator that is not tagged symmetric; symmetric "
            "Lanczos then gives a wrong log-determinant. Tag it with "
            "lineax.symmetric_tag or lineax.positive_semidefinite_tag. "
            "This will raise a ValueError in a future release."
        )


class SLQLogdet(AbstractLogdetStrategy):
    """Stochastic log-determinant via Lanczos quadrature (SLQ).

    Estimates ``log det(A)`` for PSD operators using stochastic
    trace estimation: ``logdet(A) = tr(log(A))``.  Uses matfree's
    Lanczos decomposition with sign-flip ("Rademacher") probe vectors
    by default.

    With no ``key``, `logdet` uses ``PRNGKey(seed)``, so every call sees the
    same probes: common random numbers, which suit stochastic-gradient
    training but give no variance reduction when estimates are averaged and
    a fixed pseudo-likelihood under MCMC. Pass a fresh ``key`` (or wrap a
    strategy in `gaussx.KeyedSolver`) to decorrelate calls.

    Attributes:
        num_probes: Number of probe vectors for Hutchinson estimator.
        lanczos_order: Order of the Lanczos decomposition.
        seed: Seed for probe vector generation (used when no
            ``key`` is passed to `logdet`).
        sampler: Probe distribution (``"signs"``, ``"normal"``,
            ``"sphere"``).
    """

    num_probes: int = eqx.field(static=True, default=20)
    lanczos_order: int = eqx.field(static=True, default=30)
    seed: int = eqx.field(static=True, default=0)
    sampler: SamplerName = eqx.field(static=True, default="signs")

    def _integrand(self, n: int):
        return _logdet_integrand(min(self.lanczos_order, n))

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Stochastic ``log det(A)`` via Lanczos quadrature.

        Args:
            operator: A PSD linear operator.
            key: PRNG key for probe vector sampling.  If ``None``,
                uses ``jax.random.PRNGKey(self.seed)``.

        Returns:
            Scalar estimate of ``log det(A)``.
        """
        _check_symmetric(operator, "SLQLogdet")
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        n = operator.in_size()
        point, _ = _slq_estimators(
            self._integrand(n),
            n,
            self.num_probes,
            self.sampler,
            operator.in_structure().dtype,
        )
        return point(operator.mv, key)

    def logdet_and_error(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Stochastic ``log det(A)`` with its standard error.

        Args:
            operator: A PSD linear operator.
            key: PRNG key for probe vector sampling.  If ``None``,
                uses ``jax.random.PRNGKey(self.seed)``.

        Returns:
            Tuple ``(estimate, standard_error)`` where the standard
            error is the standard error of the mean across probes.
        """
        _check_symmetric(operator, "SLQLogdet")
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        n = operator.in_size()
        _, with_sem = _slq_estimators(
            self._integrand(n),
            n,
            self.num_probes,
            self.sampler,
            operator.in_structure().dtype,
        )
        return with_sem(operator.mv, key)


class IndefiniteSLQLogdet(AbstractLogdetStrategy):
    """Stochastic ``log|det(A)|`` for symmetric (possibly indefinite) operators.

    Like `SLQLogdet` but uses ``log(|lambda|)`` as the matrix
    function, so it works on indefinite and negative-definite matrices.
    Supports a diagonal shift ``(A + shift * I)``.

    Attributes:
        num_probes: Number of probe vectors for Hutchinson estimator.
        lanczos_order: Order of the Lanczos decomposition.
        shift: Diagonal shift applied before computing the logdet.
        seed: Seed for probe vector generation.
        sampler: Probe distribution (``"signs"``, ``"normal"``,
            ``"sphere"``).
    """

    num_probes: int = eqx.field(static=True, default=20)
    lanczos_order: int = eqx.field(static=True, default=30)
    shift: float = eqx.field(static=True, default=0.0)
    seed: int = eqx.field(static=True, default=0)
    sampler: SamplerName = eqx.field(static=True, default="signs")

    def _shifted_matvec(self, operator: lx.AbstractLinearOperator):
        shift = self.shift

        def matvec(v):
            return operator.mv(v) + shift * v

        return matvec

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        r"""Stochastic ``log|det(A + shift I)|`` via Lanczos quadrature.

        Args:
            operator: A symmetric linear operator.
            key: PRNG key for probe vector sampling.  If ``None``,
                uses ``jax.random.PRNGKey(self.seed)``.

        Returns:
            Scalar estimate of ``log|det(A + shift I)|``.
        """
        _check_symmetric(operator, "IndefiniteSLQLogdet")
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        n = operator.in_size()
        order = min(self.lanczos_order, n)
        point, _ = _slq_estimators(
            _logabsdet_integrand(order),
            n,
            self.num_probes,
            self.sampler,
            operator.in_structure().dtype,
        )
        return point(self._shifted_matvec(operator), key)

    def logdet_and_error(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Stochastic ``log|det(A + shift I)|`` with its standard error.

        Args:
            operator: A symmetric linear operator.
            key: PRNG key for probe vector sampling.  If ``None``,
                uses ``jax.random.PRNGKey(self.seed)``.

        Returns:
            Tuple ``(estimate, standard_error)``.
        """
        _check_symmetric(operator, "IndefiniteSLQLogdet")
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        n = operator.in_size()
        order = min(self.lanczos_order, n)
        _, with_sem = _slq_estimators(
            _logabsdet_integrand(order),
            n,
            self.num_probes,
            self.sampler,
            operator.in_structure().dtype,
        )
        return with_sem(self._shifted_matvec(operator), key)


class DenseLogdet(AbstractLogdetStrategy):
    """Dense log-determinant via gaussx structural dispatch.

    Delegates to `gaussx.logdet` which automatically selects
    the best algorithm based on operator structure (Diagonal,
    BlockDiag, Kronecker, LowRankUpdate, or dense fallback).
    """

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Compute ``log |det(A)|`` via structural dispatch.

        Args:
            operator: A linear operator.
            key: Ignored (deterministic).

        Returns:
            Scalar ``log |det(A)|``.
        """
        from gaussx._primitives._logdet import logdet as _logdet

        return _logdet(operator)
