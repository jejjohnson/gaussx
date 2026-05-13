"""EigenPro spectral preconditioning for kernel SGD."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, Int

from gaussx._operators._implicit_kernel import ImplicitKernelOperator
from gaussx._operators._kernel import KernelOperator
from gaussx._primitives._eig import eig


# Number of rows held in memory at once when streaming the residual
# kernel diagonal estimate; chosen empirically so that ``CHUNK * m``
# stays well under typical GPU memory.
_RESIDUAL_CHUNK: int = 256


class EigenProPreconditioner(eqx.Module):
    r"""Spectral correction state for EigenPro kernel SGD.

    Stores the top eigenspace of ``K_mm / m`` on a subsample and the
    corresponding EigenPro correction weights.

    Args:
        V: Top eigenvectors of ``K_mm / m``, shape ``(m, k)``.
        D: Correction weights, shape ``(k,)``.
        subsample_indices: Indices used for the subsample, shape ``(m,)``.
        max_eigenvalue: Largest eigenvalue of ``K_mm / m``.
        beta: Maximum residual kernel diagonal used for the step size.
    """

    V: Float[Array, "m k"]
    D: Float[Array, " k"]
    subsample_indices: Int[Array, " m"]
    max_eigenvalue: Float[Array, ""]
    beta: Float[Array, ""]


def eigenpro_preconditioner(
    kernel_op: lx.AbstractLinearOperator,
    *,
    subsample_size: int = 4000,
    n_components: int = 100,
    alpha: float = 0.95,
    key: jax.Array | None = None,
) -> EigenProPreconditioner:
    r"""Build an EigenPro spectral preconditioner from a kernel operator.

    The helper samples ``m`` rows/columns, eigendecomposes ``K_mm / m``, and
    stores the top-``k`` eigenspace correction
    ``D_i = (1 - (λ_{k+1} / λ_i)^α) / λ_i``.

    Args:
        kernel_op: Square kernel linear operator.
        subsample_size: Number of points in the eigendecomposition subsample.
        n_components: Number of leading eigenvectors to keep.
        alpha: Spectral decay exponent in ``(0, 1]``.
        key: Optional PRNG key. If omitted, the first ``subsample_size`` points
            are used deterministically.

    Returns:
        EigenPro preconditioner state for kernel SGD.
    """

    if subsample_size <= 1:
        raise ValueError("subsample_size must be greater than 1.")
    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    if n_components >= subsample_size:
        raise ValueError("n_components must be smaller than subsample_size.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")
    if not lx.is_symmetric(kernel_op):
        raise ValueError(
            "kernel_op must be symmetric (tag with ``lx.symmetric_tag`` or "
            "``lx.positive_semidefinite_tag``). EigenPro relies on K_mm and "
            "the data-side diagonal coming from the same symmetric kernel."
        )

    in_shape = kernel_op.in_structure().shape
    out_shape = kernel_op.out_structure().shape
    if len(in_shape) != 1 or len(out_shape) != 1 or in_shape != out_shape:
        raise ValueError("kernel_op must be an unbatched square operator.")

    n = in_shape[0]
    if subsample_size > n:
        raise ValueError("subsample_size cannot exceed the operator size.")

    subsample_indices = _subsample_indices(n, subsample_size, key)
    K_mm = _subsample_matrix(kernel_op, subsample_indices)
    m = K_mm.shape[0]
    K_mm_scaled = 0.5 * (K_mm + K_mm.T) / m

    # Partial eigendecomposition: top (n_components + 1) eigenpairs via
    # matfree Lanczos. The "+1" gives us the tail eigenvalue needed for
    # the EigenPro correction weights. Falls back to dense eigh for very
    # small ``m`` where Lanczos isn't worth the overhead.
    rank = n_components + 1
    if rank >= m:
        eigvals_all, eigvecs = jnp.linalg.eigh(K_mm_scaled)
        eigvals_all = eigvals_all[::-1]
        eigvecs = eigvecs[:, ::-1]
    else:
        K_mm_op = lx.MatrixLinearOperator(K_mm_scaled, lx.positive_semidefinite_tag)
        eig_key = key if key is not None else jr.PRNGKey(0)
        raw_vals, raw_vecs = eig(K_mm_op, rank=rank, key=eig_key)
        # eig may return complex/unsorted for partial Lanczos; project
        # back to real and sort descending so we keep the top-k.
        eigvals_all = jnp.real(raw_vals)
        eigvecs = jnp.real(raw_vecs)
        order = jnp.argsort(eigvals_all)[::-1]
        eigvals_all = eigvals_all[order]
        eigvecs = eigvecs[:, order]

    top_eigvals = eigvals_all[:n_components]
    tail_eigenvalue = eigvals_all[n_components]
    eps = jnp.finfo(K_mm.dtype).eps
    safe_top_eigvals = jnp.maximum(top_eigvals, eps)
    safe_tail_eigenvalue = jnp.maximum(tail_eigenvalue, eps)
    ratio = jnp.minimum(safe_tail_eigenvalue / safe_top_eigvals, 1.0)
    weights = (1.0 - ratio**alpha) / safe_top_eigvals

    V = eigvecs[:, :n_components]
    beta = _residual_kernel_diagonal(kernel_op, subsample_indices, V, safe_top_eigvals)

    return EigenProPreconditioner(
        V=V,
        D=weights,
        subsample_indices=subsample_indices,
        max_eigenvalue=safe_top_eigvals[0],
        beta=beta,
    )


def eigenpro_step_size(
    precond: EigenProPreconditioner,
    batch_size: int | Float[Array, ""],
) -> Float[Array, ""]:
    r"""Return the EigenPro preconditioned step size for a batch size.

    ``batch_size`` may be a Python int or a scalar JAX array; the
    function is JIT-friendly with either. Non-positive batch sizes raise
    (eagerly when ``batch_size`` is a Python int; traced via
    ``eqx.error_if`` when it is a JAX array).
    """
    if isinstance(batch_size, int) and batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    batch = jnp.asarray(batch_size, dtype=precond.beta.dtype)
    batch = eqx.error_if(batch, batch < 1.0, "batch_size must be >= 1.")
    beta = precond.beta
    lambda_1 = precond.max_eigenvalue
    return jnp.where(
        batch < beta / lambda_1,
        batch / beta,
        (2.0 * batch) / (beta + (batch - 1.0) * lambda_1),
    )


def eigenpro_correction(
    precond: EigenProPreconditioner,
    K_batch_sub: Float[Array, "B m"],
    gradient: Float[Array, "B C"],
    step_size: float | Float[Array, ""],
) -> Float[Array, "m C"]:
    r"""Compute the EigenPro eigenspace correction for one SGD mini-batch.

    ``step_size`` may be a Python float or a scalar JAX array. Passing a
    JAX scalar avoids recompilation under ``jax.jit`` when the value
    changes between training steps.
    """
    step_size_arr = jnp.asarray(step_size, dtype=gradient.dtype)
    projected_gradient = precond.V.T @ (K_batch_sub.T @ gradient)
    weight_shape = (-1,) + (1,) * (projected_gradient.ndim - 1)
    weighted_gradient = precond.D.reshape(weight_shape) * projected_gradient
    return step_size_arr * (precond.V @ weighted_gradient)


def _subsample_indices(
    n: int,
    subsample_size: int,
    key: jax.Array | None,
) -> Int[Array, " m"]:
    if key is None:
        return jnp.arange(subsample_size)
    return jr.choice(key, n, shape=(subsample_size,), replace=False)


def _subsample_matrix(
    kernel_op: lx.AbstractLinearOperator,
    subsample_indices: Int[Array, " m"],
) -> Float[Array, "m m"]:
    if isinstance(kernel_op, ImplicitKernelOperator):
        X_sub = kernel_op.X[subsample_indices]
        K_mm = _implicit_kernel_matrix(kernel_op, X_sub, X_sub)
        if kernel_op.noise_var != 0.0:
            K_mm = K_mm + kernel_op.noise_var * jnp.eye(
                subsample_indices.shape[0], dtype=K_mm.dtype
            )
        return K_mm

    if isinstance(kernel_op, KernelOperator):
        X1_sub = kernel_op.X1[subsample_indices]
        X2_sub = kernel_op.X2[subsample_indices]
        return _kernel_operator_matrix(kernel_op, X1_sub, X2_sub)

    K = kernel_op.as_matrix()
    return K[jnp.ix_(subsample_indices, subsample_indices)]


def _cross_kernel_matrix(
    kernel_op: lx.AbstractLinearOperator,
    subsample_indices: Int[Array, " m"],
) -> Float[Array, "N m"]:
    if isinstance(kernel_op, ImplicitKernelOperator):
        X_sub = kernel_op.X[subsample_indices]
        K_nm = _implicit_kernel_matrix(kernel_op, kernel_op.X, X_sub)
        if kernel_op.noise_var != 0.0:
            noise = jnp.zeros_like(K_nm)
            noise = noise.at[
                subsample_indices, jnp.arange(subsample_indices.shape[0])
            ].set(kernel_op.noise_var)
            K_nm = K_nm + noise
        return K_nm

    if isinstance(kernel_op, KernelOperator):
        X2_sub = kernel_op.X2[subsample_indices]
        return _kernel_operator_matrix(kernel_op, kernel_op.X1, X2_sub)

    K = kernel_op.as_matrix()
    return K[:, subsample_indices]


def _kernel_diagonal(kernel_op: lx.AbstractLinearOperator) -> Float[Array, " N"]:
    if isinstance(kernel_op, ImplicitKernelOperator):
        diag = jax.vmap(
            lambda x: (
                kernel_op.kernel_fn(kernel_op.params, x, x)
                if kernel_op.params is not None
                else kernel_op.kernel_fn(x, x)
            )
        )(kernel_op.X)
        if kernel_op.noise_var != 0.0:
            diag = diag + kernel_op.noise_var
        return diag

    if isinstance(kernel_op, KernelOperator):
        return jax.vmap(lambda x1, x2: kernel_op.kernel_fn(kernel_op.params, x1, x2))(
            kernel_op.X1, kernel_op.X2
        )

    return jnp.diag(kernel_op.as_matrix())


def _residual_kernel_diagonal(
    kernel_op: lx.AbstractLinearOperator,
    subsample_indices: Int[Array, " m"],
    V: Float[Array, "m k"],
    eigenvalues: Float[Array, " k"],
) -> Float[Array, ""]:
    """Estimate ``max_x diag(K)(x) - m Σ_i φ_i(x)^2``.

    Streams the cross-kernel matrix in row-chunks (``_RESIDUAL_CHUNK``
    rows at a time) so we never hold the full ``(N, m)`` ``K_nm`` in
    memory at once — important for typical EigenPro settings where
    ``N`` can be 10^6+ and ``m ≈ 4000``.
    """
    m = subsample_indices.shape[0]
    diag = _kernel_diagonal(kernel_op)
    n = diag.shape[0]
    # K_mm V_i = m λ_i V_i for eigenpairs of K_mm / m, so
    # φ_i(x) = K_xm V_i / (m λ_i sqrt(m)).
    scaled_V = V / (eigenvalues[None, :] * m * jnp.sqrt(m))
    dtype = scaled_V.dtype

    def chunk_max(start: int, max_so_far: Float[Array, ""]) -> Float[Array, ""]:
        # ``start`` is a Python int from ``range()``; keep chunk_size as
        # a Python int too so static shapes propagate through the
        # specialised kernel-row builders.
        stop = min(start + _RESIDUAL_CHUNK, n)
        chunk_size = stop - start
        # Build only K_{x_chunk, m} — never the full K_nm.
        K_chunk = _cross_kernel_rows(kernel_op, subsample_indices, start, chunk_size)
        eigfns = K_chunk @ scaled_V  # (chunk, k)
        chunk_residual = diag[start:stop] - m * jnp.sum(eigfns**2, axis=1)
        return jnp.maximum(max_so_far, jnp.max(chunk_residual))

    init = jnp.asarray(-jnp.inf, dtype=dtype)
    # Iterate chunks in pure Python (compile-time loop) so the chunk
    # build can dispatch on ``kernel_op``'s concrete type.
    max_residual = init
    for start in range(0, n, _RESIDUAL_CHUNK):
        max_residual = chunk_max(start, max_residual)
    eps = jnp.finfo(dtype).eps
    return jnp.maximum(max_residual, eps)


def _cross_kernel_rows(
    kernel_op: lx.AbstractLinearOperator,
    subsample_indices: Int[Array, " m"],
    start: int,
    chunk_size: int,
) -> Float[Array, "chunk m"]:
    """Build the ``(chunk, m)`` slice of ``K[start:start+chunk, subsample]``."""
    stop = start + chunk_size
    if isinstance(kernel_op, ImplicitKernelOperator):
        X_rows = kernel_op.X[start:stop]
        X_sub = kernel_op.X[subsample_indices]
        K_block = _implicit_kernel_matrix(kernel_op, X_rows, X_sub)
        if kernel_op.noise_var != 0.0:
            row_idx = jnp.arange(start, stop)
            mask = row_idx[:, None] == subsample_indices[None, :]
            K_block = K_block + kernel_op.noise_var * mask.astype(K_block.dtype)
        return K_block

    if isinstance(kernel_op, KernelOperator):
        X1_rows = kernel_op.X1[start:stop]
        X2_sub = kernel_op.X2[subsample_indices]
        return _kernel_operator_matrix(kernel_op, X1_rows, X2_sub)

    # Dense fallback: materialise once, slice. Acceptable because the
    # streaming branch is the one that matters for matrix-free kernels.
    K = kernel_op.as_matrix()
    return K[start:stop][:, subsample_indices]


def _implicit_kernel_matrix(
    kernel_op: ImplicitKernelOperator,
    X1: Float[Array, "N D"],
    X2: Float[Array, "M D"],
) -> Float[Array, "N M"]:
    if kernel_op.params is not None:
        return jax.vmap(
            lambda x_i: jax.vmap(
                lambda x_j: kernel_op.kernel_fn(kernel_op.params, x_i, x_j)
            )(X2)
        )(X1)
    return jax.vmap(
        lambda x_i: jax.vmap(lambda x_j: kernel_op.kernel_fn(x_i, x_j))(X2)
    )(X1)


def _kernel_operator_matrix(
    kernel_op: KernelOperator,
    X1: Float[Array, "N D"],
    X2: Float[Array, "M D"],
) -> Float[Array, "N M"]:
    return jax.vmap(
        lambda x_i: jax.vmap(
            lambda x_j: kernel_op.kernel_fn(kernel_op.params, x_i, x_j)
        )(X2)
    )(X1)
