"""Composed SDE kernels: sum, product, and quasi-periodic."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float

from gaussx._linalg._symmetrize import symmetrize
from gaussx._ssm._sde_kernel import SDEKernel, SDEParams


class SumSDE(SDEKernel):
    """Sum of SDE kernels via block-diagonal composition.

    Attributes:
        kernels: Tuple of component SDE kernels.
    """

    kernels: tuple[SDEKernel, ...] = eqx.field()

    @property
    def state_dim(self) -> int:
        return sum(k.state_dim for k in self.kernels)

    def sde_params(self) -> SDEParams:
        """Return block-diagonal SDE parameters."""
        params_list = [k.sde_params() for k in self.kernels]

        F = jsl.block_diag(*[p.F for p in params_list])
        P_inf = jsl.block_diag(*[p.P_inf for p in params_list])

        L_blocks = [p.L for p in params_list]
        total_rows = sum(b.shape[0] for b in L_blocks)
        total_cols = sum(b.shape[1] for b in L_blocks)
        L = jnp.zeros((total_rows, total_cols))
        row_offset = 0
        col_offset = 0
        for block in L_blocks:
            r, c = block.shape
            L = L.at[row_offset : row_offset + r, col_offset : col_offset + c].set(
                block
            )
            row_offset += r
            col_offset += c

        Q_c = jsl.block_diag(*[p.Q_c for p in params_list])
        H = jnp.concatenate([p.H for p in params_list], axis=1)

        return SDEParams(F=F, L=L, H=H, Q_c=Q_c, P_inf=P_inf)


class ProductSDE(SDEKernel):
    """Product of two SDE kernels via Kronecker composition.

    Attributes:
        kernel1: First component kernel.
        kernel2: Second component kernel.
    """

    kernel1: SDEKernel
    kernel2: SDEKernel

    @property
    def state_dim(self) -> int:
        return self.kernel1.state_dim * self.kernel2.state_dim

    def sde_params(self) -> SDEParams:
        """Return Kronecker-structured SDE parameters.

        Note:
            ``SDEParams`` currently types its fields as dense
            ``jaxtyping.Float[Array, ...]``. The Kronecker products
            below are dense materializations of size
            ``(state_dim, state_dim)``, where ``state_dim`` is
            ``kernel1.state_dim * kernel2.state_dim`` — for typical SSM
            kernels (Matérn-3/2, periodic) this is ≤ 32, so the
            materialization is bounded and cheap. A future refactor
            could expose a parallel ``sde_operators()`` method that
            returns :class:`gaussx.Kronecker` operators for downstream
            filters that can exploit the structure (issue #153).
        """
        p1 = self.kernel1.sde_params()
        p2 = self.kernel2.sde_params()

        d1 = self.kernel1.state_dim
        d2 = self.kernel2.state_dim

        F = jnp.kron(p1.F, jnp.eye(d2)) + jnp.kron(jnp.eye(d1), p2.F)
        L = jnp.kron(p1.L, p2.L)
        H = jnp.kron(p1.H, p2.H)
        Q_c = jnp.kron(p1.Q_c, p2.Q_c)
        P_inf = jnp.kron(p1.P_inf, p2.P_inf)

        return SDEParams(F=F, L=L, H=H, Q_c=Q_c, P_inf=P_inf)

    def discretise(
        self,
        dt: Float[Array, ""],
    ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
        r"""Discretise via the Kronecker matrix-exponential identity.

        For a product kernel ``F = F_1 \oplus F_2 = F_1 \otimes I + I \otimes F_2``,
        the factors ``F_1 \otimes I`` and ``I \otimes F_2`` commute, so

        .. math::

            \exp(F \, dt) = \exp(F_1 \, dt) \otimes \exp(F_2 \, dt).

        This computes two ``expm`` calls of size ``d_1`` and ``d_2``
        each, plus one Kronecker product, instead of one ``expm`` of
        size ``d_1 \cdot d_2``. Numerically equivalent to the dense
        ``expm`` on ``F`` but cheaper for moderate factor sizes.

        ``Q = P_\infty - A P_\infty A^T`` is computed densely from the
        resulting ``A``; with ``P_\infty = P_{\infty,1} \otimes
        P_{\infty,2}`` this could itself be expressed as a Kronecker
        difference, but is left dense to keep the consumer-facing
        ``(A, Q)`` interface unchanged.

        Args:
            dt: Time step (scalar, positive).

        Returns:
            Tuple ``(A, Q)`` matching :meth:`SDEKernel.discretise`.
        """
        p1 = self.kernel1.sde_params()
        p2 = self.kernel2.sde_params()
        A1 = jsl.expm(p1.F * dt)
        A2 = jsl.expm(p2.F * dt)
        A = jnp.kron(A1, A2)

        # Use the per-factor stationary covariances directly; building
        # the full ``F`` via ``self.sde_params()`` would defeat the
        # whole point of this override.
        P_inf = jnp.kron(p1.P_inf, p2.P_inf)
        Q = P_inf - A @ P_inf @ A.T
        Q = symmetrize(Q)
        return A, Q


class QuasiPeriodicSDE(ProductSDE):
    """Quasi-periodic kernel: product of Matern and Periodic SDE.

    Attributes:
        kernel1: Modulating kernel (typically Matern).
        kernel2: Periodic kernel.
    """

    pass
