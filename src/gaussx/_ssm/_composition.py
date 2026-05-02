"""Composed SDE kernels: sum, product, and quasi-periodic."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from gaussx._ssm._sde_kernel import SDEKernel, SDEParams


class SumSDE(SDEKernel):
    """Sum of SDE kernels via block-diagonal composition.

    Args:
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

    Args:
        kernel1: First component kernel.
        kernel2: Second component kernel.
    """

    kernel1: SDEKernel
    kernel2: SDEKernel

    @property
    def state_dim(self) -> int:
        return self.kernel1.state_dim * self.kernel2.state_dim

    def sde_params(self) -> SDEParams:
        """Return Kronecker-structured SDE parameters."""
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


class QuasiPeriodicSDE(ProductSDE):
    """Quasi-periodic kernel: product of Matern and Periodic SDE.

    Args:
        kernel1: Modulating kernel (typically Matern).
        kernel2: Periodic kernel.
    """

    pass
