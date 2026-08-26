"""Composed SDE kernels: sum, product, and quasi-periodic."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lineax as lx
from jaxtyping import Array, Float

from gaussx._linalg._symmetrize import symmetrize
from gaussx._operators._kronecker import Kronecker
from gaussx._ssm._discretise import process_noise_covariance
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

    @property
    def stationary(self) -> bool:
        """Stationary only if every component is."""
        return all(k.stationary for k in self.kernels)

    def initial_covariance(self) -> Float[Array, "d d"]:
        """Return the block-diagonal initial covariance.

        The components are independent, so their initial covariances
        stack block-diagonally — which lets a sum mix stationary and
        non-stationary components (a local linear trend plus a Matern
        seasonal, say), each started from its own.
        """
        return jsl.block_diag(*[k.initial_covariance() for k in self.kernels])

    def sde_params(self) -> SDEParams:
        """Return block-diagonal SDE parameters."""
        params_list = [k.sde_params() for k in self.kernels]

        F = jsl.block_diag(*[p.F for p in params_list])
        # A component with no closed-form stationary covariance leaves the
        # sum without one either; propagating ``None`` routes the composite
        # through ``discretise_mfd`` rather than fabricating a ``P_inf``.
        component_p_inf = [p.P_inf for p in params_list]
        P_inf = (
            None
            if any(block is None for block in component_p_inf)
            else jsl.block_diag(*component_p_inf)
        )

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

    @property
    def stationary(self) -> bool:
        """Stationary only if both factors are."""
        return self.kernel1.stationary and self.kernel2.stationary

    def sde_params(self) -> SDEParams:
        r"""Return Kronecker-structured SDE parameters.

        The drift is the Kronecker **sum** $F_1 \oplus F_2$ and the
        stationary covariance the Kronecker **product**
        $P_1 \otimes P_2$. Substituting those into the Lyapunov equation
        $F P + P F^\top + B = 0$ fixes the composite diffusion at

        $$
        B \;=\; B_1 \otimes P_2 \;+\; P_1 \otimes B_2,
        \qquad B_i = L_i Q_{c,i} L_i^\top,
        $$

        which is **not** $B_1 \otimes B_2$ — the value a naive
        $L_1 \otimes L_2$, $Q_{c,1} \otimes Q_{c,2}$ pair would imply.
        Reporting the latter used to hand out a tuple that failed its own
        Lyapunov equation (gh-219); for a Matérn ⊗ Cosine product it was
        identically zero, since `CosineSDE` has $Q_c = 0$.

        The sum of two Kronecker products is still expressible in the
        ``(L, Q_c)`` form, by widening the noise dimension and carrying
        each factor's stationary covariance in the spectral density:

        $$
        L = \bigl[\, L_1 \otimes I_{d_2} \;\;\big|\;\; I_{d_1} \otimes L_2 \,\bigr],
        \qquad
        Q_c = \operatorname{blockdiag}\!\bigl(
            Q_{c,1} \otimes P_2,\; P_1 \otimes Q_{c,2}
        \bigr),
        $$

        so that $L Q_c L^\top$ telescopes to exactly the $B$ above by the
        mixed-product property. Writing it this way rather than through
        square roots $S_i S_i^\top = P_i$ keeps the result exact for
        singular or zero $P_\infty$ (where a Cholesky would need jitter,
        and would then violate the very Lyapunov equation this enforces)
        and keeps ``sde_params`` reverse-mode differentiable.

        Note:
            ``SDEParams`` currently types its fields as dense
            ``jaxtyping.Float[Array, ...]``. The Kronecker products
            below are dense materializations of size
            ``(state_dim, state_dim)``, where ``state_dim`` is
            ``kernel1.state_dim * kernel2.state_dim`` — for typical SSM
            kernels (Matérn-3/2, periodic) this is ≤ 32, so the
            materialization is bounded and cheap. A future refactor
            could expose a parallel ``sde_operators()`` method that
            returns `gaussx.Kronecker` operators for downstream
            filters that can exploit the structure (issue #153).

        Raises:
            NotImplementedError: If either factor lacks a stationary
                covariance. The composite diffusion needs both, so the
                tuple cannot be built — and reporting the inconsistent
                Kronecker product instead is what gh-219 was about.
        """
        p1 = self.kernel1.sde_params()
        p2 = self.kernel2.sde_params()

        d1 = self.kernel1.state_dim
        d2 = self.kernel2.state_dim

        if p1.P_inf is None or p2.P_inf is None:
            msg = (
                f"ProductSDE cannot report SDE parameters when a factor has "
                f"no stationary covariance: "
                f"{type(self.kernel1).__name__}.P_inf is "
                f"{'None' if p1.P_inf is None else 'set'} and "
                f"{type(self.kernel2).__name__}.P_inf is "
                f"{'None' if p2.P_inf is None else 'set'}. The composite "
                f"diffusion of a product kernel is B1 (x) P2 + P1 (x) B2, "
                f"which needs both. Use the factors' own parameters, or "
                f"give the factor a P_inf."
            )
            raise NotImplementedError(msg)

        # Identities carry the factor dtypes: an untyped ``jnp.eye`` is
        # float64 under x64 and would promote float32 kernels.
        eye1 = jnp.eye(d1, dtype=p1.F.dtype)
        eye2 = jnp.eye(d2, dtype=p2.F.dtype)

        F = jnp.kron(p1.F, eye2) + jnp.kron(eye1, p2.F)
        H = jnp.kron(p1.H, p2.H)
        P_inf = jnp.kron(p1.P_inf, p2.P_inf)

        # B = B1 (x) P2 + P1 (x) B2, kept in the (L, Q_c) pair by putting
        # each factor's P_inf in the *spectral density* rather than taking
        # its square root:
        #
        #     B1 (x) P2 = (L1 (x) I) (Q_c1 (x) P2) (L1 (x) I)^T
        #     P1 (x) B2 = (I (x) L2) (P1 (x) Q_c2) (I (x) L2)^T
        #
        # by the mixed-product property. Exact for any PSD P_inf --
        # including singular or zero ones, where a Cholesky would need
        # jitter and stop satisfying the Lyapunov equation this is here to
        # enforce. The noise dimension widens from s1*s2 to s1*d2 + d1*s2.
        L = jnp.concatenate(
            [jnp.kron(p1.L, eye2), jnp.kron(eye1, p2.L)],
            axis=1,
        )
        Q_c = jsl.block_diag(
            jnp.kron(p1.Q_c, p2.P_inf),
            jnp.kron(p1.P_inf, p2.Q_c),
        )

        return SDEParams(F=F, L=L, H=H, Q_c=Q_c, P_inf=P_inf)

    def discretise(
        self,
        dt: Float[Array, ""],
    ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
        r"""Discretise via the Kronecker matrix-exponential identity.

        For a product kernel ``F = F_1 \oplus F_2 = F_1 \otimes I + I \otimes F_2``,
        the factors ``F_1 \otimes I`` and ``I \otimes F_2`` commute, so

        $$
        \exp(F \, dt) = \exp(F_1 \, dt) \otimes \exp(F_2 \, dt).
        $$

        This computes two ``expm`` calls of size ``d_1`` and ``d_2``
        each, plus one Kronecker product, instead of one ``expm`` of
        size ``d_1 \cdot d_2``. Numerically equivalent to the dense
        ``expm`` on ``F`` but cheaper for moderate factor sizes.

        ``Q = P_\infty - A P_\infty A^T`` exploits the same factorisation.
        By the mixed-product property,

        $$
        (A_1 \otimes A_2)(P_1 \otimes P_2)(A_1 \otimes A_2)^\top
            = (A_1 P_1 A_1^\top) \otimes (A_2 P_2 A_2^\top),
        $$

        so the congruence is evaluated per factor via
        `gaussx.process_noise_covariance` on `gaussx.Kronecker`
        operands — ``O(d_1^3 + d_2^3)`` instead of the
        ``O((d_1 d_2)^3)`` triple product on the full matrix. Only the
        final ``Q`` is materialised, to keep the consumer-facing
        ``(A, Q)`` interface unchanged.

        Args:
            dt: Time step (scalar, positive).

        Returns:
            Tuple ``(A, Q)`` matching `SDEKernel.discretise`.
        """
        p1 = self.kernel1.sde_params()
        p2 = self.kernel2.sde_params()
        A1 = jsl.expm(p1.F * dt)
        A2 = jsl.expm(p2.F * dt)
        A = jnp.kron(A1, A2)

        # Use the per-factor stationary covariances directly; building
        # the full ``F`` via ``self.sde_params()`` would defeat the
        # whole point of this override. Keeping both operands as
        # ``Kronecker`` lets the shared helper contract each factor
        # separately instead of forming the (d1 d2)-square triple product.
        if p1.P_inf is None or p2.P_inf is None:
            # Deferring to the base MFD path here would be *wrong*, not
            # merely slower: the composite diffusion
            #
            #     B = B_1 (x) P_2  +  P_1 (x) B_2
            #
            # needs both factor stationary covariances -- exactly what is
            # missing -- so MFD has nothing correct to consume. Checked
            # against the factors directly rather than via
            # ``self.sde_params()`` (which now raises for the same reason)
            # so the message names the offending factor, and so the
            # happy path never builds the full-size drift it would
            # otherwise have to discard.
            msg = (
                f"ProductSDE cannot be discretised when a factor has no "
                f"stationary covariance: "
                f"{type(self.kernel1).__name__}.P_inf is "
                f"{'None' if p1.P_inf is None else 'set'} and "
                f"{type(self.kernel2).__name__}.P_inf is "
                f"{'None' if p2.P_inf is None else 'set'}. The composite "
                f"diffusion of a product kernel is B1 (x) P2 + P1 (x) B2, "
                f"which needs both. Discretise the factors separately, or "
                f"give the factor a P_inf."
            )
            raise NotImplementedError(msg)

        A_op = Kronecker(
            lx.MatrixLinearOperator(A1),
            lx.MatrixLinearOperator(A2),
        )
        P_op = Kronecker(
            lx.MatrixLinearOperator(p1.P_inf, lx.symmetric_tag),
            lx.MatrixLinearOperator(p2.P_inf, lx.symmetric_tag),
        )
        Q = symmetrize(process_noise_covariance(A_op, P_op).as_matrix())
        return A, Q


class QuasiPeriodicSDE(ProductSDE):
    """Quasi-periodic kernel: product of Matern and Periodic SDE.

    Attributes:
        kernel1: Modulating kernel (typically Matern).
        kernel2: Periodic kernel.
    """

    pass
