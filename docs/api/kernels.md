# Kernels & Approximations

Low-rank kernel approximations, spectral preconditioning for kernel SGD,
kernel two-sample / independence statistics, and the grid helpers behind
interpolation-based (KISS-GP style) operators.

## Low-rank kernel operators

Nyström ($K \approx K_{nm} K_{mm}^{-1} K_{mn}$) and random-Fourier-feature
approximations, returned as [`LowRankUpdate`](operators.md) operators so solves
and logdets go through Woodbury automatically.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [nystrom_operator, rff_operator]

## EigenPro preconditioning

Spectral preconditioning for kernel stochastic gradient descent: damp the top
eigendirections of the kernel operator so the step size is governed by the
residual spectrum (Ma & Belkin, 2017).

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [eigenpro_preconditioner, eigenpro_step_size, eigenpro_correction, EigenProPreconditioner]

## Falkon kernel ridge regression

Nyström kernel ridge regression solves
$(K_{nm}^\top K_{nm} + \lambda n K_{mm})\alpha = K_{nm}^\top y$. Falkon
(Rudi et al., 2017; Meanti et al., 2020) preconditions it with the Nyström
approximation $K_{nm}^\top K_{nm} \approx (n/m) K_{mm}^2$, factored as two
upper-triangular $M \times M$ Choleskys. In the preconditioned variable
$K_{mm}$ cancels, so conjugate gradients needs only triangular solves and
matvecs with $K_{nm}$, which an `ImplicitCrossKernelOperator` provides without
ever forming the $N \times M$ matrix.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [falkon_preconditioner, FalkonPreconditioner]

## Kernel statistics

Centering, the Hilbert-Schmidt independence criterion, and maximum mean
discrepancy.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [center_kernel, centering_operator, hsic, mmd_squared]

## Grids & interpolation

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [create_grid, grid_data, cubic_interpolation_weights]
