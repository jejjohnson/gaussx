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
