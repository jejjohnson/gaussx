"""GaussX kernel utilities -- Nystrom / RFF approximations and grids."""

from gaussx._kernels._eigenpro import (
    EigenProPreconditioner,
    eigenpro_correction,
    eigenpro_preconditioner,
    eigenpro_step_size,
)
from gaussx._kernels._grid import (
    create_grid,
    cubic_interpolation_weights,
    grid_data,
)
from gaussx._kernels._kernel_approx import (
    center_kernel,
    centering_operator,
    hsic,
    mmd_squared,
    nystrom_operator,
    rff_operator,
)


__all__ = [
    "EigenProPreconditioner",
    "center_kernel",
    "centering_operator",
    "create_grid",
    "cubic_interpolation_weights",
    "eigenpro_correction",
    "eigenpro_preconditioner",
    "eigenpro_step_size",
    "grid_data",
    "hsic",
    "mmd_squared",
    "nystrom_operator",
    "rff_operator",
]
