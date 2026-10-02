"""Precision builders for latent Gaussian models (INLA components)."""

from gaussx._gmrf._areal import (
    besag_structure,
    bym2_precision,
    generalized_variance_scale,
)
from gaussx._gmrf._fem import fem_matrices, fem_projector
from gaussx._gmrf._spde import matern_spde_params, spde_precision, spde_precision_grid
from gaussx._gmrf._temporal import (
    ar1_precision,
    iid_precision,
    rw1_structure,
    rw2_structure,
)


__all__ = [
    "ar1_precision",
    "besag_structure",
    "bym2_precision",
    "fem_matrices",
    "fem_projector",
    "generalized_variance_scale",
    "iid_precision",
    "matern_spde_params",
    "rw1_structure",
    "rw2_structure",
    "spde_precision",
    "spde_precision_grid",
]
