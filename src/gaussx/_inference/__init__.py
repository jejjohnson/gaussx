"""GaussX inference updates -- variational, natural-gradient, BLR, EnKF."""

from gaussx._inference._blr import (
    blr_diag_update,
    blr_full_update,
    ggn_diagonal,
    hutchinson_hessian_diag,
)
from gaussx._inference._ensemble import (
    ensemble_covariance,
    ensemble_cross_covariance,
)
from gaussx._inference._inference import (
    cavity_distribution,
    gaussian_expected_log_lik,
    log_marginal_likelihood,
    newton_update,
    process_noise_covariance,
    trace_correction,
)
from gaussx._inference._natural_gradient import (
    damped_natural_update,
    gauss_newton_precision,
    riemannian_psd_correction,
)


__all__ = [
    "blr_diag_update",
    "blr_full_update",
    "cavity_distribution",
    "damped_natural_update",
    "ensemble_covariance",
    "ensemble_cross_covariance",
    "gauss_newton_precision",
    "gaussian_expected_log_lik",
    "ggn_diagonal",
    "hutchinson_hessian_diag",
    "log_marginal_likelihood",
    "newton_update",
    "process_noise_covariance",
    "riemannian_psd_correction",
    "trace_correction",
]
