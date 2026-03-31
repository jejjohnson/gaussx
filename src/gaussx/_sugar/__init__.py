"""GaussX sugar -- compound operations combining multiple primitives."""

from gaussx._sugar._blr import (
    blr_diag_update,
    blr_full_update,
    ggn_diagonal,
    hutchinson_hessian_diag,
)
from gaussx._sugar._elbo import variational_elbo_gaussian, variational_elbo_mc
from gaussx._sugar._gaussian import (
    add_jitter,
    gaussian_entropy,
    gaussian_log_prob,
    kl_standard_normal,
    quadratic_form,
)
from gaussx._sugar._inference import (
    cavity_distribution,
    gaussian_expected_log_lik,
    log_marginal_likelihood,
    newton_update,
    process_noise_covariance,
    trace_correction,
)
from gaussx._sugar._joseph import joseph_update
from gaussx._sugar._kernel_approx import (
    center_kernel,
    centering_operator,
    hsic,
    mmd_squared,
    nystrom_operator,
    rff_operator,
)
from gaussx._sugar._linalg import (
    cov_transform,
    diag_conditional_variance,
    trace_product,
)
from gaussx._sugar._natural_gradient import (
    damped_natural_update,
    gauss_newton_precision,
    riemannian_psd_correction,
)
from gaussx._sugar._project import project
from gaussx._sugar._quadrature import (
    cubature_points,
    gauss_hermite_points,
    sigma_points,
)
from gaussx._sugar._schur import conditional_variance, schur_complement
from gaussx._sugar._svgp import whitened_svgp_predict
from gaussx._sugar._svgp_variance import svgp_variance_adjustment
from gaussx._sugar._unwhiten import unwhiten, whiten_covariance
from gaussx._sugar._woodbury import woodbury_solve


__all__ = [
    "add_jitter",
    "blr_diag_update",
    "blr_full_update",
    "cavity_distribution",
    "center_kernel",
    "centering_operator",
    "conditional_variance",
    "cov_transform",
    "cubature_points",
    "damped_natural_update",
    "diag_conditional_variance",
    "gauss_hermite_points",
    "gauss_newton_precision",
    "gaussian_entropy",
    "gaussian_expected_log_lik",
    "gaussian_log_prob",
    "ggn_diagonal",
    "hsic",
    "hutchinson_hessian_diag",
    "joseph_update",
    "kl_standard_normal",
    "log_marginal_likelihood",
    "mmd_squared",
    "newton_update",
    "nystrom_operator",
    "process_noise_covariance",
    "project",
    "quadratic_form",
    "rff_operator",
    "riemannian_psd_correction",
    "schur_complement",
    "sigma_points",
    "svgp_variance_adjustment",
    "trace_correction",
    "trace_product",
    "unwhiten",
    "variational_elbo_gaussian",
    "variational_elbo_mc",
    "whiten_covariance",
    "whitened_svgp_predict",
    "woodbury_solve",
]
