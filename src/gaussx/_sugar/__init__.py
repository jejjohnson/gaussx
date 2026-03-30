"""GaussX sugar -- compound operations combining multiple primitives."""

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
from gaussx._sugar._linalg import (
    cov_transform,
    diag_conditional_variance,
    trace_product,
)
from gaussx._sugar._project import project
from gaussx._sugar._schur import conditional_variance, schur_complement
from gaussx._sugar._unwhiten import unwhiten, whiten_covariance
from gaussx._sugar._woodbury import woodbury_solve


__all__ = [
    "add_jitter",
    "cavity_distribution",
    "conditional_variance",
    "cov_transform",
    "diag_conditional_variance",
    "gaussian_entropy",
    "gaussian_expected_log_lik",
    "gaussian_log_prob",
    "kl_standard_normal",
    "log_marginal_likelihood",
    "newton_update",
    "process_noise_covariance",
    "project",
    "quadratic_form",
    "schur_complement",
    "trace_correction",
    "trace_product",
    "unwhiten",
    "whiten_covariance",
    "woodbury_solve",
]
