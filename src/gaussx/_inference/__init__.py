"""GaussX inference updates -- variational, natural-gradient, BLR, EnKF."""

from typing import Any

from gaussx._inference._blr import (
    blr_diag_update,
    blr_full_update,
    ggn_diagonal,
    hutchinson_hessian_diag,
)
from gaussx._inference._ensemble import (
    discrepancy_step_size,
    eki_step,
    enkf_analysis,
    ensemble_covariance,
    ensemble_cross_covariance,
    ensemble_kalman_gain,
    etkf_transform,
    euclidean_distance,
    gaspari_cohn,
    haversine_distance,
    inflate_multiplicative,
    inflate_rtpp,
    inflate_rtps,
    localization_matrix,
    localized_kalman_gain,
    tikhonov_augment,
)
from gaussx._inference._inference import (
    cavity_distribution,
    gaussian_expected_log_lik,
    log_marginal_likelihood,
    newton_update,
    trace_correction,
)
from gaussx._inference._natural_gradient import (
    damped_natural_update,
    gauss_newton_precision,
    riemannian_psd_correction,
)
from gaussx._inference._vb_correction import vb_mean_correction

# ``process_noise_covariance`` is a state-space concept and now lives in
# ``gaussx._ssm``. Re-exported here so the historical import path keeps
# working. Imported from the leaf module rather than the ``_ssm`` package to
# avoid pulling the whole subpackage in (and any future import cycle).
from gaussx._ssm._discretise import process_noise_covariance


__all__ = [
    "LaplaceResult",
    "blr_diag_update",
    "blr_full_update",
    "cavity_distribution",
    "damped_natural_update",
    "discrepancy_step_size",
    "eki_step",
    "enkf_analysis",
    "ensemble_covariance",
    "ensemble_cross_covariance",
    "ensemble_kalman_gain",
    "etkf_transform",
    "euclidean_distance",
    "gaspari_cohn",
    "gauss_newton_precision",
    "gaussian_expected_log_lik",
    "ggn_diagonal",
    "haversine_distance",
    "hutchinson_hessian_diag",
    "inflate_multiplicative",
    "inflate_rtpp",
    "inflate_rtps",
    "laplace_mode",
    "localization_matrix",
    "localized_kalman_gain",
    "log_marginal_likelihood",
    "newton_update",
    "process_noise_covariance",
    "riemannian_psd_correction",
    "tikhonov_augment",
    "trace_correction",
    "vb_mean_correction",
]


# laplace_mode takes the numpyro-based GMRF priors (G6), so it needs the
# optional ``numpyro`` dependency. As in ``gaussx._distributions``, resolve it
# lazily (PEP 562) so importing ``gaussx._inference`` works in a base install.
def __getattr__(name: str) -> Any:
    if name in ("LaplaceResult", "laplace_mode"):
        from gaussx._inference import _laplace

        return getattr(_laplace, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
