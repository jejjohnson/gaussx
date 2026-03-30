"""Structured linear algebra and Gaussian primitives for JAX."""

__version__ = "0.0.4"

from gaussx._expfam import (
    GaussianExpFam as GaussianExpFam,
    fisher_info as fisher_info,
    kl_divergence as kl_divergence,
    log_partition as log_partition,
    sufficient_stats as sufficient_stats,
    to_expectation as to_expectation,
    to_natural as to_natural,
)
from gaussx._operators import (
    BlockDiag as BlockDiag,
    Kronecker as Kronecker,
    LowRankUpdate as LowRankUpdate,
    low_rank_plus_diag as low_rank_plus_diag,
    low_rank_plus_identity as low_rank_plus_identity,
    svd_low_rank_plus_diag as svd_low_rank_plus_diag,
)
from gaussx._primitives import (
    cholesky as cholesky,
    diag as diag,
    eig as eig,
    eigvals as eigvals,
    inv as inv,
    logdet as logdet,
    solve as solve,
    sqrt as sqrt,
    svd as svd,
    trace as trace,
)
from gaussx._recipes import (
    FilterState as FilterState,
    ensemble_covariance as ensemble_covariance,
    ensemble_cross_covariance as ensemble_cross_covariance,
    expectation_to_natural as expectation_to_natural,
    kalman_filter as kalman_filter,
    kalman_gain as kalman_gain,
    natural_to_expectation as natural_to_expectation,
    rts_smoother as rts_smoother,
)
from gaussx._recipes._parallel_kalman import (
    parallel_kalman_filter as parallel_kalman_filter,
    parallel_rts_smoother as parallel_rts_smoother,
)
from gaussx._strategies import (
    AbstractSolverStrategy as AbstractSolverStrategy,
    AutoSolver as AutoSolver,
    BBMMSolver as BBMMSolver,
    CGSolver as CGSolver,
    DenseSolver as DenseSolver,
    LSMRSolver as LSMRSolver,
    PreconditionedCGSolver as PreconditionedCGSolver,
)
from gaussx._sugar import (
    add_jitter as add_jitter,
    cavity_distribution as cavity_distribution,
    conditional_variance as conditional_variance,
    cov_transform as cov_transform,
    diag_conditional_variance as diag_conditional_variance,
    gaussian_entropy as gaussian_entropy,
    gaussian_expected_log_lik as gaussian_expected_log_lik,
    gaussian_log_prob as gaussian_log_prob,
    kl_standard_normal as kl_standard_normal,
    log_marginal_likelihood as log_marginal_likelihood,
    newton_update as newton_update,
    process_noise_covariance as process_noise_covariance,
    project as project,
    quadratic_form as quadratic_form,
    schur_complement as schur_complement,
    trace_correction as trace_correction,
    trace_product as trace_product,
    unwhiten as unwhiten,
    whiten_covariance as whiten_covariance,
    woodbury_solve as woodbury_solve,
)
from gaussx._tags import (
    block_diagonal_tag as block_diagonal_tag,
    diagonal_tag as diagonal_tag,
    is_block_diagonal as is_block_diagonal,
    is_diagonal as is_diagonal,
    is_kronecker as is_kronecker,
    is_low_rank as is_low_rank,
    is_lower_triangular as is_lower_triangular,
    is_negative_semidefinite as is_negative_semidefinite,
    is_positive_semidefinite as is_positive_semidefinite,
    is_symmetric as is_symmetric,
    is_upper_triangular as is_upper_triangular,
    kronecker_tag as kronecker_tag,
    low_rank_tag as low_rank_tag,
    lower_triangular_tag as lower_triangular_tag,
    negative_semidefinite_tag as negative_semidefinite_tag,
    positive_semidefinite_tag as positive_semidefinite_tag,
    symmetric_tag as symmetric_tag,
    tridiagonal_tag as tridiagonal_tag,
    unit_diagonal_tag as unit_diagonal_tag,
    upper_triangular_tag as upper_triangular_tag,
)


try:
    from gaussx._distributions import (
        MultivariateNormal as MultivariateNormal,
        MultivariateNormalPrecision as MultivariateNormalPrecision,
    )
except ModuleNotFoundError as _e:
    if _e.name != "numpyro":
        raise
