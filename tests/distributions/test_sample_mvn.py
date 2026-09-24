"""Tests for structure-aware multivariate-normal sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
import gaussx._distributions._sample as sample_module
import gaussx._primitives._cholesky as cholesky_module
from gaussx._operators import low_rank_plus_diag
from gaussx._testing import assert_sample_moments, random_pd_operator


# 4096 draws per test; each bound comes from assert_sample_moments, which
# scales with the estimator's own standard error rather than a fixed atol.
_NUM_SAMPLES = 4096


def _pd(seed: int, n: int) -> lx.MatrixLinearOperator:
    return random_pd_operator(jr.key(seed), n)


def _structured_covariances() -> dict[str, lx.AbstractLinearOperator]:
    """Small instances of every structure `sample_mvn` dispatches on."""
    lags = jnp.arange(6.0)
    kronecker = gaussx.Kronecker(_pd(1, 2), _pd(2, 3))
    return {
        "dense": _pd(0, 5),
        "diagonal": lx.DiagonalLinearOperator(jnp.array([0.5, 1.0, 2.0, 4.0])),
        "kronecker": kronecker,
        "block_diag": gaussx.BlockDiag(_pd(3, 2), _pd(4, 3)),
        # Diagonally dominant, so positive definite: 3 blocks of 2x2.
        "block_tridiag": gaussx.BlockTriDiag(
            jnp.stack([4.0 * jnp.eye(2)] * 3),
            jnp.stack([jnp.array([[0.5, 0.2], [0.1, 0.4]])] * 2),
        ),
        "toeplitz": gaussx.Toeplitz(jnp.exp(-0.5 * lags**2)),
        "kronecker_sum": gaussx.KroneckerSum(_pd(5, 2), _pd(6, 3)),
        "sum_of_kroneckers": gaussx.SumOfKroneckers(
            gaussx.Kronecker(_pd(7, 2), _pd(8, 3)),
            gaussx.Kronecker(_pd(9, 2), _pd(10, 3)),
        ),
        "low_rank": low_rank_plus_diag(
            jnp.linspace(0.5, 1.5, 5), jr.normal(jr.key(11), (5, 2))
        ),
        "tagged": lx.TaggedLinearOperator(kronecker, lx.positive_semidefinite_tag),
        "scaled": 2.5 * kronecker,
    }


_COVARIANCES = _structured_covariances()


@pytest.mark.parametrize("name", list(_COVARIANCES))
def test_samples_have_the_requested_moments(name: str) -> None:
    covariance = _COVARIANCES[name]
    n = covariance.in_size()
    mean = jnp.linspace(-1.0, 1.0, n)

    samples = gaussx.sample_mvn(
        mean, covariance, key=jr.key(12), num_samples=_NUM_SAMPLES
    )

    assert samples.shape == (_NUM_SAMPLES, n)
    assert_sample_moments(samples, mean, covariance.as_matrix())


@pytest.mark.parametrize(
    "name",
    [
        "diagonal",
        "kronecker",
        "block_diag",
        "block_tridiag",
        "toeplitz",
        "kronecker_sum",
        "low_rank",
        "tagged",
        "scaled",
    ],
)
def test_structured_covariances_are_never_densified(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Small dense factors (a Kronecker factor, one block) are expected to be
    # factorised densely; the full covariance never is. A full-size dense
    # factor is poisoned with NaN rather than refused, because a fallback held
    # in an unexecuted `lax.cond` branch is traced but must never run.
    covariance = _COVARIANCES[name]
    dense_cholesky = cholesky_module._cholesky_dense
    dense_root = sample_module.dense_symmetric_sqrt

    def poison_full_size_cholesky(operator):
        factor = dense_cholesky(operator)
        if operator.in_size() >= covariance.in_size():
            return lx.MatrixLinearOperator(jnp.full_like(factor.as_matrix(), jnp.nan))
        return factor

    def poison_full_size_root(matrix):
        root = dense_root(matrix)
        if matrix.shape[0] >= covariance.in_size():
            return jnp.full_like(root, jnp.nan)
        return root

    monkeypatch.setattr(cholesky_module, "_cholesky_dense", poison_full_size_cholesky)
    monkeypatch.setattr(sample_module, "dense_symmetric_sqrt", poison_full_size_root)

    samples = gaussx.sample_mvn(
        jnp.zeros(covariance.in_size()), covariance, key=jr.key(13), num_samples=4
    )

    assert jnp.all(jnp.isfinite(samples)), f"{name} took a full-size dense factor"


def test_batched_mean_gets_independent_noise() -> None:
    covariance = lx.DiagonalLinearOperator(jnp.array([1.0, 4.0]))
    mean = jnp.array([[0.0, 0.0], [10.0, -10.0], [5.0, 5.0]])

    samples = gaussx.sample_mvn(mean, covariance, key=jr.key(14), num_samples=4)

    assert samples.shape == (4, 3, 2)
    # Same covariance, different noise: no two batch elements share a draw.
    centred = samples - mean[None]
    assert not jnp.allclose(centred[:, 0], centred[:, 1])
    assert not jnp.allclose(centred[:, 1], centred[:, 2])


def test_batched_draws_have_the_requested_moments() -> None:
    covariance = gaussx.Kronecker(_pd(15, 2), _pd(16, 2))
    mean = jnp.stack([jnp.zeros(4), jnp.ones(4)])

    samples = gaussx.sample_mvn(
        mean, covariance, key=jr.key(17), num_samples=_NUM_SAMPLES
    )

    for index in range(2):
        assert_sample_moments(samples[:, index], mean[index], covariance.as_matrix())


@pytest.mark.parametrize("name", ["toeplitz", "dense", "kronecker"])
def test_an_empty_batch_returns_an_empty_draw(name: str) -> None:
    covariance = _COVARIANCES[name]
    mean = jnp.zeros((0, covariance.in_size()))

    samples = gaussx.sample_mvn(mean, covariance, key=jr.key(44), num_samples=3)

    assert samples.shape == (3, 0, covariance.in_size())


def test_negative_low_rank_weights_fall_back_to_dense() -> None:
    # A Woodbury downdate D - uuᵀ is still positive definite here, but it has
    # no U√D factor; the draw must come from the dense Cholesky instead.
    u = jnp.array([[0.5], [0.3], [0.2]])
    covariance = gaussx.LowRankUpdate(
        lx.DiagonalLinearOperator(jnp.array([2.0, 2.0, 2.0])), u, jnp.array([-1.0])
    )
    mean = jnp.zeros(3)

    samples = gaussx.sample_mvn(
        mean, covariance, key=jr.key(18), num_samples=_NUM_SAMPLES
    )

    assert_sample_moments(samples, mean, covariance.as_matrix())


@pytest.mark.parametrize(
    ("name", "covariance"),
    [
        # Both factors negative definite; the product is the identity.
        (
            "indefinite_kronecker_factors",
            gaussx.Kronecker(
                lx.DiagonalLinearOperator(-jnp.ones(2)),
                lx.MatrixLinearOperator(-jnp.eye(3)),
            ),
        ),
        # Plain, untagged symmetric factors.
        (
            "untagged_kronecker_sum",
            gaussx.KroneckerSum(
                lx.MatrixLinearOperator(_pd(22, 2).as_matrix()),
                lx.MatrixLinearOperator(_pd(23, 3).as_matrix()),
            ),
        ),
        # Positive definite, but its 2x circulant embedding is not.
        (
            "toeplitz_needing_a_larger_embedding",
            gaussx.Toeplitz(jnp.array([3.2900615, -1.8108547, -0.6982663])),
        ),
        # diag(-1, 2) + [2, 0][2, 0]^T = diag(3, 2): the base alone is indefinite.
        (
            "low_rank_with_indefinite_base",
            gaussx.LowRankUpdate(
                lx.DiagonalLinearOperator(jnp.array([-1.0, 2.0])),
                jnp.array([[2.0], [0.0]]),
                jnp.array([1.0]),
            ),
        ),
        # Tagged symmetric, but V = 2U: the update is 2UU^T, not UU^T.
        (
            "low_rank_with_scaled_right_factor",
            gaussx.LowRankUpdate(
                lx.DiagonalLinearOperator(jnp.ones(3)),
                jnp.array([[0.5], [0.3], [0.2]]),
                jnp.array([1.0]),
                jnp.array([[1.0], [0.6], [0.4]]),
                tags=lx.symmetric_tag,
            ),
        ),
        # Above the old 50-step Lanczos cap: the draw must stay exact.
        (
            "large_sum_of_kroneckers",
            gaussx.SumOfKroneckers(
                gaussx.Kronecker(_pd(24, 8), _pd(25, 8)),
                gaussx.Kronecker(_pd(26, 8), _pd(27, 8)),
            ),
        ),
    ],
)
def test_awkward_but_valid_covariances_are_sampled_exactly(
    name: str, covariance: lx.AbstractLinearOperator
) -> None:
    del name
    mean = jnp.zeros(covariance.in_size())

    samples = gaussx.sample_mvn(
        mean, covariance, key=jr.key(28), num_samples=_NUM_SAMPLES
    )

    assert jnp.all(jnp.isfinite(samples))
    assert_sample_moments(samples, mean, covariance.as_matrix())


@pytest.mark.parametrize(
    ("name", "covariance"),
    [
        # (-1) * (-I) = I and (-I) / (-1) = I: valid, but not through sqrt(c).
        ("negative_scalar_multiple", -1.0 * lx.DiagonalLinearOperator(-jnp.ones(3))),
        (
            "negative_scalar_quotient",
            lx.DiagonalLinearOperator(-jnp.ones(3)) / -1.0,
        ),
        # Square product of rectangular factors: the 2x2 all-ones matrix.
        (
            "rectangular_kronecker_factors",
            gaussx.Kronecker(
                lx.MatrixLinearOperator(jnp.ones((1, 2))),
                lx.MatrixLinearOperator(jnp.ones((2, 1))),
            ),
        ),
        # Square overall from rectangular blocks: diag(1, 0, 2).
        (
            "rectangular_blocks",
            gaussx.BlockDiag(
                lx.MatrixLinearOperator(jnp.array([[1.0, 0.0]])),
                lx.MatrixLinearOperator(jnp.array([[0.0], [2.0]])),
            ),
        ),
        # Singular: each diagonal block is [[1, 1], [1, 1]], no coupling.
        (
            "singular_block_tridiag",
            gaussx.BlockTriDiag(jnp.ones((3, 2, 2)), jnp.zeros((2, 2, 2))),
        ),
    ],
)
def test_more_awkward_covariances_are_sampled_exactly(
    name: str, covariance: lx.AbstractLinearOperator
) -> None:
    del name
    mean = jnp.zeros(covariance.in_size())

    samples = gaussx.sample_mvn(
        mean, covariance, key=jr.key(34), num_samples=_NUM_SAMPLES
    )

    assert jnp.all(jnp.isfinite(samples))
    assert_sample_moments(samples, mean, covariance.as_matrix())


@pytest.mark.parametrize("structure", ["kronecker", "kronecker_sum"])
def test_pathwise_gradients_are_finite_at_repeated_eigenvalues(structure: str) -> None:
    # t I has one eigenvalue repeated three times; differentiating through
    # eigh's eigenvectors there gives NaN.
    other = _pd(36, 2).as_matrix()

    def loss(t):
        isotropic = lx.MatrixLinearOperator(t * jnp.eye(3))
        if structure == "kronecker":
            covariance = gaussx.Kronecker(isotropic, lx.MatrixLinearOperator(other))
        else:
            covariance = gaussx.KroneckerSum(isotropic, lx.MatrixLinearOperator(other))
        samples = gaussx.sample_mvn(
            jnp.zeros(6), covariance, key=jr.key(37), num_samples=16
        )
        return jnp.sum(samples**2)

    t = 1.3
    gradient = jax.grad(loss)(t)
    step = 1e-6
    finite_difference = (loss(t + step) - loss(t - step)) / (2 * step)

    assert jnp.isfinite(gradient)
    assert jnp.allclose(gradient, finite_difference, rtol=1e-5)


def test_kronecker_sum_root_jvp_matches_the_dense_root() -> None:
    # The structured Sylvester JVP against dense_symmetric_sqrt's own JVP on
    # the materialised A ⊕ B, for random symmetric tangents.
    a, b = _pd(38, 3).as_matrix(), _pd(39, 2).as_matrix()
    noise = jr.normal(jr.key(40), (5, 3, 2))
    tangent_a = jr.normal(jr.key(41), (3, 3))
    tangent_b = jr.normal(jr.key(42), (2, 2))
    tangent_noise = jr.normal(jr.key(43), (5, 3, 2))

    def structured(a, b, noise):
        return sample_module._kronecker_sum_root_action(a, b, noise)

    def dense(a, b, noise):
        kron_sum = jnp.kron(a, jnp.eye(2)) + jnp.kron(jnp.eye(3), b)
        root = sample_module.dense_symmetric_sqrt(kron_sum)
        return jax.vmap(lambda z: (root @ z.reshape(-1)).reshape(3, 2))(noise)

    args = (a, b, noise)
    tangents = (tangent_a, tangent_b, tangent_noise)
    structured_out, structured_tangent = jax.jvp(structured, args, tangents)
    dense_out, dense_tangent = jax.jvp(dense, args, tangents)

    assert jnp.allclose(structured_out, dense_out, atol=1e-10)
    assert jnp.allclose(structured_tangent, dense_tangent, atol=1e-10)


def test_kronecker_sum_with_traced_factors() -> None:
    # The KroneckerSumSqrt route used a Python bool on a traced eigenvalue.
    a, b = _pd(29, 2).as_matrix(), _pd(30, 3).as_matrix()

    @jax.jit
    def draw(a, b):
        covariance = gaussx.KroneckerSum(
            lx.MatrixLinearOperator(a, lx.positive_semidefinite_tag),
            lx.MatrixLinearOperator(b, lx.positive_semidefinite_tag),
        )
        return gaussx.sample_mvn(
            jnp.zeros(6), covariance, key=jr.key(31), num_samples=_NUM_SAMPLES
        )

    samples = draw(a, b)
    expected = gaussx.KroneckerSum(
        lx.MatrixLinearOperator(a), lx.MatrixLinearOperator(b)
    ).as_matrix()
    assert_sample_moments(samples, jnp.zeros(6), expected)


def test_a_wider_mean_dtype_is_kept() -> None:
    covariance = lx.DiagonalLinearOperator(jnp.ones(2, dtype=jnp.float32))
    mean = jnp.array([1e8 + 0.25, -3.0], dtype=jnp.float64)

    samples = gaussx.sample_mvn(mean, covariance, key=jr.key(33), num_samples=2)

    assert samples.dtype == jnp.float64
    # float32 would round 1e8 + 0.25 to 1e8; unit-variance noise cannot hide it.
    assert jnp.all(jnp.abs(samples[:, 0] - mean[0]) < 10.0)
    assert samples[0, 0] != jnp.float32(samples[0, 0])


@pytest.mark.parametrize(
    ("name", "build", "argument"),
    [
        # 2I - uu^T: a valid downdate, but it has no U√D factor.
        (
            "negative_weight",
            lambda w: gaussx.LowRankUpdate(
                lx.DiagonalLinearOperator(jnp.full((3,), 2.0)),
                jnp.array([[0.5], [0.3], [0.2]]),
                w,
            ),
            jnp.array([-1.0]),
        ),
        # diag(-1, 2) + [2, 0][2, 0]^T = diag(3, 2); the base alone is not PSD.
        (
            "indefinite_diagonal_base",
            lambda d: gaussx.LowRankUpdate(
                lx.DiagonalLinearOperator(d),
                jnp.array([[2.0], [0.0]]),
                jnp.array([1.0]),
            ),
            jnp.array([-1.0, 2.0]),
        ),
        # (-2)(-I) = 2I: valid, but not through sqrt(c).
        (
            "negative_scalar",
            lambda c: c * lx.DiagonalLinearOperator(-jnp.ones(3)),
            jnp.asarray(-2.0),
        ),
    ],
)
def test_traced_values_that_rule_out_a_structured_route_fall_back_exactly(
    name: str, build, argument
) -> None:
    # The structured route's precondition is only known at run time here, so
    # the dense square root must take over rather than raise or return NaN.
    del name

    @jax.jit
    def draw(argument):
        covariance = build(argument)
        return gaussx.sample_mvn(
            jnp.zeros(covariance.in_size()),
            covariance,
            key=jr.key(45),
            num_samples=_NUM_SAMPLES,
        )

    samples = draw(argument)
    expected = build(argument).as_matrix()
    assert jnp.all(jnp.isfinite(samples))
    assert_sample_moments(samples, jnp.zeros(expected.shape[0]), expected)


def test_an_empty_batch_keeps_the_represented_dtype() -> None:
    # float32 base, float64 U: the matrix is float64, but in_structure says
    # float32. Empty and non-empty draws must agree on the dtype.
    covariance = gaussx.LowRankUpdate(
        lx.DiagonalLinearOperator(jnp.ones(3, dtype=jnp.float32)),
        jnp.array([[0.5], [0.3], [0.2]], dtype=jnp.float64),
    )
    mean = jnp.zeros(3, dtype=jnp.float32)

    empty = gaussx.sample_mvn(mean[None][:0], covariance, key=jr.key(46))
    full = gaussx.sample_mvn(mean, covariance, key=jr.key(46))

    assert empty.dtype == full.dtype == jnp.float64


@pytest.mark.parametrize(
    ("name", "loss"),
    [
        # [1, 0.5] is PD, but its 2x circulant embedding has spectrum [2, 1, 0].
        (
            "toeplitz_with_a_singular_embedding",
            lambda t: gaussx.Toeplitz(jnp.array([1.0, t])),
        ),
        # A zero low-rank weight: B + 0·uuᵀ is still PD, but √0 is not smooth.
        (
            "low_rank_with_a_zero_weight",
            lambda t: low_rank_plus_diag(
                jnp.array([1.0, 2.0, 1.5]),
                jnp.array([[0.5, 0.1], [0.3, -0.2], [0.2, 0.4]]),
                jnp.stack([t - 0.5, jnp.asarray(1.0)]),
            ),
        ),
    ],
)
def test_pathwise_gradients_are_finite_at_boundary_values(name: str, loss) -> None:
    # At the boundary the structured route is ruled out and the dense root
    # takes over, so the gradient must equal the dense route's gradient for
    # the same key. (A finite difference would straddle the two routes, which
    # map the noise differently.)
    del name

    def objective(t, dense):
        covariance = loss(t)
        if dense:
            covariance = lx.MatrixLinearOperator(covariance.as_matrix())
        samples = gaussx.sample_mvn(
            jnp.zeros(covariance.in_size()), covariance, key=jr.key(47), num_samples=8
        )
        return jnp.sum(samples**2)

    t = 0.5
    gradient = jax.grad(objective)(t, False)
    reference = jax.grad(objective)(t, True)

    assert jnp.isfinite(gradient)
    assert jnp.allclose(gradient, reference, rtol=1e-8)


def test_jit_matches_eager() -> None:
    covariance = _COVARIANCES["kronecker"]
    mean = jnp.ones(covariance.in_size())

    eager = gaussx.sample_mvn(mean, covariance, key=jr.key(20), num_samples=3)
    jitted = jax.jit(
        lambda m, key: gaussx.sample_mvn(m, covariance, key=key, num_samples=3)
    )(mean, jr.key(20))

    assert jnp.allclose(eager, jitted)


def test_vmap_over_keys() -> None:
    covariance = _COVARIANCES["diagonal"]
    keys = jr.split(jr.key(21), 5)

    samples = jax.vmap(
        lambda key: gaussx.sample_mvn(jnp.zeros(4), covariance, key=key)
    )(keys)

    assert samples.shape == (5, 1, 4)
    assert not jnp.allclose(samples[0], samples[1])


def test_rejects_rectangular_covariance() -> None:
    with pytest.raises(ValueError, match="square"):
        gaussx.sample_mvn(
            jnp.zeros(3), lx.MatrixLinearOperator(jnp.ones((3, 2))), key=jr.key(0)
        )


def test_rejects_mismatched_mean() -> None:
    with pytest.raises(ValueError, match="mean must have shape"):
        gaussx.sample_mvn(jnp.zeros(4), _COVARIANCES["dense"], key=jr.key(0))


def test_rejects_non_positive_sample_count() -> None:
    with pytest.raises(ValueError, match="num_samples"):
        gaussx.sample_mvn(
            jnp.zeros(5), _COVARIANCES["dense"], key=jr.key(0), num_samples=0
        )
