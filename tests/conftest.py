import os

import equinox.internal as eqxi
import jax
import pytest


# The suite runs with x64 on by default. GAUSSX_TEST_X64=0 runs it in JAX's
# default configuration (float32, no float64) instead: the no-x64 CI lane.
X64 = os.environ.get("GAUSSX_TEST_X64", "1") != "0"
jax.config.update("jax_enable_x64", X64)


def pytest_collection_modifyitems(config, items):
    """Skip ``x64_only`` tests when the lane runs without x64.

    Every ``x64_only`` marker must say why the test needs float64, as
    ``@pytest.mark.x64_only(reason="...")``; one without a reason is an error
    in both lanes, so it cannot slip in through the x64 lane.
    """
    for item in items:
        marker = item.get_closest_marker("x64_only")
        if marker is None:
            continue
        reason = marker.kwargs.get("reason")
        if not reason:
            raise pytest.UsageError(
                f'{item.nodeid}: x64_only needs a reason, as x64_only(reason="...")'
            )
        if not X64:
            item.add_marker(pytest.mark.skip(reason=f"x64_only: {reason}"))


@pytest.fixture(autouse=True, scope="module")
def _clear_jax_caches():
    """Drop JAX's compilation caches after each test module.

    Each xdist worker otherwise keeps every executable it has compiled, and
    under jax 0.10.2 a fast-lane worker grows to ~5 GB. Four of them exceed a
    16 GB GitHub runner, which then shuts down at ~90%. Clearing per module
    caps a worker at ~2.4 GB (tests/ssm, measured) at no cost in run time.
    """
    yield
    jax.clear_caches()


@pytest.fixture
def getkey():
    return eqxi.GetKey()
