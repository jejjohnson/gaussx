import equinox.internal as eqxi
import jax
import pytest


jax.config.update("jax_enable_x64", True)


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
