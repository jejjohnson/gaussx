"""gaussx's Earth defaults equal geophysix's.

gaussx sits below geophysix in the stack and must not import it at run time
(geophysix design decision 5), so it repeats the constants. This test, the only
place gaussx touches geophysix, keeps the copies equal. It is skipped where
geophysix is not installed.
"""

from __future__ import annotations

import inspect
import math

import pytest

from gaussx import haversine_distance
from gaussx._inference._localization import EARTH_MEAN_RADIUS_M


def test_haversine_default_radius_is_the_iugg_mean_radius() -> None:
    earth = pytest.importorskip("geophysix.earth")
    default = inspect.signature(haversine_distance).parameters["radius"].default
    assert math.isclose(default, earth.SPHERE.semi_major, rel_tol=1e-12)
    assert math.isclose(default, earth.WGS84.mean_radius, rel_tol=1e-12)


def test_module_constant_is_the_haversine_default() -> None:
    default = inspect.signature(haversine_distance).parameters["radius"].default
    assert default == EARTH_MEAN_RADIUS_M
