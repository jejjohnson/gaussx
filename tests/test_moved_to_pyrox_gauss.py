"""The NumPyro distribution classes left gaussx for pyrox-gauss (gh-843).

Accessing one of them on ``gaussx`` raises an ``AttributeError`` that names
the new import, so a stale caller learns where the class went.
"""

from __future__ import annotations

import pytest

import gaussx


MOVED = [
    "AbstractMultivariateNormal",
    "MultivariateNormal",
    "MultivariateNormalPrecision",
    "MarkovGaussian",
    "LGSSM",
    "MaskedLGSSM",
    "LGSSMFactory",
]


@pytest.mark.parametrize("name", MOVED)
def test_moved_name_says_where_it_went(name: str) -> None:
    with pytest.raises(AttributeError, match=f"from pyrox_gauss import {name}"):
        getattr(gaussx, name)


@pytest.mark.parametrize("name", MOVED)
def test_moved_name_is_not_public(name: str) -> None:
    assert name not in gaussx.__all__
    assert not hasattr(gaussx._distributions, name)


def test_unknown_name_keeps_the_plain_message() -> None:
    with pytest.raises(AttributeError, match=r"has no attribute 'NotAName'$"):
        gaussx.NotAName  # noqa: B018
