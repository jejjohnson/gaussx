"""gaussx's own deprecation warning category and helper (gh-332)."""

from __future__ import annotations

import os
import warnings

import equinox


class GaussxDeprecationWarning(DeprecationWarning):
    """A gaussx API is deprecated.

    A subclass of `DeprecationWarning`, so ``pytest.warns(DeprecationWarning)``
    and ``-W ignore::DeprecationWarning`` still match it, while filters (such
    as the test suite's ``error::`` entry) can single out gaussx's own.
    """


# Frames inside these are skipped, so a warning lands on the caller's line:
# gaussx itself (the deprecated function may be reached through gaussx
# helpers) and equinox (an eqx.Module.__init__ is called from equinox's
# metaclass, so a fixed stacklevel would point into equinox).
_SKIP = (
    os.path.dirname(__file__),
    os.path.dirname(equinox.__file__),
)


def warn_deprecated(message: str) -> None:
    """Emit a `GaussxDeprecationWarning` attributed to the caller's code."""
    warnings.warn(
        message, GaussxDeprecationWarning, stacklevel=2, skip_file_prefixes=_SKIP
    )
