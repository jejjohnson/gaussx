"""GaussX randomized linear algebra: range finders, low-rank factorisations, CUR.

Each method reaches an M × N operator A only through matvecs, or a PSD matrix only
through its diagonal and some columns, so A is never materialised to sketch it.
"""

from gaussx._randomized._interpolative import CUR, ColumnID, column_id, cur
from gaussx._randomized._nystrom import randomized_nystrom
from gaussx._randomized._range_finder import qb, range_finder
from gaussx._randomized._rpcholesky import rp_cholesky
from gaussx._randomized._svd import randomized_eigh, randomized_svd


__all__ = [
    "CUR",
    "ColumnID",
    "column_id",
    "cur",
    "qb",
    "randomized_eigh",
    "randomized_nystrom",
    "randomized_svd",
    "range_finder",
    "rp_cholesky",
]
