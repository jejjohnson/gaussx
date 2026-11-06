"""Random sketches S ∈ ℝ^(d × M), subspace embeddings for randomized linear algebra."""

from gaussx._sketching._base import AbstractSketch
from gaussx._sketching._dense import GaussianSketch, OrthonormalSketch
from gaussx._sketching._hadamard import hadamard_transform
from gaussx._sketching._sampling import RowSamplingSketch
from gaussx._sketching._sparse_sign import SparseSignSketch
from gaussx._sketching._srht import SRHTSketch


__all__ = [
    "AbstractSketch",
    "GaussianSketch",
    "OrthonormalSketch",
    "RowSamplingSketch",
    "SRHTSketch",
    "SparseSignSketch",
    "hadamard_transform",
]
