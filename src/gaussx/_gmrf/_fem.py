r"""P1 finite elements on triangle meshes: mass, stiffness and projector.

For a triangle $T$ with vertices $x_0, x_1, x_2$ and $e_i$ the edge opposite
vertex $i$ (oriented $e_0 = x_2 - x_1$, $e_1 = x_0 - x_2$, $e_2 = x_1 - x_0$),
the P1 hat functions give the local matrices

$$
C^T = \tfrac{|T|}{12}(\mathbf 1\mathbf 1^\top + I),\qquad
\tilde C^T = \tfrac{|T|}{3}I,\qquad
G^T_{ij} = \frac{e_i\cdot e_j}{4|T|}.
$$

They only use edge lengths and angles, so the same code serves planar
(``V × 2``) and surface (``V × 3``, e.g. an icosahedral sphere) meshes. Mesh
generation is out of scope: bring meshes from fmesher, pygmsh or meshio.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int

from gaussx._einx import einsum, rearrange, repeat
from gaussx._gmrf._temporal import _as_float
from gaussx._operators._sparse import SparseOperator


# Local (row, col) pairs of a triangle's 3 × 3 matrix, each unordered pair once.
_LOCAL_ROWS = np.array([0, 1, 2, 0, 0, 1])
_LOCAL_COLS = np.array([0, 1, 2, 1, 2, 2])
# Points located per block: block size × triangles stays below this.
_LOCATE_BLOCK_ENTRIES = 1 << 22
# Barycentric slack for points on edges and vertices.
_INSIDE_TOLERANCE = 1e-8


def fem_matrices(
    vertices: Float[ArrayLike, "V D"],
    triangles: Int[ArrayLike, "T 3"],
) -> tuple[lx.DiagonalLinearOperator, SparseOperator]:
    r"""Lumped mass ``C̃`` and stiffness ``G`` of P1 elements on a triangle mesh.

    The local matrices (module docstring) are computed for all triangles at
    once with einx and scattered with a ``segment_sum`` into ``G``'s
    pattern, which is built on the host from ``triangles`` (two vertices are
    coupled iff they share a triangle). ``vertices`` may be traced; only
    ``triangles`` must be concrete.

    Args:
        vertices: Vertex coordinates, shape ``(V, 2)`` for a planar mesh or
            ``(V, 3)`` for a surface mesh.
        triangles: Vertex indices of each triangle, shape ``(T, 3)``
            (concrete host integers).

    Returns:
        ``(C_lumped, G)``: the diagonal lumped mass matrix as a
        `lineax.DiagonalLinearOperator` and the stiffness matrix as a
        symmetric positive-semidefinite `SparseOperator` (its null space is
        the constants on each connected component).

    Raises:
        ValueError: If the shapes are wrong or an index is out of range.

    Examples:
        ```python
        import numpy as np
        import gaussx

        # The unit right triangle: |T| = 1/2
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        C, G = gaussx.fem_matrices(vertices, np.array([[0, 1, 2]]))
        C.as_matrix()  # I / 6
        G.as_matrix()  # [[1, -1/2, -1/2], [-1/2, 1/2, 0], [-1/2, 0, 1/2]]
        ```
    """
    X = _as_float(vertices)
    tri = _host_triangles(triangles, X.shape[0])
    if X.ndim != 2 or X.shape[1] not in (2, 3):
        raise ValueError(f"vertices must have shape (V, 2) or (V, 3), got {X.shape}.")
    gram, area = _local_geometry(X[tri])
    G_local = einx.divide("t i j, t -> t i j", gram, 4.0 * area)
    values = rearrange(G_local[:, _LOCAL_ROWS, _LOCAL_COLS], "t k -> (t k)")
    n = X.shape[0]
    G = SparseOperator.from_coo(
        tri[:, _LOCAL_ROWS].ravel(),
        tri[:, _LOCAL_COLS].ravel(),
        values,
        (n, n),
        symmetric=True,
        tags=frozenset({lx.positive_semidefinite_tag}),
    )
    mass = jax.ops.segment_sum(
        repeat(area / 3.0, "t -> (t k)", k=3), tri.ravel(), num_segments=n
    )
    return lx.DiagonalLinearOperator(mass), G


def fem_projector(
    vertices: Float[ArrayLike, "V D"],
    triangles: Int[ArrayLike, "T 3"],
    points: Float[ArrayLike, "n D"],
    *,
    triangle_index: Int[ArrayLike, " n"] | None = None,
) -> SparseOperator:
    r"""Observation matrix ``A`` of P1 interpolation: ``(A w)_k = Σ_i ψ_i(s_k) w_i``.

    Row ``k`` holds the three barycentric weights of point ``s_k`` in its
    triangle, so ``A`` has three non-zeros per row and, in a Laplace
    Hessian ``Q + AᵀWA``, never couples vertices that are not already
    neighbours.

    Point location (on the host, blocked brute force over all triangles):

    - **Planar meshes** (``V × 2``): the barycentric test; a point outside
      every triangle raises.
    - **Surface meshes** (``V × 3``) that are star-shaped about the
      centroid of their vertices (spheres, icospheres; every triangle must
      face away from the centroid, which also requires a consistent
      orientation): the ray from the centroid through each point is
      intersected with the triangles, and the weights are taken at the
      intersection, i.e. the point is projected radially onto the mesh (its
      distance to the mesh, the chord error, is ignored).
    - Any other surface mesh needs ``triangle_index``.

    With ``triangle_index`` no location is done and the weights are those of
    the point's orthogonal projection onto the triangle's plane; ``points``
    may then be traced.

    Args:
        vertices: Vertex coordinates, shape ``(V, 2)`` or ``(V, 3)``.
        triangles: Vertex indices of each triangle, shape ``(T, 3)``
            (concrete host integers).
        points: Observation locations, shape ``(n, D)``; concrete unless
            ``triangle_index`` is given.
        triangle_index: The triangle containing each point, shape ``(n,)``
            (concrete host integers).

    Returns:
        A `SparseOperator` of shape ``(n, V)``.

    Raises:
        ValueError: If a point is not on the mesh, or a surface mesh is not
            star-shaped about its centroid and ``triangle_index`` is
            missing.

    Examples:
        ```python
        import numpy as np
        import gaussx

        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        A = gaussx.fem_projector(vertices, triangles, np.array([[0.5, 0.25]]))
        A.as_matrix()  # [[0.5, 0.25, 0.25, 0]]: barycentric weights
        ```
    """
    X = _as_float(vertices)
    tri = _host_triangles(triangles, X.shape[0])
    if X.ndim != 2 or X.shape[1] not in (2, 3):
        raise ValueError(f"vertices must have shape (V, 2) or (V, 3), got {X.shape}.")
    P = jnp.asarray(points, dtype=X.dtype)
    if P.ndim != 2 or P.shape[1] != X.shape[1]:
        raise ValueError(
            f"points must have shape (n, {X.shape[1]}) like vertices, got {P.shape}."
        )
    centre = None
    if triangle_index is None:
        try:
            X_host, P_host = np.asarray(X), np.asarray(P)
        except jax.errors.TracerArrayConversionError as err:
            raise TypeError(
                "Point location needs concrete vertices and points; pass "
                "triangle_index to build the projector from traced values."
            ) from err
        if X.shape[1] == 2:
            index = _locate_planar(X_host, tri, P_host)
        else:
            index = _locate_radial(X_host, tri, P_host)
            centre = einx.mean("v d -> d", X)
    else:
        index = np.asarray(triangle_index)
        if index.shape != (P.shape[0],) or not np.issubdtype(index.dtype, np.integer):
            raise ValueError(
                f"triangle_index must be {P.shape[0]} integers, got {index.shape}."
            )
        if index.size and (index.min() < 0 or index.max() >= tri.shape[0]):
            raise ValueError("triangle_index out of range.")
    corners = X[tri[index]]  # (n, 3, D)
    if centre is not None:
        P = _radial_projection(corners, P, centre)
    weights = _barycentric(corners, P)
    n = P.shape[0]
    return SparseOperator.from_coo(
        np.repeat(np.arange(n), 3),
        tri[index].ravel(),
        rearrange(weights, "n k -> (n k)"),
        (n, X.shape[0]),
    )


# ---------------------------------------------------------------------------
# Geometry (JAX)
# ---------------------------------------------------------------------------


def _host_triangles(triangles: ArrayLike, n_vertices: int) -> np.ndarray:
    try:
        tri = np.asarray(triangles)
    except jax.errors.TracerArrayConversionError as err:
        raise TypeError("triangles must be a concrete host integer array.") from err
    if tri.ndim != 2 or tri.shape[1] != 3 or not np.issubdtype(tri.dtype, np.integer):
        raise ValueError(
            f"triangles must be integers of shape (T, 3), got {tri.shape}."
        )
    if tri.size and (tri.min() < 0 or tri.max() >= n_vertices):
        raise ValueError(f"triangle indices out of range for {n_vertices} vertices.")
    return tri.astype(np.int64)


def _local_geometry(corners: Float[Array, "T 3 D"]) -> tuple[Array, Array]:
    """Gram matrix ``e_i · e_j`` of the opposite edges, and the areas."""
    edges = corners[:, [2, 0, 1]] - corners[:, [1, 2, 0]]
    gram = einsum(edges, edges, "t i d, t j d -> t i j")
    # |T|² = (|e_0|²|e_1|² − (e_0·e_1)²) / 4 in any dimension.
    area = 0.5 * jnp.sqrt(gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] ** 2)
    return gram, area


def _barycentric(
    corners: Float[Array, "n 3 D"], points: Float[Array, "n D"]
) -> Float[Array, "n 3"]:
    """Barycentric weights of each point's projection onto its triangle's plane."""
    origin = corners[:, 0]
    spans = corners[:, 1:] - rearrange(origin, "n d -> n 1 d")  # (n, 2, D)
    gram = einsum(spans, spans, "n i d, n j d -> n i j")
    rhs = einsum(spans, points - origin, "n i d, n d -> n i")
    det = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] ** 2
    u = (gram[:, 1, 1] * rhs[:, 0] - gram[:, 0, 1] * rhs[:, 1]) / det
    v = (gram[:, 0, 0] * rhs[:, 1] - gram[:, 0, 1] * rhs[:, 0]) / det
    return rearrange(jnp.stack([1.0 - u - v, u, v]), "k n -> n k")


def _radial_projection(
    corners: Float[Array, "n 3 D"], points: Float[Array, "n D"], centre: Array
) -> Float[Array, "n D"]:
    """Where the ray from ``centre`` through each point meets its triangle's plane."""
    normal = jnp.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    direction = points - centre
    t = einsum(normal, corners[:, 0] - centre, "n d, n d -> n") / einsum(
        normal, direction, "n d, n d -> n"
    )
    return centre + einx.multiply("n, n d -> n d", t, direction)


# ---------------------------------------------------------------------------
# Point location (host, NumPy)
# ---------------------------------------------------------------------------


def _blocks(n_points: int, n_triangles: int):
    size = max(1, _LOCATE_BLOCK_ENTRIES // max(n_triangles, 1))
    for start in range(0, n_points, size):
        yield slice(start, min(start + size, n_points))


def _pick(score: np.ndarray, offset: int) -> np.ndarray:
    """Best triangle per point (largest minimum barycentric coordinate)."""
    best = einx.argmax("b [t] -> b", score)
    best_score = einx.get_at("b [t], b -> b", score, best)
    outside = np.flatnonzero(~(best_score >= -_INSIDE_TOLERANCE))
    if outside.size:
        raise ValueError(
            f"point {offset + int(outside[0])} does not lie on the mesh "
            f"({outside.size} such points in this block)."
        )
    return best


def _locate_planar(
    vertices: np.ndarray, tri: np.ndarray, points: np.ndarray
) -> np.ndarray:
    origin = vertices[tri[:, 0]]
    e1 = vertices[tri[:, 1]] - origin
    e2 = vertices[tri[:, 2]] - origin
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    det = np.where(det == 0.0, np.nan, det)  # degenerate triangles never match
    index = np.empty(points.shape[0], dtype=np.int64)
    for block in _blocks(points.shape[0], tri.shape[0]):
        r = einx.subtract("b d, t d -> b t d", points[block], origin)
        u = (
            einx.multiply("t, b t -> b t", e2[:, 1], r[..., 0])
            - einx.multiply("t, b t -> b t", e2[:, 0], r[..., 1])
        ) / det
        v = (
            einx.multiply("t, b t -> b t", e1[:, 0], r[..., 1])
            - einx.multiply("t, b t -> b t", e1[:, 1], r[..., 0])
        ) / det
        score = np.minimum(np.minimum(u, v), 1.0 - u - v)
        index[block] = _pick(np.nan_to_num(score, nan=-np.inf), block.start)
    return index


def _locate_radial(
    vertices: np.ndarray, tri: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Möller-Trumbore along rays from the vertex centroid (star-shaped meshes)."""
    centre = einx.mean("v d -> d", vertices)
    origin = vertices[tri[:, 0]]
    e1 = vertices[tri[:, 1]] - origin
    e2 = vertices[tri[:, 2]] - origin
    normal = np.cross(e1, e2)
    offset = centre - origin  # T in Möller-Trumbore
    facing = einx.dot("t d, t d -> t", normal, -offset)
    scale = np.sqrt(einx.dot("t d, t d -> t", normal, normal)) * np.sqrt(
        einx.dot("t d, t d -> t", offset, offset)
    )
    if not (np.all(facing > 1e-12 * scale) or np.all(facing < -1e-12 * scale)):
        raise ValueError(
            "The surface mesh is not star-shaped about the centroid of its "
            "vertices (or its triangles are not consistently oriented), so "
            "points cannot be located by rays from the centroid; pass "
            "triangle_index with the triangle containing each point."
        )
    w_u = np.cross(e2, offset)
    w_v = np.cross(offset, e1)
    s_t = einx.dot("t d, t d -> t", e2, w_v)
    index = np.empty(points.shape[0], dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for block in _blocks(points.shape[0], tri.shape[0]):
            index[block] = _radial_block(
                points[block] - centre, normal, w_u, w_v, s_t, block.start
            )
    return index


def _radial_block(
    direction: np.ndarray,
    normal: np.ndarray,
    w_u: np.ndarray,
    w_v: np.ndarray,
    s_t: np.ndarray,
    offset: int,
) -> np.ndarray:
    """Best triangle for one block of rays ``centre + t · direction``."""
    det = -einx.dot("b d, t d -> b t", direction, normal)
    u = einx.dot("b d, t d -> b t", direction, w_u) / det
    v = einx.dot("b d, t d -> b t", direction, w_v) / det
    t = einx.divide("t, b t -> b t", s_t, det)
    score = np.minimum(np.minimum(u, v), 1.0 - u - v)
    score = np.where((t > 0.0) & np.isfinite(score), score, -np.inf)
    return _pick(score, offset)
