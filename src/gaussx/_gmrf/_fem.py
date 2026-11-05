"""P1 finite elements on triangle meshes: mass, stiffness and projector.

For a triangle T with vertices s_0, s_1, s_2 and e_i the edge opposite vertex i
(oriented e_0 = s_2 − s_1, e_1 = s_0 − s_2, e_2 = s_1 − s_0), the P1 hat functions
give the local matrices

    C^T = (|T| / 12) (1 1ᵀ + I),   C̃^T = (|T| / 3) I,   [G^T]_ij = e_iᵀ e_j / (4 |T|)

They only use edge lengths and angles, so the same code serves planar (``V × 2``)
and surface (``V × 3``, e.g. an icosahedral sphere) meshes. Mesh generation is out
of scope: bring meshes from fmesher, pygmsh or meshio.
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
    """Lumped mass C̃ and stiffness G of P1 elements on a triangle mesh.

        [C̃]_vv = ∑_T∋v |T| / 3,   [G]_vw = ∑_T∋v,w e_vᵀ e_w / (4 |T|)

    The local matrices (module docstring) are computed for all triangles at once
    with einx and scattered with a ``segment_sum`` into the pattern of G, which is
    built on the host from ``triangles`` (two vertices are coupled iff they share a
    triangle). ``vertices`` may be traced; only ``triangles`` must be concrete.

    Algorithm:
        1. Compute the edge Gram matrix e_iᵀ e_j and the area |T| of each triangle.
        2. Form the local stiffness [G^T]_ij and assemble it into G.
        3. Assemble one third of each area onto its three vertices, for C̃.

    Args:
        vertices: Vertex coordinates, shape ``(V, 2)`` for a planar mesh or
            ``(V, 3)`` for a surface mesh.
        triangles: Vertex indices of each triangle, shape ``(T, 3)`` (concrete host
            integers).

    Returns:
        Tuple ``(C_lumped, G)``: C̃ as a `lineax.DiagonalLinearOperator` and G as a
        symmetric positive-semidefinite `SparseOperator` (its null space is the
        constants on each connected component).

    Raises:
        ValueError: If the shapes are wrong or an index is out of range.
        TypeError: If ``triangles`` is traced.

    Examples:
        The unit right triangle has |T| = 1 / 2:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        >>> mass, stiffness = gaussx.fem_matrices(vertices, np.array([[0, 1, 2]]))
        >>> bool(jnp.allclose(mass.as_matrix(), jnp.eye(3) / 6.0))
        True
        >>> expected = jnp.array(
        ...     [[1.0, -0.5, -0.5], [-0.5, 0.5, 0.0], [-0.5, 0.0, 0.5]]
        ... )
        >>> bool(jnp.allclose(stiffness.as_matrix(), expected))
        True

        An index past the last vertex is rejected:

        >>> gaussx.fem_matrices(vertices, np.array([[0, 1, 3]]))
        Traceback (most recent call last):
        ...
        ValueError: triangle indices out of range for 3 vertices.
    """
    vertex_coords = _as_float(vertices)
    triangle_vertices = _host_triangles(triangles, vertex_coords.shape[0])
    if vertex_coords.ndim != 2 or vertex_coords.shape[1] not in (2, 3):
        raise ValueError(
            f"vertices must have shape (V, 2) or (V, 3), got {vertex_coords.shape}."
        )
    corners = vertex_coords[triangle_vertices]
    # e_iᵀ e_j and |T|
    gram, area = _local_geometry(corners)
    # 4 |T|
    four_area = 4.0 * area
    # [G^T]_ij = e_iᵀ e_j / (4 |T|)
    local_stiffness = einx.divide(
        "triangle row col, triangle -> triangle row col", gram, four_area
    )
    # Each unordered pair of [G^T]_ij once
    local_entries = local_stiffness[:, _LOCAL_ROWS, _LOCAL_COLS]
    values = rearrange(local_entries, "triangle entry -> (triangle entry)")
    n = vertex_coords.shape[0]
    coo_rows = triangle_vertices[:, _LOCAL_ROWS].ravel()
    coo_cols = triangle_vertices[:, _LOCAL_COLS].ravel()
    # G = ∑_T G^T
    stiffness = SparseOperator.from_coo(
        coo_rows,
        coo_cols,
        values,
        (n, n),
        symmetric=True,
        tags=frozenset({lx.positive_semidefinite_tag}),
    )
    # |T| / 3
    third_area = area / 3.0
    corner_area = repeat(third_area, "triangle -> (triangle corner)", corner=3)
    corner_vertices = triangle_vertices.ravel()
    # [C̃]_vv = ∑_T∋v |T| / 3
    mass = jax.ops.segment_sum(corner_area, corner_vertices, num_segments=n)
    return lx.DiagonalLinearOperator(mass), stiffness


def fem_projector(
    vertices: Float[ArrayLike, "V D"],
    triangles: Int[ArrayLike, "T 3"],
    points: Float[ArrayLike, "n D"],
    *,
    triangle_index: Int[ArrayLike, " n"] | None = None,
) -> SparseOperator:
    """Observation matrix A of P1 interpolation: (A w)_k = ∑_i ψ_i(s_k) w_i.

    Row k holds the three barycentric weights of point s_k in its triangle, so A has
    three non-zeros per row and, in a Laplace Hessian Q + Aᵀ W A, never couples
    vertices that are not already neighbours.

    Point location (on the host, blocked brute force over all triangles):

    - **Planar meshes** (``V × 2``): the barycentric test; a point outside every
      triangle raises.
    - **Surface meshes** (``V × 3``) that are star-shaped about the centroid of their
      vertices (spheres, icospheres; every triangle must face away from the
      centroid, which also requires a consistent orientation): the ray from the
      centroid through each point is intersected with the triangles, and the weights
      are taken at the intersection, i.e. the point is projected radially onto the
      mesh (its distance to the mesh, the chord error, is ignored).
    - Any other surface mesh needs ``triangle_index``.

    With ``triangle_index`` no location is done and the weights are those of the
    orthogonal projection of the point onto the plane of its triangle; ``points`` may
    then be traced.

    Algorithm:
        1. Locate each point's triangle, unless ``triangle_index`` gives it.
        2. On a located surface mesh, project each point radially onto its triangle.
        3. Compute the barycentric weights and assemble one row per point.

    Args:
        vertices: Vertex coordinates, shape ``(V, 2)`` or ``(V, 3)``.
        triangles: Vertex indices of each triangle, shape ``(T, 3)`` (concrete host
            integers).
        points: Observation locations s_k, shape ``(n, D)``; concrete unless
            ``triangle_index`` is given.
        triangle_index: The triangle containing each point, shape ``(n,)`` (concrete
            host integers).

    Returns:
        A as a `SparseOperator` of shape ``(n, V)``.

    Raises:
        ValueError: If a point is not on the mesh, a surface mesh is not star-shaped
            about its centroid and ``triangle_index`` is missing, or the shapes or
            indices are wrong.
        TypeError: If ``triangles`` is traced, or ``vertices`` or ``points`` are
            traced and ``triangle_index`` is missing.

    Examples:
        A point in the lower triangle of the unit square:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        >>> triangles = np.array([[0, 1, 2], [0, 2, 3]])
        >>> projector = gaussx.fem_projector(
        ...     vertices, triangles, np.array([[0.5, 0.25]])
        ... )
        >>> expected = jnp.array([[0.5, 0.25, 0.25, 0.0]])
        >>> bool(jnp.allclose(projector.as_matrix(), expected))
        True

        A point off the mesh is rejected:

        >>> gaussx.fem_projector(vertices, triangles, np.array([[2.0, 2.0]]))
        Traceback (most recent call last):
        ...
        ValueError: point 0 does not lie on the mesh (1 such points in this block).
    """
    vertex_coords = _as_float(vertices)
    triangle_vertices = _host_triangles(triangles, vertex_coords.shape[0])
    if vertex_coords.ndim != 2 or vertex_coords.shape[1] not in (2, 3):
        raise ValueError(
            f"vertices must have shape (V, 2) or (V, 3), got {vertex_coords.shape}."
        )
    point_coords = jnp.asarray(points, dtype=vertex_coords.dtype)
    if point_coords.ndim != 2 or point_coords.shape[1] != vertex_coords.shape[1]:
        raise ValueError(
            f"points must have shape (n, {vertex_coords.shape[1]}) like vertices, "
            f"got {point_coords.shape}."
        )
    centre = None
    if triangle_index is None:
        try:
            host_vertices = np.asarray(vertex_coords)
            host_points = np.asarray(point_coords)
        except jax.errors.TracerArrayConversionError as err:
            raise TypeError(
                "Point location needs concrete vertices and points; pass "
                "triangle_index to build the projector from traced values."
            ) from err
        if vertex_coords.shape[1] == 2:
            index = _locate_planar(host_vertices, triangle_vertices, host_points)
        else:
            index = _locate_radial(host_vertices, triangle_vertices, host_points)
            # c = (1 / V) ∑_v s_v, the vertex centroid
            centre = einx.mean("v d -> d", vertex_coords)
    else:
        index = np.asarray(triangle_index)
        if index.shape != (point_coords.shape[0],) or not np.issubdtype(
            index.dtype, np.integer
        ):
            raise ValueError(
                f"triangle_index must be {point_coords.shape[0]} integers, "
                f"got {index.shape}."
            )
        if index.size and (
            index.min() < 0 or index.max() >= triangle_vertices.shape[0]
        ):
            raise ValueError("triangle_index out of range.")
    point_triangles = triangle_vertices[index]
    corners = vertex_coords[point_triangles]
    if centre is not None:
        point_coords = _radial_projection(corners, point_coords, centre)
    weights = _barycentric(corners, point_coords)
    n = point_coords.shape[0]
    coo_rows = np.repeat(np.arange(n), 3)
    coo_cols = point_triangles.ravel()
    values = rearrange(weights, "point corner -> (point corner)")
    # [A]_k,v = ψ_v(s_k)
    return SparseOperator.from_coo(
        coo_rows,
        coo_cols,
        values,
        (n, vertex_coords.shape[0]),
    )


# ---------------------------------------------------------------------------
# Geometry (JAX)
# ---------------------------------------------------------------------------


def _host_triangles(triangles: ArrayLike, n_vertices: int) -> np.ndarray:
    """Validate the triangles as a concrete host integer array.

    Args:
        triangles: Vertex indices of each triangle, shape ``(T, 3)``.
        n_vertices: Number of vertices V.

    Returns:
        The triangles as an ``int64`` NumPy array.

    Raises:
        TypeError: If they are traced.
        ValueError: If the shape or dtype is wrong or an index is out of range.
    """
    try:
        host_triangles = np.asarray(triangles)
    except jax.errors.TracerArrayConversionError as err:
        raise TypeError("triangles must be a concrete host integer array.") from err
    if (
        host_triangles.ndim != 2
        or host_triangles.shape[1] != 3
        or not np.issubdtype(host_triangles.dtype, np.integer)
    ):
        raise ValueError(
            f"triangles must be integers of shape (T, 3), got {host_triangles.shape}."
        )
    if host_triangles.size and (
        host_triangles.min() < 0 or host_triangles.max() >= n_vertices
    ):
        raise ValueError(f"triangle indices out of range for {n_vertices} vertices.")
    return host_triangles.astype(np.int64)


def _local_geometry(corners: Float[Array, "T 3 D"]) -> tuple[Array, Array]:
    """Gram matrix e_iᵀ e_j of the opposite edges, and the areas |T|.

        4 |T|² = |e_0|² |e_1|² − (e_0ᵀ e_1)²   in any dimension

    Args:
        corners: Vertex coordinates of each triangle.

    Returns:
        Tuple ``(gram, area)`` holding e_iᵀ e_j and |T|.
    """
    # e_0 = s_2 − s_1, e_1 = s_0 − s_2, e_2 = s_1 − s_0
    edges = corners[:, [2, 0, 1]] - corners[:, [1, 2, 0]]
    # e_iᵀ e_j. The pattern is quoted by the AxisSizeError of no triangles: keep it.
    gram = einsum(edges, edges, "t i d, t j d -> t i j")
    # |T| = ½ √(|e_0|² |e_1|² − (e_0ᵀ e_1)²)
    area = 0.5 * jnp.sqrt(gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] ** 2)
    return gram, area


def _barycentric(
    corners: Float[Array, "n 3 D"], points: Float[Array, "n D"]
) -> Float[Array, "n 3"]:
    """Barycentric weights of each point's projection onto its triangle's plane.

        s = s_0 + u (s_1 − s_0) + v (s_2 − s_0),   weights (1 − u − v, u, v)

    Args:
        corners: Vertex coordinates of each point's triangle.
        points: The points.

    Returns:
        The three weights of each point.
    """
    origin = corners[:, 0]
    tips = corners[:, 1:]
    # The pattern is quoted by the AxisSizeError of no points: keep its text.
    origin_row = rearrange(origin, "n d -> n 1 d")
    # s_1 − s_0, s_2 − s_0
    spans = tips - origin_row
    # g = spansᵀ spans. The entries of g stay inline below: naming a shared slice
    # would reorder the reverse-mode sum.
    gram = einsum(spans, spans, "point row dim, point col dim -> point row col")
    # s − s_0
    offsets = points - origin
    # r = spansᵀ (s − s_0)
    rhs = einsum(spans, offsets, "point row dim, point dim -> point row")
    # det g = g₀₀ g₁₁ − g₀₁²
    determinant = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] ** 2
    # u = (g₁₁ r₀ − g₀₁ r₁) / det g
    bary_u = (gram[:, 1, 1] * rhs[:, 0] - gram[:, 0, 1] * rhs[:, 1]) / determinant
    # v = (g₀₀ r₁ − g₀₁ r₀) / det g
    bary_v = (gram[:, 0, 0] * rhs[:, 1] - gram[:, 0, 1] * rhs[:, 0]) / determinant
    # 1 − u − v
    bary_first = 1.0 - bary_u - bary_v
    stacked = jnp.stack([bary_first, bary_u, bary_v])
    return rearrange(stacked, "corner point -> point corner")


def _radial_projection(
    corners: Float[Array, "n 3 D"], points: Float[Array, "n D"], centre: Array
) -> Float[Array, "n D"]:
    """Where the ray from the centre through each point meets its triangle's plane.

        c + t (s − c),   t = nᵀ (s_0 − c) / (nᵀ (s − c))

    Args:
        corners: Vertex coordinates of each point's triangle.
        points: The points s.
        centre: The centre c of the rays.

    Returns:
        The intersection of each ray with the plane of its triangle.
    """
    # s_1 − s_0
    first_edge = corners[:, 1] - corners[:, 0]
    # s_2 − s_0
    second_edge = corners[:, 2] - corners[:, 0]
    # n = (s_1 − s_0) × (s_2 − s_0)
    normal = jnp.cross(first_edge, second_edge)
    # s − c
    direction = points - centre
    # s_0 − c
    origin_offset = corners[:, 0] - centre
    # nᵀ (s_0 − c). The pattern is quoted by the AxisSizeError of no points: keep it.
    numerator = einsum(normal, origin_offset, "n d, n d -> n")
    # nᵀ (s − c)
    denominator = einsum(normal, direction, "n d, n d -> n")
    # t = nᵀ (s_0 − c) / (nᵀ (s − c))
    ray_t = numerator / denominator
    # t (s − c)
    ray_step = einx.multiply("n, n d -> n d", ray_t, direction)
    # c + t (s − c)
    return centre + ray_step


# ---------------------------------------------------------------------------
# Point location (host, NumPy)
# ---------------------------------------------------------------------------


def _blocks(n_points: int, n_triangles: int):
    """Slices of the points, small enough that block × triangles fits the budget.

    Args:
        n_points: Number of points.
        n_triangles: Number of triangles.

    Yields:
        One slice of the points per block.
    """
    size = max(1, _LOCATE_BLOCK_ENTRIES // max(n_triangles, 1))
    for start in range(0, n_points, size):
        yield slice(start, min(start + size, n_points))


def _pick(score: np.ndarray, offset: int) -> np.ndarray:
    """Best triangle per point (largest minimum barycentric coordinate).

    Args:
        score: Minimum barycentric coordinate of each point in each triangle.
        offset: Position of this block's first point, for the error message.

    Returns:
        The best triangle of each point.

    Raises:
        ValueError: If a point is outside every triangle.
    """
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
    """Triangle of each point on a planar mesh, by the barycentric test.

        s = s_0 + u (s_1 − s_0) + v (s_2 − s_0),   score = min(u, v, 1 − u − v)

    Args:
        vertices: Vertex coordinates.
        tri: Vertex indices of each triangle.
        points: The points.

    Returns:
        The triangle of each point.

    Raises:
        ValueError: If a point is outside every triangle.
    """
    origin = vertices[tri[:, 0]]
    # s_1 − s_0
    first_edge = vertices[tri[:, 1]] - origin
    # s_2 − s_0
    second_edge = vertices[tri[:, 2]] - origin
    # det = (s_1 − s_0) × (s_2 − s_0)
    determinant = (
        first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]
    )
    # Degenerate triangles never match.
    determinant = np.where(determinant == 0.0, np.nan, determinant)
    index = np.empty(points.shape[0], dtype=np.int64)
    for block in _blocks(points.shape[0], tri.shape[0]):
        # s − s_0. The pattern is quoted by the AxisSizeError of no triangles: keep it.
        offsets = einx.subtract("b d, t d -> b t d", points[block], origin)
        # u = ((s − s_0) × (s_2 − s_0)) / det
        bary_u = (
            einx.multiply("t, b t -> b t", second_edge[:, 1], offsets[..., 0])
            - einx.multiply("t, b t -> b t", second_edge[:, 0], offsets[..., 1])
        ) / determinant
        # v = ((s_1 − s_0) × (s − s_0)) / det
        bary_v = (
            einx.multiply("t, b t -> b t", first_edge[:, 0], offsets[..., 1])
            - einx.multiply("t, b t -> b t", first_edge[:, 1], offsets[..., 0])
        ) / determinant
        # min(u, v, 1 − u − v)
        score = np.minimum(np.minimum(bary_u, bary_v), 1.0 - bary_u - bary_v)
        index[block] = _pick(np.nan_to_num(score, nan=-np.inf), block.start)
    return index


def _locate_radial(
    vertices: np.ndarray, tri: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Möller-Trumbore along rays from the vertex centroid (star-shaped meshes).

    Args:
        vertices: Vertex coordinates.
        tri: Vertex indices of each triangle.
        points: The points.

    Returns:
        The triangle of each point.

    Raises:
        ValueError: If the mesh is not star-shaped about its centroid, or a ray hits
            no triangle.
    """
    # c = (1 / V) ∑_v s_v
    centre = einx.mean("v d -> d", vertices)
    origin = vertices[tri[:, 0]]
    # s_1 − s_0
    first_edge = vertices[tri[:, 1]] - origin
    # s_2 − s_0
    second_edge = vertices[tri[:, 2]] - origin
    # n = (s_1 − s_0) × (s_2 − s_0)
    normal = np.cross(first_edge, second_edge)
    # c − s_0, the T of Möller-Trumbore
    offset = centre - origin
    # nᵀ (s_0 − c). The pattern is quoted by the AxisSizeError of no triangles: keep it.
    facing = einx.dot("t d, t d -> t", normal, -offset)
    # ‖n‖₂ ‖c − s_0‖₂
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
    # (s_2 − s_0) × (c − s_0)
    u_normal = np.cross(second_edge, offset)
    # (c − s_0) × (s_1 − s_0)
    v_normal = np.cross(offset, first_edge)
    # (s_2 − s_0)ᵀ ((c − s_0) × (s_1 − s_0))
    t_numerator = einx.dot("t d, t d -> t", second_edge, v_normal)
    index = np.empty(points.shape[0], dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for block in _blocks(points.shape[0], tri.shape[0]):
            index[block] = _radial_block(
                points[block] - centre,
                normal,
                u_normal,
                v_normal,
                t_numerator,
                block.start,
            )
    return index


def _radial_block(
    direction: np.ndarray,
    normal: np.ndarray,
    u_normal: np.ndarray,
    v_normal: np.ndarray,
    t_numerator: np.ndarray,
    offset: int,
) -> np.ndarray:
    """Best triangle for one block of rays c + t (s − c).

    Args:
        direction: The ray directions s − c.
        normal: Triangle normals n.
        u_normal: Vector whose product with a direction gives the numerator of u.
        v_normal: Vector whose product with a direction gives the numerator of v.
        t_numerator: Numerator of the ray parameter t.
        offset: Position of this block's first point, for the error message.

    Returns:
        The best triangle of each ray.

    Raises:
        ValueError: If a ray hits no triangle.
    """
    # det = −(s − c)ᵀ n
    determinant = -einx.dot("b d, t d -> b t", direction, normal)
    bary_u = einx.dot("b d, t d -> b t", direction, u_normal) / determinant
    bary_v = einx.dot("b d, t d -> b t", direction, v_normal) / determinant
    ray_t = einx.divide("t, b t -> b t", t_numerator, determinant)
    # min(u, v, 1 − u − v)
    score = np.minimum(np.minimum(bary_u, bary_v), 1.0 - bary_u - bary_v)
    # Rays pointing away from a triangle never match it.
    score = np.where((ray_t > 0.0) & np.isfinite(score), score, -np.inf)
    return _pick(score, offset)
