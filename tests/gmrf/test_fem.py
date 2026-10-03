"""Tests for fem_matrices and fem_projector."""

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import gaussx


UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
SQUARE_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit icosphere with outward (counter-clockwise) triangles."""
    p = (1.0 + 5.0**0.5) / 2.0
    vertices = [
        [-1, p, 0], [1, p, 0], [-1, -p, 0], [1, -p, 0],
        [0, -1, p], [0, 1, p], [0, -1, -p], [0, 1, -p],
        [p, 0, -1], [p, 0, 1], [-p, 0, -1], [-p, 0, 1],
    ]  # fmt: skip
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]  # fmt: skip
    vertices = [list(np.asarray(v, float) / np.linalg.norm(v)) for v in vertices]
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}

        def midpoint(i, j, cache=cache):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = np.add(vertices[i], vertices[j])
                vertices.append(list(m / np.linalg.norm(m)))
                cache[key] = len(vertices) - 1
            return cache[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    return np.array(vertices), np.array(faces)


def torus(n_major: int = 12, n_minor: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A consistently oriented torus: not star-shaped about its centroid."""
    u, v = np.meshgrid(
        np.linspace(0, 2 * np.pi, n_major, endpoint=False),
        np.linspace(0, 2 * np.pi, n_minor, endpoint=False),
        indexing="ij",
    )
    radius = 2.0 + 0.5 * np.cos(v)
    vertices = np.stack(
        [(radius * np.cos(u)).ravel(), (radius * np.sin(u)).ravel(),
         (0.5 * np.sin(v)).ravel()], 1
    )  # fmt: skip
    idx = lambda i, j: (i % n_major) * n_minor + (j % n_minor)
    faces = []
    for i in range(n_major):
        for j in range(n_minor):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    return vertices, np.array(faces)


class TestFemMatrices:
    @pytest.mark.slow
    def test_single_triangle(self):
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        C, G = gaussx.fem_matrices(vertices, np.array([[0, 1, 2]]))
        assert jnp.allclose(C.as_matrix(), jnp.eye(3) / 6.0)
        expected = jnp.array([[1.0, -0.5, -0.5], [-0.5, 0.5, 0.0], [-0.5, 0.0, 0.5]])
        assert jnp.allclose(G.as_matrix(), expected)

    @pytest.mark.slow
    def test_unit_square(self):
        C, G = gaussx.fem_matrices(UNIT_SQUARE, SQUARE_TRIANGLES)
        assert jnp.allclose(lx_diag(C), jnp.array([1, 0.5, 1, 0.5]) / 3.0)
        expected = jnp.array(
            [
                [1.0, -0.5, 0.0, -0.5],
                [-0.5, 1.0, -0.5, 0.0],
                [0.0, -0.5, 1.0, -0.5],
                [-0.5, 0.0, -0.5, 1.0],
            ]
        )
        assert jnp.allclose(G.as_matrix(), expected)
        assert G.pattern.symmetric

    def test_surface_mesh_matches_planar(self):
        # The unit square embedded in 3-D and rotated: same matrices.
        angle = 0.3
        rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )
        lifted = einx.dot("v j, i j -> v i", np.c_[UNIT_SQUARE, np.zeros(4)], rotation)
        C2, G2 = gaussx.fem_matrices(UNIT_SQUARE, SQUARE_TRIANGLES)
        C3, G3 = gaussx.fem_matrices(lifted, SQUARE_TRIANGLES)
        assert jnp.allclose(C2.as_matrix(), C3.as_matrix())
        assert jnp.allclose(G2.as_matrix(), G3.as_matrix())

    @pytest.mark.slow
    def test_sphere_area_and_null_space(self):
        vertices, triangles = icosphere(2)
        C, G = gaussx.fem_matrices(vertices, triangles)
        # Total lumped mass is the polyhedron's area, just below 4π.
        assert 0.97 * 4 * np.pi < jnp.sum(lx_diag(C)) < 4 * np.pi
        assert jnp.allclose(G.mv(jnp.ones(vertices.shape[0])), 0.0, atol=1e-12)

    def test_jit_and_grad_through_vertices(self):
        def total_stiffness_trace(v):
            _, G = gaussx.fem_matrices(v, SQUARE_TRIANGLES)
            return jnp.sum(G.diagonal())

        grad = jax.jit(jax.grad(total_stiffness_trace))(jnp.asarray(UNIT_SQUARE))
        assert grad.shape == UNIT_SQUARE.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_bad_shapes_raise(self):
        with pytest.raises(ValueError, match="triangles"):
            gaussx.fem_matrices(UNIT_SQUARE, np.array([0, 1, 2]))
        with pytest.raises(ValueError, match="out of range"):
            gaussx.fem_matrices(UNIT_SQUARE, np.array([[0, 1, 4]]))
        with pytest.raises(ValueError, match="vertices"):
            gaussx.fem_matrices(np.zeros((4, 1)), SQUARE_TRIANGLES)


def lx_diag(C):
    return jnp.diag(C.as_matrix())


class TestFemProjectorPlanar:
    def test_barycentric_weights(self):
        A = gaussx.fem_projector(UNIT_SQUARE, SQUARE_TRIANGLES, np.array([[0.5, 0.25]]))
        assert A.pattern.shape == (1, 4)
        assert jnp.allclose(A.as_matrix(), jnp.array([[0.5, 0.25, 0.25, 0.0]]))

    @pytest.mark.slow
    def test_reproduces_linear_functions(self):
        points = np.asarray(jr.uniform(jr.key(0), (50, 2)))
        A = gaussx.fem_projector(UNIT_SQUARE, SQUARE_TRIANGLES, points)
        f = lambda x: 1.0 + 2.0 * x[:, 0] - 3.0 * x[:, 1]
        assert jnp.allclose(A.mv(f(UNIT_SQUARE)), f(points), atol=1e-12)
        assert jnp.allclose(A.mv(jnp.ones(4)), 1.0)

    @pytest.mark.slow
    def test_vertices_and_edges(self):
        points = np.r_[UNIT_SQUARE, [[0.5, 0.5], [1.0, 0.5]]]
        A = gaussx.fem_projector(UNIT_SQUARE, SQUARE_TRIANGLES, points)
        assert jnp.allclose(A.as_matrix()[:4], jnp.eye(4), atol=1e-12)
        assert jnp.allclose(A.mv(UNIT_SQUARE[:, 0]), points[:, 0], atol=1e-12)

    def test_outside_point_raises(self):
        with pytest.raises(ValueError, match="does not lie on the mesh"):
            gaussx.fem_projector(UNIT_SQUARE, SQUARE_TRIANGLES, np.array([[1.5, 0.5]]))

    def test_triangle_index_allows_traced_points(self):
        @jax.jit
        def project(points):
            A = gaussx.fem_projector(
                UNIT_SQUARE, SQUARE_TRIANGLES, points, triangle_index=np.array([0])
            )
            return A.values

        values = project(jnp.array([[0.5, 0.25]]))
        assert jnp.allclose(jnp.sort(values), jnp.array([0.25, 0.25, 0.5]))

    def test_traced_points_without_index_raise(self):
        with pytest.raises(TypeError, match="triangle_index"):
            jax.jit(
                lambda p: gaussx.fem_projector(UNIT_SQUARE, SQUARE_TRIANGLES, p).values
            )(jnp.array([[0.5, 0.25]]))


class TestFemProjectorSphere:
    vertices, triangles = icosphere(2)

    def test_recovers_each_vertex(self):
        A = gaussx.fem_projector(self.vertices, self.triangles, self.vertices)
        n = self.vertices.shape[0]
        assert A.pattern.shape == (n, n)
        assert jnp.allclose(A.as_matrix(), jnp.eye(n), atol=1e-10)

    @pytest.mark.slow
    def test_linear_function_at_random_points(self):
        z = jr.normal(jr.key(0), (200, 3))
        norms = jnp.sqrt(einx.dot("n d, n d -> n", z, z))
        points = np.asarray(einx.divide("n d, n -> n d", z, norms))
        A = gaussx.fem_projector(self.vertices, self.triangles, points)
        assert jnp.allclose(A.mv(jnp.ones(self.vertices.shape[0])), 1.0)
        a = jnp.array([0.3, -1.2, 0.5])
        interpolated = A.mv(self.vertices @ a)
        # The weights are taken at the radial projection q onto the mesh, so
        # a·q is reproduced exactly and a·p only up to the chord error:
        # |p − q| ≤ 1 − r_min, r_min the smallest distance of a face to 0.
        corners = self.vertices[self.triangles]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        lengths = np.sqrt(einx.dot("t d, t d -> t", normals, normals))
        r_min = np.min(
            np.abs(einx.dot("t d, t d -> t", normals, corners[:, 0])) / lengths
        )
        chord = 1.0 - r_min
        assert jnp.max(jnp.abs(interpolated - points @ a)) <= jnp.linalg.norm(a) * chord

    def test_non_star_shaped_surface_raises(self):
        vertices, triangles = torus()
        with pytest.raises(ValueError, match="triangle_index"):
            gaussx.fem_projector(vertices, triangles, vertices[:3])

    @pytest.mark.slow
    def test_triangle_index_on_non_star_shaped_surface(self):
        vertices, triangles = torus()
        centroids = einx.mean("t k d -> t d", vertices[triangles])
        A = gaussx.fem_projector(
            vertices,
            triangles,
            centroids,
            triangle_index=np.arange(triangles.shape[0]),
        )
        assert jnp.allclose(A.values, 1.0 / 3.0)
