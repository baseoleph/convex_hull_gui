"""Point cloud generators for convex hull testing."""

from generators.benchmarks import (
    controlled_h_8,
    controlled_h_32,
    controlled_h_128,
    controlled_h_half_n,
    controlled_h_sqrt_n,
    cube_with_many_edge_points,
    cube_with_many_face_points,
    narrow_lens_rim,
    prism_polygon,
    stretched_moment_alpha0,
)
from generators.degeneracies import (
    coplanar_square_with_center,
    cube_with_edge_midpoints,
    cube_with_face_centers,
    rectangular_box_exact,
    tetrahedron_with_edge_points,
    tetrahedron_with_face_points,
    unit_cube_exact,
)
from generators.general_position import (
    points_on_sphere,
    uniform_cube,
    uniform_sphere,
)

__all__ = [
    "controlled_h_8",
    "controlled_h_32",
    "controlled_h_128",
    "controlled_h_half_n",
    "controlled_h_sqrt_n",
    "coplanar_square_with_center",
    "cube_with_edge_midpoints",
    "cube_with_face_centers",
    "cube_with_many_edge_points",
    "cube_with_many_face_points",
    "narrow_lens_rim",
    "points_on_sphere",
    "prism_polygon",
    "rectangular_box_exact",
    "stretched_moment_alpha0",
    "tetrahedron_with_edge_points",
    "tetrahedron_with_face_points",
    "uniform_cube",
    "uniform_sphere",
    "unit_cube_exact",
]
