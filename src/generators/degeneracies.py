"""Deterministic degenerate 3D point-set generators."""

from __future__ import annotations

from domain.entities import Point3D


def _p(x: float, y: float, z: float) -> Point3D:
    return Point3D(x, y, z)


def unit_cube_exact() -> list[Point3D]:
    """Return the 8 exact vertices of the unit cube [0, 1]^3."""
    return [
        _p(0, 0, 0),
        _p(1, 0, 0),
        _p(1, 1, 0),
        _p(0, 1, 0),
        _p(0, 0, 1),
        _p(1, 0, 1),
        _p(1, 1, 1),
        _p(0, 1, 1),
    ]


def rectangular_box_exact() -> list[Point3D]:
    """Return the 8 exact vertices of a non-unit axis-aligned rectangular box."""
    return [
        _p(0, 0, 0),
        _p(2, 0, 0),
        _p(2, 3, 0),
        _p(0, 3, 0),
        _p(0, 0, 5),
        _p(2, 0, 5),
        _p(2, 3, 5),
        _p(0, 3, 5),
    ]


def cube_with_face_centers() -> list[Point3D]:
    """Return unit cube vertices plus one center point on each cube face."""
    return [
        *unit_cube_exact(),
        _p(0.5, 0.5, 0.0),
        _p(0.5, 0.5, 1.0),
        _p(0.5, 0.0, 0.5),
        _p(0.5, 1.0, 0.5),
        _p(0.0, 0.5, 0.5),
        _p(1.0, 0.5, 0.5),
    ]

def cube_with_edge_midpoints() -> list[Point3D]:
    """Return unit cube vertices plus the 12 cube edge midpoints."""
    return [
        *unit_cube_exact(),
        _p(0.5, 0.0, 0.0),
        _p(1.0, 0.5, 0.0),
        _p(0.5, 1.0, 0.0),
        _p(0.0, 0.5, 0.0),
        _p(0.5, 0.0, 1.0),
        _p(1.0, 0.5, 1.0),
        _p(0.5, 1.0, 1.0),
        _p(0.0, 0.5, 1.0),
        _p(0.0, 0.0, 0.5),
        _p(1.0, 0.0, 0.5),
        _p(1.0, 1.0, 0.5),
        _p(0.0, 1.0, 0.5),
    ]


def tetrahedron_with_edge_points() -> list[Point3D]:
    """Return tetrahedron vertices plus points lying on tetrahedron edges."""
    return [
        _p(0, 0, 0),
        _p(1, 0, 0),
        _p(0, 1, 0),
        _p(0, 0, 1),
        _p(0.5, 0.0, 0.0),
        _p(0.0, 0.5, 0.0),
        _p(0.0, 0.0, 0.5),
        _p(0.5, 0.5, 0.0),
        _p(0.5, 0.0, 0.5),
        _p(0.0, 0.5, 0.5),
    ]


def coplanar_square_with_center() -> list[Point3D]:
    """Return a unit square in the z=0 plane plus its center point."""
    return [
        _p(0, 0, 0),
        _p(1, 0, 0),
        _p(1, 1, 0),
        _p(0, 1, 0),
        _p(0.5, 0.5, 0),
    ]


def tetrahedron_with_face_points() -> list[Point3D]:
    """Return tetrahedron vertices plus interior points on each tetrahedron face."""
    return [
        _p(0, 0, 0),
        _p(1, 0, 0),
        _p(0, 1, 0),
        _p(0, 0, 1),
        _p(0.25, 0.25, 0.0),
        _p(0.25, 0.0, 0.25),
        _p(0.0, 0.25, 0.25),
        _p(0.5, 0.25, 0.25),
    ]
