"""Helpers for coplanar 3D point sets and polygonal facets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from domain.entities import Point3D
from geometry.exact_numbers import as_fraction
from geometry.plane_keys import PlaneKey, point_lies_on_plane

__all__ = [
    "PlaneBasis2D",
    "convex_hull_2d_indices",
    "make_plane_basis",
    "order_coplanar_facet_vertices",
    "project_points_to_plane",
]


@dataclass(frozen=True)
class PlaneBasis2D:
    """Dominant-axis projection basis for a 3D plane."""

    plane: PlaneKey
    drop_axis: int
    axes: tuple[int, int]
    reverse_orientation: bool


def _point_axis_value(point: Point3D, axis: int) -> float:
    """Return the selected point coordinate by axis index."""
    if axis == 0:
        return point.x
    if axis == 1:
        return point.y
    if axis == 2:
        return point.z
    raise ValueError(f"Axis must be 0, 1, or 2, got {axis}.")


def _orient2d_exact(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> Fraction:
    """Return the exact signed 2D orientation determinant."""
    ax = as_fraction(a[0])
    ay = as_fraction(a[1])
    bx = as_fraction(b[0])
    by = as_fraction(b[1])
    cx = as_fraction(c[0])
    cy = as_fraction(c[1])
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def make_plane_basis(plane: PlaneKey) -> PlaneBasis2D:
    """Choose a stable dominant-axis 2D projection for a plane."""
    if plane.a == 0 and plane.b == 0 and plane.c == 0:
        raise ValueError("Plane normal must be non-zero.")

    abs_a = abs(plane.a)
    abs_b = abs(plane.b)
    abs_c = abs(plane.c)

    if abs_a >= abs_b and abs_a >= abs_c:
        component = plane.a
        return PlaneBasis2D(
            plane=plane,
            drop_axis=0,
            axes=(1, 2),
            reverse_orientation=component < 0,
        )

    if abs_b >= abs_c:
        component = plane.b
        return PlaneBasis2D(
            plane=plane,
            drop_axis=1,
            axes=(2, 0),
            reverse_orientation=component < 0,
        )

    component = plane.c
    return PlaneBasis2D(
        plane=plane,
        drop_axis=2,
        axes=(0, 1),
        reverse_orientation=component < 0,
    )


def project_points_to_plane(
    points: Sequence[Point3D],
    basis: PlaneBasis2D,
) -> tuple[tuple[float, float], ...]:
    """Project 3D points to 2D using the selected dominant-axis projection."""
    axis_u, axis_v = basis.axes
    return tuple(
        (_point_axis_value(point, axis_u), _point_axis_value(point, axis_v)) for point in points
    )


def convex_hull_2d_indices(points: Sequence[tuple[float, float]]) -> tuple[int, ...]:
    """Return indices of the 2D convex hull vertices in CCW order."""
    seen: dict[tuple[float, float], int] = {}
    unique_points: list[tuple[tuple[float, float], int]] = []

    for original_index, point in enumerate(points):
        x, y = point
        as_fraction(x)
        as_fraction(y)
        if point in seen:
            continue
        seen[point] = original_index
        unique_points.append((point, original_index))

    if not unique_points:
        return ()

    unique_points.sort(key=lambda item: item[0])
    if len(unique_points) == 1:
        return (unique_points[0][1],)
    if len(unique_points) == 2:
        return unique_points[0][1], unique_points[1][1]

    lower: list[tuple[tuple[float, float], int]] = []
    for entry in unique_points:
        while len(lower) >= 2 and _orient2d_exact(lower[-2][0], lower[-1][0], entry[0]) <= 0:
            lower.pop()
        lower.append(entry)

    upper: list[tuple[tuple[float, float], int]] = []
    for entry in reversed(unique_points):
        while len(upper) >= 2 and _orient2d_exact(upper[-2][0], upper[-1][0], entry[0]) <= 0:
            upper.pop()
        upper.append(entry)

    hull = lower[:-1] + upper[:-1]
    return tuple(entry[1] for entry in hull)


def order_coplanar_facet_vertices(
    points: Sequence[Point3D],
    plane: PlaneKey,
    candidate_indices: Sequence[int],
) -> tuple[int, ...]:
    """Return ordered extreme vertices of a coplanar facet."""
    if not candidate_indices:
        return ()

    for index in candidate_indices:
        if not point_lies_on_plane(points[index], plane):
            raise ValueError("Candidate point does not lie on the given plane.")

    basis = make_plane_basis(plane)
    selected_points = tuple(points[index] for index in candidate_indices)
    projected = project_points_to_plane(selected_points, basis)
    local_hull_indices = convex_hull_2d_indices(projected)
    result = tuple(candidate_indices[index] for index in local_hull_indices)

    if basis.reverse_orientation and len(result) >= 3:
        return tuple(reversed(result))
    return result
