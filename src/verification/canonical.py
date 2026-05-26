"""Canonical form and equality comparison for HullResult3D."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from domain.entities import Face3D, HullResult3D, Point3D
from geometry.planar import order_coplanar_facet_vertices
from geometry.plane_keys import PlaneKey, canonical_plane_key, point_lies_on_plane


def _min_cyclic(indices: tuple[int, ...]) -> tuple[int, ...]:
    n = len(indices)
    best = indices
    for i in range(1, n):
        rot = indices[i:] + indices[:i]
        if rot < best:
            best = rot
    return best


def _canonical_cycle_unoriented(indices: tuple[int, ...]) -> tuple[int, ...]:
    forward = _min_cyclic(indices)
    backward = _min_cyclic(tuple(reversed(indices)))
    return min(forward, backward)


def _face_plane_key(result: HullResult3D, face: Face3D) -> PlaneKey:
    indices = face.vertex_indices
    if len(indices) < 3:
        raise RuntimeError("Face must have at least 3 vertices.")
    for i, j, k in combinations(indices, 3):
        try:
            return canonical_plane_key(result.points[i], result.points[j], result.points[k])
        except ValueError:
            continue
    raise RuntimeError(f"Cannot determine plane for collinear face {indices}.")


@dataclass(frozen=True)
class CanonicalPolygonalHull:
    """Canonical representation of a polygonal hull for equality comparison."""

    vertices: tuple[Point3D, ...]
    facets: tuple[tuple[int, ...], ...]


def to_polygonal_canonical(result: HullResult3D) -> CanonicalPolygonalHull:
    """Convert a hull result to an orientation-insensitive polygonal canonical form."""
    pts = result.points

    sorted_items = sorted(
        ((old_index, pts[old_index]) for old_index in result.vertex_indices),
        key=lambda item: (item[1].x, item[1].y, item[1].z),
    )
    vertices = tuple(point for _, point in sorted_items)
    old_to_new = {old_index: new_index for new_index, (old_index, _) in enumerate(sorted_items)}

    plane_to_old_indices: dict[PlaneKey, set[int]] = {}
    for face in result.faces:
        plane = _face_plane_key(result, face)
        group = plane_to_old_indices.setdefault(plane, set())
        group.update(face.vertex_indices)

    hull_vertex_indices = tuple(result.vertex_indices)
    for plane in plane_to_old_indices:
        plane_to_old_indices[plane] = {
            old_index
            for old_index in hull_vertex_indices
            if point_lies_on_plane(result.points[old_index], plane)
        }

    canonical_facets: list[tuple[int, ...]] = []
    for plane in sorted(plane_to_old_indices):
        candidate_old_indices = tuple(sorted(plane_to_old_indices[plane]))
        ordered_old_indices = order_coplanar_facet_vertices(
            result.points,
            plane,
            candidate_old_indices,
        )
        canonical_indices = tuple(old_to_new[old_index] for old_index in ordered_old_indices)
        if len(canonical_indices) < 3:
            raise RuntimeError("Polygonal facet has fewer than 3 vertices.")
        canonical_facets.append(_canonical_cycle_unoriented(canonical_indices))

    return CanonicalPolygonalHull(vertices=vertices, facets=tuple(sorted(canonical_facets)))


def polygonal_hulls_equal(a: HullResult3D, b: HullResult3D) -> bool:
    """Return True if hulls have the same vertices and maximal polygonal facets."""
    return to_polygonal_canonical(a) == to_polygonal_canonical(b)
