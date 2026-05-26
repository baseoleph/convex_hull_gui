"""Shared builders for polygonal HullResult3D topology."""

from __future__ import annotations

from collections.abc import Sequence

from domain.entities import Face3D, OrientedEdge, Point3D, Vector3D


def unit_normal_for_face(
    points: Sequence[Point3D],
    vertex_indices: tuple[int, ...],
) -> Vector3D:
    """Return the unit normal induced by the first three vertices of a face."""
    p0 = points[vertex_indices[0]]
    p1 = points[vertex_indices[1]]
    p2 = points[vertex_indices[2]]

    normal = (p1 - p0).cross(p2 - p0)
    length = normal.norm()
    if length == 0.0:
        raise RuntimeError(f"Face vertices are collinear: {vertex_indices}.")

    return Vector3D(normal.x / length, normal.y / length, normal.z / length)


def build_oriented_edges(faces: Sequence[Face3D]) -> tuple[OrientedEdge, ...]:
    """Build a closed half-edge list with twin links for polygonal faces."""
    raw_edges: list[OrientedEdge] = []
    directed_to_idx: dict[tuple[int, int], int] = {}

    for face_id, face in enumerate(faces):
        vertices = face.vertex_indices
        for offset, tail in enumerate(vertices):
            head = vertices[(offset + 1) % len(vertices)]
            key = (tail, head)
            if key in directed_to_idx:
                raise RuntimeError(f"Duplicate directed half-edge {tail}->{head}.")

            directed_to_idx[key] = len(raw_edges)
            raw_edges.append(OrientedEdge(tail=tail, head=head, face_id=face_id))

    final_edges: list[OrientedEdge] = []
    for edge_id, edge in enumerate(raw_edges):
        twin_id = directed_to_idx.get((edge.head, edge.tail))
        if twin_id is None:
            raise RuntimeError(f"No twin for half-edge {edge_id} ({edge.tail}->{edge.head}).")

        final_edges.append(
            OrientedEdge(
                tail=edge.tail,
                head=edge.head,
                face_id=edge.face_id,
                twin_id=twin_id,
            )
        )

    return tuple(final_edges)
