"""PyVista scene assembly functions for domain objects.

Pure conversion layer: no Plotter, no .show(), no side effects.
All functions accept domain objects and return pv.PolyData instances
that can be inspected in tests without a graphical context.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pyvista as pv

from domain.entities import Face3D, HullResult3D, Point3D


def points_to_polydata(points: Sequence[Point3D]) -> pv.PolyData:
    """Convert a sequence of Point3D to a PyVista point-cloud PolyData.

    Args:
        points: Input point cloud (may be empty).

    Returns:
        A ``pv.PolyData`` with ``n_points == len(points)``.
    """
    arr: npt.NDArray[np.float64] = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float64)
    return pv.PolyData(arr)


def _encode_faces(faces: Sequence[Face3D]) -> npt.NDArray[np.int_]:
    """Encode a sequence of Face3D into PyVista's flat face-connectivity format.

    PyVista represents a polygon mesh as a 1-D array of the form::

        [k0, i0_0, i0_1, ..., k1, i1_0, i1_1, ...]

    where ``k_j`` is the vertex count of face *j* and the subsequent
    entries are vertex indices.  Supports polygons of any arity ≥ 3.

    Args:
        faces: Faces to encode.

    Returns:
        A 1-D integer numpy array in PyVista's connectivity format.
    """
    parts: list[int] = []
    for face in faces:
        parts.append(len(face.vertex_indices))
        parts.extend(face.vertex_indices)
    return np.array(parts, dtype=np.int_)


def hull_to_polydata(result: HullResult3D) -> pv.PolyData:
    """Convert a HullResult3D to a surface PolyData mesh.

    The returned mesh uses *all* input points from ``result.points`` as its
    vertex buffer.  Only the faces defined by ``result.faces`` are encoded.

    Args:
        result: The convex hull to convert.

    Returns:
        A ``pv.PolyData`` surface mesh with ``n_points == len(result.points)``
        and ``n_cells == len(result.faces)``.
    """
    pts: npt.NDArray[np.float64] = np.array(
        [[p.x, p.y, p.z] for p in result.points], dtype=np.float64
    )
    faces_arr = _encode_faces(result.faces)
    return pv.PolyData(pts, faces_arr.tolist())


def hull_normals_to_polydata(
    result: HullResult3D,
    *,
    normal_length: float | None = None,
) -> pv.PolyData:
    """Build a PolyData of line segments visualising per-face outward normals.

    One segment per face, starting at the face centroid and extending in the
    direction of ``Face3D.normal`` scaled by *normal_length*.  The segments
    are in the same order as ``result.faces``.

    Args:
        result: The convex hull whose face normals are visualised.
        normal_length: Length of each normal arrow in world units.
            When ``None`` (default), 10 % of the bounding-box diagonal of the
            hull vertices is used; falls back to ``1.0`` for degenerate cases
            (fewer than 2 vertices or zero-length diagonal).

    Returns:
        A ``pv.PolyData`` whose cells are VTK_LINE segments, one per face.
    """
    if normal_length is None:
        normal_length = _auto_normal_length(result)

    n_faces = len(result.faces)
    if n_faces == 0:
        return pv.PolyData()

    seg_pts: list[list[float]] = []
    for face in result.faces:
        verts = [result.points[i] for i in face.vertex_indices]
        k = len(verts)
        cx = math.fsum(p.x for p in verts) / k
        cy = math.fsum(p.y for p in verts) / k
        cz = math.fsum(p.z for p in verts) / k
        n = face.normal
        seg_pts.append([cx, cy, cz])
        seg_pts.append(
            [
                cx + n.x * normal_length,
                cy + n.y * normal_length,
                cz + n.z * normal_length,
            ]
        )

    pts_arr: npt.NDArray[np.float64] = np.array(seg_pts, dtype=np.float64)
    lines_list: list[int] = []
    for i in range(n_faces):
        lines_list += [2, 2 * i, 2 * i + 1]

    return pv.PolyData(pts_arr, lines=lines_list)


def _auto_normal_length(result: HullResult3D) -> float:
    """Return 10 % of the bounding-box diagonal of hull vertices, or 1.0."""
    if len(result.vertex_indices) < 2:
        return 1.0
    verts = [result.points[i] for i in result.vertex_indices]
    xs = [p.x for p in verts]
    ys = [p.y for p in verts]
    zs = [p.z for p in verts]
    diag = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2)
    return 0.1 * diag if diag > 0.0 else 1.0
