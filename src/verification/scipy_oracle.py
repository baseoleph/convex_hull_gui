"""SciPy-based oracle for 3D convex hull computation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

from domain.entities import (
    Face3D,
    HullMetadata,
    HullResult3D,
    OrientedEdge,
    Point3D,
    Vector3D,
)


class SciPyOracle3D:
    """Convex hull oracle backed by ``scipy.spatial.ConvexHull``.

    Wraps QHull-based scipy implementation and converts the triangulated result
    into a :class:`~domain.entities.HullResult3D` with a full
    half-edge mesh.  Twin links are assigned by a reverse-edge lookup table.

    The winding order of each simplex from scipy is verified against the
    outward normal stored in ``equations`` and corrected when needed.
    """

    def compute(
        self,
        points: Sequence[Point3D],
        *,
        seed: int | None = None,
    ) -> HullResult3D:
        """Compute the convex hull using scipy and convert to HullResult3D.

        Args:
            points: Input point cloud (must have at least 4 non-coplanar points).
            seed: Ignored; scipy is deterministic.

        Returns:
            A fully populated :class:`~domain.entities.HullResult3D`
            with all half-edges and twin links set.

        Raises:
            RuntimeError: If the resulting mesh is not closed (twin missing for
                some half-edge), which indicates a scipy output inconsistency.
        """
        pts = list(points)
        n = len(pts)

        coords = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64)
        hull = ConvexHull(coords)

        # simplices: shape (F, 3) — vertex indices of each triangular facet.
        # equations: shape (F, 4) — [nx, ny, nz, d] with ||[nx,ny,nz]|| == 1.
        simplices = hull.simplices
        equations = hull.equations

        faces: list[Face3D] = []
        triples: list[tuple[int, int, int]] = []

        for si, eq in zip(simplices, equations, strict=True):
            v0, v1, v2 = int(si[0]), int(si[1]), int(si[2])
            nx, ny, nz = float(eq[0]), float(eq[1]), float(eq[2])
            outward = Vector3D(nx, ny, nz)

            # Verify that the winding (v0→v1→v2) cross product aligns with
            # the outward normal; swap v1↔v2 if antiparallel.
            p0, p1, p2 = pts[v0], pts[v1], pts[v2]
            cross = (p1 - p0).cross(p2 - p0)
            if cross.dot(outward) < 0.0:
                v1, v2 = v2, v1

            # scipy's equations already carry a unit normal; renormalize only
            # to guard against minor floating-point deviation from unit length.
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            unit_normal = Vector3D(nx / length, ny / length, nz / length)

            triples.append((v0, v1, v2))
            faces.append(Face3D(vertex_indices=(v0, v1, v2), normal=unit_normal))

        # Build half-edges from the corrected winding order.
        raw_edges: list[OrientedEdge] = []
        for fid, (i, j, k) in enumerate(triples):
            raw_edges.append(OrientedEdge(tail=i, head=j, face_id=fid))
            raw_edges.append(OrientedEdge(tail=j, head=k, face_id=fid))
            raw_edges.append(OrientedEdge(tail=k, head=i, face_id=fid))

        # Build reverse-edge lookup: (tail, head) → edge index.
        directed_to_idx: dict[tuple[int, int], int] = {
            (e.tail, e.head): eid for eid, e in enumerate(raw_edges)
        }

        final_edges: list[OrientedEdge] = []
        for eid, e in enumerate(raw_edges):
            twin_id = directed_to_idx.get((e.head, e.tail))
            if twin_id is None:
                raise RuntimeError(
                    f"No twin for half-edge {eid} ({e.tail}→{e.head}).  "
                    "Hull mesh is not closed; this may indicate a scipy output error."
                )
            final_edges.append(
                OrientedEdge(tail=e.tail, head=e.head, face_id=e.face_id, twin_id=twin_id)
            )

        vertex_indices = frozenset(idx for i, j, k in triples for idx in (i, j, k))

        meta = HullMetadata(
            algorithm="SciPyOracle3D",
            seed=seed,
            n_points_input=n,
            n_vertices_hull=len(vertex_indices),
            n_faces=len(faces),
        )

        return HullResult3D(
            points=tuple(pts),
            vertex_indices=vertex_indices,
            faces=tuple(faces),
            oriented_edges=tuple(final_edges),
            metadata=meta,
        )
