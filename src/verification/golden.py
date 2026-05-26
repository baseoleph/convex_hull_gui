"""Golden-file serialization and deserialization for HullResult3D."""

from __future__ import annotations

import json
from pathlib import Path

from domain.entities import (
    Face3D,
    HullMetadata,
    HullResult3D,
    OrientedEdge,
    Point3D,
    Vector3D,
)

SCHEMA_VERSION: int = 1


class GoldenSchemaError(Exception):
    """Raised when a golden file has an unsupported or missing schema_version."""


def _min_cyclic(indices: tuple[int, ...]) -> tuple[int, ...]:
    """Return the lexicographically smallest cyclic rotation of *indices*."""
    n = len(indices)
    best = indices
    for i in range(1, n):
        rot = indices[i:] + indices[:i]
        if rot < best:
            best = rot
    return best


def to_golden_dict(result: HullResult3D) -> dict[str, object]:
    """Build the canonical golden dictionary for *result* without writing to disk.

    The dictionary has the same structure as the JSON files produced by
    :func:`dump_golden` and accepted by :func:`load_golden`.

    Args:
        result: Hull result to convert.

    Returns:
        A JSON-serializable :class:`dict` with ``schema_version`` = 1.
    """
    pts = result.points

    sorted_faces = sorted(result.faces, key=lambda f: _min_cyclic(tuple(f.vertex_indices)))
    canonical_vi_list = [_min_cyclic(f.vertex_indices) for f in sorted_faces]

    raw_edges: list[OrientedEdge] = []
    for fid, vi in enumerate(canonical_vi_list):
        n = len(vi)
        for i in range(n):
            raw_edges.append(OrientedEdge(tail=vi[i], head=vi[(i + 1) % n], face_id=fid))

    dir_to_idx: dict[tuple[int, int], int] = {
        (e.tail, e.head): eid for eid, e in enumerate(raw_edges)
    }
    final_edges = [
        OrientedEdge(
            tail=e.tail,
            head=e.head,
            face_id=e.face_id,
            twin_id=dir_to_idx[(e.head, e.tail)],
        )
        for e in raw_edges
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "points": [[p.x, p.y, p.z] for p in pts],
        "vertex_indices": sorted(result.vertex_indices),
        "faces": [
            {
                "vertex_indices": list(vi),
                "normal": [f.normal.x, f.normal.y, f.normal.z],
            }
            for vi, f in zip(canonical_vi_list, sorted_faces, strict=True)
        ],
        "oriented_edges": [
            {
                "tail": e.tail,
                "head": e.head,
                "face_id": e.face_id,
                "twin_id": e.twin_id,
            }
            for e in final_edges
        ],
        "metadata": {
            "algorithm": result.metadata.algorithm,
            "seed": result.metadata.seed,
            "n_points_input": result.metadata.n_points_input,
            "n_vertices_hull": result.metadata.n_vertices_hull,
            "n_faces": result.metadata.n_faces,
        },
    }


def dump_golden(result: HullResult3D, path: Path | str) -> None:
    """Serialize *result* to a JSON golden file at *path*.

    Delegates to :func:`to_golden_dict` for the canonical representation and
    writes the result as indented JSON with a trailing newline.

    Args:
        result: Hull result to serialize.
        path: Destination file path (created or overwritten).
    """
    data = to_golden_dict(result)
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_golden(path: Path | str) -> HullResult3D:
    """Load a golden file and return a :class:`~domain.entities.HullResult3D`.

    Args:
        path: Path to the golden JSON file.

    Returns:
        A fully populated :class:`~domain.entities.HullResult3D`.

    Raises:
        GoldenSchemaError: If the file's ``schema_version`` is absent or not
            equal to :data:`SCHEMA_VERSION`.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise GoldenSchemaError(
            f"Unsupported golden schema version: {version!r}. Expected {SCHEMA_VERSION}."
        )

    points = tuple(Point3D(float(p[0]), float(p[1]), float(p[2])) for p in raw["points"])

    vertex_indices: frozenset[int] = frozenset(int(i) for i in raw["vertex_indices"])

    faces = tuple(
        Face3D(
            vertex_indices=tuple(int(i) for i in f["vertex_indices"]),
            normal=Vector3D(float(f["normal"][0]), float(f["normal"][1]), float(f["normal"][2])),
        )
        for f in raw["faces"]
    )

    oriented_edges = tuple(
        OrientedEdge(
            tail=int(e["tail"]),
            head=int(e["head"]),
            face_id=int(e["face_id"]),
            twin_id=int(e["twin_id"]),
        )
        for e in raw["oriented_edges"]
    )

    meta_raw = raw["metadata"]
    metadata = HullMetadata(
        algorithm=str(meta_raw["algorithm"]),
        seed=int(meta_raw["seed"]) if meta_raw["seed"] is not None else None,
        n_points_input=int(meta_raw["n_points_input"]),
        n_vertices_hull=int(meta_raw["n_vertices_hull"]),
        n_faces=int(meta_raw["n_faces"]),
    )

    return HullResult3D(
        points=points,
        vertex_indices=vertex_indices,
        faces=faces,
        oriented_edges=oriented_edges,
        metadata=metadata,
    )
