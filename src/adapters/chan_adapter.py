"""Adapter wrapping convex_hull/src/chan.py in the standard project algorithm contract."""

from __future__ import annotations

import importlib
import types
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from algorithms.polygonal_result import build_oriented_edges, unit_normal_for_face
from domain.entities import Face3D, HullMetadata, HullResult3D, Point3D, Vector3D


class ChanInputError(ValueError):
    """Input rejected before calling convex_hull/src/chan.py."""


class ChanExecutionError(RuntimeError):
    """convex_hull/src/chan.py raised an unexpected exception."""


class ChanOutputError(RuntimeError):
    """Legacy mesh cannot be converted to a valid HullResult3D."""


def _load_chan_module(chan_path: Path | None = None) -> ModuleType:
    # SourceFileLoader.exec_module fails on Windows with non-ASCII paths (Cyrillic),
    # so we load the source via Path.read_text and compile+exec it ourselves.
    path = chan_path
    if path is None:
        try:
            return importlib.import_module("convex_hull.chan")
        except ModuleNotFoundError:
            pass
        repo_root = Path(__file__).parent.parent.parent
        path = repo_root / "convex_hull" / "src" / "chan.py"

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModuleNotFoundError(f"Cannot locate chan.py at {path}") from exc

    mod = types.ModuleType("convex_hull.chan")
    mod.__file__ = str(path)
    exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


def _to_chan_points(
    points: Sequence[Point3D],
    chan: ModuleType,
) -> tuple[list[object], dict[tuple[float, float, float], int]]:
    coord_to_index: dict[tuple[float, float, float], int] = {}
    legacy: list[object] = []
    for i, p in enumerate(points):
        key = (p.x, p.y, p.z)
        if key in coord_to_index:
            raise ChanInputError("Legacy chan.py adapter does not support duplicate points.")
        coord_to_index[key] = i
        legacy.append(chan.Point3D(p.x, p.y, p.z))
    return legacy, coord_to_index


def _extract_unique_faces(legacy_mesh: object) -> list[object]:
    face_map = legacy_mesh.face_map  # type: ignore[attr-defined]
    return list({id(face): face for face in face_map.values()}.values())


def _orient_face_outward(
    raw_indices: tuple[int, int, int],
    points: Sequence[Point3D],
    cx: float,
    cy: float,
    cz: float,
) -> tuple[tuple[int, int, int], Vector3D]:
    normal = unit_normal_for_face(points, raw_indices)
    p = points[raw_indices[0]]
    dot = normal.x * (p.x - cx) + normal.y * (p.y - cy) + normal.z * (p.z - cz)
    if dot < 0:
        i, j, k = raw_indices
        return (i, k, j), Vector3D(-normal.x, -normal.y, -normal.z)
    return raw_indices, normal


def _legacy_mesh_to_hull_result(
    points: Sequence[Point3D],
    legacy_mesh: object,
    coord_to_index: dict[tuple[float, float, float], int],
    seed: int | None,
    source: str = "convex_hull/src/chan.py",
) -> HullResult3D:
    legacy_faces = _extract_unique_faces(legacy_mesh)
    if not legacy_faces:
        raise ChanOutputError("Legacy chan.py returned an empty mesh.")

    n = len(points)
    cx = sum(p.x for p in points) / n
    cy = sum(p.y for p in points) / n
    cz = sum(p.z for p in points) / n

    faces: list[Face3D] = []
    for legacy_face in legacy_faces:
        verts = legacy_face.get_vertices()  # type: ignore[attr-defined]
        if len(verts) != 3:
            raise ChanOutputError(f"Only triangular faces are supported; got {len(verts)}-gon.")
        try:
            raw = tuple(coord_to_index[(v.x, v.y, v.z)] for v in verts)
        except KeyError as exc:
            raise ChanOutputError(f"Legacy mesh vertex not found in input: {exc}") from exc

        indices, normal = _orient_face_outward(raw, points, cx, cy, cz)  # type: ignore[arg-type]
        faces.append(Face3D(vertex_indices=indices, normal=normal))

    try:
        edges = build_oriented_edges(faces)
    except RuntimeError as exc:
        raise ChanOutputError(f"Legacy chan.py returned an open mesh: {exc}") from exc

    vertex_indices: frozenset[int] = frozenset(idx for face in faces for idx in face.vertex_indices)

    return HullResult3D(
        points=tuple(points),
        vertex_indices=vertex_indices,
        faces=tuple(faces),
        oriented_edges=edges,
        metadata=HullMetadata(
            algorithm="chan",
            seed=seed,
            n_points_input=len(points),
            n_vertices_hull=len(vertex_indices),
            n_faces=len(faces),
            trace={
                "source": source,
                "adapter": "ChanHull3D",
            },
        ),
    )


class ChanHull3D:
    """Adapter wrapping external convex_hull/src/chan.py as a standard project algorithm."""

    def __init__(self, chan_path: Path | None = None) -> None:
        """Store optional path override for chan.py."""
        self._chan_path = chan_path

    def compute(
        self,
        points: Sequence[Point3D],
        *,
        seed: int | None = None,
    ) -> HullResult3D:
        """Compute convex hull via convex_hull/src/chan.py and return HullResult3D."""
        if len(points) < 4:
            raise ChanInputError(f"At least 4 points required; got {len(points)}.")

        try:
            chan = _load_chan_module(self._chan_path)
        except ModuleNotFoundError as exc:
            raise ChanExecutionError(str(exc)) from exc

        legacy_points, coord_to_index = _to_chan_points(points, chan)

        try:
            legacy_mesh = chan.chans_algorithm(legacy_points)
        except Exception as exc:
            raise ChanExecutionError(
                f"convex_hull/src/chan.py failed: {type(exc).__name__}: {exc}"
            ) from exc

        if legacy_mesh is None:
            raise ChanExecutionError(
                "convex_hull/src/chan.py returned None (algorithm did not converge)."
            )

        source = str(self._chan_path) if self._chan_path is not None else "convex_hull/src/chan.py"
        return _legacy_mesh_to_hull_result(points, legacy_mesh, coord_to_index, seed, source)
