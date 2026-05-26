"""Adapter wrapping convex_hull/src/bruteforce_degenerate.py in the project algorithm contract."""

from __future__ import annotations

import sys
import types
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from domain.entities import Face3D, HullMetadata, HullResult3D, OrientedEdge, Point3D, Vector3D


class UnsupportedAffineDimensionError(ValueError):
    """Raised when the input does not have sufficient affine dimension."""


def _load_bruteforce_module(bf_path: Path | None = None) -> ModuleType:
    # Use read_text + exec to avoid SourceFileLoader issues with non-ASCII paths on Windows.
    path = bf_path
    if path is None:
        repo_root = Path(__file__).parent.parent.parent
        path = repo_root / "convex_hull" / "src" / "bruteforce_degenerate.py"

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModuleNotFoundError(
            f"Cannot locate bruteforce_degenerate.py at {path}"
        ) from exc

    mod = types.ModuleType("convex_hull.bruteforce_degenerate")
    mod.__file__ = str(path)
    # Register before exec: dataclasses look up cls.__module__ in sys.modules at definition time.
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(source, str(path), "exec"), mod.__dict__)
    except Exception:
        sys.modules.pop(mod.__name__, None)
        raise
    return mod


def _to_bf_points(points: Sequence[Point3D], bf: ModuleType) -> list[Any]:
    return [bf.Point3D(p.x, p.y, p.z) for p in points]


def _bf_result_to_domain(bf_result: Any, original_points: tuple[Point3D, ...]) -> HullResult3D:
    domain_faces = tuple(
        Face3D(
            vertex_indices=face.vertex_indices,
            normal=Vector3D(face.normal.x, face.normal.y, face.normal.z),
        )
        for face in bf_result.faces
    )
    domain_edges = tuple(
        OrientedEdge(
            tail=edge.tail,
            head=edge.head,
            face_id=edge.face_id,
            twin_id=edge.twin_id,
        )
        for edge in bf_result.oriented_edges
    )
    domain_metadata = HullMetadata(
        algorithm=bf_result.metadata.algorithm,
        seed=bf_result.metadata.seed,
        n_points_input=bf_result.metadata.n_points_input,
        n_vertices_hull=bf_result.metadata.n_vertices_hull,
        n_faces=bf_result.metadata.n_faces,
        trace=bf_result.metadata.trace,
    )
    return HullResult3D(
        points=original_points,
        vertex_indices=bf_result.vertex_indices,
        faces=domain_faces,
        oriented_edges=domain_edges,
        metadata=domain_metadata,
    )


class DegenerateBruteforce3D:
    """Adapter wrapping convex_hull/src/bruteforce_degenerate.py."""

    def __init__(self, bf_path: Path | None = None) -> None:
        """Store optional path override for bruteforce_degenerate.py."""
        self._bf_path = bf_path

    def compute(
        self,
        points: Sequence[Point3D],
        *,
        seed: int | None = None,
    ) -> HullResult3D:
        """Compute convex hull via convex_hull/src/bruteforce_degenerate.py."""
        bf = _load_bruteforce_module(self._bf_path)
        original_points = tuple(points)
        bf_points = _to_bf_points(original_points, bf)

        try:
            bf_result = bf.DegenerateBruteforce3D().compute(bf_points, seed=seed)
        except bf.UnsupportedAffineDimensionError as exc:
            raise UnsupportedAffineDimensionError(str(exc)) from exc

        return _bf_result_to_domain(bf_result, original_points)
