"""Core geometric value objects for the chanhull3d domain."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Vector3D:
    """Immutable 3-D vector with basic linear algebra operations."""

    x: float
    y: float
    z: float

    def dot(self, other: Vector3D) -> float:
        """Return the dot product of this vector with *other*."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3D) -> Vector3D:
        """Return the cross product of this vector with *other*."""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        """Return the Euclidean length of this vector."""
        return math.sqrt(self.dot(self))

    def is_zero(self, tol: float = 0.0) -> bool:
        """Return True if the vector's norm is at most *tol*."""
        return self.norm() <= tol


@dataclass(frozen=True)
class Point3D:
    """Immutable 3-D point."""

    x: float
    y: float
    z: float

    def __sub__(self, other: Point3D) -> Vector3D:
        """Return the displacement vector from *other* to this point."""
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)


@dataclass(frozen=True)
class Face3D:
    """A convex polygonal face defined by vertex indices and an outward normal."""

    vertex_indices: tuple[int, ...]
    normal: Vector3D

    def __post_init__(self) -> None:
        """Validate vertex count, uniqueness, and non-zero normal."""
        if len(self.vertex_indices) < 3:
            raise ValueError(
                f"Face3D requires at least 3 vertices, got {len(self.vertex_indices)}."
            )
        if len(set(self.vertex_indices)) != len(self.vertex_indices):
            raise ValueError(f"Face3D vertex indices must be unique, got {self.vertex_indices}.")
        if self.normal.is_zero():
            raise ValueError("Face3D normal must be a non-zero vector.")


@dataclass(frozen=True)
class OrientedEdge:
    """A directed half-edge in the hull's edge structure."""

    tail: int
    head: int
    face_id: int
    twin_id: int | None = field(default=None)


@dataclass(frozen=True)
class HullMetadata:
    """Metadata recorded alongside a hull computation result."""

    algorithm: str
    seed: int | None = field(default=None)
    n_points_input: int = field(default=0)
    n_vertices_hull: int = field(default=0)
    n_faces: int = field(default=0)
    trace: Any = field(default=None)


@dataclass(frozen=True)
class HullResult3D:
    """Complete output of a 3-D convex hull computation."""

    points: tuple[Point3D, ...]
    vertex_indices: frozenset[int]
    faces: tuple[Face3D, ...]
    oriented_edges: tuple[OrientedEdge, ...]
    metadata: HullMetadata
