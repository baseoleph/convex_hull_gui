"""Point cloud generators for non-degenerate (general-position) test scenarios."""

from __future__ import annotations

import numpy as np

from domain.entities import Point3D


def _rng(seed: int | None) -> np.random.Generator:
    """Return a seeded numpy Generator."""
    return np.random.default_rng(seed)


def uniform_cube(n: int, *, seed: int | None = None) -> list[Point3D]:
    """Generate *n* points drawn uniformly from the unit cube [0, 1]^3.

    Args:
        n: Number of points to generate.
        seed: Random seed for reproducibility.

    Returns:
        A list of *n* :class:`~domain.entities.Point3D` instances.
    """
    rng = _rng(seed)
    coords: np.ndarray = rng.uniform(0.0, 1.0, (n, 3))
    return [Point3D(float(row[0]), float(row[1]), float(row[2])) for row in coords]


def uniform_sphere(n: int, *, seed: int | None = None) -> list[Point3D]:
    """Generate *n* points drawn uniformly from the interior of the unit ball.

    Uses rejection sampling from [-1, 1]^3.

    Args:
        n: Number of points to generate.
        seed: Random seed for reproducibility.

    Returns:
        A list of *n* :class:`~domain.entities.Point3D` instances,
        all satisfying x² + y² + z² ≤ 1.
    """
    rng = _rng(seed)
    pts: list[Point3D] = []
    batch_size = max(n * 2, 64)
    while len(pts) < n:
        batch: np.ndarray = rng.uniform(-1.0, 1.0, (batch_size, 3))
        r2: np.ndarray = (batch**2).sum(axis=1)
        accepted = batch[r2 <= 1.0]
        for row in accepted:
            pts.append(Point3D(float(row[0]), float(row[1]), float(row[2])))
            if len(pts) == n:
                break
    return pts


def points_on_sphere(n: int, *, seed: int | None = None) -> list[Point3D]:
    """Generate *n* points distributed uniformly on the unit sphere surface.

    All returned points satisfy x² + y² + z² = 1 (up to floating-point
    precision), so every point is a vertex of its convex hull when the
    input is in general position.

    Uses the Box-Muller / normal-distribution projection method.

    Args:
        n: Number of points to generate.
        seed: Random seed for reproducibility.

    Returns:
        A list of *n* :class:`~domain.entities.Point3D` instances.
    """
    rng = _rng(seed)
    raw: np.ndarray = rng.standard_normal((n, 3))
    norms: np.ndarray = np.linalg.norm(raw, axis=1, keepdims=True)
    unit: np.ndarray = raw / norms
    return [Point3D(float(row[0]), float(row[1]), float(row[2])) for row in unit]


