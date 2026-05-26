"""Scalable benchmark point-cloud generators for degenerate 3D hull testing."""

from __future__ import annotations

import math

import numpy as np

from domain.entities import Point3D
from generators.degeneracies import unit_cube_exact


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _p(x: float, y: float, z: float) -> Point3D:
    return Point3D(x, y, z)


def _stretched_moment_coords(n: int, *, alpha: float) -> np.ndarray:
    """Return scaled twisted-cubic coordinates on a deterministic stretched grid."""
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")

    u = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    e = u if alpha == 0.0 else np.expm1(alpha * u) / math.expm1(alpha)

    t = 2.0 * e - 1.0
    raw = np.column_stack((t, t * t, t * t * t))
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    coords = raw.copy()
    for axis in range(coords.shape[1]):
        lo = float(mins[axis])
        hi = float(maxs[axis])
        if hi > lo:
            coords[:, axis] = 2.0 * (coords[:, axis] - lo) / (hi - lo) - 1.0
    return coords


def _order_points(
    pts: list[Point3D],
    *,
    seed: int | None,
    shuffle: bool,
    order: str,
) -> list[Point3D]:
    """Apply a reproducible output ordering to a point list."""
    if order not in {"ordered", "reversed", "shuffled"}:
        raise ValueError(f"unknown order: {order!r}")
    effective_order = "shuffled" if shuffle else order
    if effective_order == "ordered":
        return pts
    if effective_order == "reversed":
        return list(reversed(pts))
    if effective_order == "shuffled":
        rng = _rng(seed)
        rng.shuffle(pts)
        return pts
    raise AssertionError(f"unreachable order state: {effective_order!r}")


def cube_with_many_face_points(n: int, *, seed: int | None = None) -> list[Point3D]:
    """Return 8 cube vertices plus n-8 points lying exactly on cube faces.

    Args:
        n: Total number of points. Must be >= 8.
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If n < 8.
    """
    if n < 8:
        raise ValueError(f"n must be >= 8, got {n}")
    cube = unit_cube_exact()
    if n == 8:
        return cube
    rng = _rng(seed)
    extra = n - 8
    face_ids = rng.integers(0, 6, size=extra)
    uv = rng.uniform(0.0, 1.0, (extra, 2))
    face_pts: list[Point3D] = []
    for i in range(extra):
        u, v = float(uv[i, 0]), float(uv[i, 1])
        fid = int(face_ids[i])
        if fid == 0:
            face_pts.append(_p(u, v, 0.0))
        elif fid == 1:
            face_pts.append(_p(u, v, 1.0))
        elif fid == 2:
            face_pts.append(_p(0.0, u, v))
        elif fid == 3:
            face_pts.append(_p(1.0, u, v))
        elif fid == 4:
            face_pts.append(_p(u, 0.0, v))
        else:
            face_pts.append(_p(u, 1.0, v))
    return cube + face_pts


def cube_with_many_edge_points(n: int, *, seed: int | None = None) -> list[Point3D]:
    """Return 8 cube vertices plus n-8 points lying exactly on cube edges.

    Args:
        n: Total number of points. Must be >= 8.
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If n < 8.
    """
    if n < 8:
        raise ValueError(f"n must be >= 8, got {n}")
    cube = unit_cube_exact()
    if n == 8:
        return cube
    rng = _rng(seed)
    extra = n - 8
    edges = [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ((1.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ]
    edge_ids = rng.integers(0, 12, size=extra)
    ts = rng.uniform(0.0, 1.0, size=extra)
    edge_pts: list[Point3D] = []
    for i in range(extra):
        (ox, oy, oz), (dx, dy, dz) = edges[int(edge_ids[i])]
        t = float(ts[i])
        edge_pts.append(_p(ox + t * dx, oy + t * dy, oz + t * dz))
    return cube + edge_pts


def narrow_lens_rim(
    n: int,
    *,
    lens_radius: float = 1e9,
    rim_width: float = 1e-9,
    seed: int | None = None,
) -> list[Point3D]:
    """Return n adversarial points on the equatorial rim of a narrow lens.

    Args:
        n: Number of points. Must be >= 4.
        lens_radius: Radius of the lens-defining spheres.
        rim_width: Half-thickness of the equatorial sampling band.
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If n < 4, lens_radius <= 1, or rim_width <= 0.
    """
    if n < 4:
        raise ValueError(f"n must be >= 4, got {n}")
    if lens_radius <= 1.0:
        raise ValueError(f"lens_radius must be > 1, got {lens_radius}")
    if rim_width <= 0.0:
        raise ValueError(f"rim_width must be > 0, got {rim_width}")

    rng = _rng(seed)
    r = float(lens_radius)

    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    z = rng.uniform(-rim_width, rim_width, n)
    sign = np.where(rng.random(n) < 0.5, +1.0, -1.0)
    cz = sign * (r - 1.0)

    r_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)

    dx, dy, dz = x, y, z - cz
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    scale = r / norm
    coords = np.column_stack((dx * scale, dy * scale, cz + dz * scale))
    return [_p(float(row[0]), float(row[1]), float(row[2])) for row in coords]


def prism_polygon(k: int, *, height: float = 1.0) -> list[Point3D]:
    """Return a regular k-gon prism: k bottom vertices then k top vertices.

    Args:
        k: Number of sides. Must be >= 3.
        height: Height of the prism.

    Raises:
        ValueError: If k < 3.
    """
    if k < 3:
        raise ValueError(f"k must be >= 3, got {k}")
    pts: list[Point3D] = []
    for z in (0.0, height):
        for i in range(k):
            angle = 2.0 * math.pi * i / k
            pts.append(_p(math.cos(angle), math.sin(angle), z))
    return pts


def _controlled_h_general_position(
    n: int,
    *,
    h: int,
    seed: int | None = None,
    interior_scale: float = 0.75,
) -> list[Point3D]:
    """Return n general-position points with exactly h hull vertices.

    Args:
        n: Total number of points. Must be >= h.
        h: Number of hull vertices. Must be >= 4.
        seed: Random seed for reproducibility.
        interior_scale: Shrink factor toward centroid in (0, 1).

    Raises:
        ValueError: If h < 4, n < h, or interior_scale not in (0, 1).
        RuntimeError: If extreme points do not span 3D.
    """
    if h < 4:
        raise ValueError(f"h must be >= 4, got {h}")
    if n < h:
        raise ValueError(f"n must be >= h={h}, got n={n}")
    if not (0.0 < interior_scale < 1.0):
        raise ValueError(f"interior_scale must be in (0, 1), got {interior_scale}")

    rng = _rng(seed)

    raw = rng.standard_normal((h, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    extreme_arr = raw / norms

    diffs = extreme_arr[1:] - extreme_arr[0]
    if np.linalg.matrix_rank(diffs) < 3:
        raise RuntimeError(f"Extreme points do not span 3D (affine rank < 3) for seed={seed!r}")

    centroid = extreme_arr.mean(axis=0)
    extreme_pts = [_p(float(r[0]), float(r[1]), float(r[2])) for r in extreme_arr]

    num_interior = n - h
    if num_interior == 0:
        return extreme_pts

    chosen = rng.integers(0, h, size=(num_interior, 4))
    raw_w = rng.uniform(0.0, 1.0, size=(num_interior, 4))
    weights = raw_w / raw_w.sum(axis=1, keepdims=True)

    interior_arr = np.einsum("ij,ijk->ik", weights, extreme_arr[chosen])
    interior_arr = centroid + interior_scale * (interior_arr - centroid)

    interior_pts = [_p(float(r[0]), float(r[1]), float(r[2])) for r in interior_arr]
    return extreme_pts + interior_pts


def controlled_h_8(n: int, *, seed: int | None = None) -> list[Point3D]:
    """Controlled-h scenario with h=8 hull vertices."""
    return _controlled_h_general_position(n, h=8, seed=seed)


def controlled_h_32(n: int, *, seed: int | None = None) -> list[Point3D]:
    """Controlled-h scenario with h=32 hull vertices."""
    return _controlled_h_general_position(n, h=32, seed=seed)


def controlled_h_128(n: int, *, seed: int | None = None) -> list[Point3D]:
    """Controlled-h scenario with h=128 hull vertices. Requires n >= 128."""
    return _controlled_h_general_position(n, h=128, seed=seed)


def controlled_h_sqrt_n(n: int, *, seed: int | None = None) -> list[Point3D]:
    """Controlled-h scenario with h=floor(sqrt(n)), minimum 4."""
    h = max(4, int(math.sqrt(n)))
    return _controlled_h_general_position(n, h=h, seed=seed)


def controlled_h_half_n(n: int, *, seed: int | None = None) -> list[Point3D]:
    """Controlled-h scenario with h=n//2, minimum 4."""
    h = max(4, n // 2)
    return _controlled_h_general_position(n, h=h, seed=seed)


def stretched_moment_curve(
    n: int,
    *,
    alpha: float = 8.0,
    seed: int | None = None,
    shuffle: bool = False,
    order: str = "ordered",
) -> list[Point3D]:
    """Return n general-position points on a stretched moment curve (h == n).

    Args:
        n: Number of points. Must be >= 4.
        alpha: Stretching exponent.
        seed: Random seed used only for shuffled output order.
        shuffle: Backwards-compatible alias for ``order="shuffled"``.
        order: One of ``"ordered"``, ``"reversed"``, or ``"shuffled"``.

    Raises:
        ValueError: If n < 4, alpha < 0, or order is unknown.
    """
    if n < 4:
        raise ValueError(f"n must be >= 4, got {n}")
    coords = _stretched_moment_coords(n, alpha=alpha)
    pts = [_p(float(x), float(y), float(z)) for x, y, z in coords]
    return _order_points(pts, seed=seed, shuffle=shuffle, order=order)


def stretched_moment_alpha0(n: int, *, seed: int | None = None) -> list[Point3D]:
    """Stretched moment curve with alpha=0 (near-uniform spacing)."""
    return stretched_moment_curve(n, alpha=0.0, seed=seed, shuffle=False)
