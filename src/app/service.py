"""Registry and helpers for GUI algorithms, scenarios, and point loading."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from adapters.bruteforce_adapter import DegenerateBruteforce3D
from adapters.chan_adapter import ChanHull3D
from domain.entities import HullResult3D, Point3D
from generators import (
    coplanar_square_with_center,
    cube_with_edge_midpoints,
    cube_with_face_centers,
    points_on_sphere,
    rectangular_box_exact,
    tetrahedron_with_edge_points,
    tetrahedron_with_face_points,
    uniform_cube,
    uniform_sphere,
    unit_cube_exact,
)
from generators.benchmarks import (
    controlled_h_8,
    controlled_h_32,
    controlled_h_128,
    controlled_h_half_n,
    controlled_h_sqrt_n,
    cube_with_many_edge_points,
    cube_with_many_face_points,
    narrow_lens_rim,
    prism_polygon,
    stretched_moment_alpha0,
)


class HullAlgorithm3D(Protocol):
    """Protocol for 3-D convex hull algorithms."""

    def compute(self, points: Sequence[Point3D], *, seed: int | None = None) -> HullResult3D:
        """Compute the convex hull of *points* and return a HullResult3D."""
        ...


def _fixed_scenario(
    generator: Callable[[], list[Point3D]],
) -> Callable[..., list[Point3D]]:
    def wrapped(_n: int, *, seed: int | None) -> list[Point3D]:
        return generator()

    return wrapped


def _prism_scenario(n: int, *, seed: int | None = None) -> list[Point3D]:
    return prism_polygon(max(3, n))


SCENARIOS: list[tuple[str, Callable[..., list[Point3D]]]] = [
    ("uniform_sphere", uniform_sphere),
    ("uniform_cube", uniform_cube),
    ("points_on_sphere", points_on_sphere),
    ("narrow_lens_rim", narrow_lens_rim),
    ("cube_with_many_face_points", cube_with_many_face_points),
    ("cube_with_many_edge_points", cube_with_many_edge_points),
    ("stretched_moment_alpha0", stretched_moment_alpha0),
    ("prism_polygon", _prism_scenario),
    ("unit_cube_exact", _fixed_scenario(unit_cube_exact)),
    ("rectangular_box_exact", _fixed_scenario(rectangular_box_exact)),
    ("cube_with_edge_midpoints", _fixed_scenario(cube_with_edge_midpoints)),
    ("cube_with_face_centers", _fixed_scenario(cube_with_face_centers)),
    ("tetrahedron_with_edge_points", _fixed_scenario(tetrahedron_with_edge_points)),
    ("tetrahedron_with_face_points", _fixed_scenario(tetrahedron_with_face_points)),
    ("coplanar_square_with_center", _fixed_scenario(coplanar_square_with_center)),
    ("controlled_h_8", controlled_h_8),
    ("controlled_h_32", controlled_h_32),
    ("controlled_h_128", controlled_h_128),
    ("controlled_h_sqrt_n", controlled_h_sqrt_n),
    ("controlled_h_half_n", controlled_h_half_n),
]

SCENARIO_MAP: dict[str, Callable[..., list[Point3D]]] = dict(SCENARIOS)

ALGORITHMS: dict[str, Callable[[], HullAlgorithm3D]] = {
    "bruteforce_degenerate": DegenerateBruteforce3D,
    "chan": ChanHull3D,
}


def load_points_from_file(path: Path) -> list[Point3D]:
    """Load a point list from a JSON file in ``{"points": [[x, y, z], ...]}`` format."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc
    if "points" not in data:
        raise ValueError(f'{path}: missing "points" key')
    raw = data["points"]
    if not isinstance(raw, list):
        raise ValueError(f'{path}: "points" must be a JSON array')
    pts: list[Point3D] = []
    for i, item in enumerate(raw):
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError(f"{path}: item {i} is not a 3-element array")
        try:
            pts.append(Point3D(float(item[0]), float(item[1]), float(item[2])))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: item {i} has non-numeric coordinate: {exc}") from exc
    return pts


def generate_points(scenario: str, n: int, *, seed: int | None) -> list[Point3D]:
    """Generate *n* points using the named scenario generator."""
    return SCENARIO_MAP[scenario](n, seed=seed)


def get_algorithm(name: str) -> HullAlgorithm3D:
    """Return a fresh instance of the named hull algorithm."""
    return ALGORITHMS[name]()

