"""Pure GUI state for the demo application."""

from __future__ import annotations

from dataclasses import dataclass, field

from app import service as app_service
from domain.entities import HullResult3D, Point3D


def _default_algorithm() -> str:
    """Return the preferred default algorithm name for the GUI."""
    if "chan" in app_service.ALGORITHMS:
        return "chan"
    return sorted(app_service.ALGORITHMS)[0]


def _default_scenario() -> str:
    """Return the preferred default scenario name for the GUI."""
    if "uniform_sphere" in app_service.SCENARIO_MAP:
        return "uniform_sphere"
    return app_service.SCENARIOS[0][0]


@dataclass
class GuiState:
    """Current state of the GUI controller.

    Args:
        points: Currently loaded or generated point cloud.
        result: Current hull result, if available.
        report: Current verification report, if available.
        last_error: Last user-visible error message, if any.
        last_used_seed: Actual seed used by the last completed GUI action, if any.
        last_build_seconds: Duration of the last successful hull build, if any.
        algorithm: Selected algorithm name.
        scenario: Selected generator scenario name.
        n: Requested number of points for generation.
        seed: Random seed for generation and algorithm execution.
    """

    points: list[Point3D] | None = None
    result: HullResult3D | None = None
    report: str | None = None
    last_error: str | None = None
    last_used_seed: int | None = None
    last_build_seconds: float | None = None
    generation_seed: int | None = None
    algorithm: str = field(default_factory=_default_algorithm)
    scenario: str = field(default_factory=_default_scenario)
    n: int = 20
    seed: int | None = None
