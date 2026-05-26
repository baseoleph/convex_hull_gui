"""Pure controller layer for the demo GUI."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Literal

from app import service as app_service
from app.random_source import make_random_seed
from domain.entities import HullResult3D, Point3D
from gui.state import GuiState
from verification.golden import dump_golden, load_golden


def _format_error(exc: Exception) -> str:
    """Return a compact human-readable error message."""
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


class GuiController:
    """Stateful orchestration layer for the GUI without any Qt dependencies."""

    def __init__(self, state: GuiState | None = None) -> None:
        """Initialise the controller with existing or default state."""
        self._state = state if state is not None else GuiState()

    @property
    def state(self) -> GuiState:
        """Return the mutable GUI state object."""
        return self._state

    def available_algorithms(self) -> list[str]:
        """Return algorithm names for UI display."""
        return ["bruteforce_degenerate", "chan"]

    def available_scenarios(self) -> list[str]:
        """Return registered scenario names in definition order."""
        return [name for name, _ in app_service.SCENARIOS]

    def set_parameters(
        self,
        *,
        algorithm: str,
        scenario: str,
        n: int,
        seed: int | None,
    ) -> None:
        """Validate and store UI parameters in the state."""
        if algorithm not in app_service.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        if scenario not in app_service.SCENARIO_MAP:
            raise ValueError(f"Unknown scenario: {scenario}")
        if n < 4:
            raise ValueError(f"Point count must be at least 4, got {n}.")

        self._state.algorithm = algorithm
        self._state.scenario = scenario
        self._state.n = n
        self._state.seed = seed
        self._state.last_used_seed = None
        self._state.last_build_seconds = None

    def _resolve_seed(self) -> int:
        """Resolve the current seed selection into the actual seed to use now."""
        if self._state.seed is not None:
            self._state.last_used_seed = self._state.seed
            return self._state.seed
        resolved = make_random_seed()
        self._state.last_used_seed = resolved
        return resolved

    def _generate_points_with_seed(self, seed: int) -> list[Point3D]:
        """Generate points using an already resolved seed."""
        points = app_service.generate_points(
            self._state.scenario,
            self._state.n,
            seed=seed,
        )
        self._state.points = list(points)
        self._state.result = None
        self._state.last_error = None
        self._state.last_used_seed = seed
        self._state.generation_seed = seed
        self._state.last_build_seconds = None
        return self._state.points

    def generate_points(self) -> list[Point3D]:
        """Generate points from the current state parameters."""
        seed = self._resolve_seed()
        return self._generate_points_with_seed(seed)

    def build_hull(self) -> HullResult3D:
        """Build a hull for the current or newly generated points."""
        seed = self._resolve_seed()
        points = (
            self._state.points
            if self._state.points is not None
            else self._generate_points_with_seed(seed)
        )
        algorithm = app_service.get_algorithm(self._state.algorithm)
        self._state.last_build_seconds = None

        started_at = perf_counter()
        try:
            result = algorithm.compute(points, seed=seed)
        except Exception as exc:
            self._state.last_error = _format_error(exc)
            self._state.last_build_seconds = None
            raise

        self._state.points = list(result.points)
        self._state.result = result
        self._state.last_error = None
        self._state.last_used_seed = seed
        self._state.last_build_seconds = perf_counter() - started_at
        return result

    def apply_hull_result(
        self,
        result: HullResult3D,
        *,
        build_seconds: float | None = None,
    ) -> HullResult3D:
        """Store an externally computed hull result in state."""
        self._state.points = list(result.points)
        self._state.result = result
        self._state.last_error = None
        self._state.last_build_seconds = build_seconds
        return result

    def load_result(self, path: Path) -> HullResult3D:
        """Load a saved hull result and update state from it."""
        result = load_golden(path)
        self._state.result = result
        self._state.points = list(result.points)
        self._state.report = None
        self._state.last_error = None
        self._state.last_used_seed = result.metadata.seed
        self._state.generation_seed = result.metadata.seed
        self._state.last_build_seconds = None
        return result

    def load_json(
        self, path: Path
    ) -> tuple[Literal["hull"], HullResult3D] | tuple[Literal["points"], list[Point3D]]:
        """Load a JSON file in either golden (hull) or points-only format.

        Returns a tagged tuple ``("hull", result)`` or ``("points", points)``
        depending on the detected format.  Raises ``ValueError`` if neither
        ``schema_version`` nor ``points`` is present.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "schema_version" in raw:
            result = self.load_result(path)
            return ("hull", result)
        if "points" in raw:
            points = [Point3D(float(p[0]), float(p[1]), float(p[2])) for p in raw["points"]]
            self._state.points = points
            self._state.result = None
            self._state.report = None
            self._state.last_error = None
            self._state.last_used_seed = None
            self._state.generation_seed = None
            self._state.last_build_seconds = None
            return ("points", points)
        raise ValueError("JSON must contain 'schema_version' (hull) or 'points' key.")

    def save_result(self, path: Path) -> None:
        """Save the current hull result to a golden JSON file."""
        if self._state.result is None:
            raise ValueError("No hull result is available to save.")

        dump_golden(self._state.result, path)
        self._state.last_error = None

    def load_manual_points(self, points: list[Point3D]) -> None:
        """Store a manually entered point set in state without generating."""
        self._state.points = list(points)
        self._state.result = None
        self._state.last_error = None
        self._state.last_used_seed = None
        self._state.generation_seed = None
        self._state.last_build_seconds = None

    def clear_error(self) -> None:
        """Clear the last recorded user-visible error."""
        self._state.last_error = None
