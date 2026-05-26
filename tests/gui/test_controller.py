from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from adapters.bruteforce_adapter import DegenerateBruteforce3D
from app import service as app_service
from domain.entities import HullMetadata, HullResult3D, Point3D
from gui.controller import GuiController
from gui.state import GuiState
from verification.golden import load_golden, to_golden_dict

_TETRA_POINTS = [
    Point3D(0.0, 0.0, 0.0),
    Point3D(1.0, 0.0, 0.0),
    Point3D(0.0, 1.0, 0.0),
    Point3D(0.0, 0.0, 1.0),
]


class _FakeAlgorithm:
    def __init__(self, result: HullResult3D | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[list[Point3D], int | None]] = []

    def compute(self, points: list[Point3D], *, seed: int | None = None) -> HullResult3D:
        self.calls.append((list(points), seed))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _build_result(
    points: list[Point3D] | None = None,
    *,
    algorithm: str = "chan",
    seed: int | None = 1,
) -> HullResult3D:
    source_points = list(points) if points is not None else list(_TETRA_POINTS)
    result = DegenerateBruteforce3D().compute(source_points, seed=seed)
    metadata = HullMetadata(
        algorithm=algorithm,
        seed=seed,
        n_points_input=result.metadata.n_points_input,
        n_vertices_hull=result.metadata.n_vertices_hull,
        n_faces=result.metadata.n_faces,
        trace=result.metadata.trace,
    )
    return HullResult3D(
        points=result.points,
        vertex_indices=result.vertex_indices,
        faces=result.faces,
        oriented_edges=result.oriented_edges,
        metadata=metadata,
    )


def _make_temp_dir() -> Path:
    temp_dir = Path.cwd() / "out" / "gui_controller_test_tmp" / uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _patch_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch service with minimal algorithms and scenarios."""
    monkeypatch.setattr(
        app_service, "ALGORITHMS", {"bruteforce_degenerate": object, "chan": object}
    )
    monkeypatch.setattr(
        app_service,
        "SCENARIOS",
        [("uniform_sphere", object), ("uniform_cube", object)],
    )
    monkeypatch.setattr(
        app_service,
        "SCENARIO_MAP",
        {"uniform_sphere": object, "uniform_cube": object},
    )


def test_controller_lists_algorithms_hardcoded() -> None:
    controller = GuiController()

    assert controller.available_algorithms() == ["bruteforce_degenerate", "chan"]


def test_controller_lists_scenarios_in_definition_order(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        app_service,
        "SCENARIOS",
        [("uniform_sphere", object), ("uniform_cube", object), ("prism_polygon", object)],
    )

    controller = GuiController()

    assert controller.available_scenarios() == ["uniform_sphere", "uniform_cube", "prism_polygon"]


def test_set_parameters_validates_and_updates_state(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _patch_service(monkeypatch)

    controller = GuiController()
    controller.set_parameters(
        algorithm="chan",
        scenario="uniform_sphere",
        n=12,
        seed=7,
    )

    assert controller.state.algorithm == "chan"
    assert controller.state.scenario == "uniform_sphere"
    assert controller.state.n == 12
    assert controller.state.seed == 7

    with pytest.raises(ValueError, match="Unknown algorithm"):
        controller.set_parameters(
            algorithm="missing",
            scenario="uniform_sphere",
            n=12,
            seed=7,
        )

    with pytest.raises(ValueError, match="Unknown scenario"):
        controller.set_parameters(
            algorithm="chan",
            scenario="missing",
            n=12,
            seed=7,
        )

    with pytest.raises(ValueError, match="at least 4"):
        controller.set_parameters(
            algorithm="chan",
            scenario="uniform_sphere",
            n=3,
            seed=7,
        )


def test_generate_points_stores_points_and_clears_previous_result(  # type: ignore[no-untyped-def]
    monkeypatch,
) -> None:
    generated = [
        Point3D(1.0, 2.0, 3.0),
        Point3D(4.0, 5.0, 6.0),
        Point3D(7.0, 8.0, 9.0),
        Point3D(10.0, 11.0, 12.0),
    ]
    old_result = _build_result()
    state = GuiState(
        points=None,
        result=old_result,
        last_error="old error",
        seed=7,
    )
    controller = GuiController(state=state)

    def _fake_generate_points(scenario: str, n: int, *, seed: int | None) -> list[Point3D]:
        assert scenario == state.scenario
        assert n == state.n
        assert seed == state.seed
        return list(generated)

    monkeypatch.setattr(app_service, "generate_points", _fake_generate_points)

    returned = controller.generate_points()

    assert returned == generated
    assert controller.state.points == generated
    assert controller.state.result is None
    assert controller.state.last_error is None


def test_generate_points_resolves_random_seed_and_records_last_used_seed(  # type: ignore[no-untyped-def]
    monkeypatch,
) -> None:
    import gui.controller as controller_module

    state = GuiState(seed=None)
    controller = GuiController(state=state)
    generate_calls: list[tuple[str, int, int | None]] = []

    def _fake_generate_points(scenario: str, n: int, *, seed: int | None) -> list[Point3D]:
        generate_calls.append((scenario, n, seed))
        return list(_TETRA_POINTS)

    monkeypatch.setattr(controller_module, "make_random_seed", lambda: 123456)
    monkeypatch.setattr(app_service, "generate_points", _fake_generate_points)

    returned = controller.generate_points()

    assert returned == list(_TETRA_POINTS)
    assert generate_calls == [(state.scenario, state.n, 123456)]
    assert controller.state.last_used_seed == 123456
    assert controller.state.seed is None


def test_build_hull_uses_existing_points_and_does_not_regenerate(  # type: ignore[no-untyped-def]
    monkeypatch,
) -> None:
    import gui.controller as controller_module

    state = GuiState(
        points=list(_TETRA_POINTS),
        seed=1,
    )
    controller = GuiController(state=state)
    expected = _build_result(list(_TETRA_POINTS), algorithm=state.algorithm, seed=state.seed)
    fake_algorithm = _FakeAlgorithm(result=expected)
    generate_calls = 0

    def _fake_generate_points(_scenario: str, _n: int, *, seed: int | None) -> list[Point3D]:
        nonlocal generate_calls
        generate_calls += 1
        return list(_TETRA_POINTS)

    monkeypatch.setattr(app_service, "generate_points", _fake_generate_points)
    monkeypatch.setattr(app_service, "get_algorithm", lambda _name: fake_algorithm)
    perf_counter_values = iter([10.0, 10.125])
    monkeypatch.setattr(controller_module, "perf_counter", lambda: next(perf_counter_values))

    result = controller.build_hull()

    assert result == expected
    assert generate_calls == 0
    assert fake_algorithm.calls == [(list(_TETRA_POINTS), state.seed)]
    assert controller.state.result == expected
    assert controller.state.last_error is None
    assert controller.state.last_build_seconds == pytest.approx(0.125)


def test_build_hull_reuses_resolved_random_seed_for_generation_and_algorithm(  # type: ignore[no-untyped-def]
    monkeypatch,
) -> None:
    import gui.controller as controller_module

    state = GuiState(points=None, seed=None)
    controller = GuiController(state=state)
    expected = _build_result(list(_TETRA_POINTS), algorithm=state.algorithm, seed=654321)
    fake_algorithm = _FakeAlgorithm(result=expected)
    generate_calls: list[tuple[str, int, int | None]] = []

    def _fake_generate_points(scenario: str, n: int, *, seed: int | None) -> list[Point3D]:
        generate_calls.append((scenario, n, seed))
        return list(_TETRA_POINTS)

    monkeypatch.setattr(controller_module, "make_random_seed", lambda: 654321)
    monkeypatch.setattr(app_service, "generate_points", _fake_generate_points)
    monkeypatch.setattr(app_service, "get_algorithm", lambda _name: fake_algorithm)
    perf_counter_values = iter([50.0, 50.25])
    monkeypatch.setattr(controller_module, "perf_counter", lambda: next(perf_counter_values))

    result = controller.build_hull()

    assert result == expected
    assert generate_calls == [(state.scenario, state.n, 654321)]
    assert fake_algorithm.calls == [(list(_TETRA_POINTS), 654321)]
    assert controller.state.last_used_seed == 654321
    assert controller.state.seed is None
    assert controller.state.last_build_seconds == pytest.approx(0.25)


def test_build_hull_records_errors_in_last_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import gui.controller as controller_module

    state = GuiState(points=list(_TETRA_POINTS), seed=1)
    controller = GuiController(state=state)
    fake_algorithm = _FakeAlgorithm(error=ValueError("boom"))

    monkeypatch.setattr(app_service, "get_algorithm", lambda _name: fake_algorithm)
    perf_counter_values = iter([100.0])
    monkeypatch.setattr(controller_module, "perf_counter", lambda: next(perf_counter_values))

    with pytest.raises(ValueError, match="boom"):
        controller.build_hull()

    assert controller.state.last_error == "ValueError: boom"
    assert controller.state.last_build_seconds is None


def test_load_manual_points_stores_points_and_clears_state() -> None:
    old_result = _build_result()
    state = GuiState(points=list(_TETRA_POINTS), result=old_result, last_error="old")
    controller = GuiController(state=state)

    new_points = [Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0), Point3D(0, 0, 1)]
    controller.load_manual_points(new_points)

    assert controller.state.points == new_points
    assert controller.state.result is None
    assert controller.state.last_error is None
    assert controller.state.last_used_seed is None
    assert controller.state.generation_seed is None


def test_load_save_roundtrip() -> None:
    temp_dir = _make_temp_dir()
    try:
        source_path = temp_dir / "source.json"
        saved_path = temp_dir / "saved.json"
        expected = _build_result()

        from verification.golden import dump_golden

        dump_golden(expected, source_path)

        controller = GuiController()
        loaded = controller.load_result(source_path)
        controller.save_result(saved_path)
        saved = load_golden(saved_path)

        assert to_golden_dict(loaded) == to_golden_dict(expected)
        assert controller.state.points == list(expected.points)
        assert to_golden_dict(saved) == to_golden_dict(expected)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_save_without_result_raises_value_error() -> None:
    controller = GuiController()

    with pytest.raises(ValueError, match="No hull result is available to save"):
        controller.save_result(Path("unused.json"))


def test_clear_error() -> None:
    controller = GuiController(state=GuiState(last_error="something failed"))

    controller.clear_error()

    assert controller.state.last_error is None


# ---------------------------------------------------------------------------
# Integration tests вЂ” real algorithms, no mocks
# ---------------------------------------------------------------------------

_SPHERE_POINTS_N10_S42 = [
    Point3D(0.374540, 0.950714, -0.163485),
    Point3D(0.731994, 0.598658, -0.325259),
    Point3D(0.156019, 0.058084, -0.985932),
    Point3D(0.866176, 0.601115, -0.352990),
    Point3D(0.708073, 0.020584, 0.705790),
    Point3D(0.969910, 0.832443, -0.213896),
    Point3D(0.181825, 0.183405, 0.966040),
    Point3D(0.611853, 0.139494, -0.779288),
    Point3D(0.292145, 0.366362, -0.882820),
    Point3D(0.456070, 0.785176, -0.419983),
]


def test_integration_chan_builds_valid_hull_for_tetrahedron() -> None:
    state = GuiState(points=list(_TETRA_POINTS), seed=1, algorithm="chan")
    controller = GuiController(state=state)

    result = controller.build_hull()

    assert len(result.vertex_indices) == 4
    assert len(result.faces) == 4
    assert all(len(f.vertex_indices) == 3 for f in result.faces)


def test_integration_bruteforce_degenerate_builds_valid_hull_for_unit_cube() -> None:
    from generators.degeneracies import unit_cube_exact

    state = GuiState(
        points=unit_cube_exact(),
        seed=None,
        algorithm="bruteforce_degenerate",
    )
    controller = GuiController(state=state)

    result = controller.build_hull()

    assert len(result.vertex_indices) == 8
    assert len(result.faces) == 6


def test_integration_chan_generate_and_build_end_to_end() -> None:
    state = GuiState(algorithm="chan", scenario="uniform_sphere", n=20, seed=42)
    controller = GuiController(state=state)

    points = controller.generate_points()
    assert len(points) == 20
    assert controller.state.result is None

    result = controller.build_hull()

    assert result is not None
    assert controller.state.result is result
    assert controller.state.points == list(result.points)
    assert len(result.vertex_indices) >= 4


def test_integration_bruteforce_degenerate_build_cube_with_edge_midpoints() -> None:
    from generators.degeneracies import cube_with_edge_midpoints

    state = GuiState(
        points=cube_with_edge_midpoints(),
        seed=None,
        algorithm="bruteforce_degenerate",
    )
    controller = GuiController(state=state)

    result = controller.build_hull()

    assert len(result.vertex_indices) == 8
    assert len(result.faces) == 6
