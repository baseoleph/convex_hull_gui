from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from PySide6.QtWidgets import QWidget

from adapters.bruteforce_adapter import DegenerateBruteforce3D
from domain.entities import HullMetadata, HullResult3D, Point3D
from gui.window import MainWindow

_TETRA_POINTS = [
    Point3D(0.0, 0.0, 0.0),
    Point3D(1.0, 0.0, 0.0),
    Point3D(0.0, 1.0, 0.0),
    Point3D(0.0, 0.0, 1.0),
]


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


@dataclass
class _FakeState:
    algorithm: str = "chan"
    scenario: str = "uniform_sphere"
    n: int = 20
    seed: int | None = None
    points: list[Point3D] | None = None
    result: HullResult3D | None = None
    last_error: str | None = None
    last_used_seed: int | None = None
    last_build_seconds: float | None = None
    generation_seed: int | None = None


class _FakeController:
    def __init__(self) -> None:
        self.state = _FakeState()
        self.parameters_calls: list[tuple[str, str, int, int | None]] = []
        self.generate_calls = 0
        self.build_calls = 0
        self.load_calls: list[Path] = []
        self.save_calls: list[Path] = []
        self.manual_points_calls: list[list[Point3D]] = []
        self.random_seed_to_return = 424242
        self.points_to_return = list(_TETRA_POINTS)
        self.result_to_return = _build_result(list(_TETRA_POINTS), seed=424242)
        self.apply_hull_result_calls = 0
        self.generate_error: Exception | None = None
        self.load_error: Exception | None = None
        self.save_error: Exception | None = None

    def available_algorithms(self) -> list[str]:
        return ["bruteforce_degenerate", "chan"]

    def available_scenarios(self) -> list[str]:
        return ["uniform_sphere", "uniform_cube"]

    def set_parameters(
        self,
        *,
        algorithm: str,
        scenario: str,
        n: int,
        seed: int | None,
    ) -> None:
        self.parameters_calls.append((algorithm, scenario, n, seed))
        self.state.algorithm = algorithm
        self.state.scenario = scenario
        self.state.n = n
        self.state.seed = seed

    def generate_points(self) -> list[Point3D]:
        self.generate_calls += 1
        if self.generate_error is not None:
            raise self.generate_error
        self.state.last_used_seed = (
            self.random_seed_to_return if self.state.seed is None else self.state.seed
        )
        self.state.generation_seed = self.state.last_used_seed
        points = list(self.points_to_return)
        self.state.points = points
        self.state.result = None
        return points

    def apply_hull_result(
        self,
        result: HullResult3D,
        *,
        build_seconds: float | None = None,
    ) -> HullResult3D:
        self.apply_hull_result_calls += 1
        self.state.result = result
        self.state.points = list(result.points)
        self.state.last_build_seconds = build_seconds
        return result

    def load_json(self, path: Path) -> tuple[str, HullResult3D] | tuple[str, list[Point3D]]:
        self.load_calls.append(path)
        if self.load_error is not None:
            raise self.load_error
        self.state.last_used_seed = self.result_to_return.metadata.seed
        self.state.generation_seed = self.result_to_return.metadata.seed
        self.state.last_build_seconds = None
        self.state.result = self.result_to_return
        self.state.points = list(self.result_to_return.points)
        self.state.report = None
        return ("hull", self.result_to_return)

    def save_result(self, path: Path) -> None:
        self.save_calls.append(path)
        if self.save_error is not None:
            raise self.save_error

    def load_manual_points(self, points: list[Point3D]) -> None:
        self.manual_points_calls.append(list(points))
        self.state.points = list(points)
        self.state.result = None
        self.state.last_error = None
        self.state.last_used_seed = None
        self.state.generation_seed = None


class _FakeSceneWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.render_points_calls: list[tuple[list[Point3D], bool]] = []
        self.render_hull_calls: list[tuple[HullResult3D, bool, bool]] = []
        self.reset_camera_calls = 0
        self.set_axes_visible_calls: list[bool] = []
        self.set_coords_visible_calls: list[bool] = []

    def render_points(
        self,
        points: list[Point3D],
        *,
        preserve_camera: bool = False,
    ) -> None:
        self.render_points_calls.append((list(points), preserve_camera))

    def render_hull(
        self,
        result: HullResult3D,
        *,
        show_points: bool,
        preserve_camera: bool = False,
    ) -> None:
        self.render_hull_calls.append((result, show_points, preserve_camera))

    def reset_camera(self) -> None:
        self.reset_camera_calls += 1

    def set_axes_visible(self, show: bool) -> None:
        self.set_axes_visible_calls.append(show)

    def set_coords_visible(self, show: bool) -> None:
        self.set_coords_visible_calls.append(show)


def test_window_populates_comboboxes(qapp) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        assert [window.algorithm_box.itemText(i) for i in range(window.algorithm_box.count())] == [
            "bruteforce_degenerate",
            "chan",
        ]
        assert [window.scenario_box.itemText(i) for i in range(window.scenario_box.count())] == [
            "uniform_sphere",
            "uniform_cube",
        ]
        assert window.n_spin.value() == 20
        assert window.seed_spin.text() == ""
        assert window.seed_spin.isEnabled() is False
        assert window.random_seed_box.isChecked() is True
        assert window.show_input_points_box.isChecked() is True
        assert window.show_axes_box.isChecked() is True
        assert window.show_coords_box.isChecked() is False
        assert window.load_button.isEnabled() is True
        assert window.save_button.isEnabled() is True
    finally:
        window.close()


def test_generate_button_calls_controller_and_scene(qapp) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        window.random_seed_box.setChecked(False)
        window.n_spin.setValue(12)
        window.seed_spin.setText("7")
        window.generate_button.click()

        assert controller.parameters_calls == [("chan", "uniform_sphere", 12, 7)]
        assert controller.generate_calls == 1
        assert scene.render_points_calls == [(controller.points_to_return, False)]
        status = window.status_label.toPlainText()
        assert "Points generated." in status
        assert "scenario: uniform_sphere" in status
        assert "n: 12" in status
        assert "seed: 7" in status
    finally:
        window.close()


def test_random_seed_checkbox_disables_seed_input(qapp) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        window.random_seed_box.setChecked(False)
        assert window.seed_spin.isEnabled() is True

        window.random_seed_box.setChecked(True)
        assert window.seed_spin.isEnabled() is False
        assert window.seed_spin.text() == ""
    finally:
        window.close()


def test_random_seed_mode_passes_none_to_controller(qapp) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        window.random_seed_box.setChecked(True)
        window.generate_button.click()

        assert controller.parameters_calls == [("chan", "uniform_sphere", 20, None)]
        assert "seed: 424242" in window.status_label.toPlainText()
    finally:
        window.close()


def test_seed_input_starts_disabled_for_none_seed_state(qapp) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    controller.state.seed = None
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        assert window.random_seed_box.isChecked() is True
        assert window.seed_spin.isEnabled() is False
        assert window.seed_spin.text() == ""
    finally:
        window.close()


def test_build_button_launches_subprocess(qapp) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        window.build_button.click()

        assert controller.parameters_calls == [("chan", "uniform_sphere", 20, None)]
        assert not window.build_button.isEnabled()
        assert window.cancel_button.isEnabled()
        assert window.status_label.toPlainText() == "Building hull..."
    finally:
        if window._hull_process is not None:
            with contextlib.suppress(RuntimeError):
                window._hull_process.finished.disconnect()
            window._hull_process.kill()
            window._hull_process = None
        window.close()


def test_hull_process_finished_renders_result(qapp, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtCore import QProcess

    from verification.golden import dump_golden

    controller = _FakeController()
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        result = controller.result_to_return
        result_path = tmp_path / "result.json"
        dump_golden(result, result_path)

        controller.state.generation_seed = 424242
        window._hull_tmp_output = result_path
        window._hull_tmp_input = None
        window._hull_process = QProcess()
        window._hull_build_start = perf_counter() - 0.125
        window.build_button.setEnabled(False)
        window.cancel_button.setEnabled(True)
        window.show_input_points_box.setChecked(True)

        window._on_hull_process_finished(0, QProcess.ExitStatus.NormalExit)

        assert window.build_button.isEnabled()
        assert not window.cancel_button.isEnabled()
        assert controller.apply_hull_result_calls == 1
        assert len(scene.render_hull_calls) == 1
        rendered_result, show_points, _ = scene.render_hull_calls[0]
        assert show_points is True
        status = window.status_label.toPlainText()
        assert "Hull built." in status
        assert "algorithm: chan" in status
        assert "input points: 4" in status
        assert "seed: 424242" in status
    finally:
        window.close()


def test_controller_error_is_shown_in_status(qapp) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    controller.generate_error = ValueError("bad input")
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        window.generate_button.click()

        assert scene.render_points_calls == []
        assert window.status_label.toPlainText() == "Error: ValueError: bad input"
    finally:
        window.close()


def test_reset_camera_button_calls_scene(qapp) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        calls_before = scene.reset_camera_calls
        window.reset_camera_button.click()

        assert scene.reset_camera_calls == calls_before + 1
    finally:
        window.close()


def test_load_json_hull_renders_result(qapp, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)
    selected_path = Path("C:/tmp/example.json")

    monkeypatch.setattr(
        "gui.window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(selected_path), "JSON Files (*.json)"),
    )

    try:
        window.show_input_points_box.setChecked(True)
        window.load_button.click()

        assert controller.load_calls == [selected_path]
        assert len(scene.render_hull_calls) == 1
        rendered_result, show_points, _ = scene.render_hull_calls[0]
        assert show_points is True
        status = window.status_label.toPlainText()
        assert f"Loaded JSON: {selected_path}" in status
        assert "algorithm: chan" in status
    finally:
        window.close()


def test_save_json_calls_controller_and_shows_path(  # type: ignore[no-untyped-def]
    qapp,
    monkeypatch,
) -> None:
    controller = _FakeController()
    controller.state.result = controller.result_to_return
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)
    selected_path = Path("C:/tmp/saved.json")

    monkeypatch.setattr(
        "gui.window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(selected_path), "JSON Files (*.json)"),
    )

    try:
        window.save_button.click()

        assert controller.save_calls == [selected_path]
        assert window.status_label.toPlainText() == f"Saved JSON: {selected_path}"
    finally:
        window.close()


def test_save_without_result_shows_error(qapp, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    controller.save_error = ValueError("No hull result is available to save.")
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)
    selected_path = Path("C:/tmp/missing.json")

    monkeypatch.setattr(
        "gui.window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(selected_path), "JSON Files (*.json)"),
    )

    try:
        window.save_button.click()

        assert controller.save_calls == [selected_path]
        assert window.status_label.toPlainText() == (
            "Error: ValueError: No hull result is available to save."
        )
    finally:
        window.close()


def test_show_points_checkbox_refreshes_scene_when_result_exists(qapp) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    controller.state.result = controller.result_to_return
    controller.state.points = list(controller.result_to_return.points)
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        scene.render_hull_calls.clear()
        was_checked = window.show_input_points_box.isChecked()
        window.show_input_points_box.setChecked(not was_checked)

        assert len(scene.render_hull_calls) == 1
        _result, show_points, preserve_camera = scene.render_hull_calls[0]
        assert show_points is not was_checked
        assert preserve_camera is True
    finally:
        window.close()


def test_show_axes_toggle_calls_set_axes_visible(qapp) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        scene.set_axes_visible_calls.clear()
        window.show_axes_box.setChecked(False)

        assert False in scene.set_axes_visible_calls
    finally:
        window.close()


def test_load_text_button_parses_and_renders_points(qapp) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        window.points_text_edit.setPlainText("0 0 0\n1 0 0\n0 1 0\n0 0 1")
        window.load_text_button.click()

        assert len(controller.manual_points_calls) == 1
        assert len(controller.manual_points_calls[0]) == 4
        assert len(scene.render_points_calls) == 1
        assert "Loaded 4 points" in window.status_label.toPlainText()
    finally:
        window.close()


def test_load_text_button_too_few_points_shows_error(qapp) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        window.points_text_edit.setPlainText("0 0 0\n1 0 0\n0 1 0")
        window.load_text_button.click()

        assert controller.manual_points_calls == []
        assert "At least 4 points required" in window.status_label.toPlainText()
    finally:
        window.close()


def test_load_text_button_bad_line_shows_error(qapp) -> None:  # type: ignore[no-untyped-def]
    controller = _FakeController()
    scene = _FakeSceneWidget()
    window = MainWindow(controller=controller, scene_widget=scene)

    try:
        window.points_text_edit.setPlainText("0 0 0\n1 0\n0 1 0\n0 0 1")
        window.load_text_button.click()

        assert controller.manual_points_calls == []
        assert "Line 2" in window.status_label.toPlainText()
    finally:
        window.close()
