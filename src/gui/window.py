"""Main window for the chanhull3d GUI."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from domain.entities import HullResult3D, Point3D
from gui.controller import GuiController
from verification.golden import load_golden


class _SceneWidgetProtocol(Protocol):
    """Required scene widget API for MainWindow integration."""

    def render_points(
        self,
        points: list[Point3D],
        *,
        preserve_camera: bool = False,
    ) -> None:
        """Render a point cloud."""

    def render_hull(
        self,
        result: HullResult3D,
        *,
        show_points: bool,
        preserve_camera: bool = False,
    ) -> None:
        """Render a hull result."""

    def reset_camera(self) -> None:
        """Reset the current viewport camera."""

    def set_axes_visible(self, show: bool) -> None:
        """Show or hide coordinate axes."""

    def set_coords_visible(self, show: bool) -> None:
        """Show or hide per-point coordinate labels."""


class MainWindow(QMainWindow):
    """Top-level GUI window with controls and 3-D viewport."""

    def __init__(
        self,
        *,
        controller: GuiController | None = None,
        scene_widget: QWidget | None = None,
    ) -> None:
        """Initialise the main window and wire up controls."""
        super().__init__()

        self.controller = controller if controller is not None else GuiController()
        # None until lazy init fires (or immediately if injected by tests).
        self.scene_widget: QWidget | None = scene_widget
        self._scene_container_layout: QVBoxLayout | None = None
        self._scene_placeholder: QLabel | None = None

        self.setWindowTitle("convex_hull")

        self.algorithm_box = QComboBox(self)
        self.algorithm_box.addItems(self.controller.available_algorithms())

        self.scenario_box = QComboBox(self)
        self.scenario_box.addItems(self.controller.available_scenarios())

        self.n_spin = QSpinBox(self)
        self.n_spin.setMinimum(4)
        self.n_spin.setMaximum(1_000_000)

        self.seed_spin = QLineEdit(self)
        self.seed_spin.setPlaceholderText("seed")
        self.random_seed_box = QCheckBox("random", self)

        self.generate_button = QPushButton("Generate", self)
        self.build_button = QPushButton("Build Hull", self)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setEnabled(False)

        self._hull_process: QProcess | None = None
        self._hull_tmp_input: Path | None = None
        self._hull_tmp_output: Path | None = None
        self._hull_build_start: float | None = None
        self.load_button = QPushButton("Load JSON", self)
        self.save_button = QPushButton("Save JSON", self)
        self.reset_camera_button = QPushButton("Reset Camera", self)

        self.show_input_points_box = QCheckBox("show points", self)
        self.show_axes_box = QCheckBox("show axes", self)
        self.show_coords_box = QCheckBox("show coordinates", self)
        self.status_label = QPlainTextEdit(self)

        self.points_text_edit = QPlainTextEdit(self)
        self.points_text_edit.setPlaceholderText("x y z\nx y z\n...")
        self.load_text_button = QPushButton("Load Points", self)

        self._apply_state_defaults()
        self._build_layout()
        self._connect_signals()

        if self.scene_widget is not None:
            # Injected widget (e.g. in tests): configure scene immediately.
            self._on_scene_ready()
        else:
            # Runtime: defer heavy PyVista/VTK import until after the window appears.
            QTimer.singleShot(0, self._init_scene_widget)

        self._set_status("Ready.")

    def _apply_state_defaults(self) -> None:
        """Populate widgets from the current controller state."""
        state = self.controller.state

        algorithm_index = self.algorithm_box.findText(state.algorithm)
        if algorithm_index >= 0:
            self.algorithm_box.setCurrentIndex(algorithm_index)

        scenario_index = self.scenario_box.findText(state.scenario)
        if scenario_index >= 0:
            self.scenario_box.setCurrentIndex(scenario_index)

        self.n_spin.setValue(state.n)
        self.seed_spin.setText("" if state.seed is None else str(state.seed))
        self.random_seed_box.setChecked(state.seed is None)
        self._update_seed_input_state()

        self.show_input_points_box.setChecked(True)
        self.show_axes_box.setChecked(True)
        self.show_coords_box.setChecked(False)

        self.status_label.setReadOnly(True)
        self.status_label.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)

    def _build_layout(self) -> None:
        """Construct the two-column main window layout."""
        central_widget = QWidget(self)
        root_layout = QHBoxLayout(central_widget)

        controls_widget = QWidget(central_widget)
        controls_widget.setMinimumWidth(280)
        controls_layout = QVBoxLayout(controls_widget)

        # Algorithm selector (shared across both input modes)
        form_top = QFormLayout()
        form_top.addRow("Algorithm", self.algorithm_box)
        controls_layout.addLayout(form_top)

        # ---------- Tab widget ----------
        tab_widget = QTabWidget(controls_widget)

        # Tab: Generate
        gen_tab = QWidget()
        gen_layout = QVBoxLayout(gen_tab)

        seed_widget = QWidget(gen_tab)
        seed_layout = QHBoxLayout(seed_widget)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.addWidget(self.seed_spin, stretch=1)
        seed_layout.addWidget(self.random_seed_box, stretch=0)

        form_gen = QFormLayout()
        form_gen.addRow("Scenario", self.scenario_box)
        form_gen.addRow("n", self.n_spin)
        form_gen.addRow("seed", seed_widget)
        gen_layout.addLayout(form_gen)
        gen_layout.addWidget(self.generate_button)
        gen_layout.addStretch()

        # Tab: Input
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        input_layout.addWidget(self.points_text_edit, stretch=1)
        input_layout.addWidget(self.load_text_button)

        tab_widget.addTab(input_tab, "Input")
        tab_widget.addTab(gen_tab, "Generate")
        controls_layout.addWidget(tab_widget)
        # --------------------------------

        controls_layout.addWidget(self.build_button)
        controls_layout.addWidget(self.cancel_button)
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.reset_camera_button)
        controls_layout.addWidget(self.show_input_points_box)
        controls_layout.addWidget(self.show_axes_box)
        controls_layout.addWidget(self.show_coords_box)
        controls_layout.addWidget(self.status_label, stretch=1)

        # 3-D viewport area — holds either a placeholder or the real scene widget.
        scene_container = QWidget(central_widget)
        self._scene_container_layout = QVBoxLayout(scene_container)
        self._scene_container_layout.setContentsMargins(0, 0, 0, 0)

        if self.scene_widget is not None:
            self._scene_container_layout.addWidget(self.scene_widget)
        else:
            self._scene_placeholder = QLabel("Loading 3D viewport…")
            self._scene_container_layout.addWidget(self._scene_placeholder)

        root_layout.addWidget(controls_widget, stretch=0)
        root_layout.addWidget(scene_container, stretch=1)

        self.setCentralWidget(central_widget)
        self.resize(1200, 720)

    def _connect_signals(self) -> None:
        """Connect UI events to handlers."""
        self.generate_button.clicked.connect(self._handle_generate)
        self.build_button.clicked.connect(self._handle_build_hull)
        self.cancel_button.clicked.connect(self._handle_cancel_build)
        self.load_button.clicked.connect(self._handle_load_json)
        self.save_button.clicked.connect(self._handle_save_json)
        self.reset_camera_button.clicked.connect(self._handle_reset_camera)
        self.random_seed_box.toggled.connect(self._handle_random_seed_toggled)
        self.show_input_points_box.toggled.connect(self._handle_display_options_changed)
        self.show_axes_box.toggled.connect(self._handle_show_axes_toggled)
        self.show_coords_box.toggled.connect(self._handle_show_coords_toggled)
        self.load_text_button.clicked.connect(self._handle_load_from_text)

    def _init_scene_widget(self) -> None:
        """Import and construct the heavy 3-D scene widget (deferred until after show).

        Exceptions are caught and surfaced in the UI — PySide6 silently swallows
        exceptions raised inside timer/signal callbacks.
        """
        try:
            self._load_scene_widget()
        except Exception as exc:
            import traceback

            traceback.print_exc()
            msg = f"3D viewport failed to load: {type(exc).__name__}: {exc}"
            if self._scene_placeholder is not None:
                self._scene_placeholder.setText(msg)
            self._set_status(msg)

    def _load_scene_widget(self) -> None:
        try:
            from app.startup_profile import mark
        except ImportError:

            def mark(name: str) -> None:
                pass

        mark("before_scene_import")
        from gui.scene_widget import HullSceneWidget

        mark("after_scene_import")
        widget = HullSceneWidget()
        mark("after_scene_construct")

        self.scene_widget = widget

        assert self._scene_container_layout is not None
        if self._scene_placeholder is not None:
            self._scene_container_layout.removeWidget(self._scene_placeholder)
            self._scene_placeholder.deleteLater()
            self._scene_placeholder = None
        self._scene_container_layout.addWidget(widget)

        self._on_scene_ready()
        mark("scene_ready")

    def _on_scene_ready(self) -> None:
        """Apply initial viewport state once the scene widget is available."""
        self._scene().set_axes_visible(self.show_axes_box.isChecked())
        self._scene().reset_camera()

    def _scene(self) -> _SceneWidgetProtocol:
        """Return the scene widget under the required protocol."""
        assert self.scene_widget is not None, "scene_widget accessed before initialization"
        return cast(_SceneWidgetProtocol, self.scene_widget)

    def _current_parameters(self) -> tuple[str, str, int, int | None]:
        """Read current parameter values from the UI."""
        seed: int | None = None
        if not self.random_seed_box.isChecked():
            text = self.seed_spin.text().strip()
            try:
                seed = int(text) if text else 0
            except ValueError:
                seed = 0
        return (
            self.algorithm_box.currentText(),
            self.scenario_box.currentText(),
            self.n_spin.value(),
            seed,
        )

    def _set_status(self, message: str) -> None:
        """Update the status/metadata panel."""
        self.status_label.setPlainText(message)

    def _format_result_metadata(
        self,
        result: HullResult3D,
        *,
        build_seconds: float | None = None,
        display_seed: int | None = None,
    ) -> str:
        """Format hull metadata for display in the status panel."""
        metadata = result.metadata
        seed_value = display_seed if display_seed is not None else metadata.seed
        lines = [
            "Hull built.",
            f"algorithm: {metadata.algorithm}",
            f"input points: {metadata.n_points_input}",
            f"hull vertices: {metadata.n_vertices_hull}",
            f"faces: {metadata.n_faces}",
            f"seed: {seed_value}",
        ]
        if build_seconds is not None:
            lines.append(f"time: {build_seconds:.3f} s")
        return "\n".join(lines)

    def _update_seed_input_state(self) -> None:
        """Enable or disable the manual seed input based on the random-seed checkbox."""
        self.seed_spin.setEnabled(not self.random_seed_box.isChecked())

    def _sync_controller_parameters_from_ui(self) -> tuple[str, str, int, int | None]:
        """Validate and store current UI parameters in the controller."""
        algorithm, scenario, n_points, seed = self._current_parameters()
        self.controller.set_parameters(
            algorithm=algorithm,
            scenario=scenario,
            n=n_points,
            seed=seed,
        )
        return algorithm, scenario, n_points, seed

    def _render_current_state(self, *, preserve_camera: bool = False) -> None:
        """Render the current controller state without re-running algorithms."""
        state = self.controller.state
        if state.result is not None:
            self._scene().render_hull(
                state.result,
                show_points=self.show_input_points_box.isChecked(),
                preserve_camera=preserve_camera,
            )
            return

        if state.points is not None:
            self._scene().render_points(state.points, preserve_camera=preserve_camera)

    def _handle_generate(self) -> None:
        """Generate points and render them in the viewport."""
        try:
            _, scenario, n_points, seed = self._sync_controller_parameters_from_ui()
            points = self.controller.generate_points()
            self._scene().render_points(points)
        except Exception as exc:
            self._set_status(f"Error: {type(exc).__name__}: {exc}")
            return

        self._set_status(
            "\n".join(
                [
                    "Points generated.",
                    f"scenario: {scenario}",
                    f"n: {n_points}",
                    f"seed: {self.controller.state.generation_seed}",
                ]
            )
        )

    def _handle_build_hull(self) -> None:
        """Start hull computation in a subprocess."""
        if self._hull_process is not None:
            return

        self._sync_controller_parameters_from_ui()

        if self.controller.state.points is None:
            try:
                points = self.controller.generate_points()
                self._scene().render_points(points)
            except Exception as exc:
                self._set_status(f"Error: {type(exc).__name__}: {exc}")
                return

        assert self.controller.state.points is not None
        points = self.controller.state.points

        fd, tmp_in_str = tempfile.mkstemp(suffix=".json")
        try:
            import os

            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"points": [[p.x, p.y, p.z] for p in points]}, f)
        except Exception as exc:
            self._set_status(f"Preparation error: {type(exc).__name__}: {exc}")
            return

        self._hull_tmp_input = Path(tmp_in_str)
        self._hull_tmp_output = Path(tmp_in_str + ".out.json")

        seed = self.controller.state.generation_seed
        args = [
            "-X",
            "utf8",
            "-m",
            "app.hull_worker",
            "--algorithm",
            self.controller.state.algorithm,
            "--input",
            str(self._hull_tmp_input),
            "--output",
            str(self._hull_tmp_output),
        ]
        if seed is not None:
            args += ["--seed", str(seed)]

        process = QProcess(self)
        process.finished.connect(self._on_hull_process_finished)
        process.readyReadStandardOutput.connect(self._forward_hull_stdout)
        self._hull_process = process
        self._hull_build_start = perf_counter()

        process.start(sys.executable, args)

        self.build_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._set_status("Building hull...")

    def _handle_cancel_build(self) -> None:
        """Kill the running hull subprocess and restore UI."""
        if self._hull_process is not None:
            with contextlib.suppress(RuntimeError):
                self._hull_process.finished.disconnect(self._on_hull_process_finished)
            self._hull_process.kill()
            self._hull_process = None
        self._hull_build_start = None
        self._cleanup_hull_tmp()
        self.build_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._set_status("Build cancelled.")

    def _forward_hull_stdout(self) -> None:
        if self._hull_process is None:
            return
        data: bytes = self._hull_process.readAllStandardOutput().data()  # type: ignore[assignment]
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    def _on_hull_process_finished(self, exit_code: int, _exit_status: object) -> None:
        """Handle hull subprocess completion."""
        process = self._hull_process
        self._hull_process = None
        build_seconds = (
            (perf_counter() - self._hull_build_start) if self._hull_build_start else None
        )
        self._hull_build_start = None

        self.build_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        if exit_code != 0:
            self._cleanup_hull_tmp()
            stderr_text = ""
            if process is not None:
                stderr_bytes: bytes = process.readAllStandardError().data()  # type: ignore[assignment]
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            msg = f"Build error (exit code {exit_code})."
            if stderr_text:
                msg += f"\n\n{stderr_text}"
            self._set_status(msg)
            return

        try:
            assert self._hull_tmp_output is not None
            result = load_golden(self._hull_tmp_output)
        except Exception as exc:
            self._cleanup_hull_tmp()
            self._set_status(f"Error reading result: {type(exc).__name__}: {exc}")
            return

        self._cleanup_hull_tmp()
        self.controller.apply_hull_result(result, build_seconds=build_seconds)
        self._render_current_state()
        self._set_status(
            self._format_result_metadata(
                result,
                build_seconds=build_seconds,
                display_seed=self.controller.state.generation_seed,
            )
        )

    def _cleanup_hull_tmp(self) -> None:
        """Delete temporary files used for subprocess communication."""
        for path in (self._hull_tmp_input, self._hull_tmp_output):
            if path is not None:
                path.unlink(missing_ok=True)
        self._hull_tmp_input = None
        self._hull_tmp_output = None

    def _handle_load_json(self) -> None:
        """Load a hull result or a points-only JSON file."""
        path_text, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load JSON",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path_text:
            return

        try:
            _kind, payload = self.controller.load_json(Path(path_text))
        except Exception as exc:
            self._set_status(f"Error: {type(exc).__name__}: {exc}")
            return

        if isinstance(payload, HullResult3D):
            self._render_current_state()
            self._set_status(f"Loaded JSON: {path_text}\n{self._format_result_metadata(payload)}")
        else:
            self._scene().render_points(payload)
            self._set_status(f"Loaded {len(payload)} points from: {path_text}")

    def _handle_save_json(self) -> None:
        """Save the current hull result to JSON."""
        path_text, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save JSON",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path_text:
            return

        try:
            self.controller.save_result(Path(path_text))
        except Exception as exc:
            self._set_status(f"Error: {type(exc).__name__}: {exc}")
            return

        self._set_status(f"Saved JSON: {path_text}")

    def _handle_display_options_changed(self) -> None:
        """Refresh the viewport after checkbox changes without recomputation."""
        try:
            self._render_current_state(preserve_camera=True)
        except Exception as exc:
            self._set_status(f"Error: {type(exc).__name__}: {exc}")

    def _handle_random_seed_toggled(self) -> None:
        """Update the seed input state after toggling random-seed mode."""
        self.seed_spin.clear()
        self._update_seed_input_state()

    def _handle_reset_camera(self) -> None:
        """Reset the viewport camera."""
        try:
            self._scene().reset_camera()
        except Exception as exc:
            self._set_status(f"Error: {type(exc).__name__}: {exc}")

    def _handle_show_axes_toggled(self, checked: bool) -> None:
        """Toggle coordinate axes visibility in the viewport."""
        try:
            self._scene().set_axes_visible(checked)
        except Exception as exc:
            self._set_status(f"Error: {type(exc).__name__}: {exc}")

    def _handle_show_coords_toggled(self, checked: bool) -> None:
        """Toggle per-point coordinate label visibility in the viewport."""
        try:
            self._scene().set_coords_visible(checked)
        except Exception as exc:
            self._set_status(f"Error: {type(exc).__name__}: {exc}")

    def _handle_load_from_text(self) -> None:
        """Parse the text input area and load points into the viewport."""
        text = self.points_text_edit.toPlainText()
        points: list[Point3D] = []
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                self._set_status(f"Line {lineno}: expected 'x y z', got: '{line}'")
                return
            try:
                points.append(Point3D(float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                self._set_status(f"Line {lineno}: non-numeric coordinates: '{line}'")
                return
        if len(points) < 4:
            self._set_status("At least 4 points required.")
            return
        self.controller.load_manual_points(points)
        self._scene().render_points(points)
        self._set_status(f"Loaded {len(points)} points from text input.")
