from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtWidgets import QWidget

import gui.scene_widget as scene_widget_module
from adapters.bruteforce_adapter import DegenerateBruteforce3D
from domain.entities import HullResult3D, Point3D
from gui.scene_widget import HullSceneWidget

_TETRA_POINTS = [
    Point3D(0.0, 0.0, 0.0),
    Point3D(1.0, 0.0, 0.0),
    Point3D(0.0, 1.0, 0.0),
    Point3D(0.0, 0.0, 1.0),
]


def _build_result() -> HullResult3D:
    return DegenerateBruteforce3D().compute(_TETRA_POINTS, seed=1)


class _FakeQtInteractor(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.renderer = SimpleNamespace(actors={})
        self.camera_position: object = ("initial-camera",)
        self.add_mesh_calls: list[tuple[object, dict[str, object]]] = []
        self.reset_camera_clipping_range_calls = 0
        self.view_isometric_calls = 0

    def add_mesh(self, mesh: object, **kwargs: object) -> None:
        actor_key = f"actor-{len(self.add_mesh_calls)}"
        self.renderer.actors[actor_key] = mesh
        self.add_mesh_calls.append((mesh, kwargs))

    def add_actor(self, actor: object, name: object = None, reset_camera: bool = False) -> None:
        pass

    def remove_actor(self, name: object) -> None:
        pass

    def clear(self) -> None:
        self.renderer.actors.clear()

    def view_isometric(self, render: bool = True) -> None:
        self.view_isometric_calls += 1

    def reset_camera(self) -> None:
        pass

    def reset_camera_clipping_range(self) -> None:
        self.reset_camera_clipping_range_calls += 1

    def render(self) -> None:
        pass


def test_render_hull_preserves_camera_when_toggling_normals(
    qapp,  # type: ignore[no-untyped-def]
    monkeypatch,
) -> None:
    del qapp
    show_hull_calls: list[bool] = []

    def fake_show_hull(
        result: HullResult3D,
        *,
        show_normals: bool = False,
        interactive: bool = True,
        plotter: object | None = None,
    ) -> object:
        del result, interactive
        assert plotter is not None
        show_hull_calls.append(show_normals)
        plotter.add_mesh("hull")
        return plotter

    monkeypatch.setattr(scene_widget_module, "QtInteractor", _FakeQtInteractor)
    monkeypatch.setattr(scene_widget_module, "show_hull", fake_show_hull)
    monkeypatch.setattr(scene_widget_module, "hull_normals_to_polydata", lambda result: "normals")

    widget = HullSceneWidget()

    try:
        widget.render_hull(_build_result(), show_points=False, show_normals=False)
        plotter = widget._plotter
        plotter.camera_position = ("user-camera",)

        widget.render_hull(
            _build_result(),
            show_points=False,
            show_normals=True,
            preserve_camera=True,
        )

        assert show_hull_calls == [False, False]
        assert plotter.camera_position == ("user-camera",)
        assert plotter.add_mesh_calls[-1] == (
            "normals",
            {"color": "tomato", "line_width": 2.0},
        )
    finally:
        widget.close()
        widget.deleteLater()


def test_render_hull_without_preserve_camera_calls_view_isometric(
    qapp,  # type: ignore[no-untyped-def]
    monkeypatch,
) -> None:
    del qapp

    def fake_show_hull(
        result: HullResult3D,
        *,
        show_normals: bool = False,
        interactive: bool = True,
        plotter: object | None = None,
    ) -> object:
        del result, interactive
        assert plotter is not None
        plotter.add_mesh("hull")
        return plotter

    monkeypatch.setattr(scene_widget_module, "QtInteractor", _FakeQtInteractor)
    monkeypatch.setattr(scene_widget_module, "show_hull", fake_show_hull)
    monkeypatch.setattr(scene_widget_module, "hull_normals_to_polydata", lambda result: "normals")

    widget = HullSceneWidget()

    try:
        widget.render_hull(_build_result(), show_points=False, show_normals=False)
        plotter = widget._plotter

        assert plotter.view_isometric_calls == 1
        assert plotter.reset_camera_clipping_range_calls == 2
    finally:
        widget.close()
        widget.deleteLater()


def test_render_hull_with_points_adds_normals_after_camera_fit(
    qapp,  # type: ignore[no-untyped-def]
    monkeypatch,
) -> None:
    del qapp
    show_hull_with_points_calls: list[bool] = []

    def fake_show_hull_with_points(
        result: HullResult3D,
        *,
        show_normals: bool = False,
        point_size: float = 8.0,
        interactive: bool = True,
        plotter: object | None = None,
    ) -> object:
        del result, point_size, interactive
        assert plotter is not None
        show_hull_with_points_calls.append(show_normals)
        plotter.add_mesh("hull-with-points")
        return plotter

    monkeypatch.setattr(scene_widget_module, "QtInteractor", _FakeQtInteractor)
    monkeypatch.setattr(scene_widget_module, "show_hull_with_points", fake_show_hull_with_points)
    monkeypatch.setattr(scene_widget_module, "hull_normals_to_polydata", lambda result: "normals")

    widget = HullSceneWidget()

    try:
        widget.render_hull(_build_result(), show_points=True, show_normals=True)
        plotter = widget._plotter

        assert show_hull_with_points_calls == [False]
        assert plotter.view_isometric_calls == 1
        assert plotter.add_mesh_calls[-1] == (
            "normals",
            {"color": "tomato", "line_width": 2.0},
        )
    finally:
        widget.close()
        widget.deleteLater()
