"""Qt widget that embeds the PyVista viewport for the demo GUI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor  # type: ignore[import-untyped]

from domain.entities import HullResult3D, Point3D
from rendering import show_hull, show_hull_with_points, show_points
from rendering.scene import hull_normals_to_polydata

_NORMAL_COLOR: str = "tomato"
_NORMAL_LINE_WIDTH: float = 2.0


class HullSceneWidget(QWidget):
    """Widget wrapper around a PyVistaQt interactor."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the scene widget and embed the PyVista interactor."""
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plotter = QtInteractor(parent=self)
        layout.addWidget(self._plotter)

        self._show_axes: bool = True
        self._axes_actor: Any = None
        self._show_coords: bool = False
        self._coords_actor: Any = None
        self._last_points: list[Point3D] = []

    def clear_scene(self) -> None:
        """Clear all actors from the current scene."""
        self._plotter.clear()
        self._axes_actor = None
        self._coords_actor = None

    def reset_camera(self) -> None:
        """Reset camera to isometric view, then realign axes."""
        self._plotter.view_isometric(render=False)
        self._plotter.reset_camera_clipping_range()
        self._refresh_axes()
        self._plotter.render()

    def set_axes_visible(self, show: bool) -> None:
        """Show or hide coordinate axes at the world origin."""
        self._show_axes = show
        self._refresh_axes()
        self._plotter.render()

    def set_coords_visible(self, show: bool) -> None:
        """Show or hide per-point coordinate labels."""
        self._show_coords = show
        self._refresh_coord_labels()
        self._plotter.render()

    _AXES_NAME = "_hull_axes_origin"

    def _refresh_axes(self) -> None:
        """Remove and re-add axes actor scaled to current scene bounds."""
        self._plotter.remove_actor(self._AXES_NAME)
        self._axes_actor = None
        if not self._show_axes:
            return
        marker = pv.create_axes_marker(
            cone_radius=0.1,
            tip_length=0.08,
            shaft_length=1.0,
            label_size=(0.01, 0.01),
        )
        self._plotter.add_actor(marker, name=self._AXES_NAME, reset_camera=False)
        self._axes_actor = marker

    def _refresh_coord_labels(self) -> None:
        """Redraw coordinate labels based on current visibility and point set."""
        if self._coords_actor is not None:
            self._plotter.remove_actor(self._coords_actor)
            self._coords_actor = None
        if self._show_coords and self._last_points:
            pts_arr = np.array([[p.x, p.y, p.z] for p in self._last_points], dtype=np.float64)
            cloud = pv.PolyData(pts_arr)
            labels = [f"({p.x:.2f}, {p.y:.2f}, {p.z:.2f})" for p in self._last_points]
            cloud["labels"] = labels  # type: ignore[type-var]
            self._coords_actor = self._plotter.add_point_labels(
                cloud,
                "labels",
                font_size=10,
                always_visible=True,
                shape=None,
            )

    def _camera_position(self) -> Any:
        """Return the current plotter camera position object."""
        return self._plotter.camera_position

    def _restore_or_reset_camera(
        self,
        *,
        preserve_camera: bool,
        saved_camera_position: Any,
    ) -> None:
        """Restore the prior camera when requested, otherwise fit the scene."""
        if preserve_camera:
            self._plotter.camera_position = saved_camera_position
            return
        self.reset_camera()

    def _add_normals_overlay(self, result: HullResult3D) -> None:
        """Draw face normals without affecting the current camera scale."""
        normals_pd = hull_normals_to_polydata(result)
        self._plotter.add_mesh(
            normals_pd,
            color=_NORMAL_COLOR,
            line_width=_NORMAL_LINE_WIDTH,
        )

    def render_points(
        self,
        points: Sequence[Point3D],
        *,
        preserve_camera: bool = False,
    ) -> None:
        """Render a point cloud in the embedded viewport."""
        saved_camera_position = self._camera_position()
        self.clear_scene()
        self._last_points = list(points)
        show_points(points, interactive=False, plotter=self._plotter)
        self._refresh_axes()
        self._refresh_coord_labels()
        self._restore_or_reset_camera(
            preserve_camera=preserve_camera,
            saved_camera_position=saved_camera_position,
        )
        self._plotter.reset_camera_clipping_range()

    def render_hull(
        self,
        result: HullResult3D,
        *,
        show_points: bool,
        show_normals: bool = False,
        preserve_camera: bool = False,
    ) -> None:
        """Render a hull result in the embedded viewport."""
        saved_camera_position = self._camera_position()
        self.clear_scene()
        self._last_points = list(result.points)
        if show_points:
            show_hull_with_points(
                result,
                show_normals=False,
                interactive=False,
                plotter=self._plotter,
            )
        else:
            show_hull(
                result,
                show_normals=False,
                interactive=False,
                plotter=self._plotter,
            )
        self._refresh_axes()
        self._refresh_coord_labels()
        self._restore_or_reset_camera(
            preserve_camera=preserve_camera,
            saved_camera_position=saved_camera_position,
        )
        if show_normals:
            self._add_normals_overlay(result)
        self._plotter.reset_camera_clipping_range()
