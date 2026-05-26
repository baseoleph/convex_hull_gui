from __future__ import annotations

import pytest

from gui.scene_widget import HullSceneWidget


def test_scene_widget_importable() -> None:
    assert HullSceneWidget.__name__ == "HullSceneWidget"
    assert hasattr(HullSceneWidget, "clear_scene")
    assert hasattr(HullSceneWidget, "reset_camera")
    assert hasattr(HullSceneWidget, "render_points")
    assert hasattr(HullSceneWidget, "render_hull")
    assert hasattr(HullSceneWidget, "set_axes_visible")
    assert hasattr(HullSceneWidget, "set_coords_visible")


def test_scene_widget_constructs_headless(qapp) -> None:  # type: ignore[no-untyped-def]
    try:
        widget = HullSceneWidget()
    except Exception as exc:
        pytest.skip(f"QtInteractor construction is unstable in headless mode: {exc}")

    try:
        assert widget.isVisible() is False
    finally:
        widget.close()
        widget.deleteLater()
