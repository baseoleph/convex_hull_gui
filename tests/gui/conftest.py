from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import pyvista as pv  # type: ignore[import-untyped]
from PySide6.QtWidgets import QApplication

pv.OFF_SCREEN = True


@pytest.fixture()
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
