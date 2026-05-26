"""Entry point for the convex-hull GUI application."""

from __future__ import annotations

import sys


def main() -> None:
    """Launch the demo GUI application."""
    from app.startup_profile import mark

    mark("gui_main_start")

    from PySide6.QtWidgets import QApplication

    mark("after_pyside6_import")

    from gui.window import MainWindow

    mark("after_window_import")

    existing_app = QApplication.instance()
    app = existing_app if existing_app is not None else QApplication(sys.argv)
    mark("app_created")

    window = MainWindow()
    mark("window_created")

    window.show()
    mark("window_shown")

    app.exec()
