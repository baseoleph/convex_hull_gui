from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication


def _import_fresh(module_name: str) -> object:
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_import_gui_main_does_not_create_qapplication() -> None:
    assert QApplication.instance() is None

    module = _import_fresh("gui.main")

    assert hasattr(module, "main")
    assert QApplication.instance() is None


def test_import_main_window_does_not_create_qapplication() -> None:
    assert QApplication.instance() is None

    module = _import_fresh("gui.window")

    assert module.MainWindow.__name__ == "MainWindow"
    assert QApplication.instance() is None
