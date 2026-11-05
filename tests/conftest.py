"""One Qt application lifetime for all offscreen editor checks."""

import gc
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def app():
    application = QApplication.instance() or QApplication([])
    yield application
    # Dispose test windows and queued QObject deletions before QApplication teardown.
    for widget in application.topLevelWidgets():
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()
    gc.collect()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()


@pytest.fixture
def load_example():
    from opencv_dataflow.core import Graph

    examples = Path(__file__).resolve().parents[1] / "examples"

    def load(window, folder):
        path = examples / folder / f"{folder}.json"
        window.replace_graph(Graph.load(path, window.registry))

    return load
