"""One Qt application lifetime for all offscreen editor checks."""

import gc
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "local_examples: needs the optional local example collection"
    )


def pytest_collection_modifyitems(config, items):
    if (config.rootpath / "examples" / "README.md").is_file():
        return
    skip = pytest.mark.skip(reason="Optional local example collection is absent")
    for item in items:
        if item.get_closest_marker("local_examples"):
            item.add_marker(skip)


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
        paths = list(examples.glob(f"*/{folder}/{folder}.json"))
        assert len(paths) == 1, (folder, paths)
        window.replace_graph(Graph.load(paths[0], window.registry))

    return load
