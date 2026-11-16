"""Offscreen draft lifecycle; no desktop input or screen capture."""

import copy
import json
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("opencv_dataflow")
from PySide6.QtWidgets import QApplication

from opencv_dataflow.simple_gdrnpp.qt_setup import GDRNPPSetupDialog

PILOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/gdrnpp"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class Client:
    def __init__(self):
        self.hold = Event()
        self.hold.set()
        self.started = Event()
        self.closed = Event()

    def call(self, op, data, arrays=None, *, cancelled=lambda: False):
        self.started.set()
        while not self.hold.wait(0.01):
            if cancelled():
                raise RuntimeError("cancelled")
        summary = json.loads((PILOT / "assets/setup_summary.json").read_text())
        summary["settings"]["minimum_coverage"] = 0.733333333333
        image = np.full((540, 720, 3), [10, 20, 30], np.uint8)
        if op == "inspect":
            return dict(setup=summary), dict(
                rgb=image,
                depth=np.ones((540, 720), np.float32),
                camera=np.eye(3),
                mask=np.ones((540, 720), bool),
            )
        if op == "stage":
            return dict(recipe_path=data["relative_asset"], sha256="f" * 64), {}
        if op == "preview":
            return dict(
                token="preview",
                candidate_index=0,
                report=dict(candidates=[], accepted_candidates=0, target_frame="camera"),
            ), dict(
                overlay=image,
                points=np.zeros((1, 3), np.float32),
                colors=np.zeros((1, 3), np.uint8),
            )
        raise ValueError(op)

    def close(self):
        self.closed.set()


def settle(app, dialog):
    end = time.monotonic() + 5
    while dialog.pending and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)
    assert dialog.pending is None, dialog.status.text()


def opened(app, tmp_path):
    applied = dict(recipe_path="", sha256="", models_dir=str(tmp_path))
    draft = copy.deepcopy(applied)
    client = Client()
    dialog = GDRNPPSetupDialog(None, draft, SimpleNamespace(project_dir=tmp_path), client=client)
    dialog.load_base(dict(kind="recipe", path="fixture.gdrnpp"))
    settle(app, dialog)
    return dialog, client, applied, draft


def test_reopens_reference_boxes_units_and_preserves_unedited_precision(app, tmp_path):
    dialog, client, applied, draft = opened(app, tmp_path)
    assert dialog.viewer.image_item.pixmap().width() == 720 and dialog.boxes.rowCount() == 1
    assert dialog.viewer.image_item.pixmap().toImage().pixelColor(0, 0).getRgb()[:3] == (10, 20, 30)
    assert dialog.fields["distance_tolerance_m"].value() == 5
    assert dialog._draft()["settings"]["minimum_coverage"] == 0.733333333333
    dialog.name.setText("changed name")
    assert dialog._draft()["settings"]["minimum_coverage"] == 0.733333333333
    dialog.fields["minimum_coverage"].setValue(75)
    assert dialog._draft()["settings"]["minimum_coverage"] == 0.75
    dialog._add_box((10, 10, 30, 35))
    assert dialog.boxes.rowCount() == 2
    dialog.viewer.zoom(1.25)
    dialog.reject()
    assert client.closed.wait(2) and draft == applied


def test_stage_updates_only_the_supplied_inspector_draft(app, tmp_path):
    dialog, client, applied, draft = opened(app, tmp_path)
    dialog.name.setText("draft revision")
    dialog._stage()
    settle(app, dialog)
    assert dialog.result() == dialog.DialogCode.Accepted
    assert draft["sha256"] == "f" * 64 and draft["recipe_path"].startswith("assets/gdrnpp/")
    assert applied["recipe_path"] == "" and client.closed.wait(2)


def test_depth_alignment_method_is_an_explicit_draft_choice(app, tmp_path):
    dialog, client, applied, draft = opened(app, tmp_path)
    assert dialog._draft()["settings"]["depth_refinement"] == "median_translation"
    dialog.depth_refine.setChecked(True)
    dialog.refine_method.setCurrentIndex(dialog.refine_method.findData("projective_icp_v1"))
    assert dialog._draft()["settings"]["depth_refinement"] == "projective_icp_v1"
    assert dialog._draft()["settings"]["distance_tolerance_m"] == 0.005
    dialog.reject()
    assert client.closed.wait(2) and draft == applied


def test_edit_cancels_pending_preview_and_discards_older_result(app, tmp_path):
    dialog, client, applied, draft = opened(app, tmp_path)
    client.started.clear()
    client.hold.clear()
    dialog._preview_reference()
    assert client.started.wait(2)
    dialog.name.setText("edited while computing")
    client.hold.set()
    settle(app, dialog)
    assert dialog.token is None and dialog.preview_report is None
    assert not dialog.result_cloud_button.isEnabled() and draft == applied
    dialog.reject()
    assert client.closed.wait(2)
