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
        summary = json.loads((PILOT / "setup/assets/setup_summary.json").read_text())
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
    assert dialog.fields["maximum_overlap"].value() == 100
    assert dialog._draft()["settings"]["maximum_overlap"] == 1
    assert dialog._draft()["settings"]["minimum_coverage"] == 0.733333333333
    dialog.name.setText("changed name")
    assert dialog._draft()["settings"]["minimum_coverage"] == 0.733333333333
    dialog.fields["minimum_coverage"].setValue(75)
    assert dialog._draft()["settings"]["minimum_coverage"] == 0.75
    dialog.fields["maximum_overlap"].setValue(60)
    assert dialog._draft()["settings"]["maximum_overlap"] == 0.6
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


@pytest.mark.parametrize("suppressed", [False, True])
def test_preview_selects_best_later_candidate_and_switches_cached_review(app, tmp_path, suppressed):
    from opencv_dataflow.simple_gdrnpp.selection import describe_review

    dialog, client, applied, draft = opened(app, tmp_path)
    report = dict(
        candidates=[
            dict(id="rejected_first", fit=dict(status="rejected", coverage=0.99)),
            dict(id="accepted_second", fit=dict(status="accepted", coverage=0.8)),
        ],
        accepted_candidates=1,
        target_frame="camera",
    )
    if suppressed:
        report["candidates"][0].update(
            depth_fit=dict(status="accepted", coverage=0.79),
            selection=dict(status="overlap_suppressed", retained_by="accepted_second", overlap=1.0),
        )
    arrays = dict(
        overlay=np.zeros((54, 72, 3), np.uint8),
        points=np.zeros((1, 3), np.float32),
        colors=np.zeros((1, 3), np.uint8),
    )
    calls = []

    def response(number):
        review, _ = describe_review(report, number)
        return dict(
            token="cached", report=report, review=review, candidate_index=review["candidate_index"]
        )

    def cached(op, data, *args, **kwargs):
        calls.append(op)
        assert op == "render" and data["token"] == "cached"
        return response(data["index"] + 1), arrays

    client.call = cached
    dialog._previewed(response(0), arrays)
    assert dialog.candidates.currentIndex() == 1
    assert "accepted_second" in dialog.result_info.text()
    dialog._candidate(0)
    settle(app, dialog)
    assert calls == ["render"] and dialog.candidates.currentIndex() == 0
    status = "overlap_suppressed" if suppressed else "rejected"
    assert f"rejected_first ({status})" in dialog.result_info.text()
    assert status in dialog.candidates.itemText(0)
    clouds = []
    dialog._cloud = clouds.append
    dialog._result_cloud()
    assert clouds[0].metadata["review"]["candidate_number"] == 1
    assert not clouds[0].metadata["review"]["is_best_accepted"]
    assert draft == applied
    dialog.reject()
    assert client.closed.wait(2)
