"""Host paired captures, graph ownership, frozen browsing, and native Qt lifecycle."""

import copy
import hashlib
import json
import time
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("opencv_dataflow")
from opencv_dataflow.core import Graph
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.plugin_dialog import PluginDialogContext
from opencv_dataflow.simple_gdrnpp.completed import BINDING, SCHEMA, report_digest
from opencv_dataflow.simple_gdrnpp.dataflow import SimpleGDRNPP
from opencv_dataflow.simple_gdrnpp.selection import describe_review


@pytest.fixture
def run(tmp_path):
    raw = b"sealed archive fixture - not a real trained model"
    pose = dict(model="pose_3d_v1", source_to_target=np.eye(4).tolist())
    pose["source_to_target"][2][3] = 1
    evidence = dict(
        points=np.array([[0.01, 0, 1], [0.02, 0, 0.8], [0.03, 0, 1.2]], np.float32),
        pixels=np.array([[1, 1], [2, 1], [3, 1]], np.int32),
        inside=np.ones(3, bool),
        observed_depth_m=np.ones(3, np.float32),
        target=np.ones(3, bool),
        known=np.ones(3, bool),
        delta_m=np.array([0, -0.2, 0.2], np.float32),
        supported=np.array([1, 0, 0], bool),
        contradicted=np.array([0, 1, 0], bool),
        occluded=np.array([0, 0, 1], bool),
        outside_target=np.zeros(3, bool),
    )
    report = dict(
        candidates=[
            dict(
                id="A",
                pose=pose,
                crop=dict(source_xyxy=[0, 0, 5, 3]),
                fit=dict(
                    status="rejected", coverage=1 / 3, contradiction=1 / 3, rendered_samples=3
                ),
            )
        ],
        best_pose=None,
        accepted_candidates=0,
        capture_id="fixture-B",
        target_frame="camera",
    )
    identity = dict(
        schema=SCHEMA,
        token="a" * 32,
        session="b" * 32,
        archive_sha256=hashlib.sha256(raw).hexdigest(),
        content_sha256="c" * 64,
        report_sha256=report_digest(report),
        candidates=1,
    )
    values = dict(
        image=ImageArray(np.full((3, 5, 3), [11, 22, 33], np.uint8), "rgb"),
        depth=DepthArray(np.ones((3, 5), np.float32) * 1000, "mm"),
        camera=dict(matrix=[[20, 0, 2], [0, 20, 1], [0, 0, 1]], image_size=[5, 3], frame="camera"),
        detections=[dict(identifier="A", box=[0, 0, 5, 3])],
        capture_id="fixture-B",
    )
    source = dict(
        rgb=np.asarray(values["image"]),
        depth=np.ones((3, 5), np.float32),
        camera=np.array(values["camera"]["matrix"], np.float64),
        mask=np.ones((3, 5), bool),
        recipe_bytes=np.frombuffer(raw, np.uint8).reshape(1, -1),
    )
    source.update({"reference_" + k: v for k, v in list(source.items()) if k != "recipe_bytes"})

    class Client:
        def __init__(self):
            self.closed = 0
            self.calls = []
            self.identity = copy.deepcopy(identity)

        def close(self):
            self.closed += 1

        def call(self, op, data, arrays=None, **kwargs):
            assert not self.closed
            self.calls.append(op)
            if op == "preview":
                selection, table = describe_review(report, 0)
                return dict(
                    report=copy.deepcopy(report),
                    completed=copy.deepcopy(identity),
                    review=selection,
                    candidate_table=table,
                ), {}
            if data["identity"] != self.identity:
                raise ValueError("Completed execution expired")
            meta = dict(identity=copy.deepcopy(identity))
            if op == "capture_completed":
                meta.update(
                    report=copy.deepcopy(report),
                    archive_bytes=len(raw),
                    setup=dict(content_sha256=identity["content_sha256"]),
                    capture=dict(
                        capture_id="fixture-B",
                        camera_frame="camera",
                        detections=[
                            dict(
                                object_id=6,
                                score=None,
                                source="external_box",
                                identifier="A",
                                box=[0.0, 0.0, 5.0, 3.0],
                            )
                        ],
                    ),
                    scope="Unit fixture, no real inference",
                )
                return meta, {k: v.copy() for k, v in source.items()}
            if op == "capture_completed_candidate":
                return dict(meta, index=0, candidate_id="A"), {
                    k: v.copy() for k, v in evidence.items()
                }
            assert op == "capture_completed_graphics"
            return meta, {}

    graph = Graph()
    graph.project_dir = tmp_path
    block = graph.add(
        SimpleGDRNPP(
            parameters=dict(
                recipe_path="fixture.gdrnpp",
                sha256=identity["archive_sha256"],
                models_dir="external-model",
            )
        )
    )
    client = Client()
    block._client = client
    payloads = {name: dict(spatial=frame([5, 3], "fixture-grid")) for name in ("image", "depth")}
    assert (
        block.execute({k: ValuePacket(v, payloads.get(k, {})) for k, v in values.items()})
        is not None
    ), block.error
    context = PluginDialogContext(graph, block)
    yield SimpleNamespace(
        block=block,
        graph=graph,
        context=context,
        client=client,
        values=values,
        evidence=evidence,
        report=report,
        identity=identity,
    )
    context._close()
    graph.close()


def test_capture_survives_graph_close_and_uses_no_analysis(run, tmp_path):
    job = run.block.prepare_completed_review(run.context)
    frozen = job.fetch()
    calls = run.client.calls.copy()
    run.graph.close()
    assert run.client.closed == 1
    assert not frozen.arrays["rgb"].flags.writeable
    assert not frozen.candidates[0]["points"].flags.writeable
    assert frozen.bundle.inputs["depth"].value.units == "mm"
    assert frozen.arrays["depth"].dtype == np.float32
    assert frozen.report == run.report
    assert frozen.sample(0, 1)["contradicted"] is True
    image = frozen.image(0)
    cloud = frozen.cloud(0)
    assert image.shape == (3, 5, 3) and cloud.metadata["stored_review"]
    exported = frozen.export(tmp_path / "review")
    meta = json.loads((exported / "review.json").read_bytes())
    assert meta["original_inputs"]["depth"]["units"] == "mm"
    assert meta["path_redactions"] == ["parameters.models_dir"]
    assert run.client.calls == calls
    with pytest.raises(FileExistsError):
        frozen.export(exported)


def test_display_and_execution_graph_own_worker_until_last_close(run):
    display = SimpleGDRNPP()
    display.copy_runtime_from(run.block)
    display.copy_runtime_from(run.block)  # on_frame + copy_results must not double-acquire.
    assert display._lease is run.block._lease
    run.block.close()
    assert run.client.closed == 0 and display._completed_record is not None
    next_execution = SimpleGDRNPP()
    next_execution.copy_runtime_from(display)
    display.close()
    assert run.client.closed == 0
    next_execution.close()
    next_execution.close()
    assert run.client.closed == 1


def test_pending_capture_owns_a_lease_and_failure_or_cancel_releases_it(run):
    job = run.block.prepare_completed_review(run.context)
    run.block.close()
    assert not run.client.closed
    with pytest.raises(RuntimeError, match="cancelled"):
        job.fetch(cancelled=lambda: True)
    assert run.client.closed == 1 and job.closed
    job.close()


@pytest.mark.parametrize("change", ("binding", "report", "source", "boxes", "token"))
def test_mixed_or_stale_evidence_is_rejected(run, change):
    if change == "binding":
        run.block.last_result.payloads["report"][BINDING]["token"] = "foreign"
        with pytest.raises(ValueError, match="different completed"):
            run.block.prepare_completed_review(run.context)
        return
    job = run.block.prepare_completed_review(run.context)
    if change == "report":
        job.bundle.outputs["report"].value["capture_id"] = "another"
    elif change == "source":
        job.bundle.inputs["depth"].value[0, 0] = 900
    elif change == "boxes":
        job.bundle.inputs["detections"].value[0]["box"] = [0, 0, 4, 3]
    else:
        run.client.identity["token"] = "expired"
    with pytest.raises(ValueError, match="differ|another|expired"):
        job.fetch()
    assert job.closed
    assert run.block._lease._owners == 1


def test_lean_and_missing_history_require_another_debug_completion(run):
    run.graph.set_run_profile("lean")
    with pytest.raises(ValueError, match="Debug"):
        run.block.prepare_completed_review(run.context)
    run.graph.set_run_profile("debug")
    with pytest.raises(ValueError, match="unavailable|Debug"):
        run.block.prepare_completed_review(run.context)


@pytest.mark.parametrize("port", ("accepted_count", "poses", "candidate_table"))
def test_modified_paired_summary_is_not_accepted(run, port):
    run.block.last_result.outputs[port] = 42
    with pytest.raises(ValueError, match="differ"):
        run.block.prepare_completed_review(run.context)


def test_native_review_remains_frozen_after_publication_and_graph_close(run, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from opencv_dataflow.simple_gdrnpp.qt_completed_review import CompletedReviewDialog

    app = QApplication.instance() or QApplication([])
    dialog = CompletedReviewDialog(
        None, run.context, lambda: run.block.prepare_completed_review(run.context)
    )
    dialog.capture_latest()
    deadline = time.monotonic() + 5
    while dialog.pending is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert dialog.frozen is not None, dialog.status.text()
    assert dialog.table.rowCount() == 1 and dialog.tabs.count() == 5
    assert dialog.cloud.cloud is not None and dialog.cloud.flyaround.isEnabled()
    dialog.sample.setValue(1)
    assert '"contradicted": true' in dialog.sample_info.text()
    before = dialog.frozen
    dialog.published(2)
    assert "stays frozen" in dialog.status.text()
    run.graph.close()
    dialog.sample.setValue(2)
    assert dialog.frozen is before and '"occluded": true' in dialog.sample_info.text()
    dialog.select_cloud_sample({"row": 0})
    assert dialog.sample.value() == 0
    dialog.select_pixel(3, 1)
    assert dialog.sample.value() == 2 and dialog.pick.last_pick["row"] == 2
    dialog.reject()
    assert dialog.closed


def test_active_and_closed_context_reject_before_acquiring_another_lease(run):
    with run.block._execution_lock:
        with pytest.raises(RuntimeError, match="active"):
            run.block.prepare_completed_review(run.context)
    assert run.block._lease._owners == 1
    run.context._close()
    with pytest.raises(RuntimeError, match="closed"):
        run.block.prepare_completed_review(run.context)
    assert run.block._lease._owners == 1


def test_closing_during_transfer_does_not_terminate_another_graphs_worker(run, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from opencv_dataflow.simple_gdrnpp.qt_completed_review import CompletedReviewDialog

    app = QApplication.instance() or QApplication([])
    entered, finish = Event(), Event()
    original = run.client.call

    def delayed(operation, *args, **kwargs):
        if operation == "capture_completed":
            entered.set()
            assert finish.wait(5)
        return original(operation, *args, **kwargs)

    monkeypatch.setattr(run.client, "call", delayed)
    dialog = CompletedReviewDialog(
        None, run.context, lambda: run.block.prepare_completed_review(run.context)
    )
    dialog.capture_latest()
    future = dialog.pending
    assert entered.wait(2)
    dialog.reject()
    assert run.client.closed == 0
    finish.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        future.result(timeout=3)
    assert run.block._lease._owners == 1 and run.client.closed == 0
    assert "capture_completed_candidate" not in run.client.calls
    app.processEvents()


def test_background_export_needs_no_worker(run, tmp_path):
    from PySide6.QtWidgets import QApplication

    from opencv_dataflow.simple_gdrnpp.qt_completed_review import CompletedReviewDialog

    app = QApplication.instance() or QApplication([])
    frozen = run.block.prepare_completed_review(run.context).fetch()
    dialog = CompletedReviewDialog(None, run.context, lambda: None)
    dialog.set_capture(frozen)
    run.graph.close()
    calls = run.client.calls.copy()
    dialog.start_export(tmp_path / "native_export")
    deadline = time.monotonic() + 5
    while dialog.pending is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert (tmp_path / "native_export/review.json").is_file(), dialog.status.text()
    assert (tmp_path / "native_export/candidate_01.ply").is_file()
    assert dialog.export_button.isEnabled() and run.client.calls == calls
    dialog.reject()
