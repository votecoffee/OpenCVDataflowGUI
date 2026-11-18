"""Independent score decay, native parity, metadata, geometry and admission checks."""

import copy
import hashlib
import json
import math
import os
import runpy
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.objects import collection
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.soft_nms import SuppressDetections, native_box
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolbox import block_entry


def item(box, score, cid=0, name="a"):
    x, y, w, h = box
    return {
        "id": name,
        "class_id": cid,
        "label": f"class-{cid}",
        "score": score,
        "metrics": {"score": score, "width": w, "custom": 42},
        "geometry": {"bbox": box, "unclipped_xyxy": [x, y, x + w, y + h]},
        "custom": {"serial": name},
        "orientation_measured": False,
    }


def arguments(items=None, dtype=np.uint8):
    image = np.zeros((80, 120, 3), dtype)
    context = {"spatial": frame([120, 80], "camera"), "ticket": TicketClock().issue("image")}
    context["pyramid"] = {"level": 0, "largest_size": [120, 80], "to_largest": np.eye(3).tolist()}
    data = collection(
        "boxes",
        items
        if items is not None
        else [
            item([10, 10, 20, 20], 0.9),
            item([12, 10, 20, 20], 0.8, name="b"),
            item([30, 10, 20, 20], 0.6, name="c"),
            item([10, 10, 20, 20], 0.7, 1, "d"),
        ],
        image,
        context,
    )
    return {
        "image": ValuePacket(image, copy.deepcopy(context)),
        "collection": ValuePacket(data, copy.deepcopy(context)),
    }


def run(args=None, **parameters):
    block = SuppressDetections(parameters=parameters)
    result = block.execute(arguments() if args is None else args)
    assert result is not None, block.error
    return block, result


def reference(boxes, scores, method, threshold, iou_threshold=0.5, sigma=0.5, top_k=0):
    # Independently implement greedy decay and tie-breaking; never call OpenCV here.
    current = [float(np.float32(s)) for s in scores]
    pending, selected = list(range(len(boxes))), []
    while pending and (not top_k or len(selected) < top_k):
        winner = min(pending, key=lambda i: (-current[i], i))
        if current[winner] < float(np.float32(threshold)):
            break
        selected.append((winner, current[winner]))
        pending.remove(winner)
        a = boxes[winner]
        for i in pending:
            b = boxes[i]
            intersection = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
                0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
            )
            overlap = float(np.float32(intersection / (a[2] * a[3] + b[2] * b[3] - intersection)))
            if method == "hard":
                if overlap > iou_threshold:
                    current[i] = -1
                continue
            if method == "none":
                continue
            factor = (
                (1 - overlap if overlap > iou_threshold else 1)
                if method == "linear"
                else math.exp(-(overlap**2) / sigma)
            )
            current[i] = float(np.float32(current[i] * factor))
    return selected


@pytest.mark.parametrize("method", ["linear", "gaussian"])
@pytest.mark.parametrize("agnostic,top_k", [(False, 0), (True, 0), (False, 1), (True, 2)])
def test_native_and_independent_decay_keep_ids_classes_scores_and_source_indices(
    method, agnostic, top_k
):
    args = arguments()
    args["collection"].value["context"]["custom_note"] = {"calibration": "original"}
    before = copy.deepcopy(args["collection"].value)
    b, r = run(
        args,
        method=method,
        class_agnostic=agnostic,
        top_k=top_k,
        score_threshold=0.1,
        include_candidate_details=True,
    )
    expected = []
    groups = [[0, 1, 2, 3]] if agnostic else [[0, 1, 2], [3]]
    for indices in groups:
        boxes = [before["items"][i]["geometry"]["bbox"] for i in indices]
        scores = [before["items"][i]["score"] for i in indices]
        chosen = reference(boxes, scores, method, 0.1, top_k=top_k)
        native_scores, native_indices = cv2.dnn.softNMSBoxes(
            boxes,
            scores,
            0.1,
            0.5,
            top_k=top_k,
            sigma=0.5,
            method=cv2.dnn.SoftNMSMethod_SOFTNMS_LINEAR
            if method == "linear"
            else cv2.dnn.SoftNMSMethod_SOFTNMS_GAUSSIAN,
        )
        assert [i for i, _ in chosen] == list(native_indices)
        np.testing.assert_allclose([s for _, s in chosen], native_scores, atol=1e-7)
        expected += [(indices[i], s) for i, s in chosen]
    expected.sort(key=lambda pair: (-pair[1], pair[0]))
    assert r.outputs["indices"] == [i for i, _ in expected]
    np.testing.assert_allclose(r.outputs["scores"], [s for _, s in expected], atol=1e-7)
    for found, (index, score) in zip(r.outputs["collection"]["items"], expected, strict=True):
        original = before["items"][index]
        assert found["id"] == original["id"] and found["class_id"] == original["class_id"]
        assert found["geometry"] == original["geometry"] and found["custom"] == original["custom"]
        assert found["metrics"]["custom"] == 42 and found["metrics"]["score"] == found["score"]
        assert found["soft_nms"]["source_index"] == index
        assert found["soft_nms"]["original_score"] == original["score"]
        assert found["score"] == pytest.approx(score, abs=1e-7)
    assert args["collection"].value == before and b.process_inputs_readonly
    assert r.outputs["collection"]["context"]["custom_note"] == {"calibration": "original"}
    assert ticket_key(r.payloads["collection"]["ticket"]) == ticket_key(
        args["image"].payload["ticket"]
    )
    assert r.outputs["collection"]["context"]["pyramid"] == before["context"]["pyramid"]


def test_decay_changes_rank_and_hard_removes_an_adjacent_candidate():
    _, soft = run(score_threshold=0.1)
    _, hard = run(method="hard", score_threshold=0.1)
    assert soft.outputs["indices"] == [0, 3, 2, 1]
    assert hard.outputs["indices"] == [0, 3, 2]
    assert soft.outputs["scores"][-1] < 0.2
    details = run(score_threshold=0.25, include_candidate_details=True)[1].outputs["report"][
        "candidates"
    ]
    assert details[1]["status"] == "suppressed_or_top_k" and details[1]["updated_score"] is None


@pytest.mark.parametrize("method", ["linear", "gaussian", "hard", "none"])
def test_float32_threshold_equality_zero_scores_and_stable_ties(method):
    _, r = run(
        arguments([item([10, 10, 20, 20], 0.25), item([80, 10, 10, 10], 0.25, name="b")]),
        method=method,
    )
    assert r.outputs["indices"] == ([] if method == "hard" else [0, 1])
    _, zero = run(arguments([item([0, 0, 10, 10], 0)]), method=method, score_threshold=0)
    assert zero.outputs["count"] == (0 if method == "hard" else 1)


def test_gaussian_ignores_iou_cutoff_but_sigma_controls_decay():
    a = run(method="gaussian", score_threshold=0, iou_threshold=0)[1]
    b = run(method="gaussian", score_threshold=0, iou_threshold=1)[1]
    assert (
        a.outputs["indices"] == b.outputs["indices"] and a.outputs["scores"] == b.outputs["scores"]
    )
    c = run(method="gaussian", score_threshold=0, sigma=0.1)[1]
    assert c.outputs["scores"][-1] < a.outputs["scores"][-1]
    assert a.outputs["report"]["iou_threshold_used"] is False


@pytest.mark.parametrize(
    "rounding,expected", [("cover", [-1, 1, 12, 11]), ("nearest", [0, 2, 10, 10])]
)
def test_fractional_native_boxes_are_explicit_and_original_geometry_is_preserved(
    rounding, expected
):
    args = arguments([item([-0.2, 1.5, 10.4, 10.1], 0.9)])
    _, r = run(args, rounding=rounding)
    found = r.outputs["collection"]["items"][0]
    assert found["soft_nms"]["native_bbox"] == expected
    assert found["geometry"] == args["collection"].value["items"][0]["geometry"]
    with pytest.raises(ValidationError):
        native_box(np.array([1.1, 1.1, 1.2, 1.2]), "nearest")
    with pytest.raises(ValidationError):
        native_box(np.array([0, 0, 32768, 32768]), "cover")


def test_unclipped_vs_visible_box_policy_has_no_silent_fallback():
    args = arguments([item([0, 10, 10, 20], 0.9), item([0, 10, 10, 20], 0.8, name="b")])
    args["collection"].value["items"][0]["geometry"]["unclipped_xyxy"] = [-30, 10, 10, 30]
    full = run(args, score_threshold=0.1)[1]
    clipped = run(args, score_threshold=0.1, box_source="bbox")[1]
    assert full.outputs["count"] == 2 and clipped.outputs["count"] == 1
    del args["collection"].value["items"][0]["geometry"]["unclipped_xyxy"]
    b = SuppressDetections()
    assert b.execute(args) is None and "four numeric" in b.error


def test_class_filter_and_center_mask_region_intersection_leave_pixels_unchanged():
    args = arguments(dtype=np.uint16)
    mask = np.zeros((80, 120), np.uint8)
    mask[:, 30:] = 255
    args["mask"] = mask
    b, r = run(args, accepted_class_ids=[0], include_candidate_details=True)
    assert r.outputs["indices"] == [2] and r.outputs["image"].dtype == np.uint16
    np.testing.assert_array_equal(r.outputs["image"], args["image"].value)
    assert r.outputs["report"]["filtered_counts"]["class_filter"] == 1
    assert r.drawings
    b.run_profile = profile("lean")
    lean = b.execute(args)
    assert not lean.drawings and lean.outputs["indices"] == r.outputs["indices"]
    args["region"] = {
        "model": "image_region_v1",
        "image_size": [120, 80],
        "shape": "rectangle",
        "rectangle": [70, 0, 30, 80],
        "invert": False,
    }
    assert b.execute(args).outputs["count"] == 0


@pytest.mark.parametrize(
    "field",
    ["ticket", "frame", "root_size", "pyramid", "recipe", "internal_frame", "size", "origin"],
)
def test_known_source_identity_or_collection_grid_conflicts_reject(field):
    args = arguments()
    wire = args["collection"].payload
    data = args["collection"].value
    if field == "ticket":
        wire["ticket"] = TicketClock().issue("another")
    elif field == "frame":
        wire["spatial"]["root_id"] = "other"
    elif field == "root_size":
        wire["spatial"]["root_size"] = [121, 80]
    elif field == "pyramid":
        wire["pyramid"]["level"] = 1
    elif field == "recipe":
        args["image"].payload["recipe_ref"] = {"id": "a", "revision": 1, "digest": "a" * 64}
        wire["recipe_ref"] = {"id": "a", "revision": 2, "digest": "b" * 64}
    elif field == "internal_frame":
        data["context"]["spatial"]["root_id"] = "other"
    elif field == "size":
        data["image_size"] = [119, 80]
    else:
        data["origin"] = [1, 0]
    b = SuppressDetections()
    assert b.execute(args) is None and not b.current_emissions


@pytest.mark.parametrize(
    "params",
    [
        {"sigma": 0},
        {"sigma": float("nan")},
        {"top_k": True},
        {"rounding": "truncate"},
        {"class_agnostic": 1},
        {"accepted_class_ids": [0, 0]},
        {"accepted_class_ids": [True]},
        {"minimum_count": 3, "maximum_count": 2},
        {"maximum_candidates": 4097},
    ],
)
def test_invalid_settings_validate_without_processing(params):
    with pytest.raises(ValidationError):
        SuppressDetections(parameters=params).validate_parameters()


@pytest.mark.parametrize(
    "params",
    [
        {"maximum_raw_detections": 2},
        {"maximum_candidates": 2},
        {"maximum_results": 1},
        {"maximum_image_pixels": 10},
    ],
)
def test_budgets_reject_instead_of_emitting_partial_results(params):
    b = SuppressDetections(parameters=params)
    assert b.execute(arguments()) is None
    assert not b.current_emissions and "budget" in b.error


def test_native_exception_containment_recovery_admission_and_empty_count_failure(monkeypatch):
    args = arguments()
    b = SuppressDetections()
    first = b.execute(args)
    before = b.last_inputs
    b.input_failure_policies["collection"] = "reject_failed"
    args["collection"].payload["outcome"] = {
        "model": "tool_outcome_v1",
        "passed": False,
        "chain_passed": False,
        "source": {"id": "prior"},
        "fail_reasons": [],
    }
    assert b.execute(args) is None and b.input_rejected and not b.current_emissions
    assert b.last_result is first
    assert b.last_inputs is before
    args["collection"].payload.pop("outcome")
    original = cv2.dnn.softNMSBoxes
    monkeypatch.setattr(
        cv2.dnn,
        "softNMSBoxes",
        lambda *a, **k: (_ for _ in ()).throw(cv2.error("controlled native failure")),
    )
    b.execute(args)
    assert "controlled native failure" in b.error and not b.current_emissions
    monkeypatch.setattr(cv2.dnn, "softNMSBoxes", original)
    assert b.execute(args) is not None
    empty = b.execute(arguments([]))
    assert empty is not None and not empty.passed and empty.outputs["count"] == 0
    assert empty.fail_reasons[0]["code"] == "suppression.count" and b.current_emissions


@pytest.mark.parametrize(
    "returned",
    [
        ([0.9, 0.8], [0]),
        ([0.9, 0.8], [0, 0]),
        ([0.9], [7]),
        ([float("nan")], [0]),
        ([1.0], [0]),
        ([0.9], [True]),
        ([0.9], [0.0]),
    ],
)
def test_malformed_native_score_index_pairs_emit_no_current_result(monkeypatch, returned):
    monkeypatch.setattr(cv2.dnn, "softNMSBoxes", lambda *a, **k: returned)
    b = SuppressDetections()
    assert b.execute(arguments()) is None and not b.current_emissions


def test_unknown_contexts_remain_usable_and_none_mode_skips_native_calls(monkeypatch):
    args = arguments()
    args = {"image": args["image"].value, "collection": args["collection"].value}
    args["collection"]["context"] = {}

    def forbidden(*a, **k):
        pytest.fail("Disabled suppression called native NMS")

    monkeypatch.setattr(cv2.dnn, "softNMSBoxes", forbidden)
    monkeypatch.setattr(cv2.dnn, "NMSBoxes", forbidden)
    _, r = run(args, method="none")
    assert r.outputs["indices"] == [0, 1, 3, 2]
    assert r.outputs["report"]["context_checks"] == {
        "ticket_checked": False,
        "frame_checked": False,
    }


BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/soft_nms"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("mode", ["debug", "lean"])
def test_saved_image_dependent_projects_match_scalar_reference_and_filmstrip(case, mode):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    try:
        graph.set_run_profile(mode)
        graph.run()
        block = graph.blocks["suppress"]
        result = block.last_result
        assert result is not None, block.error
        image = graph.blocks["source"].last_result.outputs["image"]
        # CNN scores = known base + red mean * 0.02. No model call in the reference.
        maximum = 65535 if case == "high_depth" else 255
        raw_scores = np.float32(TRUTH["base_scores"]) + np.float32(
            image[..., 2].mean() / maximum * 0.02
        )
        ordered = sorted(range(6), key=lambda i: (-raw_scores[i], i))
        decoded = graph.blocks["decode"].last_result.outputs["collection"]["items"]
        assert [v["id"] for v in decoded] == ordered
        np.testing.assert_allclose([v["score"] for v in decoded], raw_scores[ordered], atol=1e-6)
        # Quantization here is independently expressed, using pinned source edges.
        boxes = []
        for i in ordered:
            x, y, r, b = TRUTH["boxes_xyxy"][i]
            if case == "visible_boxes":
                x, y, r, b = max(0, x), max(0, y), min(64, r), min(64, b)
            if case == "nearest":
                x, y, r, b = map(round, [x, y, r, b])
            else:
                x, y, r, b = math.floor(x), math.floor(y), math.ceil(r), math.ceil(b)
            boxes.append([x, y, r - x, b - y])
        params = block.parameters
        groups = {}
        for index, original in enumerate(ordered):
            center = (
                np.array(TRUTH["boxes_xyxy"][original][:2]) + TRUTH["boxes_xyxy"][original][2:]
            ) / 2
            if case == "selected" and not (28 <= math.floor(center[0]) < 64):
                continue
            if raw_scores[original] < params["score_threshold"]:
                continue
            groups.setdefault(
                0 if params["class_agnostic"] else TRUTH["classes"][original], []
            ).append(index)
        chosen = []
        for indices in groups.values():
            values = reference(
                [boxes[i] for i in indices],
                [raw_scores[ordered[i]] for i in indices],
                params["method"],
                params["score_threshold"],
                top_k=params["top_k"],
            )
            chosen += [(indices[i], s) for i, s in values]
        chosen.sort(key=lambda row: (-row[1], row[0]))
        assert result.outputs["indices"] == [i for i, _ in chosen]
        np.testing.assert_allclose(result.outputs["scores"], [s for _, s in chosen], atol=1e-6)
        for found, (i, _) in zip(result.outputs["collection"]["items"], chosen, strict=True):
            assert found["geometry"] == decoded[i]["geometry"] and found["id"] == decoded[i]["id"]
            assert found["soft_nms"]["native_bbox"] == boxes[i]
            assert found["class_id"] == TRUTH["classes"][found["id"]]
        assert result.passed == (case not in ("empty", "count_failure"))
        assert ("candidates" in result.outputs["report"]) == (case != "details_off")
        assert ticket_key(result.payloads["collection"]["ticket"]) == ticket_key(
            graph.blocks["source"].last_result.outputs["ticket"]
        )
        np.testing.assert_array_equal(result.outputs["image"], image)
        overlay = graph.blocks["overlay"].last_result
        assert overlay is not None, graph.blocks["overlay"].error
        assert graph.blocks["filmstrip"].last_result is not None
        assert bool(result.drawings) == (mode == "debug" and case != "empty")
        if mode == "debug" and case != "empty":
            assert np.any(overlay.outputs["image"] != image)
    finally:
        graph.close()


def test_exact_pinned_assets_and_authoring_regeneration_without_overwrite(tmp_path):
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build(tmp_path)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(tmp_path)).read_bytes(), path
    with pytest.raises(FileExistsError):
        build(tmp_path)


def test_portable_json_xml_parameters_geometry_and_graphics(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(BASE / "linear.json", registry)
    try:
        graph.run()
        expected = copy.deepcopy(graph.blocks["suppress"].last_result.outputs["collection"])
        assert block_entry(graph.blocks["suppress"]).path[:2] == (
            "OpenCV",
            "Deep Neural Networks (dnn)",
        )
        graph.save(tmp_path / "saved.json")
        export_project_xml(graph.to_dict(), tmp_path / "saved.xml")
    finally:
        graph.close()
    for graph in (
        Graph.load(tmp_path / "saved.json", registry),
        Graph.from_dict(read_project_xml(tmp_path / "saved.xml"), registry, project_dir=tmp_path),
    ):
        try:
            assert graph.blocks["suppress"].last_result is None
            graph.run()
            assert graph.blocks["suppress"].last_result.outputs["collection"] == expected
        finally:
            graph.close()


def test_cold_offscreen_single_image_runner_and_no_native_shutdown_error():
    script = r"""
import sys
from pathlib import Path
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry();graph=Graph.load(Path(sys.argv[1]),registry)
try:
 worker=Runner(graph,'once',registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(15000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==len(frames)==1
 assert graph.blocks['suppress'].last_result.outputs['count']==6
finally:graph.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(BASE / "linear.json")],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
