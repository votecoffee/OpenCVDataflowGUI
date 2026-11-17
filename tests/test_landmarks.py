"""Independent subpixel geometry, identity, incomplete output and pose qualification."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph, Result
from opencv_dataflow.dnn_blocks import DNNInference, preprocess
from opencv_dataflow.images import ImageArray
from opencv_dataflow.landmarks import DecodeLandmarks, LandmarkPose
from opencv_dataflow.ml_common import tensor_record
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.poses import local_offset
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.timing import advance_timing, mark

CENTERS = [(5.25, 6.4), (17.3, 10.2), (0, 0)]
BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/landmarks"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


def arguments(*, layout="NCHW", letterbox=False, values=None, direct=False):
    image = np.full((80, 160 if letterbox else 120, 3), [30, 100, 180], np.uint8)
    _, geometry = preprocess(
        {"image": image},
        {
            **DNNInference.defaults,
            "width": 24,
            "height": 16,
            "resize_mode": "letterbox" if letterbox else "stretch",
        },
    )
    if values is None:
        y, x = np.mgrid[:16, :24]
        values = np.array(
            [np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 8) for cx, cy in CENTERS], np.float32
        )
    if not direct:
        if layout.endswith("HWC"):
            values = values.transpose(1, 2, 0)
        if layout.startswith("N"):
            values = values[None]
    return {
        "image": image,
        "tensors": {"head": tensor_record(values, 8_000_000, allow_empty=True)},
        "preprocessing": geometry,
    }


def decode(args=None, **params):
    block = DecodeLandmarks(parameters=params)
    result = block.execute(args or arguments())
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("layout", ["NCHW", "NHWC", "CHW", "HWC"])
@pytest.mark.parametrize("letterbox", [False, True])
def test_log_gaussian_refinement_and_source_mapping_have_independent_analytic_reference(
    layout, letterbox
):
    args = arguments(layout=layout, letterbox=letterbox)
    _, result = decode(args, layout=layout, minimum_confidence=0.8)
    items = result.outputs["landmarks"]["items"]
    for i in range(2):
        center = np.array(CENTERS[i])
        expected = (
            (center + 0.5) * 5 - 0.5
            if not letterbox
            else (center + 0.5 - [0, 2]) / [0.15, 0.15] - 0.5
        )
        np.testing.assert_allclose(items[i]["position"], expected, atol=2e-6)
        assert items[i]["accepted"] and items[i]["refinement"] == "refined"
        assert items[i]["fit_rmse"] < 1e-6
    assert [i["index"] for i in items] == [0, 1, 2]
    assert items[2]["refinement"] in ("boundary", "selection_boundary")
    np.testing.assert_array_equal(result.outputs["image"], args["image"])
    if letterbox:
        assert not items[2]["accepted"]  # The strongest peak is padding; selected tail is weak.


@pytest.mark.parametrize(
    "mapping,expected",
    [
        ("half_pixel", [(5.25 + 0.5) * 5 - 0.5, (6.4 + 0.5) * 5 - 0.5]),
        ("asymmetric", [5.25 * 5 + 2, 6.4 * 5 + 2]),
        ("align_corners", [(0.5 + 5.25 * 23 / 23) * 5 - 0.5, (0.5 + 6.4 * 15 / 15) * 5 - 0.5]),
    ],
)
def test_declared_heatmap_alignment_conventions(mapping, expected):
    _, result = decode(heatmap_mapping=mapping, channels=[0])
    np.testing.assert_allclose(result.outputs["points"][0], expected, atol=2e-6)


@pytest.mark.parametrize("units", ["network_pixels", "normalized"])
@pytest.mark.parametrize("origin", ["pixel_centers", "pixel_edges"])
@pytest.mark.parametrize("layout", ["rows", "channels_first"])
def test_direct_coordinate_layout_units_and_origin_are_explicit(units, origin, layout):
    rows = np.array([[5, 6, 0.9], [17, 10, 0.1]], np.float32)
    if origin == "pixel_edges":
        rows[:, :2] += 0.5
    if units == "normalized":
        rows[:, :2] /= [24, 16]
    values = rows.T[None] if layout == "channels_first" else rows[None]
    _, result = decode(
        arguments(values=values, direct=True),
        format="xy_score",
        coordinate_layout=layout,
        coordinate_units=units,
        coordinate_origin=origin,
    )
    items = result.outputs["landmarks"]["items"]
    np.testing.assert_allclose(items[0]["position"], [27, 32], atol=3e-6)
    assert items[0]["accepted"] and not items[1]["accepted"]
    assert result.outputs["points"] is NO_VALUE and result.outputs["indices"] == [0]
    assert not result.passed


def test_selected_channel_order_names_and_partial_points_cannot_silently_renumber_correspondence():
    args = arguments(values=np.array([[[5, 6, 0.9], [17, 10, 0.1]]]), direct=True)
    _, result = decode(
        args,
        format="xy_score",
        channels=[1, 0],
        names=["weak", "good"],
        points_policy="accepted_only",
        require_all=False,
    )
    assert result.passed and result.outputs["indices"] == [1]
    assert [i["channel"] for i in result.outputs["landmarks"]["items"]] == [1, 0]
    np.testing.assert_allclose(result.outputs["points"], [[27, 32]])


def test_peak_search_respects_roi_instead_of_masking_after_an_outside_global_peak():
    values = np.zeros((1, 16, 24), np.float32)
    values[0, 6, 5], values[0, 8, 15] = 0.95, 0.8
    args = arguments(values=values)
    args["region"] = {
        "model": "image_region_v1",
        "image_size": [120, 80],
        "shape": "rectangle",
        "rectangle": [60, 30, 40, 40],
    }
    _, searched = decode(args, refinement="none")
    assert searched.passed
    np.testing.assert_array_equal(searched.outputs["points"], [[77, 42]])
    _, filtered = decode(args, refinement="none", peak_search="entire_head")
    assert not filtered.passed
    assert filtered.outputs["landmarks"]["items"][0]["reason"] == "outside_selection"
    args["mask"] = np.zeros((80, 120), np.uint8)
    _, empty = decode(args)
    assert (
        empty.outputs["count"] == 0 and empty.outputs["landmarks"]["items"][0]["position"] is None
    )


def test_ambiguity_refinement_fallback_and_strict_fit_policy_are_observable():
    args = arguments(values=np.ones((1, 16, 24), np.float32))
    _, rejected = decode(args)
    assert rejected.outputs["landmarks"]["items"][0]["reason"] == "ambiguous_peak"
    _, first = decode(args, ambiguity="first")
    assert first.passed and first.outputs["landmarks"]["items"][0]["ambiguous"]
    _, strict = decode(channels=[2], refinement_failure="reject")
    assert (
        not strict.passed
        and strict.outputs["landmarks"]["items"][0]["reason"] == "refinement_rejected"
    )
    _, integer = decode(channels=[0], refinement="none")
    np.testing.assert_array_equal(integer.outputs["points"], [[27, 32]])


@pytest.mark.parametrize(
    "params,field",
    [
        ({"minimum_confidence": 1.1}, "minimum_confidence"),
        ({"channels": [0, 0]}, "channels"),
        ({"channels": [True]}, "channels"),
        ({"names": [""]}, "names"),
        ({"connections": [[0, 0]]}, "connections"),
        ({"connections": [[0, 1], [1, 0]]}, "connections"),
        ({"require_all": 1}, "require_all"),
        ({"refinement_radius": 4}, "refinement_radius"),
    ],
)
def test_invalid_configuration_identifies_fields_without_processing(params, field):
    block = DecodeLandmarks(parameters=params)
    with pytest.raises(ValidationError) as err:
        block.validate_parameters()
    assert field in err.value.fields
    assert not block.last_inputs


@pytest.mark.parametrize(
    "params",
    [
        {"channels": [3]},
        {"names": ["only"]},
        {"connections": [[0, 3]]},
        {"maximum_tensor_elements": 1},
        {"maximum_landmarks": 1},
        {"maximum_heatmap_pixels": 1},
        {"maximum_image_pixels": 1},
        {"layout": "CHW"},
    ],
)
def test_head_mismatch_and_budgets_emit_nothing(params):
    block = DecodeLandmarks(parameters=params)
    assert block.execute(arguments()) is None and not block.current_emissions


def test_empty_coordinate_rows_and_one_cell_align_corners_are_valid_shapes():
    args = arguments(values=np.empty((1, 0, 3)), direct=True)
    _, result = decode(args, format="xy_score", minimum_count=0)
    assert result.passed and result.outputs["points"] == []
    _, single = decode(arguments(values=np.ones((1, 1, 1))), heatmap_mapping="align_corners")
    np.testing.assert_array_equal(single.outputs["points"], [[59.5, 39.5]])


def test_graphics_off_native_exception_and_readonly_history_contract(monkeypatch):
    block, result = decode(connections=[[0, 1]], graphics_maximum_landmarks=1)
    assert (
        result.outputs["report"]["graphics_truncated"]
        and len(result.outputs["landmarks"]["items"]) == 3
    )
    native = cv2.minMaxLoc

    def bad(*args, **kwargs):
        raise cv2.error("controlled peak error")

    monkeypatch.setattr(cv2, "minMaxLoc", bad)
    assert (
        block.execute(arguments()) is None
        and "controlled peak" in block.error
        and not block.current_emissions
    )
    monkeypatch.setattr(cv2, "minMaxLoc", native)
    block.run_profile = profile("lean")
    result = block.execute(arguments())
    assert result is not None and not result.drawings and block.timing.count == 3


def contextual():
    args = arguments()
    args["image"] = ImageArray(args["image"].astype(np.uint16) * 257, "rgb")
    spatial = frame([120, 80], "camera")
    payload = {
        "spatial": spatial,
        "ticket": TicketClock().issue("camera"),
        "recipe_ref": {"id": "part", "revision": 1, "digest": "1" * 64},
        "processing_time": advance_timing(None, mark(), mark(), source=True),
    }
    return (
        args,
        payload,
        {key: ValuePacket(value, copy.deepcopy(payload)) for key, value in args.items()},
    )


def test_context_depth_and_landmark_pose_pair_offsets_and_missing_are_preserved():
    args, payload, wrapped = contextual()
    _, result = decode(wrapped)
    assert result.outputs["image"].color_format == "rgb"
    for context in result.payloads.values():
        assert (
            context["ticket"] == payload["ticket"]
            and context["recipe_ref"] == payload["recipe_ref"]
        )
        assert context["processing_time"]["started_at"] == payload["processing_time"]["started_at"]
    pose = LandmarkPose(parameters={"pose_offset_x": 3})
    selected = pose.execute(
        {
            "image": wrapped["image"],
            "landmarks": ValuePacket(result.outputs["landmarks"], result.payloads["landmarks"]),
        }
    )
    assert selected is not None and selected.passed, pose.error
    points = np.array([i["position"] for i in result.outputs["landmarks"]["items"][:2]])
    angle = -np.degrees(np.arctan2(*(points[1] - points[0])[::-1]))
    np.testing.assert_allclose(
        selected.outputs["pose"]["source_to_target"],
        local_offset(*points[0], angle) @ local_offset(3, 0, 0),
    )
    pose.parameters["origin_index"] = 20
    missing = pose.execute({"image": wrapped["image"], "landmarks": result.outputs["landmarks"]})
    assert missing is not None and not missing.passed and missing.outputs["pose"] is NO_VALUE


@pytest.mark.parametrize("field", ["ticket", "spatial", "recipe_ref"])
def test_known_context_conflict_rejects_and_valid_input_recovers(field):
    args, payload, wrapped = contextual()
    bad = wrapped["tensors"].payload
    if field == "ticket":
        bad[field] = TicketClock().issue("other")
    elif field == "spatial":
        bad[field]["root_id"] = "other"
    else:
        bad[field]["revision"] = 2
    block = DecodeLandmarks()
    assert block.execute(wrapped) is None and not block.current_emissions
    assert block.execute(args) is not None


def test_catalog_has_all_parameter_help_and_safe_flags():
    for kind in ("ml.decode_landmarks", "ml.landmark_pose"):
        block = builtin_registry().create(kind)
        block.validate_parameters()
        assert block.defaults.keys() == block.parameter_help.keys()
        assert block.preview_safe and block.parallel_safe and block.process_inputs_readonly


@pytest.mark.parametrize("case", list(TRUTH["cases"]))
@pytest.mark.parametrize("run_profile", ["debug", "lean"])
def test_native_image_landmark_examples_known_points_identity_pose_and_filmstrip(case, run_profile):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(run_profile)
    try:
        graph.run()
        result = graph.blocks["decode"].last_result
        assert result is not None, graph.blocks["decode"].error
        expected = TRUTH["cases"][case]
        items = result.outputs["landmarks"]["items"]
        assert result.passed is expected["passed"]
        assert result.outputs["indices"] == expected["accepted"]
        assert [i["channel"] for i in items] == expected["channels"]
        for i in expected["accepted"]:
            np.testing.assert_allclose(
                items[i]["position"],
                expected["source_points"][i],
                atol=expected["source_tolerance"],
                rtol=0,
            )
        if len(expected["accepted"]) != len(items):
            assert result.outputs["points"] is NO_VALUE
        else:
            np.testing.assert_allclose(
                result.outputs["points"],
                expected["source_points"],
                atol=expected["source_tolerance"],
                rtol=0,
            )
        original = graph.blocks["source"].last_result.outputs["image"]
        np.testing.assert_array_equal(result.outputs["image"], original)
        assert original.dtype == np.uint16
        assert (
            result.payloads["landmarks"]["ticket"]
            == graph.blocks["source"].last_result.outputs["ticket"]
        )
        overlay = graph.blocks["overlay"].last_result.outputs["image"]
        np.testing.assert_array_equal(
            overlay, graph.blocks["filmstrip"].last_result.outputs["image"]
        )
        if run_profile == "debug":
            assert result.drawings
            assert np.any(overlay != original)
        else:
            assert not result.drawings
            np.testing.assert_array_equal(overlay, original)
        if case == "pose_region":
            pose = graph.blocks["pose"].last_result
            assert pose is not None and pose.outputs["found"]
            np.testing.assert_allclose(pose.outputs["pose"]["origin"], items[0]["position"])
            assert graph.blocks["roi"].last_result is not None
        if case == "pinned":
            assert graph.blocks["infer"].last_result.outputs["receipt"]["calls"][0][
                "contract_revision"
            ]
    finally:
        graph.close()


def test_save_as_json_xml_sealed_model_integrity_and_exact_regeneration(tmp_path):
    before = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    graph = Graph.load(BASE / "pinned.json", builtin_registry())
    try:
        graph.save(tmp_path / "saved.json")
        restored = Graph.load(tmp_path / "saved.json", builtin_registry())
        try:
            restored.run()
            assert restored.blocks["decode"].last_result.passed
        finally:
            restored.close()
        export_project_xml(graph.to_dict(), tmp_path / "saved.xml")
        restored = Graph.from_dict(
            read_project_xml(tmp_path / "saved.xml"), builtin_registry(), project_dir=tmp_path
        )
        try:
            restored.run()
            assert restored.blocks["decode"].last_result.passed
        finally:
            restored.close()
    finally:
        graph.close()
    spec = importlib.util.spec_from_file_location("landmark_examples", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path / "regenerated"
    module.HERE.mkdir()
    module.build()
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((module.HERE / name).read_bytes()).hexdigest() == digest
    for path in [*BASE.glob("*.json"), BASE / "bundle/contract.json"]:
        assert json.loads((module.HERE / path.relative_to(BASE)).read_bytes()) == json.loads(
            path.read_bytes()
        )
    assert all(
        hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
        for name, digest in before.items()
    )
    (module.HERE / "assets/landmarks.onnx").write_bytes(b"existing model")
    with pytest.raises(FileExistsError):
        module.build()


def test_wired_missing_region_blocks_processing_and_pose_maps_shared_root(monkeypatch):
    graph = Graph.load(BASE / "selection.json", builtin_registry())
    try:
        graph.run()
        count = graph.blocks["decode"].timing.count
        monkeypatch.setattr(
            graph.blocks["selection"],
            "process",
            lambda inputs: Result({"mask": NO_VALUE, "region": NO_VALUE, "overlay": NO_VALUE}),
        )
        graph.run()
        assert (
            graph.blocks["decode"].timing.count == count
            and not graph.blocks["decode"].current_emissions
        )
    finally:
        graph.close()
    args, payload, wrapped = contextual()
    _, result = decode(wrapped)
    target = np.zeros((160, 240, 3), np.uint8)
    context = copy.deepcopy(payload)
    context["spatial"] = frame([240, 160], "camera")
    data = copy.deepcopy(result.outputs["landmarks"])
    data["context"]["spatial"]["root_size"] = [240, 160]
    data["context"]["spatial"]["to_root"] = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    pose = LandmarkPose()
    selected = pose.execute({"image": ValuePacket(target, context), "landmarks": data})
    assert selected is not None and selected.passed, pose.error
    np.testing.assert_allclose(
        selected.outputs["pose"]["origin"], np.array(data["items"][0]["position"]) * 2
    )


@pytest.mark.parametrize("problem", ["dtype", "nonfinite", "nested", "shape", "batch"])
def test_malformed_tensors_emit_nothing(problem):
    args = arguments()
    record = args["tensors"]["head"]
    if problem == "dtype":
        record["dtype"] = "uint8"
    elif problem == "nonfinite":
        record["values"][0] = float("nan")
    elif problem == "nested":
        record["values"][0] = [1]
    elif problem == "shape":
        record["shape"] = [1, 3, 15, 24]
    else:
        record["shape"] = [3, 1, 16, 24]
    block = DecodeLandmarks()
    assert block.execute(args) is None and not block.current_emissions


def test_direct_sigmoid_threshold_equality_and_invalid_pose_settings():
    args = arguments(values=np.array([[[5, 6, 0]]]), direct=True)
    _, result = decode(args, format="xy_score", score_activation="sigmoid")
    assert result.passed and result.outputs["landmarks"]["items"][0]["confidence"] == 0.5
    for params in (
        {"origin_index": 1, "direction_index": 1},
        {"origin_index": 1.0},
        {"pose_offset_x": 1e7},
    ):
        with pytest.raises(ValidationError):
            LandmarkPose(parameters=params).validate_parameters()


@pytest.mark.parametrize("field", ["ticket", "spatial", "recipe_ref"])
def test_pose_body_wire_conflicts_emit_no_pose_and_recover(field):
    _, payload, wrapped = contextual()
    _, decoded = decode(wrapped)
    data = decoded.outputs["landmarks"]
    bad = copy.deepcopy(payload)
    if field == "ticket":
        bad[field] = TicketClock().issue("other")
    elif field == "spatial":
        bad[field]["root_id"] = "other"
    else:
        bad[field]["revision"] = 2
    block = LandmarkPose()
    assert block.execute({"image": wrapped["image"], "landmarks": ValuePacket(data, bad)}) is None
    assert not block.current_emissions
    result = block.execute({"image": wrapped["image"], "landmarks": data})
    assert result is not None and result.passed, block.error


def test_native_model_coordinates_follow_changed_source_pixels():
    graph = Graph.load(BASE / "refined.json", builtin_registry())
    try:
        graph.run()
        before = np.array(graph.blocks["decode"].last_result.outputs["points"])
        image = graph.blocks["source"].last_result.outputs["image"]
        moved = image.copy()
        moved[..., 2] = np.roll(moved[..., 2], 5, axis=1)
        infer = graph.blocks["infer"]
        forwarded = infer.execute({"image": moved})
        assert forwarded is not None, infer.error
        _, decoded = decode(
            {"image": moved, **{k: forwarded.outputs[k] for k in ("tensors", "preprocessing")}},
            names=["red", "green", "blue"],
            output_name="heatmaps",
        )
        after = np.array(decoded.outputs["points"])
        np.testing.assert_allclose(after[0] - before[0], [5, 0], atol=0.002)
        np.testing.assert_array_equal(after[1:], before[1:])
    finally:
        graph.close()
