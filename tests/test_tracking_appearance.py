"""Histogram/embedding references, assignment, context and contained state failures."""

import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.image_tracking import TrackImageParts
from opencv_dataflow.images import ImageArray
from opencv_dataflow.objects import collection
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.tracking_appearance import (
    DescribeBoxAppearance,
    unit_appearance,
    updated_appearance,
)


def row(x, vector=None, space="model/preprocessing-v1", identity=1):
    return {
        "id": identity,
        "center": [x, 20],
        "size": [12, 10],
        "class_id": 0,
        "label": "part",
        **({"appearance": {"space": space, "vector": vector}} if vector is not None else {}),
    }


def runner(**parameters):
    block = TrackImageParts(
        parameters={
            "clock": "nominal",
            "minimum_observations": 1,
            "minimum_confidence": 0,
            "association_distance": 200,
            "innovation_gate": 1e6,
            "position_noise": 0.001,
            "maximum_position_sigma": 10000,
            **parameters,
        }
    )
    clock = TicketClock()

    def step(rows):
        payload = {"spatial": frame([180, 50], "camera"), "ticket": clock.issue("camera")}
        return block.execute(
            {
                "image": ValuePacket(np.zeros((50, 180), np.uint8), payload),
                "detections": ValuePacket(rows, payload),
            }
        )

    return block, step


def boxes():
    image = np.zeros((20, 40, 3), np.uint8)
    image[:, :20] = (0, 0, 255)
    image[:, 20:] = (255, 0, 0)
    payload = {"spatial": frame([40, 20], "camera"), "ticket": TicketClock().issue("camera")}
    records = [
        {
            "id": i,
            "score": 0.9,
            "label": "part",
            "class_id": 0,
            "metrics": {"x": x, "y": 0, "width": 20, "height": 20},
            "geometry": {"bbox": [x, 0, 20, 20]},
        }
        for i, x in enumerate((0, 20))
    ]
    data = collection("boxes", records, image, payload)
    return {
        "image": ValuePacket(image, payload),
        "detections": ValuePacket(data, copy.deepcopy(payload)),
    }


def test_unit_normalization_handles_huge_values_and_matches_cosine_reference():
    value = unit_appearance({"space": "test", "vector": [3e300, 4e300]}, 2)
    np.testing.assert_allclose(value["vector"], [0.6, 0.8])
    current = {"space": "test", "vector": np.array([1.0, 0.0])}
    updated = updated_appearance(value, current, 0.25)
    expected = 0.75 * np.array([0.6, 0.8]) + 0.25 * np.array([1, 0])
    np.testing.assert_allclose(updated["vector"], expected / np.linalg.norm(expected))
    np.testing.assert_array_equal(updated_appearance(value, None, 0.5)["vector"], value["vector"])


@pytest.mark.parametrize(
    "descriptor",
    [
        {},
        {"space": "", "vector": [1]},
        {"space": "x", "vector": []},
        {"space": "x", "vector": [0, 0]},
        {"space": "x", "vector": [True]},
        {"space": "x", "vector": [float("nan")]},
        {"space": "x", "vector": [1, 2, 3]},
    ],
)
def test_malformed_descriptors_reject_before_native_state(descriptor):
    with pytest.raises(ValidationError):
        unit_appearance(descriptor, 2)


@pytest.mark.parametrize("policy", ["off", "required"])
def test_occluded_position_swap_preserves_appearance_identity_and_exposes_motion_only_switch(
    policy,
):
    block, step = runner(appearance_policy=policy)
    for _ in range(3):
        result = step([row(40, [1, 0], identity="red"), row(140, [0, 1], identity="blue")])
    original = [r["track_id"] for r in result.outputs["tracks"]["items"]]
    step([])
    result = step([row(40, [0, 1], identity="blue"), row(140, [1, 0], identity="red")])
    items = result.outputs["tracks"]["items"]
    assert [r["track_id"] for r in items] == original
    assert [r["source_detection_id"] for r in items] == (
        ["red", "blue"] if policy == "required" else ["blue", "red"]
    )
    assert all(r["appearance_available"] is (policy == "required") for r in items)
    assert all(
        a["appearance_used"] is (policy == "required")
        for a in result.outputs["report"]["associations"]
    )
    assert all(r["class_id"] == 0 and not r["angle_available"] for r in items)
    block.close()


def test_feature_space_change_retires_history_then_valid_activation_recovers_with_new_generation():
    block, step = runner(appearance_policy="required")
    first = step([row(50, [1, 0])]).outputs["tracks"]["items"][0]["track_id"]
    assert step([row(51, [1, 0], "new-model")]) is None
    assert not block.current_emissions and block.state == "error"
    result = step([row(52, [1, 0], "new-model")])
    assert result.outputs["tracks"]["items"][0]["track_id"] != first


def test_required_if_both_and_off_missing_descriptor_semantics():
    block, step = runner(appearance_policy="required")
    assert step([row(20)]) is None and not block.current_emissions
    block, step = runner(appearance_policy="if_both")
    original = step([row(20, [1, 0])]).outputs["tracks"]["items"][0]["track_id"]
    result = step([row(21)])
    assert result.outputs["tracks"]["items"][0]["track_id"] == original
    assert result.outputs["tracks"]["items"][0]["appearance_available"]
    assert result.outputs["report"]["associations"][0]["appearance_distance"] is None
    block, step = runner(appearance_policy="off")
    bad = row(20)
    bad["appearance"] = {"unparsed": "off"}
    assert step([bad]) is not None


def test_class_ids_gate_even_equal_text_labels_and_geometry_gates_still_apply():
    _, step = runner(appearance_policy="required", association_distance=10)
    first = step([row(20, [1, 0])]).outputs["tracks"]["items"][0]["track_id"]
    other = row(21, [1, 0])
    other["class_id"] = 1
    result = step([other])
    assert not result.outputs["report"]["associations"]
    result = step([row(150, [1, 0])])
    assert not result.outputs["report"]["associations"]
    assert result.outputs["tracks"]["items"][0]["track_id"] == first


def test_histograms_match_independent_native_reference_and_preserve_pixels_records_context():
    args = boxes()
    before = copy.deepcopy(args["detections"].value)
    tool = DescribeBoxAppearance(parameters={"roi_inset": 0})
    result = tool.execute(args)
    assert result is not None, tool.error
    for item, x in zip(result.outputs["collection"]["items"], (0, 20), strict=True):
        hsv = cv2.cvtColor(args["image"].value[:, x : x + 20], cv2.COLOR_BGR2HSV)
        expected = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(item["appearance"]["vector"], expected)
        assert item["score"] == 0.9
    assert args["detections"].value == before
    np.testing.assert_array_equal(result.outputs["image"], args["image"].value)
    assert result.payloads["collection"]["ticket"] == args["image"].payload["ticket"]
    assert result.drawings


def test_rgb_tags_gray_histograms_alpha_masks_and_lean_are_respected():
    args = boxes()
    args["image"].value = ImageArray(args["image"].value[..., ::-1], "rgb")
    tool = DescribeBoxAppearance(parameters={"color_model": "gray", "roi_inset": 0})
    tool.run_profile = PROFILES["lean"]
    result = tool.execute(args)
    assert result is not None, tool.error
    # RGB red is luminance 76 -> bin 9; blue is 29 -> bin 3.
    assert [
        np.argmax(i["appearance"]["vector"]) for i in result.outputs["collection"]["items"]
    ] == [9, 3]
    assert not result.drawings
    rgba = np.dstack((np.asarray(args["image"].value), np.full((20, 40), 255, np.uint8)))
    rgba[:, :20, 3] = 0
    args["image"].value = ImageArray(rgba, "rgba")
    result = tool.execute(args)
    assert result.outputs["report"]["available"] == 1
    assert "appearance" not in result.outputs["collection"]["items"][0]
    args["mask"] = ValuePacket(np.zeros((20, 40), np.uint8), args["image"].payload)
    result = tool.execute(args)
    assert result.outputs["report"]["available"] == 0


@pytest.mark.parametrize("change", ["ticket", "frame", "depth", "box", "work", "support", "bins"])
def test_histogram_guards_and_native_failure_recovery(change, monkeypatch):
    args, parameters = boxes(), {}
    if change == "ticket":
        args["detections"].payload["ticket"] = TicketClock().issue("other")
    if change == "frame":
        args["detections"].payload["spatial"] = frame([40, 20], "other")
    if change == "depth":
        args["image"].value = args["image"].value.astype(np.uint16)
    if change == "box":
        args["detections"].value["items"][0]["geometry"]["bbox"][2] = -1
    if change == "work":
        parameters["maximum_sample_pixels"] = 1
    if change == "support":
        parameters.update(minimum_pixels=1000, insufficient="error")
    if change == "bins":
        parameters.update(hue_bins=64, saturation_bins=32)
    tool = DescribeBoxAppearance(parameters=parameters)
    assert tool.execute(args) is None and not tool.current_emissions
    tool = DescribeBoxAppearance()
    native = cv2.calcHist
    monkeypatch.setattr(cv2, "calcHist", lambda *a, **k: (_ for _ in ()).throw(cv2.error("native")))
    assert tool.execute(boxes()) is None and not tool.current_emissions
    monkeypatch.setattr(cv2, "calcHist", native)
    assert tool.execute(boxes()) is not None


def test_copyable_worker_state_is_private_and_registry_includes_descriptor():
    block, step = runner(appearance_policy="required")
    step([row(20, [1, 0])])
    clone = TrackImageParts(parameters=block.parameters)
    clone.copy_runtime_from(block)
    old = copy.deepcopy(clone.snapshot_runtime())
    step([row(21, [1, 0.2])])
    assert str(clone.snapshot_runtime()) == str(old)
    assert builtin_registry().create("ml.box_appearance").type_id == "ml.box_appearance"


def test_required_detector_appearance_does_not_fabricate_fresh_local_flow_features():
    tool = builtin_registry().create(
        "sequence.track_refreshed",
        parameters={
            "appearance_policy": "required",
            "minimum_observations": 1,
            "clock": "nominal",
        },
    )
    clock = TicketClock()
    texture = np.random.default_rng(21).integers(80, 240, (32, 40), np.uint8)
    for index in range(2):
        image = np.zeros((112, 224), np.uint8)
        image[30:62, 20 + index * 3 : 60 + index * 3] = texture
        context = {"spatial": frame([224, 112], "camera"), "ticket": clock.issue("camera")}
        values = {"image": ValuePacket(image, context), "refresh": ValuePacket(index == 0, context)}
        if index == 0:
            values["detections"] = ValuePacket(
                [
                    {
                        "id": "detector-row",
                        "center": [39.5, 45.5],
                        "size": [40, 32],
                        "appearance": {"space": "test", "vector": [1, 0]},
                    }
                ],
                context,
            )
        result = tool.execute(values)
        assert result is not None, tool.error
        if index == 1:
            found = result.outputs["tracks"]["items"][0]
            assert found["measurement_source"] == "local_flow" and found["appearance_available"]
            assert found["source_detection_id"].startswith("local_flow:")
            assert not result.outputs["report"]["associations"][0]["appearance_used"]


def test_appearance_controls_and_xml_are_available_without_native_work(app, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.core import Graph
    from opencv_dataflow.project_xml import export_project_xml, read_project_xml

    monkeypatch.setattr(cv2, "calcHist", lambda *a, **k: pytest.fail("native histogram"))
    graph = Graph()
    tracker = graph.add(TrackImageParts())
    descriptor = graph.add(DescribeBoxAppearance())
    window = MainWindow()
    try:
        window.replace_graph(graph)
        for block, fields in (
            (tracker, ("appearance_policy",)),
            (descriptor, ("color_model", "insufficient")),
        ):
            window.inspector.select(block)
            for name in fields:
                assert isinstance(window.inspector.fields[name], QComboBox)
        export_project_xml(graph.to_dict(), tmp_path / "project.xml")
        assert read_project_xml(tmp_path / "project.xml") == graph.to_dict()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
