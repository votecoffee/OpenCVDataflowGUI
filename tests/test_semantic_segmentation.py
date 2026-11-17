"""Pixel-grid references, score semantics, selection and normal failure containment."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.dnn_blocks import DNNInference, preprocess
from opencv_dataflow.images import ImageArray
from opencv_dataflow.ml_common import selected_tensor, tensor_record
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.selection import rasterize
from opencv_dataflow.semantic_segmentation import DecodeSegmentation
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.timing import advance_timing, mark

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/semantic_segmentation"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


def inputs(values=None, *, shape=(7, 12), mode="stretch", layout="NCHW"):
    source = np.full((*shape, 3), [50, 100, 200], np.uint8)
    _, geometry = preprocess(
        {"image": source},
        {
            **DNNInference.defaults,
            "width": 8,
            "height": 8,
            "resize_mode": mode,
        },
    )
    if values is None:
        values = np.array([[[4, 4, 0, 0], [4, 1, 0, 0]], [[0, 0, 4, 4], [0, 1, 4, 4]]], np.float32)
    if layout in ("NHWC", "HWC"):
        values = values.transpose(1, 2, 0)
    if layout.startswith("N"):
        values = values[None]
    return {
        "image": source,
        "tensors": {"head": tensor_record(values, 8_000_000)},
        "preprocessing": geometry,
    }


def decode(args=None, **parameters):
    block = DecodeSegmentation(parameters=parameters)
    result = block.execute(args or inputs())
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("layout", ["NCHW", "NHWC", "CHW", "HWC"])
@pytest.mark.parametrize("mode", ["stretch", "letterbox"])
def test_source_labels_have_exact_independent_half_pixel_edge_reference(layout, mode):
    args = inputs(mode=mode, layout=layout)
    _, result = decode(args, layout=layout, labels=["background", "part"])
    # 12x7 letterboxes to 8x5, with centered top padding 1; x8/y8 head covers full network.
    xs = (np.arange(12) + 0.5) / 12 * 4
    ys = (np.arange(7) + 0.5) * (5 / 7 if mode == "letterbox" else 8 / 7)
    ys = (ys + (1 if mode == "letterbox" else 0)) * 2 / 8
    grid = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], np.int32)
    expected = grid[np.floor(ys).astype(int)[:, None], np.floor(xs).astype(int)]
    np.testing.assert_array_equal(result.outputs["labels"], expected)
    np.testing.assert_array_equal(result.outputs["mask"], expected * 255)
    assert result.outputs["pixel_count"] == np.count_nonzero(expected)
    assert result.outputs["fraction"] == pytest.approx(np.mean(expected))
    np.testing.assert_array_equal(result.outputs["image"], args["image"])
    assert result.outputs["labels"].dtype == np.int32
    assert result.outputs["mask"].dtype == np.uint8
    assert result.outputs["confidence"].dtype == np.float32


def test_rounded_asymmetric_letterbox_padding_does_not_become_source_foreground():
    # This head has a foreground-only padding row above and below the content.
    values = np.zeros((2, 8, 8), np.float32)
    values[0] = 4
    values[1, :1] = 8
    values[1, 6:] = 8
    _, result = decode(inputs(values, mode="letterbox"), minimum_pixels=0)
    assert not np.any(result.outputs["labels"])
    assert result.passed and result.outputs["pixel_count"] == 0


def test_linear_scores_before_argmax_differs_from_nearest_winner_and_matches_scalar_reference():
    logits = np.array([[[1, 0]], [[0, 8]]], np.float32)
    args = inputs(logits, shape=(1, 8))
    _, nearest = decode(args)
    _, linear = decode(args, resampling="scores_linear")
    np.testing.assert_array_equal(nearest.outputs["labels"], [[0, 0, 0, 0, 1, 1, 1, 1]])
    # x=(x+.5)*2/8-.5; OpenCV linear phases are multiples of 1/32.
    expected, confidence = [], []
    for x in range(8):
        t = min(max((x + 0.5) * 2 / 8 - 0.5, 0), 1)
        a, b = 1 - t, 8 * t
        winner = int(b > a)
        expected.append(winner)
        confidence.append(1 / (1 + np.exp(-abs(a - b))))
    np.testing.assert_array_equal(linear.outputs["labels"], [expected])
    np.testing.assert_allclose(linear.outputs["confidence"], [confidence], atol=1e-7)
    assert not np.array_equal(nearest.outputs["labels"], linear.outputs["labels"])


@pytest.mark.parametrize("activation", ["probability", "sigmoid"])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC", "HW", "NHW"])
def test_binary_foreground_threshold_and_background_winning_confidence(activation, layout):
    probs = np.array([[0.1, 0.4, 0.5, 0.8]], np.float32)
    values = probs if activation == "probability" else np.log(probs / (1 - probs))
    args = inputs(values if layout in ("HW", "NHW") else values[None], shape=(1, 4), layout=layout)
    _, result = decode(args, format="binary", layout=layout, score_activation=activation)
    np.testing.assert_array_equal(result.outputs["labels"], [[0, 0, 1, 1]])
    np.testing.assert_allclose(result.outputs["confidence"], [[0.9, 0.6, 0.5, 0.8]], atol=1e-7)
    assert result.outputs["pixel_count"] == 2
    _, threshold = decode(
        args,
        format="binary",
        layout=layout,
        score_activation=activation,
        binary_threshold=0.9,
        minimum_pixels=0,
    )
    assert threshold.outputs["pixel_count"] == 0


def test_softmax_extreme_logits_are_stable_and_probability_sum_is_checked():
    maximum = np.finfo(np.float32).max
    _, result = decode(
        inputs(np.array([[[maximum]], [[-maximum]]], np.float32)), selected_class_ids=[0]
    )
    assert np.all(result.outputs["labels"] == 0)
    assert np.all(result.outputs["confidence"] == 1)
    args = inputs(np.array([[[0.2]], [[0.6]]], np.float32))
    block = DecodeSegmentation(parameters={"score_activation": "probability"})
    assert block.execute(args) is None and "sum to one" in block.error
    args["tensors"]["head"]["values"] = [0.2, 0.8]
    assert block.execute(args) is not None and block.state == "pass"


def test_ignored_winners_stay_ignored_and_do_not_choose_runners_up():
    args = inputs(np.array([[[8, 0, 0]], [[7, 8, 0]], [[0, 7, 8]]]), shape=(1, 3))
    _, result = decode(args, selected_class_ids=[], ignored_class_ids=[0, 1])
    np.testing.assert_array_equal(result.outputs["labels"], [[-1, -1, 2]])
    np.testing.assert_array_equal(result.outputs["mask"], [[0, 0, 255]])
    np.testing.assert_array_equal(result.outputs["confidence"][:, :2], 0)
    assert result.outputs["report"]["ignored_pixels_in_roi"] == 2
    assert result.outputs["fraction"] == pytest.approx(1 / 3)
    assert result.outputs["report"]["classes"][2]["pixels"] == 1


def test_confidence_gating_masks_and_polygon_region_intersect_in_source_grid():
    args = inputs()
    args["mask"] = np.full((7, 12), 255, np.uint8)
    args["mask"][:2] = 0
    args["region"] = {
        "model": "image_region_v1",
        "shape": "polygon",
        "image_size": [12, 7],
        "vertices": [[2, 1], [10, 1], [8, 6], [2, 6]],
    }
    eligible = (rasterize(args["region"], (7, 12)) != 0) & (args["mask"] != 0)
    _, original = decode()
    _, result = decode(args, minimum_confidence=0.9)
    expected = original.outputs["labels"].copy()
    expected[~eligible | (original.outputs["confidence"] < 0.9)] = -1
    np.testing.assert_array_equal(result.outputs["labels"], expected)
    assert result.outputs["report"]["roi_pixels"] == np.count_nonzero(eligible)
    assert result.outputs["fraction"] == np.count_nonzero(expected == 1) / np.count_nonzero(
        eligible
    )


def test_empty_roi_and_valid_area_failure_keep_current_outputs_and_reasons():
    args = inputs()
    args["mask"] = np.zeros((7, 12), np.uint8)
    args["image"] = ValuePacket(args["image"], {"ticket": TicketClock().issue("source")})
    block, result = decode(args)
    assert not result.passed and block.state == "fail"
    assert result.outputs["pixel_count"] == 0 and result.outputs["fraction"] == 0
    assert result.fail_reasons[0]["code"] == "segmentation.area"
    assert np.all(result.outputs["labels"] == -1)
    assert all(p["outcome"]["passed"] is False for p in result.payloads.values())
    assert block.current_emissions
    _, allowed = decode(args, minimum_pixels=0)
    assert allowed.passed
    _, too_large = decode(maximum_fraction=0.1)
    assert not too_large.passed and too_large.outputs["pixel_count"] > 0


@pytest.mark.parametrize("layout", ["NCHW", "NHWC", "HW", "NHW"])
def test_class_id_head_keeps_zero_and_unknown_without_inventing_confidence(layout):
    values = np.array([[0, 1, -1, 2]], np.float32)
    args = inputs(values if layout in ("HW", "NHW") else values[None], shape=(1, 4), layout=layout)
    _, result = decode(
        args, format="class_ids", class_count=3, layout=layout, selected_class_ids=[]
    )
    np.testing.assert_array_equal(result.outputs["labels"], values)
    assert result.outputs["confidence"] is NO_VALUE and result.score is None
    assert result.outputs["pixel_count"] == 3
    assert not result.outputs["report"]["confidence_available"]
    assert all(c["mean_confidence"] is None for c in result.outputs["report"]["classes"])


@pytest.mark.parametrize(
    "parameters,field",
    [
        ({"selected_class_ids": [True]}, "selected_class_ids"),
        ({"ignored_class_ids": [1, 1]}, "ignored_class_ids"),
        ({"selected_class_ids": [-1]}, "selected_class_ids"),
        ({"ignored_class_ids": [256]}, "ignored_class_ids"),
        ({"labels": [""]}, "labels"),
        ({"minimum_confidence": 1.1}, "minimum_confidence"),
        ({"format": "binary"}, "score_activation"),
        ({"score_activation": "sigmoid"}, "score_activation"),
        ({"format": "class_ids"}, "class_count"),
        (
            {"format": "class_ids", "class_count": 2, "minimum_confidence": 0.1},
            "minimum_confidence",
        ),
        ({"format": "class_ids", "class_count": 2, "resampling": "scores_linear"}, "resampling"),
        ({"minimum_pixels": 2, "maximum_pixels": 1}, "maximum_pixels"),
        ({"minimum_fraction": 0.6, "maximum_fraction": 0.5}, "minimum_fraction"),
    ],
)
def test_configuration_errors_identify_fields_before_execution(parameters, field):
    block = DecodeSegmentation(parameters=parameters)
    with pytest.raises(ValidationError) as invalid:
        block.validate_parameters()
    assert field in invalid.value.fields
    assert not block.last_inputs and block.last_result is None


@pytest.mark.parametrize(
    "parameters,fragment",
    [
        ({"layout": "CHW"}, "dimensions"),
        ({"class_count": 3}, "Class count"),
        ({"labels": ["only"]}, "one label"),
        ({"selected_class_ids": [2]}, "class range"),
        ({"output_name": "missing"}, "missing"),
        ({"maximum_tensor_elements": 1}, "tensor elements"),
        ({"maximum_image_pixels": 1}, "pixel/axis"),
        ({"resampling": "scores_linear", "maximum_resample_elements": 1}, "resampling budget"),
    ],
)
def test_invalid_heads_and_work_budgets_emit_nothing(parameters, fragment):
    block = DecodeSegmentation(parameters=parameters)
    assert block.execute(inputs()) is None
    assert fragment in block.error
    assert not block.current_emissions


def test_context_ticket_recipe_pyramid_timing_and_original_rgb_depth_are_preserved():
    args = inputs()
    args["image"] = ImageArray(args["image"].astype(np.uint16) * 257, "rgb")
    ticket = TicketClock().issue("camera", "fixed", 42)
    spatial = frame([12, 7], "lens")
    spatial["root_size"], spatial["to_root"] = [24, 14], [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    context = {
        "ticket": ticket,
        "spatial": spatial,
        "pyramid": {
            "level": 1,
            "largest_size": [24, 14],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
    }
    context["processing_time"] = advance_timing(None, mark(), mark(), source=True)
    context["recipe_ref"] = {"id": "part", "revision": 1, "digest": "1" * 64}
    wrapped = {k: ValuePacket(v, copy.deepcopy(context)) for k, v in args.items()}
    _, result = decode(wrapped)
    assert result.outputs["image"].color_format == "rgb"
    np.testing.assert_array_equal(result.outputs["image"], args["image"])
    for payload in result.payloads.values():
        assert payload["ticket"] == ticket and payload["spatial"] == spatial
        assert payload["pyramid"] == context["pyramid"]
        assert payload["recipe_ref"] == context["recipe_ref"]
        assert payload["processing_time"]["started_at"] == context["processing_time"]["started_at"]
    assert result.outputs["report"]["context_checks"] == {
        "ticket_checked": True,
        "frame_checked": True,
    }
    bad = copy.deepcopy(wrapped)
    bad["tensors"].payload["ticket"] = TicketClock().issue("other", "fixed", "different")
    block = DecodeSegmentation()
    assert block.execute(bad) is None and "different acquisition" in block.error
    assert not block.current_emissions
    assert block.execute(wrapped) is not None


@pytest.mark.parametrize("change", ["frame", "recipe"])
def test_context_conflicts_are_rejected_and_a_valid_followup_recovers(change):
    args = inputs()
    payload = {
        "spatial": frame([12, 7], "camera"),
        "recipe_ref": {"id": "part", "revision": 1, "digest": "1" * 64},
    }
    wrapped = {k: ValuePacket(v, copy.deepcopy(payload)) for k, v in args.items()}
    if change == "frame":
        wrapped["tensors"].payload["spatial"]["root_id"] = "other"
    else:
        wrapped["tensors"].payload["recipe_ref"]["revision"] = 2
    block = DecodeSegmentation()
    assert block.execute(wrapped) is None and not block.current_emissions
    assert block.execute(args) is not None


def test_native_exception_emits_nothing_and_recovers_and_lean_skips_all_optional_graphics(
    monkeypatch,
):
    block, good = decode()
    assert good.drawings and good.graphics
    native = cv2.remap
    block.parameters["resampling"] = "scores_linear"

    def broken(*args, **kwargs):
        raise cv2.error("controlled remap failure")

    monkeypatch.setattr(cv2, "remap", broken)
    assert block.execute(inputs()) is None and "controlled remap failure" in block.error
    assert not block.current_emissions
    monkeypatch.setattr(cv2, "remap", native)
    assert block.execute(inputs()) is not None
    lean = DecodeSegmentation()
    lean.run_profile = profile("lean")
    result = lean.execute(inputs())
    assert result is not None, lean.error
    assert not result.drawings and not result.graphics


def test_graphics_budget_does_not_truncate_numerical_mask_and_metadata_are_complete():
    _, full = decode()
    _, limited = decode(graphics_maximum_points=1)
    assert limited.outputs["report"]["graphics_truncated"]
    np.testing.assert_array_equal(limited.outputs["mask"], full.outputs["mask"])
    block = builtin_registry().create("ml.decode_segmentation")
    block.validate_parameters()
    assert block.parameter_help.keys() == block.defaults.keys()
    assert block.preview_safe and block.parallel_safe and block.process_inputs_readonly


def test_tensor_reader_keeps_old_default_limit_but_supports_explicit_image_head_budget():
    tensor = tensor_record(np.zeros((1, 3, 820, 820), np.float32), 3_000_000)
    with pytest.raises(ValueError):
        selected_tensor({"head": tensor})
    assert selected_tensor({"head": tensor}, max_elements=3_000_000).shape == (1, 3, 820, 820)
    # Exercise the full executor too: image heads commonly exceed the vector limit.
    values = np.zeros((3, 820, 820), np.float32)
    values[1] = 1
    _, result = decode(inputs(values, shape=(1, 1)))
    assert result.outputs["pixel_count"] == 1 and result.passed
    assert result.outputs["confidence"][0, 0] == pytest.approx(np.e / (np.e + 2))


@pytest.mark.parametrize(
    "mutation,fragment",
    [
        ({"shape": [2, 1, 2, 4]}, "batch dimension"),
        ({"shape": [1, 2, 2, 5]}, "shape"),
        ({"shape": [1, 2, 0, 4]}, "shape"),
        ({"dtype": "uint8"}, "dequantized"),
        ({"values": []}, "tensor elements"),
        ({"values": [float("nan")] * 16}, "JSON compliant"),
        ({"values": [float("inf")] * 16}, "JSON compliant"),
        ({"values": ["not numeric"] * 16}, "float"),
    ],
)
def test_malformed_serialized_tensors_emit_nothing(mutation, fragment):
    args = inputs()
    args["tensors"]["head"].update(mutation)
    block = DecodeSegmentation()
    assert block.execute(args) is None, block.error
    assert fragment in block.error
    assert not block.current_emissions


@pytest.mark.parametrize("bad", [-2, 2, 0.5])
def test_invalid_predecoded_ids_are_rejected(bad):
    args = inputs(np.array([[bad]], np.float32), layout="HW")
    block = DecodeSegmentation(parameters={"format": "class_ids", "class_count": 2, "layout": "HW"})
    assert block.execute(args) is None and "integer IDs" in block.error


def test_probability_invalid_padding_is_not_hidden_by_source_grid_sampling():
    values = np.zeros((2, 8, 8), np.float32)
    values[1] = 1
    values[1, 0] = 1.1  # Top padding is not sampled by the 12x7 source.
    block = DecodeSegmentation(
        parameters={"score_activation": "probability", "resampling": "scores_linear"}
    )
    assert block.execute(inputs(values, mode="letterbox")) is None
    assert "[0,1]" in block.error


def test_invalid_preprocessing_maps_and_missing_source_coverage_are_rejected():
    args = inputs()
    block = DecodeSegmentation()
    args["preprocessing"]["input_size"] = [11, 7]
    assert block.execute(args) is None and "sizes" in block.error
    args = inputs()
    args["preprocessing"]["input_to_network"][0][2] = 3
    args["preprocessing"]["network_to_input"][0][2] = -4.5
    assert block.execute(args) is None and "outside the network" in block.error


@pytest.mark.parametrize("profile_name", ["debug", "lean"])
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_native_example_labels_counts_confidence_mask_blobs_and_filmstrip(case, profile_name):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile_name)
    try:
        graph.run()
        result = graph.blocks["decode"].last_result
        assert result is not None, graph.blocks["decode"].error
        expected = TRUTH["cases"][case]
        raw = cv2.imread(str(BASE / "assets" / expected["expected_labels"]), cv2.IMREAD_UNCHANGED)
        labels = np.where(raw == 255, -1, raw.astype(np.int32))
        np.testing.assert_array_equal(result.outputs["labels"], labels)
        np.testing.assert_array_equal(result.outputs["mask"], (labels >= 0).astype(np.uint8) * 255)
        assert result.passed is expected["passed"]
        assert result.outputs["pixel_count"] == expected["selected_pixels"]
        assert result.outputs["fraction"] == expected["selected_pixels"] / expected["roi_pixels"]
        assert [c["pixels"] for c in result.outputs["report"]["classes"]] == expected[
            "class_counts"
        ]
        original = graph.blocks["source"].last_result.outputs["image"]
        np.testing.assert_array_equal(result.outputs["image"], original)
        assert (
            result.payloads["mask"]["ticket"]
            == graph.blocks["source"].last_result.outputs["ticket"]
        )
        if case == "class_ids":
            assert result.outputs["confidence"] is NO_VALUE
        else:
            assert np.isfinite(result.outputs["confidence"]).all()
            assert np.all(result.outputs["confidence"][labels < 0] == 0)
        overlay = graph.blocks["overlay"].last_result.outputs["image"]
        np.testing.assert_array_equal(
            overlay, graph.blocks["filmstrip"].last_result.outputs["image"]
        )
        if profile_name == "debug":
            assert result.drawings and result.graphics
            assert np.any(overlay != original)
        else:
            assert not result.drawings and not result.graphics
        if case == "multiclass":
            assert graph.blocks["blobs"].last_result.outputs["count"] > 0
    finally:
        graph.close()


def test_examples_assets_json_xml_save_as_and_independent_regeneration(tmp_path):
    before = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    graph = Graph.load(BASE / "multiclass.json", builtin_registry())
    try:
        graph.save(tmp_path / "saved.json")
        reopened = Graph.load(tmp_path / "saved.json", builtin_registry())
        try:
            reopened.run()
            assert reopened.blocks["decode"].last_result.passed
            assert (
                reopened.blocks["decode"].last_result.outputs["pixel_count"]
                == TRUTH["cases"]["multiclass"]["selected_pixels"]
            )
        finally:
            reopened.close()
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
    spec = importlib.util.spec_from_file_location("segmentation_example", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path / "regenerated"
    module.HERE.mkdir()
    module.build()
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((module.HERE / "assets" / name).read_bytes()).hexdigest() == digest
    for path in BASE.glob("*.json"):
        assert json.loads((module.HERE / path.name).read_bytes()) == json.loads(path.read_bytes())
    assert all(
        hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
        for name, digest in before.items()
    )
    (module.HERE / "assets/color_segmentation.onnx").write_bytes(b"existing model")
    with pytest.raises(FileExistsError):
        module.build()
