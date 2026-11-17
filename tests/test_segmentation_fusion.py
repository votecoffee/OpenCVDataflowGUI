"""Independent pixel, overlap, context and bounded failure references for tile fusion."""

import copy

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.dnn_blocks import DNNInference, preprocess
from opencv_dataflow.images import ImageArray
from opencv_dataflow.ml_common import tensor_record
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame, moved_payload
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.segmentation_fusion import FuseTileSegmentation
from opencv_dataflow.semantic_segmentation import DecodeSegmentation
from opencv_dataflow.tickets import TicketClock


def arguments():
    image = ImageArray(np.full((4, 8, 3), 32768, np.uint16), "rgb")
    context = {
        "spatial": frame([8, 4], "camera"),
        "ticket": TicketClock().issue("camera"),
        "recipe_ref": {"id": "part", "revision": 1, "digest": "1" * 64},
        "inspection_notes": {"fixture": "A"},
    }
    packets = []
    for x, foreground in ((0, 0.8), (3, 0.4)):
        payload = moved_payload(context, [[1, 0, x], [0, 1, 0], [0, 0, 1]], [5, 4])
        scores = np.empty((2, 4, 5), np.float32)
        scores[0], scores[1] = 1 - foreground, foreground
        labels = scores.argmax(axis=0).astype(np.int32)
        data = {
            "model": "segmentation_tile_v1",
            "class_count": 2,
            "class_names": ["background", "part"],
            "labels": labels,
            "probabilities": scores,
            "context": copy.deepcopy(payload),
        }
        packets.append(ValuePacket(data, payload))
    return {
        "image": ValuePacket(image, context),
        "tiles": ValuePacket(
            FlowGroup(tuple(packets), ("left", "right")), {"ticket": context["ticket"]}
        ),
    }


def tiles(args):
    return args["tiles"].value.items


def run(args=None, **parameters):
    block = FuseTileSegmentation(parameters=parameters)
    result = block.execute(arguments() if args is None else args)
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("weighting", ["uniform", "feather"])
def test_overlap_has_independent_scalar_probability_and_edge_weight_reference(weighting):
    args = arguments()
    _, result = run(args, weighting=weighting, feather_width=2, export_segmentation=True)
    expected, confidence, weight, coverage = (np.zeros((4, 8)) for _ in range(4))
    for y in range(4):
        for x in range(8):
            samples = []
            for start, probability in ((0, 0.8), (3, 0.4)):
                u = x - start
                if 0 <= u < 5:
                    edge = min(u + 0.5, 4.5 - u, y + 0.5, 3.5 - y)
                    samples.append((probability, min(edge / 2, 1) if weighting == "feather" else 1))
            mean = sum(p * w for p, w in samples) / sum(w for _, w in samples)
            expected[y, x] = int(mean > 0.5)
            confidence[y, x] = max(mean, 1 - mean)
            weight[y, x] = sum(w for _, w in samples)
            coverage[y, x] = len(samples)
    np.testing.assert_array_equal(result.outputs["labels"], expected)
    np.testing.assert_allclose(result.outputs["confidence"], confidence, atol=1e-7)
    np.testing.assert_allclose(result.outputs["weight"], weight, atol=1e-7)
    np.testing.assert_array_equal(result.outputs["coverage"], coverage)
    assert result.outputs["report"]["coverage_fraction"] == 1
    assert result.outputs["segmentation"]["probabilities"].dtype == np.float32
    assert result.outputs["image"].color_format == "rgb"
    assert result.outputs["image"].dtype == np.uint16
    for payload in result.payloads.values():
        assert payload["spatial"] == args["image"].payload["spatial"]
        assert payload["ticket"] == args["image"].payload["ticket"]
        assert payload["recipe_ref"] == args["image"].payload["recipe_ref"]
        assert payload["inspection_notes"] == {"fixture": "A"}


def test_label_vote_is_explicit_and_ties_choose_lower_class():
    args = arguments()
    for packet in tiles(args):
        packet.value["probabilities"] = None
    _, result = run(args, mode="label_vote", export_segmentation=True)
    np.testing.assert_array_equal(
        result.outputs["labels"], np.tile([1, 1, 1, 0, 0, 0, 0, 0], (4, 1))
    )
    np.testing.assert_array_equal(result.outputs["confidence"][:, 3:5], 0.5)
    assert result.outputs["report"]["confidence_semantics"] == "weighted_vote_fraction"
    assert result.outputs["segmentation"]["probabilities"] is None
    block = FuseTileSegmentation()
    assert block.execute(args) is None and "full class planes" in block.error


@pytest.mark.parametrize(
    "matrix,shape",
    [
        ([[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]], (2, 4)),
        ([[-1, 0, 7], [0, 1, 0], [0, 0, 1]], (4, 8)),
        ([[0, 1, 0], [-1, 0, 3], [0, 0, 1]], (8, 4)),
    ],
)
def test_center_mapping_for_scale_flip_and_right_angle(matrix, shape):
    args = arguments()
    packet = tiles(args)[0]
    payload = moved_payload(args["image"].payload, matrix, list(shape[::-1]))
    local = np.indices(shape).sum(axis=0).astype(np.int32) % 2
    packet.value.update(labels=local, probabilities=None, context=copy.deepcopy(payload))
    packet.payload = payload
    args["tiles"].value = FlowGroup((packet,), ("only",))
    _, result = run(args, mode="label_vote")
    inverse = np.linalg.inv(np.asarray(matrix, float))
    expected = np.empty((4, 8), np.int32)
    for y in range(4):
        for x in range(8):
            u, v, _ = inverse @ [x, y, 1]
            expected[y, x] = local[
                min(int(np.floor(v + 0.5)), shape[0] - 1), min(int(np.floor(u + 0.5)), shape[1] - 1)
            ]
    np.testing.assert_array_equal(result.outputs["labels"], expected)


def test_masked_bilinear_support_does_not_bleed_ignored_class_scores():
    args = arguments()
    packet = tiles(args)[0]
    payload = moved_payload(args["image"].payload, [[2, 0, 0.5], [0, 4, 1.5], [0, 0, 1]], [4, 1])
    packet.payload = payload
    packet.value.update(
        context=copy.deepcopy(payload),
        labels=np.array([[1, -1, -1, 1]], np.int32),
        probabilities=np.array([[[0, 1, 1, 0]], [[1, 0, 0, 1]]], np.float32),
    )
    args["tiles"].value = FlowGroup((packet,), ("only",))
    _, result = run(args, minimum_coverage=0)
    np.testing.assert_array_equal(
        result.outputs["labels"], np.tile([1, 1, 1, -1, -1, 1, 1, 1], (4, 1))
    )
    assert result.outputs["report"]["coverage_fraction"] == 0.75
    assert np.all(result.outputs["confidence"][result.outputs["labels"] >= 0] == 1)


def test_uncovered_empty_roi_low_confidence_and_area_failure_keep_outputs():
    args = arguments()
    args["tiles"].value = FlowGroup((tiles(args)[0],), ("left",))
    _, result = run(args)
    assert not result.passed and result.outputs["report"]["coverage_fraction"] == 5 / 8
    assert result.fail_reasons[0]["code"] == "segmentation.coverage"
    assert np.all(result.outputs["labels"][:, 5:] == -1)
    assert result.drawings and result.graphics
    args["mask"] = np.zeros((4, 8), np.uint8)
    _, empty = run(args)
    assert not empty.passed and empty.outputs["report"]["coverage_fraction"] == 0
    assert empty.outputs["fraction"] == 0 and np.all(empty.outputs["labels"] == -1)
    _, gated = run(minimum_confidence=0.9)
    assert not gated.passed and gated.outputs["pixel_count"] == 0
    assert gated.outputs["report"]["coverage_fraction"] == 1


def test_reference_roi_is_evaluated_in_reference_frame_and_ignored_winners_are_not_reassigned():
    args = arguments()
    mask = np.zeros((4, 8), np.uint8)
    mask[:, :3] = 255
    args["mask"] = ValuePacket(mask, copy.deepcopy(args["image"].payload))
    _, result = run(args, ignored_class_ids=[0])
    assert result.passed and result.outputs["fraction"] == 1
    assert np.all(result.outputs["labels"][:, 3:] == -1)
    args["mask"].payload["spatial"]["root_id"] = "other"
    block = FuseTileSegmentation()
    assert block.execute(args) is None and not block.current_emissions


def test_failed_tile_is_reported_without_losing_current_partial_result():
    # Obtain an authentic runtime failure payload, rather than inventing its schema.
    _, failed = run(maximum_pixels=1)
    args = arguments()
    tiles(args)[1].payload["outcome"] = failed.payloads["image"]["outcome"]
    _, result = run(args)
    assert not result.passed and result.outputs["pixel_count"] > 0
    assert any(r["code"] == "segmentation.tile_failed" for r in result.fail_reasons)


@pytest.mark.parametrize(
    "change",
    [
        "ticket",
        "root",
        "root_size",
        "wire",
        "pyramid",
        "recipe",
        "classes",
        "nan",
        "probability",
        "labels",
        "shape",
        "affine",
        "count",
        "duplicates",
        "input_budget",
        "array_budget",
        "work_budget",
    ],
)
def test_bad_inputs_emit_nothing_and_recover_on_next_valid_ticket(change):
    args = arguments()
    packet = tiles(args)[1]
    parameters = {}
    if change == "ticket":
        packet.payload["ticket"] = TicketClock().issue("other")
    elif change in ("root", "root_size", "wire"):
        key, value = {
            "root": ("root_id", "other"),
            "root_size": ("root_size", [99, 4]),
            "wire": ("to_root", [[1, 0, 2], [0, 1, 0], [0, 0, 1]]),
        }[change]
        packet.payload["spatial"][key] = value
    elif change == "pyramid":
        packet.value["context"]["pyramid"] = {"custom": "mismatch"}
    elif change == "recipe":
        packet.value["context"]["recipe_ref"]["revision"] = 2
    elif change == "classes":
        packet.value["class_names"] = ["part", "background"]
    elif change in ("nan", "probability"):
        packet.value["probabilities"][0, 0, 0] = np.nan if change == "nan" else 0.9
    elif change == "labels":
        packet.value["labels"][0, 0] = 99
    elif change == "shape":
        packet.value["probabilities"] = np.zeros((2, 4, 4), np.float32)
    elif change == "affine":
        for payload in (packet.payload, packet.value["context"]):
            payload["spatial"]["to_root"] = [[1, 0.1, 3], [0, 1, 0], [0, 0, 1]]
    elif change == "count":
        parameters["expected_tiles"] = 3
    elif change == "duplicates":
        args["tiles"].value.labels = ("same", "same")
    else:
        parameters[
            {
                "input_budget": "maximum_input_elements",
                "array_budget": "maximum_resample_elements",
                "work_budget": "maximum_mapped_elements",
            }[change]
        ] = 1
    block = FuseTileSegmentation(parameters=parameters)
    assert block.execute(args) is None, change
    assert block.state == "error" and not block.current_emissions
    block.parameters = copy.deepcopy(block.defaults)
    assert block.execute(arguments()) is not None, block.error


@pytest.mark.parametrize("run_profile", ["debug", "lean"])
def test_graph_json_xml_and_run_profiles(tmp_path, run_profile):
    registry = builtin_registry()
    graph = Graph()
    graph.add(registry.create("ml.fuse_tile_segmentation", block_id="fuse"))
    graph.set_run_profile(run_profile)
    try:
        path = tmp_path / "graph.json"
        graph.save(path)
        export_project_xml(graph.to_dict(), tmp_path / "graph.xml")
        for data in (
            Graph.load(path, registry),
            Graph.from_dict(read_project_xml(tmp_path / "graph.xml"), registry),
        ):
            try:
                restored = data.blocks["fuse"]
                restored.run_profile = PROFILES[run_profile]
                result = restored.execute(arguments())
                assert result is not None, restored.error
                assert bool(result.drawings) == (run_profile == "debug")
                assert result.outputs["segmentation"] is NO_VALUE
            finally:
                data.close()
    finally:
        graph.close()


@pytest.mark.parametrize(
    "kind,activation,resampling",
    [
        (kind, activation, resampling)
        for kind, activation in (
            ("multiclass", "softmax"),
            ("multiclass", "probability"),
            ("binary", "sigmoid"),
            ("class_ids", "probability"),
        )
        for resampling in ("labels_nearest", "scores_linear")
        if not (kind == "class_ids" and resampling == "scores_linear")
    ],
)
def test_decoder_export_is_opt_in_and_keeps_real_class_planes(kind, activation, resampling):
    image = np.zeros((4, 8, 3), np.uint8)
    _, geometry = preprocess({"image": image}, {**DNNInference.defaults, "width": 8, "height": 4})
    values = np.array([[[[0.2, 0.8]], [[0.8, 0.2]]]], np.float32)
    if kind == "binary":
        values = np.array([[[[-2, 2]]]], np.float32)
    if kind == "class_ids":
        values = np.array([[[[1, 0]]]], np.float32)
    context = {"spatial": frame([8, 4], "camera"), "ticket": TicketClock().issue("camera")}
    args = {
        "image": image,
        "tensors": {"head": tensor_record(values, 100)},
        "preprocessing": geometry,
    }
    args = {key: ValuePacket(value, copy.deepcopy(context)) for key, value in args.items()}
    parameters = dict(
        format=kind, score_activation=activation, class_count=2, resampling=resampling
    )
    default = DecodeSegmentation(parameters=parameters)
    result = default.execute(args)
    assert result is not None, default.error
    assert result.outputs["segmentation"] is NO_VALUE
    export = DecodeSegmentation(parameters={**parameters, "export_segmentation": True})
    result = export.execute(args)
    assert result is not None, export.error
    record = result.outputs["segmentation"]
    np.testing.assert_array_equal(record["labels"], result.outputs["labels"])
    assert record["context"]["spatial"] == context["spatial"]
    if kind == "class_ids":
        assert record["probabilities"] is None
    else:
        scores = record["probabilities"]
        assert scores.dtype == np.float32 and scores.shape == (2, 4, 8)
        np.testing.assert_allclose(scores.sum(axis=0), 1, atol=1e-7)
        np.testing.assert_array_equal(scores.argmax(axis=0), result.outputs["labels"])
    export.parameters["maximum_resample_elements"] = 1
    assert export.execute(args) is None and not export.current_emissions
