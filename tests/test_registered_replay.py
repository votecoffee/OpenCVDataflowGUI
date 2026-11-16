"""Registered replay pixel, validity, native-placement and provenance contracts."""

import copy
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from opencv_dataflow import registered_replay
from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.mosaic_layout import MODEL, seal_layout
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.tickets import TicketClock

IDS = ("left", "middle", "right")
KINDS = ("gray8", "rgb8", "bgr8", "gray16", "rgba16", "gray32", "rgb64")


def fixture(kind="gray8", order=(0, 1, 2), masks=True):
    clock = TicketClock()
    y, x = np.indices((32, 72))
    base = ((x * 3 + y * 5) % 150 + 30).astype(np.uint8)
    packets, selections, images, validity = [], [], [], []
    for i, name in enumerate(IDS):
        image = base[:, i * 12 : i * 12 + 48].copy() + i * 3
        encoding = "gray"
        if kind.startswith(("rgb", "bgr")):
            image = np.dstack((image, image + 1, image + 2))
            encoding = "rgb" if kind.startswith("rgb") else "bgr"
            if kind == "rgba16":
                image = np.dstack((image, np.full(image.shape[:2], 255, np.uint8)))
                encoding = "rgba"
        dtype = {"16": np.uint16, "32": np.float32, "64": np.float64}.get(kind[-2:], np.uint8)
        image = image.astype(dtype)
        if kind.endswith("16"):
            image *= 200
        elif dtype != np.uint8:
            image /= 123
        mask = np.full((32, 48), 255, np.uint8)
        if masks:
            mask[10:16, 28 - i * 8 : 34 - i * 8] = 0
        context = {
            "ticket": clock.issue(name, "external", name + "-current"),
            "spatial": frame([48, 32], "camera-" + name),
            "source_id": name,
            "timestamp_ns": 1000 + i * 100,
            "custom": {"index": i},
        }
        images.append(ImageArray(image, encoding))
        validity.append(mask)
        packets.append(ValuePacket(images[-1], context))
        selections.append(ValuePacket(mask, {"spatial": context["spatial"]}))
    layout = seal_layout(
        {
            "model": MODEL,
            "image_size": [72, 32],
            "source_identity": {"mode": "payload", "field": "source_id"},
            "views": [
                {
                    "index": i,
                    "label": name,
                    "source_id": name,
                    "source_size": [48, 32],
                    "source_dtype": images[i].dtype.name,
                    "source_format": images[i].color_format,
                    "source_to_mosaic": [[1, 0, i * 12], [0, 1, 0], [0, 0, 1]],
                    "source_spatial": packets[i].payload["spatial"],
                    "source_ticket": None,
                }
                for i, name in enumerate(IDS)
            ],
        }
    )
    batch = FlowGroup(tuple(packets[i] for i in order), tuple("Display " + IDS[i] for i in order))
    mask_group = FlowGroup(tuple(selections[i] for i in order), batch.labels)
    return {"views": batch, "layout": layout, "masks": mask_group}, images, validity


def oracle(images, masks, order, crop):
    frames, valid = [], []
    for i in order:
        image = np.zeros((32, 72, *images[i].shape[2:]), images[i].dtype)
        mask = np.zeros((32, 72), bool)
        image[:, i * 12 : i * 12 + 48] = images[i]
        mask[:, i * 12 : i * 12 + 48] = masks[i] != 0
        image[~mask] = 0
        frames.append(image)
        valid.append(mask)
    union, common = np.logical_or.reduce(valid), np.logical_and.reduce(valid)
    bounds = [0, 0, 72, 32]
    if crop != "canvas":
        ys, xs = np.where(common if crop == "intersection" else union)
        if not len(xs):
            return None
        bounds = [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]
    x, y, w, h = bounds
    common = common[y : y + h, x : x + w]
    for j in range(len(frames)):
        frames[j] = frames[j][y : y + h, x : x + w].copy()
        valid[j] = valid[j][y : y + h, x : x + w].copy()
        if crop == "intersection":
            valid[j] &= common
        frames[j][~valid[j]] = 0
    return frames, valid, common, list(map(int, bounds))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("crop", ["canvas", "union", "intersection"])
def test_preserve_depth_format_exact_pixels_masks_and_member_context(kind, crop):
    inputs, images, masks = fixture(kind, (2, 0, 1))
    original = copy.deepcopy(inputs)
    block = builtin_registry().create(
        "cv.replay_registered_views",
        parameters={"crop": crop, "interpolation": "nearest", "preview_frame": 1},
    )
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    expected = oracle(images, masks, (2, 0, 1), crop)
    outputs, coverage, common, bounds = expected
    for j, (packet, support) in enumerate(
        zip(result.outputs["frames"].items, result.outputs["coverage"].items, strict=True)
    ):
        i = (2, 0, 1)[j]
        np.testing.assert_array_equal(packet.value, outputs[j])
        np.testing.assert_array_equal(support.value, coverage[j].astype(np.uint8) * 255)
        assert (
            packet.value.dtype == images[i].dtype
            and packet.value.color_format == images[i].color_format
        )
        payload = packet.payload
        assert payload["ticket"] == inputs["views"].items[j].payload["ticket"]
        assert payload["timestamp_ns"] == 1000 + i * 100 and payload["custom"] == {"index": i}
        assert payload["outcome"]["passed"] and payload["outcome"]["source"]["id"] == block.id
        assert payload["spatial"]["root_id"] == "mosaic:" + inputs["layout"]["layout_id"]
        assert payload["spatial"]["root_size"] == [72, 32]
        np.testing.assert_array_equal(
            payload["spatial"]["to_root"], [[1, 0, bounds[0]], [0, 1, bounds[1]], [0, 0, 1]]
        )
        record = payload["registered_replay"]
        np.testing.assert_array_equal(
            record["source_to_frame"], [[1, 0, i * 12 - bounds[0]], [0, 1, -bounds[1]], [0, 0, 1]]
        )
        assert record["source_spatial"] == inputs["views"].items[j].payload["spatial"]
        assert record["source_id"] == IDS[i]
        assert not np.shares_memory(packet.value, inputs["views"].items[j].value)
    np.testing.assert_array_equal(result.outputs["common"], common.astype(np.uint8) * 255)
    np.testing.assert_array_equal(result.outputs["image"], outputs[1])
    for left, right in zip(original["views"].items, inputs["views"].items, strict=True):
        np.testing.assert_array_equal(left.value, right.value)
        assert left.payload == right.payload


@pytest.mark.parametrize("kind", ["gray8", "rgb8", "bgr8"])
@pytest.mark.parametrize("crop", ["canvas", "union", "intersection"])
def test_native_detail_placement_matches_independent_reference_and_masks(kind, crop):
    inputs, images, masks = fixture(kind, (2, 1))
    block = builtin_registry().create(
        "cv.replay_registered_views",
        parameters={
            "backend": "native_detail",
            "missing_views": "allow",
            "crop": crop,
            "interpolation": "nearest",
        },
    )
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    expected, coverage, common, bounds = oracle(images, masks, (2, 1), crop)
    for packet, image in zip(result.outputs["frames"].items, expected, strict=True):
        np.testing.assert_array_equal(packet.value, image)
    np.testing.assert_array_equal(result.outputs["common"], common.astype(np.uint8) * 255)
    assert result.outputs["info"]["origin"] == bounds[:2]
    assert result.outputs["info"]["source_matching"]["missing_source_ids"] == ["left"]


def test_native_mask_argument_is_ignored_by_this_wheel_and_wrapper_enforces_it():
    native = cv2.detail.Timelapser.createDefault(cv2.detail.Timelapser_AS_IS)
    native.initialize([(0, 0)], [(8, 8)])
    source = np.full((8, 8, 3), 173, np.int16)
    native.process(source, np.zeros((8, 8), np.uint8), (0, 0))
    np.testing.assert_array_equal(native.getDst().get(), source)
    native.process(np.full_like(source, 47), np.ones((8, 8), np.uint8), (0, 0))
    assert np.all(native.getDst().get() == 47)


@pytest.mark.parametrize("kind", ["gray16", "rgba16", "gray32", "rgb64"])
def test_native_rejects_inexact_types_before_any_warp(kind, monkeypatch):
    inputs, _, _ = fixture(kind)
    monkeypatch.setattr(
        registered_replay, "warp_mosaic_view", lambda *a: pytest.fail("Entered warp")
    )
    block = builtin_registry().create(
        "cv.replay_registered_views", parameters={"backend": "native_detail"}
    )
    assert block.execute(inputs) is None and block.error_kind == "invalid"
    assert not any(block.current_emissions.values())


def test_no_common_support_is_failed_with_no_frame_events_and_no_native_initialization(monkeypatch):
    inputs, _, _ = fixture(order=(0, 2))
    inputs["masks"].items[0].value[:, 24:] = 0
    block = builtin_registry().create(
        "cv.replay_registered_views",
        parameters={"backend": "native_detail", "crop": "intersection", "missing_views": "allow"},
    )
    monkeypatch.setattr(
        registered_replay, "native_timelapser", lambda *a: pytest.fail("Empty native crop")
    )
    result = block.execute(inputs)
    assert result is not None and not result.passed and block.state == "fail"
    assert result.fail_reasons[0]["code"] == "replay_no_common_support"
    assert (
        result.outputs["frames"]
        is result.outputs["coverage"]
        is result.outputs["image"]
        is NO_VALUE
    )
    assert not block.current_emissions["frames"] and result.outputs["info"]["common_pixels"] == 0


@pytest.mark.parametrize("backend", ["preserve", "native_detail"])
def test_empty_selected_source_stays_blank_without_reusing_previous_frame(backend):
    inputs, _, _ = fixture()
    inputs["masks"].items[1].value[:] = 0
    block = builtin_registry().create(
        "cv.replay_registered_views", parameters={"backend": backend, "crop": "union"}
    )
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    frames = result.outputs["frames"].items
    assert frames[0].value.any() and not frames[1].value.any() and frames[2].value.any()
    assert result.outputs["info"]["frames"][1]["valid_pixels"] == 0


def test_missing_policies_and_calibrated_order_keep_current_tickets():
    inputs, _, _ = fixture(order=(2, 1))
    block = builtin_registry().create("cv.replay_registered_views")
    assert block.execute(inputs) is None and block.error_kind == "invalid"
    block.parameters.update(missing_views="fail", frame_order="calibrated")
    result = block.execute(inputs)
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "mosaic_sources_missing"
    assert [p.payload["source_id"] for p in result.outputs["frames"].items] == ["middle", "right"]
    assert all(not p.payload["outcome"]["passed"] for p in result.outputs["frames"].items)
    assert (
        result.outputs["frames"].items[0].payload["ticket"]
        == inputs["views"].items[1].payload["ticket"]
    )


@pytest.mark.parametrize(
    "bad", ["budget", "preview", "grid", "duplicate", "unknown", "layout", "mask"]
)
def test_guards_reject_before_warp_or_native_allocation(bad, monkeypatch):
    inputs, _, _ = fixture()
    parameters = {}
    if bad == "budget":
        parameters["maximum_workspace_mib"] = 1
        for view in inputs["layout"]["views"]:
            matrix = np.asarray(view["source_to_mosaic"], float)
            matrix[:2] *= 100
            view["source_to_mosaic"] = matrix.tolist()
        images = [p.value for p in inputs["views"].items]
        from opencv_dataflow.image_stitching import mosaic_geometry

        size, shift = mosaic_geometry(
            images, [np.asarray(v["source_to_mosaic"]) for v in inputs["layout"]["views"]]
        )
        inputs["layout"]["image_size"] = list(size)
        for view in inputs["layout"]["views"]:
            view["source_to_mosaic"] = (shift @ view["source_to_mosaic"]).tolist()
        inputs["layout"] = seal_layout(inputs["layout"])
    elif bad == "preview":
        parameters["preview_frame"] = 3
    elif bad == "grid":
        inputs["views"].items[0].payload["spatial"]["to_root"][0][2] = 1
    elif bad == "duplicate":
        inputs["source_ids"] = ["left"] * 3
    elif bad == "unknown":
        inputs["source_ids"] = ["not-trained", "middle", "right"]
    elif bad == "layout":
        inputs["layout"]["layout_id"] = "0" * 64
    else:
        inputs["masks"].items[1].payload["spatial"]["to_root"][1][2] = 1
    monkeypatch.setattr(
        registered_replay, "warp_mosaic_view", lambda *a: pytest.fail("Invalid warp")
    )
    block = builtin_registry().create("cv.replay_registered_views", parameters=parameters)
    assert block.execute(inputs) is None and block.error_kind == "invalid"
    assert not any(block.current_emissions.values())


def test_native_failure_is_contained_then_fresh_call_recovers(monkeypatch):
    inputs, _, _ = fixture()
    native = registered_replay.native_timelapser
    block = builtin_registry().create(
        "cv.replay_registered_views", parameters={"backend": "native_detail"}
    )

    def broken(*a):
        raise cv2.error("Controlled native initialize failure")

    monkeypatch.setattr(registered_replay, "native_timelapser", broken)
    assert block.execute(inputs) is None and block.error_kind == "exception"
    assert not any(block.current_emissions.values())
    monkeypatch.setattr(registered_replay, "native_timelapser", native)
    assert block.execute(inputs).passed


def test_parallel_instances_have_no_shared_timelapser_or_frame_state():
    def run(preview):
        inputs, _, _ = fixture("rgb8")
        block = builtin_registry().create(
            "cv.replay_registered_views",
            parameters={"backend": "native_detail", "preview_frame": preview},
        )
        result = block.execute(inputs)
        assert result is not None, block.error
        return result.outputs["image"], result.outputs["info"]["frames"][preview]["source_id"]

    with ThreadPoolExecutor(max_workers=3) as pool:
        answers = list(pool.map(run, range(3)))
    assert [answer[1] for answer in answers] == list(IDS)
    assert not np.array_equal(answers[0][0], answers[1][0])


def test_linear_half_pixel_uses_full_source_support_and_independent_averages():
    inputs, images, _ = fixture("rgb64", masks=False)
    inputs["layout"]["views"][1]["source_to_mosaic"][0][2] = 12.5
    inputs["layout"] = seal_layout(inputs["layout"])
    block = builtin_registry().create("cv.replay_registered_views")
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    expected = np.zeros((32, 72, 3), np.float64)
    expected[:, 13:60] = (images[1][:, :-1] + images[1][:, 1:]) / 2
    np.testing.assert_allclose(
        result.outputs["frames"].items[1].value, expected, atol=1e-15, rtol=0
    )
    valid = np.zeros((32, 72), np.uint8)
    valid[:, 13:60] = 255
    np.testing.assert_array_equal(result.outputs["coverage"].items[1].value, valid)


@pytest.mark.parametrize("backend", ["preserve", "native_detail"])
def test_rotated_source_pyramid_keeps_inverse_mapping_and_original_level(backend):
    inputs, images, masks = fixture("rgb8", order=(0,))
    layout = inputs["layout"]
    layout["views"] = layout["views"][:1]
    matrix = [[0, -1, 31], [1, 0, 0], [0, 0, 1]]
    layout["views"][0]["source_to_mosaic"] = matrix
    layout["image_size"] = [32, 48]
    inputs["layout"] = seal_layout(layout)
    inputs["views"].items[0].payload["pyramid"] = {
        "level": 1,
        "largest_size": [96, 64],
        "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
    }
    block = builtin_registry().create(
        "cv.replay_registered_views", parameters={"backend": backend, "interpolation": "nearest"}
    )
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    expected = np.rot90(images[0], -1).copy()
    expected[np.rot90(masks[0], -1) == 0] = 0
    packet = result.outputs["frames"].items[0]
    np.testing.assert_array_equal(packet.value, expected)
    record = packet.payload["registered_replay"]
    np.testing.assert_array_equal(record["source_to_frame"], matrix)
    assert "pyramid" not in packet.payload and record["source_pyramid"]["level"] == 1
    np.testing.assert_array_equal(np.linalg.inv(matrix) @ [25, 9, 1], [9, 6, 1])


def test_legacy_ordered_layout_remains_usable_and_lean_keeps_primary_frames():
    inputs, images, masks = fixture()
    layout = inputs["layout"]
    layout["model"] = "tiled_scan_mosaic_v1"
    layout.pop("layout_id")
    layout.pop("source_identity")
    for record, label in zip(layout["views"], inputs["views"].labels, strict=True):
        record["label"] = label
        record.pop("source_id")
    block = builtin_registry().create("cv.replay_registered_views")
    from opencv_dataflow.run_profiles import profile

    block.run_profile = profile("lean")
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    expected = oracle(images, masks, range(3), "canvas")[0]
    for packet, reference in zip(result.outputs["frames"].items, expected, strict=True):
        np.testing.assert_array_equal(packet.value, reference)
    assert not result.graphics and not result.drawings
    inputs["views"] = FlowGroup(inputs["views"].items[::-1], inputs["views"].labels[::-1])
    inputs["masks"] = FlowGroup(inputs["masks"].items[::-1], inputs["masks"].labels[::-1])
    assert block.execute(inputs) is None and block.error_kind == "invalid"


def test_xml_project_reopen_and_dropdowns_are_offscreen(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.opencv_reference import opencv_reference

    registry, graph = builtin_registry(), Graph()
    block = graph.add(
        registry.create(
            "cv.replay_registered_views",
            parameters={"backend": "native_detail", "crop": "intersection"},
        )
    )
    xml = tmp_path / "replay.cv.replay_registered_views.xml"
    save_block_xml(block, xml)
    assert read_block_xml(xml)["parameters"] == block.parameters
    project = tmp_path / "program.json"
    graph.save(project)
    reopened = Graph.load(project, registry)
    assert reopened.blocks[block.id].parameters == block.parameters
    reopened.close()
    assert "Timelapser.createDefault" in opencv_reference(block)
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        for key in (
            "crop",
            "backend",
            "frame_order",
            "interpolation",
            "missing_views",
            "grid_policy",
        ):
            assert isinstance(window.inspector.fields[key], QComboBox)
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
