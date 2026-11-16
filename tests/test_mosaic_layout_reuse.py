"""Stable source binding, fixed canvas identity, geometry, files and failure containment."""

import copy
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow import tiled_mosaic
from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.mosaic_layout import MODEL, seal_layout
from opencv_dataflow.mosaic_layout_io import LoadMosaicLayout, SaveMosaicLayout, layout_bytes
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.tickets import TicketClock

IDS = ("left", "middle", "right")


def fixture(kind="gray8", mode="explicit"):
    y, x = np.indices((32, 120))
    base = ((x * 7 + y * 11) % 150 + 30).astype(np.uint8)
    images = [base[:, i * 36 : i * 36 + 48].copy() + i * 5 for i in range(3)]
    if kind == "rgb16":
        images = [
            ImageArray(np.dstack((v, v + 1, v + 2)).astype(np.uint16) * 200, "rgb") for v in images
        ]
    elif kind == "gray64":
        images = [v.astype(np.float64) / 123 for v in images]
    layout = {
        "model": MODEL,
        "image_size": [120, 32],
        "source_identity": {"mode": mode, "field": "source_id"},
        "views": [],
    }
    for i, image in enumerate(images):
        layout["views"].append(
            {
                "index": i,
                "label": "Calibrated " + IDS[i],
                "source_id": IDS[i],
                "source_size": [48, 32],
                "source_dtype": image.dtype.name,
                "source_format": "rgb" if kind == "rgb16" else "gray",
                "source_to_mosaic": [[1, 0, i * 36], [0, 1, 0], [0, 0, 1]],
                "source_spatial": frame([48, 32], "cam-" + IDS[i]),
                "source_ticket": None,
            }
        )
    return images, seal_layout(layout)


def group(images, order, *, mode="explicit", changed=None):
    clock, packets = TicketClock(), []
    for i in order:
        context = {
            "spatial": frame([48, 32], "cam-" + IDS[i]),
            "source_id": IDS[i],
            "ticket": clock.issue("camera", "external", IDS[i] + "-current"),
        }
        if changed == i:
            context["spatial"]["to_root"][0][2] = 1
        packets.append(ValuePacket(images[i], context))
    labels = tuple(IDS[i] if mode == "labels" else "display-" + str(j) for j, i in enumerate(order))
    return FlowGroup(tuple(packets), labels)


def render(images, layout, order, *, masks=None, **settings):
    parameters = {
        "source_ids": [IDS[i] for i in order],
        "interpolation": "nearest",
        "tile_size": 128,
        "missing_views": "allow",
        **settings,
    }
    block = builtin_registry().create("cv.render_mosaic_tile", parameters=parameters)
    inputs = {"views": group(images, order), "layout": layout}
    if masks is not None:
        inputs["masks"] = group(masks, order)
    result = block.execute(inputs)
    assert result is not None, block.error
    return block, result


def reference(images, order, masks=None, last=False):
    shape = (32, 120, *images[0].shape[2:])
    output, count = np.zeros(shape, images[0].dtype), np.zeros(shape[:2], np.uint8)
    for i in sorted(order):
        area = np.s_[:, i * 36 : i * 36 + 48]
        valid = np.ones((32, 48), bool) if masks is None else masks[i] != 0
        use = valid if last else valid & (count[area] == 0)
        output[area][use] = images[i][use]
        count[area] += valid
    return output, (count > 0).astype(np.uint8) * 255, (count > 1).astype(np.uint8) * 255


@pytest.mark.parametrize("kind", ["gray8", "rgb16", "gray64"])
@pytest.mark.parametrize("blend", ["prefer_first", "prefer_last", "feather", "multiband"])
def test_reordering_preserves_pixels_priority_masks_and_source_provenance(kind, blend):
    images, layout = fixture(kind)
    masks = [np.full((32, 48), 255, np.uint8) for _ in images]
    masks[1][8:15, 20:27] = 0
    original = [image.copy() for image in images]
    settings = {"blend": blend, "multiband_bands": 2, "feather_width": 7}
    _, ordered = render(images, layout, [0, 1, 2], masks=masks, **settings)
    _, shuffled = render(images, layout, [2, 0, 1], masks=masks, **settings)
    for key in ("image", "coverage", "overlap"):
        np.testing.assert_array_equal(shuffled.outputs[key], ordered.outputs[key])
    if blend.startswith("prefer"):
        expected = reference(images, [0, 1, 2], masks, blend == "prefer_last")
        for key, value in zip(("image", "coverage", "overlap"), expected, strict=True):
            np.testing.assert_array_equal(shuffled.outputs[key], value)
    tile = shuffled.outputs["tile"]
    assert tile["source_matching"]["complete"]
    assert [v["source_id"] for v in tile["views"]] == list(IDS)
    assert [v["input_index"] for v in tile["views"]] == [1, 2, 0]
    assert [v["source_ticket"]["value"] for v in tile["views"]] == [
        name + "-current" for name in IDS
    ]
    for source, saved in zip(images, original, strict=True):
        np.testing.assert_array_equal(source, saved)


@pytest.mark.parametrize("order", [[2, 0], [2, 1], [2]])
def test_absent_views_keep_full_grid_root_coverage_and_per_view_maps(order):
    images, layout = fixture()
    _, result = render(images, layout, order)
    expected = reference(images, order)
    for key, value in zip(("image", "coverage", "overlap"), expected, strict=True):
        np.testing.assert_array_equal(result.outputs[key], value)
    spatial = result.payloads["image"]["spatial"]
    assert spatial["root_id"] == "mosaic:" + layout["layout_id"]
    assert spatial["root_size"] == spatial["image_size"] == [120, 32]
    np.testing.assert_array_equal(spatial["to_root"], np.eye(3))
    assert result.outputs["tile"]["source_matching"]["missing_source_ids"] == [
        IDS[i] for i in range(3) if i not in order
    ]
    for record in result.outputs["tile"]["views"]:
        np.testing.assert_array_equal(
            record["source_to_tile"], layout["views"][record["index"]]["source_to_mosaic"]
        )


def test_missing_policy_error_has_no_events_fail_and_allow_preserve_partial_data():
    images, layout = fixture()
    block = builtin_registry().create("cv.render_mosaic_tile", parameters={"source_ids": ["right"]})
    inputs = {"views": group(images, [2]), "layout": layout}
    assert block.execute(inputs) is None and block.error_kind == "invalid"
    assert not any(block.current_emissions.values())
    block.parameters["missing_views"] = "fail"
    failed = block.execute(inputs)
    assert failed is not None and not failed.passed
    assert failed.fail_reasons[0]["code"] == "mosaic_sources_missing"
    block.parameters["missing_views"] = "allow"
    allowed = block.execute(inputs)
    assert allowed.passed and not allowed.outputs["tile"]["source_matching"]["complete"]
    np.testing.assert_array_equal(allowed.outputs["image"], failed.outputs["image"])


@pytest.mark.parametrize("mode", ["labels", "payload", "explicit"])
def test_identity_modes_and_wired_override_resolve_source_records(mode):
    images, layout = fixture(mode=mode)
    inputs = {"views": group(images, [2, 0], mode=mode), "layout": layout}
    block = builtin_registry().create(
        "cv.render_mosaic_tile", parameters={"missing_views": "allow"}
    )
    if mode == "explicit":
        inputs["source_ids"] = ["right", "left"]
    result = block.execute(inputs)
    assert result is not None, block.error
    assert result.outputs["tile"]["source_matching"]["present_source_ids"] == ["left", "right"]
    inputs["source_ids"] = []
    assert block.execute(inputs) is None and block.error_kind == "invalid"


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "duplicate", "grid", "depth", "identity", "absent_transform", "absent_grid"],
)
def test_incompatible_or_tampered_layouts_fail_before_native_warp(mutation, monkeypatch):
    images, layout = fixture()
    inputs = {
        "views": group(images, [2, 0], changed=0 if mutation == "grid" else None),
        "layout": layout,
    }
    parameters = {"missing_views": "allow", "source_ids": ["right", "left"]}
    if mutation == "unknown":
        parameters["source_ids"] = ["untrained-camera", "left"]
    elif mutation == "duplicate":
        parameters["source_ids"] = ["left", "left"]
    elif mutation == "depth":
        inputs["views"] = group([image.astype(np.uint16) for image in images], [2, 0])
    elif mutation == "identity":
        layout["layout_id"] = "0" * 64
    elif mutation == "absent_transform":
        layout["views"][1]["source_to_mosaic"][0][0] = float("nan")
    elif mutation == "absent_grid":
        layout["views"][1]["source_size"][0] += 1
    monkeypatch.setattr(
        tiled_mosaic, "warp_mosaic_view", lambda *a: pytest.fail("Invalid reuse entered warp")
    )
    block = builtin_registry().create("cv.render_mosaic_tile", parameters=parameters)
    assert block.execute(inputs) is None
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())


def test_grid_override_records_caller_assertion_and_revision_identity_excludes_tickets_labels():
    images, layout = fixture()
    block = builtin_registry().create(
        "cv.render_mosaic_tile",
        parameters={
            "source_ids": ["right"],
            "missing_views": "allow",
            "grid_policy": "pixel_indices",
        },
    )
    result = block.execute({"views": group(images, [2], changed=2), "layout": layout})
    assert result is not None, block.error
    assert (
        result.outputs["tile"]["source_matching"]["grid_checks"][0]["evidence"]
        == "caller_asserted_pixel_indices"
    )
    changed = copy.deepcopy(layout)
    changed["views"][0].update(label="renamed", source_ticket={"anything": "new capture"})
    assert seal_layout(changed)["layout_id"] == layout["layout_id"]
    changed["views"][1]["source_to_mosaic"][0][2] += 0.25
    assert seal_layout(changed)["layout_id"] != layout["layout_id"]


def test_large_layout_missing_anchor_validates_bounds_without_dense_allocation(monkeypatch):
    images, layout = fixture()
    transforms = []
    for record in layout["views"]:
        matrix = np.asarray(record["source_to_mosaic"], float)
        matrix[:2] *= 1000
        transforms.append(matrix)
    size, shift = tiled_mosaic.mosaic_geometry(images, transforms)
    layout["image_size"] = list(size)
    for record, matrix in zip(layout["views"], transforms, strict=True):
        record["source_to_mosaic"] = (shift @ matrix).tolist()
    layout = seal_layout(layout)
    assert size[0] * size[1] > 256 * 1024**2
    native, windows = tiled_mosaic.warp_mosaic_view, []

    def bounded(image, matrix, window, interpolation):
        assert max(window) <= 16
        windows.append(window)
        return native(image, matrix, window, interpolation)

    monkeypatch.setattr(tiled_mosaic, "warp_mosaic_view", bounded)
    _, result = render(images, layout, [2], tile_size=16, tile_index=7000)
    assert result.outputs["image"].shape == (16, 16)
    assert result.outputs["tile"]["mosaic_size"] == list(size)


def test_layout_files_new_replace_hash_strict_json_and_source_input_is_unchanged(tmp_path):
    _, layout = fixture()
    original = copy.deepcopy(layout)
    path = tmp_path / "calibration.json"
    save = SaveMosaicLayout(parameters={"path": str(path)})
    saved = save.execute({"layout": layout})
    assert saved is not None, save.error
    raw = path.read_bytes()
    assert saved.outputs["info"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert save.execute({"layout": layout}) is None
    assert path.read_bytes() == raw and layout == original
    load = LoadMosaicLayout(
        parameters={"path": str(path), "expected_sha256": hashlib.sha256(raw).hexdigest()}
    )
    assert load.execute({}).outputs["layout"] == layout
    replacement = copy.deepcopy(layout)
    replacement["note"] = "new capture annotations"
    save.parameters["write_mode"] = "replace"
    assert save.execute({"layout": replacement}) is not None
    assert not list(tmp_path.glob(".mosaic-*.tmp"))
    assert load.execute({}) is None and load.error_kind == "invalid"
    load.parameters["expected_sha256"] = ""
    assert load.execute({}).outputs["layout"] == replacement


@pytest.mark.parametrize(
    "bad", [b'{"model":1,"model":2}', b'{"x":NaN}', b"\xef\xbb\xbf{}", b"[" + b" " * (1024**2)]
)
def test_bad_layout_files_are_invalid_and_emit_no_output(tmp_path, bad):
    path = tmp_path / "bad.json"
    path.write_bytes(bad)
    block = LoadMosaicLayout(parameters={"path": str(path)})
    assert block.execute({}) is None and block.error_kind == "invalid"
    assert not any(block.current_emissions.values())


def test_native_exception_recovery_and_legacy_order_stays_strict(monkeypatch):
    images, layout = fixture()
    block = builtin_registry().create("cv.render_mosaic_tile", parameters={"source_ids": list(IDS)})
    inputs = {"views": group(images, [0, 1, 2]), "layout": layout}
    native = tiled_mosaic.warp_mosaic_view

    def broken(*args):
        raise cv2.error("Controlled reuse warp failure")

    monkeypatch.setattr(tiled_mosaic, "warp_mosaic_view", broken)
    assert block.execute(inputs) is None and block.error_kind == "exception"
    assert not any(block.current_emissions.values())
    monkeypatch.setattr(tiled_mosaic, "warp_mosaic_view", native)
    assert block.execute(inputs).passed
    legacy = copy.deepcopy(layout)
    legacy["model"] = "tiled_scan_mosaic_v1"
    block.parameters["source_ids"] = []
    assert block.execute({"views": group(images, [2]), "layout": legacy}) is None


@pytest.mark.parametrize("mode", ["labels", "payload", "explicit"])
def test_native_registration_seals_stable_sources_for_reuse_without_refitting(mode, monkeypatch):
    folder = (
        Path(__file__).resolve().parents[1] / "examples/opencv tools/mosaic_layout_reuse/assets"
    )
    images = [cv2.imread(str(folder / (name + ".png")), cv2.IMREAD_GRAYSCALE) for name in IDS]
    packets = tuple(
        ValuePacket(image, {"source_id": name, "spatial": frame([192, 128], "cam-" + name)})
        for name, image in zip(IDS, images, strict=True)
    )
    batch = FlowGroup(packets, IDS)
    block = builtin_registry().create(
        "cv.register_scan_layout",
        parameters={"source_identity": mode, "min_inliers": 6},
    )
    inputs = {"views": batch}
    if mode == "explicit":
        inputs["source_ids"] = list(IDS)
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    layout = result.outputs["layout"]
    assert layout["model"] == MODEL and layout["layout_id"] == seal_layout(layout)["layout_id"]
    assert [v["source_id"] for v in layout["views"]] == list(IDS)
    monkeypatch.setattr(tiled_mosaic, "register_scan", lambda *a: pytest.fail("Reuse refitted"))
    renderer = builtin_registry().create(
        "cv.render_mosaic_tile",
        parameters={"missing_views": "allow", "source_ids": ["right", "middle"]},
    )
    replay = renderer.execute(
        {"views": FlowGroup(packets[::-1][:2], IDS[::-1][:2]), "layout": layout}
    )
    assert replay is not None and replay.passed, renderer.error
    assert replay.outputs["tile"]["source_matching"]["missing_source_ids"] == ["left"]
    assert replay.payloads["image"]["spatial"]["root_id"] == "mosaic:" + layout["layout_id"]
    block.parameters.update(source_identity="explicit", source_ids=["same"] * 3)
    assert block.execute({"views": batch}) is None and block.error_kind == "invalid"


@pytest.mark.parametrize("blend", ["feather", "multiband"])
def test_reused_missing_anchor_adjacent_tiles_match_dense_composition(blend):
    images, layout = fixture("rgb16")
    masks = [np.full((32, 48), 255, np.uint8) for _ in images]
    masks[2][11:19, 6:12] = 0
    settings = {"blend": blend, "feather_width": 5, "multiband_bands": 2}
    _, dense = render(images, layout, [2, 1], masks=masks, **settings)
    assembled = np.zeros_like(dense.outputs["image"])
    coverage = np.zeros((32, 120), np.uint8)
    for index in range(4):
        _, tile = render(
            images, layout, [2, 1], masks=masks, tile_size=32, tile_index=index, **settings
        )
        x, y = tile.outputs["tile"]["origin"]
        h, w = tile.outputs["image"].shape[:2]
        assembled[y : y + h, x : x + w] = tile.outputs["image"]
        coverage[y : y + h, x : x + w] = tile.outputs["coverage"]
        spatial = tile.payloads["image"]["spatial"]
        np.testing.assert_array_equal(spatial["to_root"], [[1, 0, x], [0, 1, y], [0, 0, 1]])
        assert spatial["root_id"] == dense.payloads["image"]["spatial"]["root_id"]
    np.testing.assert_allclose(assembled.astype(float), dense.outputs["image"], atol=1, rtol=0)
    np.testing.assert_array_equal(coverage, dense.outputs["coverage"])


def test_sealed_layout_asset_save_as_xml_and_inspector_choices(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    _, layout = fixture()
    original = tmp_path / "original"
    (original / "assets").mkdir(parents=True)
    raw = layout_bytes(layout)
    (original / "assets/layout.json").write_bytes(raw)
    registry, graph = builtin_registry(), Graph()
    load = graph.add(
        registry.create("cv.load_mosaic_layout", parameters={"path": "assets/layout.json"})
    )
    render_block = graph.add(
        registry.create("cv.render_mosaic_tile", parameters={"source_ids": list(IDS)})
    )
    graph.save(original / "program.json")
    other = tmp_path / "copy/program.json"
    graph.save(other)
    assert (other.parent / "assets/layout.json").read_bytes() == raw
    reopened = Graph.load(other, registry)
    assert reopened.blocks[load.id].execute({}).outputs["layout"] == layout
    reopened.close()
    xml = tmp_path / "tile.cv.render_mosaic_tile.xml"
    save_block_xml(render_block, xml)
    assert read_block_xml(xml)["parameters"] == render_block.parameters
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(render_block)
        for name in ("missing_views", "grid_policy", "blend"):
            assert isinstance(window.inspector.fields[name], QComboBox)
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
