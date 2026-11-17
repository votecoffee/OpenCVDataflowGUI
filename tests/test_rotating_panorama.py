"""Independent ray-color rendering, public geometry, source ownership and guarded failures."""

import copy
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.graphics_layers import graphics_bundle
from opencv_dataflow.images import ImageArray
from opencv_dataflow.panorama_geometry import validate_layout
from opencv_dataflow.payloads import (
    NO_VALUE,
    FlowGroup,
    ValuePacket,
    frame,
    moved_payload,
    packet_for,
)
from opencv_dataflow.poses import local_offset
from opencv_dataflow.rotating_panorama import verification as verify_geometry
from opencv_dataflow.tickets import TicketClock


def yaw(degrees):
    a = np.deg2rad(degrees)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


@pytest.fixture
def known_model(monkeypatch):
    from opencv_dataflow import rotating_panorama

    K = np.array([[60, 0, 40], [0, 60, 32], [0, 0, 1]], float)
    matrices = [yaw(-15), np.eye(3), yaw(15)]
    v, u = np.indices((64, 80))
    rays = np.stack([(u - 40) / 60, (v - 32) / 60, np.ones_like(u)], axis=2)
    clock = TicketClock("synthetic-radiance")
    items = []
    for i, R in enumerate(matrices):
        q = rays @ R.T
        values = 100 + 10 * q[..., 0] / q[..., 2] + 20 * q[..., 1] / q[..., 2]
        items.append(
            ValuePacket(
                ImageArray(values, "gray"),
                {"spatial": frame([80, 64], f"source-{i}"), "ticket": clock.issue(f"source-{i}")},
            )
        )
    packet = ValuePacket(
        FlowGroup(tuple(items), ("left", "center", "right")), {"ticket": clock.issue("batch")}
    )

    class Solver:
        def setRegistrationResol(self, *a):
            pass

        def setPanoConfidenceThresh(self, *a):
            pass

        def setWaveCorrection(self, *a):
            pass

        def estimateTransform(self, *a):
            return 0

        def component(self):
            return [0, 1, 2]

        def cameras(self):
            return [SimpleNamespace(R=R, K=lambda: K.copy()) for R in matrices]

    monkeypatch.setattr(rotating_panorama.cv2, "Stitcher_create", lambda *_: Solver())
    monkeypatch.setattr(
        rotating_panorama,
        "verification",
        lambda *a: (
            [
                {"source_index": 0, "target_index": 1, "status": "validated"},
                {"source_index": 1, "target_index": 2, "status": "validated"},
            ],
            [(0, 1), (1, 2)],
            False,
        ),
    )
    return packet, K, matrices


def stitch(packet, **parameters):
    block = builtin_registry().create(
        "cv.stitch_rotating_panorama",
        parameters={"anchor_label": "center", "projection_scale": 60.0, **parameters},
    )
    result = block.execute({"views": packet})
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("projection", ["spherical", "cylindrical", "plane"])
def test_native_inverse_maps_render_an_independent_continuous_ray_field(known_model, projection):
    packet, _, _ = known_model
    before = [p.value.copy() for p in packet.value.items]
    _, result = stitch(packet, projection=projection)
    assert result.passed
    layout = result.outputs["layout"]
    validate_layout(layout)
    yy, xx = np.indices(result.outputs["image"].shape)
    u = (xx + layout["origin"][0]) / 60
    v = (yy + layout["origin"][1]) / 60
    if projection == "plane":
        x, y, z = u, v, np.ones_like(u)
    elif projection == "cylindrical":
        x, y, z = np.sin(u), v, np.cos(u)
    else:
        x, y, z = (
            np.cos(v - np.pi / 2) * np.sin(u),
            np.sin(v - np.pi / 2),
            np.cos(v - np.pi / 2) * np.cos(u),
        )
    truth = 100 + 10 * x / z + 20 * y / z
    valid = result.outputs["coverage"] != 0
    np.testing.assert_allclose(result.outputs["image"][valid], truth[valid], rtol=0, atol=0.02)
    assert not result.outputs["image"][~valid].any()
    assert np.array_equal(result.outputs["contributors"] > 0, valid)
    assert np.count_nonzero(result.outputs["contributors"] >= 2) > 100
    for i, p in enumerate(packet.value.items):
        np.testing.assert_array_equal(p.value, before[i])
    for key in ("image", "coverage", "layout", "report", "contributors"):
        assert result.payloads[key]["ticket"] == packet.payload["ticket"]
        assert result.payloads[key]["spatial"] == layout["spatial"]
    assert result.payloads["diagnostic"]["spatial"] != layout["spatial"]
    assert result.payloads["diagnostic"]["ticket"] == packet.payload["ticket"]


def test_uint16_is_not_narrowed_to_the_registration_bytes(known_model):
    packet, _, _ = known_model
    high = ValuePacket(
        FlowGroup(
            tuple(
                ValuePacket(ImageArray(np.rint(p.value * 100).astype(np.uint16), "gray"), p.payload)
                for p in packet.value.items
            ),
            packet.value.labels,
        ),
        packet.payload,
    )
    _, result = stitch(high, projection="plane")
    assert result.passed and result.outputs["image"].dtype == np.uint16
    assert result.outputs["image"].max() > 10000


def test_source_and_target_masks_filter_points_without_confusing_coordinate_frames(known_model):
    packet, _, _ = known_model
    _, result = stitch(packet)
    registry = builtin_registry()
    mapper = registry.create("cv.panorama_points", parameters={"source_index": 0})
    points = ValuePacket([[40, 32], [-1, 3]], packet.value.items[0].payload)
    mapped = mapper.execute(
        {
            "layout": packet_for(result, "layout"),
            "points": points,
            "mask": packet_for(result, "coverage"),
        }
    )
    assert mapped is not None, mapper.error
    assert mapped.outputs["valid"] == [True, False] and mapped.outputs["indices"] == [0]
    assert mapped.payloads["points"]["ticket"] == packet.payload["ticket"]
    source_mask = np.full((64, 80), 255, np.uint8)
    source_mask[30:35, 38:43] = 0
    rejected = mapper.execute(
        {
            "layout": packet_for(result, "layout"),
            "points": points,
            "source_mask": ValuePacket(source_mask, packet.value.items[0].payload),
        }
    )
    assert rejected is not None and not rejected.passed and rejected.outputs["count"] == 0
    assert rejected.drawings
    overlay = registry.create("cv.overlay_result_graphics")
    shown = overlay.execute(
        {
            "image": packet_for(result, "image"),
            "graphics_1": ValuePacket(
                graphics_bundle(mapper, rejected), rejected.payloads["points"]
            ),
        }
    )
    assert shown is not None, overlay.error
    assert "Overlaid 1 layers" in shown.message


def test_crop_coordinates_and_native_forward_points_agree(known_model):
    packet, _, _ = known_model
    _, result = stitch(packet)
    mapper = builtin_registry().create("cv.panorama_points", parameters={"source_index": 0})
    source = packet.value.items[0].payload
    crop = moved_payload(source, local_offset(30, 25, 0), [20, 20])
    mapped = mapper.execute(
        {"layout": packet_for(result, "layout"), "points": ValuePacket([[10, 7]], crop)}
    )
    assert mapped is not None, mapper.error
    layout = result.outputs["layout"]
    camera = layout["cameras"][0]
    warper = cv2.PyRotationWarper(layout["projection"], layout["scale"])
    native = (
        np.array(
            warper.warpPoint(
                (40.0, 32.0),
                np.array(camera["K"], np.float32),
                np.array(camera["camera_to_panorama"], np.float32),
            )
        )
        - layout["origin"]
    )
    np.testing.assert_allclose(mapped.outputs["points"], [native], atol=1e-4)


def test_changed_layout_stale_tickets_and_wrong_coverage_emit_nothing(known_model):
    packet, _, _ = known_model
    _, result = stitch(packet)
    mapper = builtin_registry().create("cv.panorama_points", parameters={"source_index": 0})
    bad = copy.deepcopy(result.outputs["layout"])
    bad["cameras"][0]["K"][0][0] += 1
    assert mapper.execute({"layout": bad, "points": [[40, 32]]}) is None
    assert "identity changed" in mapper.error
    stale = copy.deepcopy(packet.value.items[0].payload)
    stale["ticket"]["value"] = 99
    assert (
        mapper.execute(
            {"layout": packet_for(result, "layout"), "points": ValuePacket([[40, 32]], stale)}
        )
        is None
    )
    assert "tickets differ" in mapper.error
    wrong = {"spatial": frame(result.outputs["layout"]["image_size"], "wrong")}
    assert (
        mapper.execute(
            {
                "layout": packet_for(result, "layout"),
                "points": [[40, 32]],
                "mask": ValuePacket(result.outputs["coverage"], wrong),
            }
        )
        is None
    )
    assert not any(mapper.current_emissions.values())


def test_partial_component_policies_are_explicit_and_failed_images_are_not_fabricated(
    known_model, monkeypatch
):
    from opencv_dataflow import rotating_panorama

    packet, _, _ = known_model
    monkeypatch.setattr(
        rotating_panorama,
        "verification",
        lambda *a: (
            [{"source_index": 0, "target_index": 1, "status": "validated"}],
            [(0, 1)],
            False,
        ),
    )
    for policy, passed in [("reject", False), ("fail_partial", False), ("allow_partial", True)]:
        block, result = stitch(packet, component_policy=policy)
        assert result.passed is passed
        assert result.outputs["report"]["dropped_labels"] == ["right"]
        if policy == "reject":
            assert result.outputs["image"] is NO_VALUE
            assert not block.current_emissions.get("image")
        else:
            assert isinstance(result.outputs["image"], np.ndarray)


def test_independent_model_disagreement_rejects_registration_and_report_stays_json_safe(
    known_model, monkeypatch
):
    from opencv_dataflow import rotating_panorama

    packet, _, _ = known_model
    monkeypatch.setattr(
        rotating_panorama,
        "verification",
        lambda *a: (
            [
                {
                    "source_index": 0,
                    "target_index": 1,
                    "status": "inconsistent_rotation_model",
                    "camera_rms_pixels": 8.0,
                }
            ],
            [],
            True,
        ),
    )
    block, result = stitch(packet)
    assert not result.passed and result.outputs["report"]["status"] == "inconsistent_rotation_model"
    assert result.outputs["image"] is NO_VALUE and not block.current_emissions.get("layout")


def test_actual_geometric_verifier_detects_a_wrong_rotating_camera_model(monkeypatch):
    from opencv_dataflow import rotating_panorama

    v, u = np.indices((6, 8))
    a = np.c_[u.ravel() * 8 + 8, v.ravel() * 8 + 8].astype(np.float32)
    angle = np.deg2rad(-15)
    x = (a[:, 0] - 40) / 60
    y = (a[:, 1] - 32) / 60
    z = -np.sin(angle) * x + np.cos(angle)
    b = np.c_[60 * (np.cos(angle) * x + np.sin(angle)) / z + 40, 60 * y / z + 32].astype(np.float32)
    monkeypatch.setattr(rotating_panorama, "_pairs", lambda *args: (a, b))
    K = [[60, 0, 40], [0, 60, 32], [0, 0, 1]]
    cameras = [
        {"index": i, "K": K, "camera_to_panorama": R.tolist()}
        for i, R in enumerate([yaw(-15), np.eye(3)])
    ]
    p = builtin_registry().create("cv.stitch_rotating_panorama").parameters
    rows, edges, bad = verify_geometry(cameras, {0: None, 1: None}, [[1, 1], [1, 1]], p)
    assert not bad and edges == [(0, 1)] and rows[0]["camera_rms_pixels"] < 1e-4
    cameras[0]["camera_to_panorama"] = np.eye(3).tolist()
    rows, edges, bad = verify_geometry(cameras, {0: None, 1: None}, [[1, 1], [1, 1]], p)
    assert bad and not edges and rows[0]["status"] == "inconsistent_rotation_model"


def test_registration_resize_restores_full_pixel_intrinsics(known_model, monkeypatch):
    from opencv_dataflow import rotating_panorama

    packet, _, matrices = known_model
    K = np.array([[120, 0, 80], [0, 120, 64], [0, 0, 1]], float)
    vv, uu = np.indices((128, 160))
    rays = np.stack([(uu - 80) / 120, (vv - 64) / 120, np.ones_like(uu)], axis=2)
    entries = []
    for i, R in enumerate(matrices):
        q = rays @ R.T
        values = 100 + 10 * q[..., 0] / q[..., 2] + 20 * q[..., 1] / q[..., 2]
        context = moved_payload(packet.value.items[i].payload, np.diag([0.5, 0.5, 1]), [160, 128])
        entries.append(ValuePacket(ImageArray(values, "gray"), context))

    class Solver:
        def setRegistrationResol(self, *a):
            pass

        def setPanoConfidenceThresh(self, *a):
            pass

        def setWaveCorrection(self, *a):
            pass

        def estimateTransform(self, work, masks):
            self.shape = work[0].shape
            return 0

        def component(self):
            return [0, 1, 2]

        def cameras(self):
            h, w = self.shape
            scaled = np.diag([w / 160, h / 128, 1]) @ K
            return [SimpleNamespace(R=R, K=lambda: scaled.copy()) for R in matrices]

    monkeypatch.setattr(rotating_panorama.cv2, "Stitcher_create", lambda *_: Solver())
    _, result = stitch(
        ValuePacket(FlowGroup(tuple(entries), packet.value.labels), packet.payload),
        registration_megapixels=0.01,
    )
    for camera in result.outputs["layout"]["cameras"]:
        np.testing.assert_allclose(camera["K"], K, atol=1e-12)
        assert max(camera["registration_scale"]) < 1


def test_serial_camera_batches_can_have_repeated_source_labels(known_model):
    packet, _, _ = known_model
    serial = ValuePacket(
        FlowGroup(packet.value.items, ("camera", "camera", "camera")), packet.payload
    )
    _, result = stitch(serial, anchor_label="", reference_index=1)
    assert result.passed
    assert [c["label"] for c in result.outputs["layout"]["cameras"]] == [
        "view_0",
        "view_1",
        "view_2",
    ]
    assert result.outputs["report"]["input_labels"] == ["camera"] * 3
    _, named = stitch(serial, source_labels=["left", "center", "right"])
    assert named.passed and named.outputs["layout"]["anchor_label"] == "center"


def test_input_budget_is_checked_before_unpacking_or_rasterizing_masks(known_model, monkeypatch):
    from opencv_dataflow import rotating_panorama

    packet, _, _ = known_model
    monkeypatch.setattr(
        rotating_panorama, "_unpack", lambda *a: (_ for _ in ()).throw(AssertionError("allocated"))
    )
    block = builtin_registry().create(
        "cv.stitch_rotating_panorama", parameters={"maximum_input_pixels": 64}
    )
    assert block.execute({"views": packet}) is None
    assert "allocated" not in block.error and not any(block.current_emissions.values())


def test_public_layout_survives_strict_json_round_trip_without_aliasing_sources(known_model):
    packet, _, _ = known_model
    _, result = stitch(packet)
    layout = json.loads(json.dumps(result.outputs["layout"], allow_nan=False))
    validate_layout(layout)
    layout["cameras"][0]["source_spatial"]["to_root"][0][0] = 2
    assert packet.value.items[0].payload["spatial"]["to_root"][0][0] == 1


@pytest.mark.parametrize(
    "parameters",
    [
        {"minimum_focal_ratio": 5, "maximum_focal_ratio": 1},
        {"minimum_inliers": 1000, "verification_features": 32},
        {"registration_megapixels": True},
        {"projection_scale": float("nan")},
        {"anchor_label": "missing"},
        {"maximum_input_pixels": 64},
        {"maximum_working_bytes": 1},
        {"maximum_output_pixels": 1},
    ],
)
def test_bad_settings_or_oversized_canvases_cannot_emit_current_images(known_model, parameters):
    packet, _, _ = known_model
    block = builtin_registry().create(
        "cv.stitch_rotating_panorama", parameters={"anchor_label": "center", **parameters}
    )
    assert block.execute({"views": packet}) is None
    assert block.error and not any(block.current_emissions.values())


def test_graphics_off_skips_charts_and_markers_and_native_failures_recover(
    known_model, monkeypatch
):
    from opencv_dataflow import rotating_panorama

    packet, _, _ = known_model
    block = builtin_registry().create(
        "cv.stitch_rotating_panorama", parameters={"anchor_label": "center"}
    )
    block.set_graphics_mode("off")
    monkeypatch.setattr(
        rotating_panorama,
        "diagnostic",
        lambda *a: (_ for _ in ()).throw(AssertionError("graphics")),
    )
    result = block.execute({"views": packet})
    assert result is not None, block.error
    assert not result.drawings and result.outputs["diagnostic"] is NO_VALUE
    factory = rotating_panorama.cv2.Stitcher_create
    monkeypatch.setattr(
        rotating_panorama.cv2,
        "Stitcher_create",
        lambda *a: (_ for _ in ()).throw(RuntimeError("native failed")),
    )
    assert block.execute({"views": packet}) is None
    assert "native failed" in block.error and not any(block.current_emissions.values())
    monkeypatch.setattr(rotating_panorama.cv2, "Stitcher_create", factory)
    assert block.execute({"views": packet}).passed


def test_xml_and_offscreen_controls_do_not_estimate_cameras(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.project_xml import export_project_xml, read_project_xml
    from opencv_dataflow.toolbox import block_entry

    graph = Graph()
    registry = builtin_registry()
    block = graph.add(registry.create("cv.stitch_rotating_panorama"))
    graph.add(registry.create("cv.panorama_points"))
    snapshot = copy.deepcopy(graph.to_dict())
    path = tmp_path / "panorama.xml"
    export_project_xml(snapshot, path)
    assert Graph.from_dict(read_project_xml(path), registry).to_dict() == snapshot
    assert "stitching" in block_entry(block).path[1]
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        for key in ("projection", "wave_correction", "component_policy", "blend", "interpolation"):
            assert isinstance(window.inspector.fields[key], QComboBox)
        assert block.last_result is None
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
