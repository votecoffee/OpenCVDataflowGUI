"""Independent sample counts, transform conventions, selection and failure contracts."""

import copy

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.poses import local_offset, make_pose


def tool(**parameters):
    return builtin_registry().create("inspect.visible_support", parameters=parameters)


def run(block, inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


def records(positions, size=(11, 9), image_size=(96, 64)):
    w, h = size
    return [
        {
            "id": f"part-{i}",
            "score": 0.95,
            "bounds": [x, y, w, h],
            "pose": make_pose(
                local_offset(x + (w - 1) / 2, y + (h - 1) / 2, 0), list(image_size), {}
            ),
        }
        for i, (x, y) in enumerate(positions)
    ]


def scene():
    template = (np.arange(99).reshape(9, 11) + 100).astype(np.uint16)
    image = np.zeros((64, 96), np.uint16)
    positions = [(8, 10), (40, 35), (51, 35)]  # Last two touch without overlap.
    for x, y in positions:
        image[y : y + 9, x : x + 11] = template
    image[35:44, 51:55] = 0  # Exact 4/11 occluded columns in the downstream candidate.
    return image, template, records(positions)


def test_clear_touching_and_covered_candidates_keep_independent_detector_scores():
    image, template, matches = scene()
    original, pixels = copy.deepcopy(matches), image.copy()
    result = run(
        tool(absolute_tolerance=0), {"image": image, "template": template, "matches": matches}
    )
    assert result.passed and result.outputs["count"] == 2
    assert [m["id"] for m in result.outputs["matches"]] == ["part-0", "part-1"]
    evidence = [m["support"] for m in result.outputs["candidates"]]
    assert [m["matching_samples"] for m in evidence] == [99, 99, 63]
    assert [m["visible_fraction"] for m in evidence] == [1, 1, 7 / 11]
    assert all(m["observed_samples"] == 99 for m in evidence)
    assert evidence[2]["reasons"] == ["appearance_support_below_minimum"]
    assert all(m["score"] == 0.95 for m in result.outputs["candidates"])
    assert matches == original
    np.testing.assert_array_equal(image, pixels)
    np.testing.assert_array_equal(result.outputs["image"], pixels)
    assert result.drawings and result.outputs["report"]["true_occlusion_measurement"] is False


def test_border_clipping_and_source_selection_are_not_hidden_by_a_smaller_denominator():
    template = np.full((5, 7), 123, np.uint8)
    image = np.full((10, 12), 123, np.uint8)
    candidate = records([(-2, 2)], size=(7, 5), image_size=(12, 10))
    result = run(
        tool(absolute_tolerance=0, minimum_visible_fraction=0.6),
        {"image": image, "template": template, "matches": candidate},
    )
    evidence = result.outputs["candidates"][0]["support"]
    assert evidence["expected_samples"] == 35 and evidence["in_frame_samples"] == 25
    assert evidence["observed_agreement_fraction"] == 1
    assert evidence["visible_fraction"] == 5 / 7
    assert evidence["reasons"] == ["insufficient_observed_support"]
    mask = np.ones(image.shape, np.uint8)
    mask[2:7, 0] = 0
    selected = run(
        tool(minimum_matches=0),
        {"image": image, "template": template, "matches": candidate, "mask": mask},
    )
    evidence = selected.outputs["candidates"][0]["support"]
    assert evidence["in_frame_samples"] == 25 and evidence["observed_samples"] == 20


@pytest.mark.parametrize("mode", ["transform", "pose"])
def test_rotation_and_projective_transforms_match_analytic_linear_scene(mode):
    yy, xx = np.indices((64, 96))
    image = (xx + 3 * yy).astype(np.float64)
    v, u = np.indices((5, 7), np.float64)
    matrix = np.array([[1.2, 0.3, 12], [-0.2, 1.1, 15], [0.002, -0.001, 1]])
    divisor = matrix[2, 0] * u + matrix[2, 1] * v + 1
    x = (matrix[0, 0] * u + matrix[0, 1] * v + 12) / divisor
    y = (matrix[1, 0] * u + matrix[1, 1] * v + 15) / divisor
    template = np.rint(x * 32) / 32 + 3 * np.rint(y * 32) / 32
    target = {"spatial": frame([96, 64], "camera")}
    candidate = {
        "score": 0.8,
        "transform": {
            "model": "image_pattern_pose_v1",
            "source_size": [7, 5],
            "target_size": [96, 64],
            "target_payload": target,
            "source_payload": {},
            "source_to_target": matrix.tolist(),
        },
        "pose": make_pose(matrix @ local_offset(3, 2, 0), [96, 64], target),
    }
    result = run(
        tool(transform_mode=mode, absolute_tolerance=1e-5),
        {"image": ValuePacket(image, target), "template": template, "matches": [candidate]},
    )
    support = result.outputs["matches"][0]["support"]
    assert support["visible_fraction"] == 1 and support["matching_samples"] == 35
    np.testing.assert_allclose(support["template_to_image"], matrix, atol=1e-12)


def test_auto_uses_model_transform_or_bounds_before_offset_pick_pose():
    image, template, matches = scene()
    matches[0]["pose"] = make_pose(local_offset(70, 50, 40), [96, 64], {})
    result = run(tool(), {"image": image, "template": template, "matches": [matches[0]]})
    assert result.outputs["matches"][0]["support"]["visible_fraction"] == 1
    assert result.outputs["matches"][0]["pose"] == matches[0]["pose"]


def test_candidates_from_a_parent_image_map_into_a_known_crop_frame():
    from opencv_dataflow.payloads import moved_payload

    image, template, matches = scene()
    context = {"spatial": frame([96, 64], "camera")}
    matches[0]["pose"] = make_pose(local_offset(13, 14, 0), [96, 64], context)
    cropped = image[5:50, 5:80]
    crop_context = moved_payload(context, local_offset(5, 5, 0), [75, 45])
    result = run(
        tool(),
        {
            "image": ValuePacket(cropped, crop_context),
            "template": template,
            "matches": ValuePacket([matches[0]], context),
        },
    )
    assert result.outputs["count"] == 1
    np.testing.assert_array_equal(
        result.outputs["matches"][0]["support"]["template_to_image"], local_offset(3, 5, 0)
    )


@pytest.mark.parametrize("origin", [[], [1, 1]])
def test_ninety_degree_rotation_with_center_or_custom_pose_origin_is_exact(origin):
    template = (np.arange(35).reshape(5, 7) + 100).astype(np.uint16)
    image = np.zeros((30, 40), np.uint16)
    image[10:17, 8:13] = np.rot90(template)
    matrix = np.array([[0, 1, 8], [-1, 0, 16], [0, 0, 1]], float)
    anchor = origin or [3, 2]
    pose = make_pose(matrix @ local_offset(*anchor, 0), [40, 30], {})
    result = run(
        tool(
            transform_mode="pose",
            template_origin=origin,
            interpolation="nearest",
            absolute_tolerance=0,
        ),
        {"image": image, "template": template, "matches": [{"score": 0.9, "pose": pose}]},
    )
    assert result.outputs["matches"][0]["support"]["matching_samples"] == 35


@pytest.mark.parametrize("dtype", [np.uint16, np.float64])
def test_rgb_bgr_agreement_ignores_alpha_preserves_raw_units_and_expected_mask_holes(dtype):
    image = np.zeros((20, 30, 3), dtype)
    template = np.full((5, 7, 4), [100, 200, 300, 0], dtype)
    image[4:9, 8:15] = [300, 200, 100]
    image[5:7, 10:12] = 0
    mask = np.ones((5, 7), np.uint8)
    mask[1:3, 2:4] = 0
    source = ImageArray(image, "bgr")
    expected = ImageArray(template, "rgba")
    matches = records([(8, 4)], size=(7, 5), image_size=(30, 20))
    result = run(
        tool(comparison="all_channels", absolute_tolerance=0),
        {"image": source, "template": expected, "matches": matches, "template_mask": mask},
    )
    assert result.outputs["matches"][0]["support"]["matching_samples"] == 31
    assert result.outputs["matches"][0]["support"]["visible_fraction"] == 1
    assert result.outputs["image"].dtype == dtype
    failure = run(
        tool(comparison="all_channels", absolute_tolerance=0),
        {"image": source, "template": expected, "matches": matches},
    )
    assert failure.outputs["candidates"][0]["support"]["visible_fraction"] == 31 / 35


def test_pattern_envelope_metadata_and_existing_pick_selection_interoperate():
    image, template, matches = scene()
    envelope = {
        "model": "pattern_matches_v1",
        "items": matches,
        "count": 3,
        "pattern_id": "keep-me",
    }
    result = run(tool(), {"image": image, "template": template, "matches": envelope})
    assert result.outputs["matches"]["count"] == 2
    assert result.outputs["matches"]["pattern_id"] == "keep-me"
    pick = run(
        builtin_registry().create(
            "inspect.select_pick", parameters={"footprint": [-5, -4, 10, 8], "clearance": 0}
        ),
        {"image": packet_for(result, "image"), "matches": packet_for(result, "matches")},
    )
    assert pick.outputs["selected"]["pose"]["origin"] == [45, 39]
    assert pick.outputs["selected"]["score"] == 0.95


@pytest.mark.parametrize(
    "parameters",
    [
        {"minimum_visible_fraction": 1.1},
        {"absolute_tolerance": -1},
        {"maximum_candidates": True},
        {"minimum_matches": 3, "maximum_candidates": 2},
        {"template_origin": [1]},
        {"maximum_template_pixels": 2},
        {"maximum_sample_count": 1},
        {"maximum_working_bytes": 1},
    ],
)
def test_invalid_configuration_and_work_limits_emit_no_current_outputs(parameters):
    image, template, matches = scene()
    block = tool(**parameters)
    assert block.execute({"image": image, "template": template, "matches": matches}) is None
    assert block.error and not any(block.current_emissions.values())


def test_malformed_candidates_empty_selection_and_extreme_values_are_contained_and_recover():
    image, template, matches = scene()
    block = tool()
    for bad in [
        None,
        [{}],
        {"model": "pattern_matches_v1", "items": matches, "count": 2},
        [{"bounds": [0, 0, 7, 5]}],
        matches * 22,
    ]:
        assert block.execute({"image": image, "template": template, "matches": bad}) is None
        assert not any(block.current_emissions.values())
    assert (
        block.execute(
            {
                "image": image,
                "template": template,
                "matches": matches,
                "template_mask": np.zeros(template.shape, np.uint8),
            }
        )
        is None
    )
    recovered = run(block, {"image": image, "template": template, "matches": matches})
    assert recovered.passed and block.state == "pass"
    extreme = np.full((10, 12), 1e308)
    expected = np.full((5, 7), -1e308)
    assert (
        block.execute(
            {
                "image": extreme,
                "template": expected,
                "matches": records([(0, 0)], size=(7, 5), image_size=(12, 10)),
            }
        )
        is None
    )
    assert "finite comparison" in block.error


def test_no_candidates_is_valid_inspection_failure_with_optional_empty_success_and_no_graphics():
    image, template, _ = scene()
    for minimum in (0, 1):
        block = tool(minimum_matches=minimum)
        block.set_graphics_mode("off")
        result = run(block, {"image": image, "template": template, "matches": []})
        assert result.passed is (minimum == 0)
        assert result.outputs["count"] == 0 and not result.drawings
        assert bool(result.fail_reasons) is (minimum == 1)


def test_graphics_off_skips_drawable_construction_and_keeps_exact_numerical_results(monkeypatch):
    from opencv_dataflow import visible_support

    image, template, matches = scene()
    block = tool()
    block.set_graphics_mode("off")
    monkeypatch.setattr(
        visible_support,
        "graphic_object",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("graphics")),
    )
    result = run(block, {"image": image, "template": template, "matches": matches})
    assert not result.drawings and result.outputs["count"] == 2


def test_known_template_and_target_frames_tickets_and_native_errors_are_contained(monkeypatch):
    from opencv_dataflow import visible_support
    from opencv_dataflow.tickets import TicketClock

    image, template, matches = scene()
    block = tool()
    target = {
        "spatial": frame([96, 64], "camera"),
        "ticket": TicketClock("session").issue("camera"),
    }
    wrong = {"spatial": frame([96, 64], "elsewhere")}
    matches[0]["pose"] = make_pose(local_offset(13, 14, 0), [96, 64], wrong)
    assert (
        block.execute(
            {"image": ValuePacket(image, target), "template": template, "matches": matches}
        )
        is None
    )
    assert "coordinate roots" in block.error
    image, template, matches = scene()
    stale = copy.deepcopy(target)
    stale["ticket"]["value"] = 100
    matches[0]["pose"] = make_pose(local_offset(13, 14, 0), [96, 64], stale)
    assert (
        block.execute(
            {"image": ValuePacket(image, target), "template": template, "matches": matches}
        )
        is None
    )
    assert "tickets differ" in block.error
    reference = {"spatial": frame([11, 9], "reference")}
    record = {
        "transform": {
            "model": "image_pattern_pose_v1",
            "source_size": [11, 9],
            "target_size": [96, 64],
            "source_to_target": local_offset(8, 10, 0).tolist(),
            "source_payload": {"spatial": frame([11, 9], "wrong-reference")},
            "target_payload": target,
        }
    }
    assert (
        block.execute(
            {
                "image": ValuePacket(image, target),
                "template": ValuePacket(template, reference),
                "matches": [record],
            }
        )
        is None
    )
    assert "trained source grids" in block.error
    block.parameters["template_grid_policy"] = "assert_pixel_alignment"
    result = run(
        block,
        {
            "image": ValuePacket(image, target),
            "template": ValuePacket(template, reference),
            "matches": [record],
        },
    )
    assert result.outputs["count"] == 1
    assert result.outputs["report"]["template_grid_policy"] == "assert_pixel_alignment"
    image, template, matches = scene()
    monkeypatch.setattr(
        visible_support.cv2,
        "remap",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("native sampler")),
    )
    assert block.execute({"image": image, "template": template, "matches": matches}) is None
    assert "native sampler" in block.error and not any(block.current_emissions.values())


def test_xml_and_offscreen_inspector_expose_modes_without_processing(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.project_xml import export_project_xml, read_project_xml
    from opencv_dataflow.toolbox import block_entry

    graph = Graph()
    block = graph.add(tool(comparison="all_channels", template_origin=[2, 3]))
    snapshot = copy.deepcopy(graph.to_dict())
    path = tmp_path / "support.xml"
    export_project_xml(snapshot, path)
    assert Graph.from_dict(read_project_xml(path), builtin_registry()).to_dict() == snapshot
    assert "imgproc" in block_entry(block).path[1]
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        for name in ("transform_mode", "template_grid_policy", "comparison", "interpolation"):
            assert isinstance(window.inspector.fields[name], QComboBox)
        assert block.last_result is None
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()


def test_no_observed_pixels_cannot_be_accepted_even_with_zero_fraction_thresholds():
    image, template, matches = scene()
    result = run(
        tool(minimum_visible_fraction=0, minimum_observed_fraction=0),
        {
            "image": image,
            "template": template,
            "matches": matches,
            "mask": np.zeros(image.shape, np.uint8),
        },
    )
    assert result.outputs["count"] == 0 and not result.passed
    assert all(
        m["support"]["reasons"] == ["no_observed_support"] for m in result.outputs["candidates"]
    )
