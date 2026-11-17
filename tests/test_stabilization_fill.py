"""Independent border truth, current support, reprojection and guarded review-only data."""

import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.timing import advance_timing, mark

TYPE = "sequence.fill_stabilization_borders"
W, H = 40, 32


def group(values, contexts):
    return FlowGroup(
        tuple(ValuePacket(a, p) for a, p in zip(values, contexts, strict=True)),
        tuple(f"frame{i}" for i in range(len(values))),
    )


def fixture(dtype=np.uint8, encoding="gray", scale=1.0):
    def truth(offset):
        y, x = np.mgrid[:H, :W]
        a = (30 + (x + offset) * 2 + y).astype(np.float64)
        if encoding != "gray":
            a = np.stack((a, a / 2 + 5, a / 3 + 10), axis=2)
        if np.dtype(dtype).kind == "u":
            a *= 257 if dtype == np.uint16 else 1
        else:
            a *= scale / 255
        a = np.rint(a).astype(dtype) if np.dtype(dtype).kind == "u" else a.astype(dtype)
        if encoding in ("rgba", "bgra"):
            maximum = np.iinfo(dtype).max if np.dtype(dtype).kind == "u" else 1
            a = np.dstack((a, np.full((H, W), maximum, dtype)))
        return ImageArray(a, encoding)

    current, donor = truth(0), truth(2)
    observed = current.copy()
    current[:, -6:] = 0
    masks = [np.full((H, W), 255, np.uint8) for _ in range(2)]
    masks[0][:, -6:] = 0
    clock = TicketClock()
    contexts = [
        {
            "spatial": frame([W, H], "stabilized"),
            "ticket": clock.issue("camera", "external", str(i)),
            "custom": i,
        }
        for i in range(2)
    ]
    contexts[1]["spatial"]["to_root"][0][2] = 2
    return [current, donor], masks, contexts, observed


def execute(values=None, masks=None, contexts=None, **parameters):
    if values is None:
        values, masks, contexts, _ = fixture()
    block = builtin_registry().create(TYPE, parameters=parameters)
    result = block.execute({"frames": group(values, contexts), "coverage": group(masks, contexts)})
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("encoding", ["gray", "rgb", "bgr", "rgba", "bgra"])
def test_spatial_neighbor_fill_matches_independent_truth_without_expanding_measurement(
    dtype, encoding
):
    values, masks, contexts, expected = fixture(dtype, encoding)
    before = [a.copy() for a in values]
    _, result = execute(values, masks, contexts, interpolation="nearest", minimum_fill_fraction=1)
    assert result.passed
    np.testing.assert_array_equal(result.outputs["display"], expected)
    np.testing.assert_array_equal(result.outputs["image"], values[0])
    np.testing.assert_array_equal(result.outputs["coverage"], masks[0])
    np.testing.assert_array_equal(result.outputs["filled_mask"] != 0, masks[0] == 0)
    assert not np.any(result.outputs["coverage"] & result.outputs["filled_mask"])
    assert np.all(result.outputs["display_support"] == 255)
    assert not result.outputs["remaining_mask"].any()
    assert np.all(result.outputs["donor_index"][:, -6:] == 1)
    assert result.outputs["display"].dtype == dtype
    assert result.outputs["display"].color_format == encoding
    for original, after in zip(before, values, strict=True):
        np.testing.assert_array_equal(after, original)
    for name in ("image", "display", "coverage", "filled_mask", "report"):
        assert result.payloads[name]["ticket"] == contexts[0]["ticket"]
        assert result.payloads[name]["spatial"] == contexts[0]["spatial"]
    assert result.payloads["display"]["stabilization_border_fill"]["review_only"]
    assert not result.payloads["image"]["stabilization_border_fill"]["review_only"]


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
@pytest.mark.parametrize("blend", ["nearest", "mean"])
def test_float_linear_warp_and_mean_stay_finite_at_tiny_and_hdr_scales(scale, blend):
    values, masks, contexts, expected = fixture(np.float64, "rgba", scale)
    # Quarter-pixel translation has an analytic bilinear ramp value.
    contexts[1]["spatial"]["to_root"][0][2] = 2.25
    _, result = execute(values, masks, contexts, blend=blend, interpolation="linear")
    assert result.passed and np.isfinite(result.outputs["display"]).all()
    filled = result.outputs["filled_mask"] != 0
    expected = np.asarray(expected).copy()
    expected[..., :3] -= np.array([0.5, 0.25, 1 / 6]) * (scale / 255)
    np.testing.assert_allclose(
        result.outputs["display"][..., :3][filled] / scale,
        expected[..., :3][filled] / scale,
        atol=1e-15,
    )
    np.testing.assert_allclose(result.outputs["display"][..., 3][filled], 1, atol=1e-15)


def test_distance_limit_and_selection_do_not_shrink_the_requested_denominator():
    _, result = execute(maximum_fill_distance=1, unfilled="fail")
    assert not result.passed
    assert result.outputs["report"]["requested_pixels"] == H * 6
    assert result.outputs["report"]["fill_fraction"] == 1 / 6
    assert np.all(result.outputs["filled_mask"][:, -6] == 255)
    assert not result.outputs["filled_mask"][:, -5:].any()


def test_border_only_preserves_interior_holes_and_all_mode_can_fill_them():
    values, masks, contexts, expected = fixture()
    masks[0][12, 10], values[0][12, 10] = 0, 0
    _, border = execute(values, masks, contexts)
    assert border.passed and not border.outputs["filled_mask"][12, 10]
    assert border.outputs["remaining_mask"][12, 10]
    _, all_holes = execute(values, masks, contexts, holes="all")
    np.testing.assert_array_equal(all_holes.outputs["display"], expected)


def test_mask_selects_fill_centers_and_mean_uses_all_eligible_donors():
    values, masks, contexts, expected = fixture()
    values.append(values[1].copy() + np.uint8(2))
    masks.append(masks[1].copy())
    contexts.append(copy.deepcopy(contexts[1]))
    block = builtin_registry().create(TYPE, parameters={"blend": "mean"})
    selected = np.zeros((H, W), np.uint8)
    selected[4:16, -6:] = 255
    result = block.execute(
        {
            "frames": group(values, contexts),
            "coverage": group(masks, contexts),
            "mask": ValuePacket(selected, contexts[0]),
        }
    )
    assert result is not None and result.passed, block.error
    fill = selected != 0
    np.testing.assert_array_equal(result.outputs["display"][fill], expected[fill] + 1)
    assert not result.outputs["filled_mask"][~fill].any()
    assert np.all(result.outputs["contributors"][fill] == 2)
    assert result.outputs["report"]["filled_pixels"] == 12 * 6


def test_changed_overlap_has_a_local_halo_and_next_nearest_donor_can_fill():
    values, masks, contexts, _ = fixture()
    values.append(values[1].copy())
    masks.append(masks[1].copy())
    contexts.append(copy.deepcopy(contexts[1]))
    # Donor x=30:32 maps onto measured target x=32:34 immediately beside its hole.
    values[1][12:16, 30:32] = 255
    _, result = execute(values, masks, contexts, change_radius=3, interpolation="nearest")
    assert result.passed
    assert np.all(result.outputs["donor_index"][12:16, 34:37] == 2)
    assert np.all(result.outputs["donor_index"][:8, 34:] == 1)
    assert result.outputs["report"]["neighbors"][1]["changed_halo_pixels"] > 0


def test_inconsistent_donor_fails_availability_but_keeps_current_measured_data():
    values, masks, contexts, _ = fixture()
    values[1] += np.uint8(60)
    _, result = execute(values, masks, contexts, unfilled="fail")
    assert not result.passed and not result.outputs["filled_mask"].any()
    assert result.outputs["report"]["neighbors"][1]["state"] == "inconsistent_overlap"
    np.testing.assert_array_equal(result.outputs["display"], result.outputs["image"])


def test_different_segments_skip_or_reject_and_pixel_grid_requires_explicit_assertion():
    values, masks, contexts, _ = fixture()
    contexts[1]["spatial"]["root_id"] = "another-segment"
    _, skipped = execute(values, masks, contexts, unfilled="fail")
    assert not skipped.passed and not skipped.outputs["filled_mask"].any()
    assert skipped.outputs["report"]["neighbors"][1]["state"] == "different_segment"
    block = builtin_registry().create(TYPE, parameters={"different_segment": "error"})
    assert (
        block.execute({"frames": group(values, contexts), "coverage": group(masks, contexts)})
        is None
    )
    assert "segment" in block.error and not block.current_emissions.get("display")
    values[1] = values[0].copy()
    values[1][:, -6:] = 80
    for context in contexts:
        context.pop("spatial")
    _, explicit = execute(values, masks, contexts, alignment="pixel_grid")
    assert explicit.passed and explicit.outputs["filled_mask"].any()


def test_wrong_mask_grid_or_ticket_is_rejected_and_review_cannot_be_reused_as_measured():
    values, masks, contexts, _ = fixture()
    block = builtin_registry().create(TYPE)
    wrong = copy.deepcopy(contexts)
    wrong[1]["ticket"]["value"] = "stale"
    assert (
        block.execute({"frames": group(values, contexts), "coverage": group(masks, wrong)}) is None
    )
    assert "tickets" in block.error
    wrong = copy.deepcopy(contexts)
    wrong[1]["spatial"]["to_root"][0][2] = 9
    assert (
        block.execute({"frames": group(values, contexts), "coverage": group(masks, wrong)}) is None
    )
    assert "grids" in block.error
    contexts[1]["stabilization_border_fill"] = {"review_only": True}
    assert (
        block.execute({"frames": group(values, contexts), "coverage": group(masks, contexts)})
        is None
    )
    assert "review images" in block.error and not block.current_emissions.get("image")


@pytest.mark.parametrize(
    "parameters",
    [
        {"target_index": True},
        {"temporal_radius": 0},
        {"maximum_error": float("nan")},
        {"maximum_fill_distance": -1},
        {"minimum_fill_fraction": 1.1},
        {"maximum_working_bytes": 1.5},
    ],
)
def test_invalid_controls_report_validation_before_processing(parameters):
    block = builtin_registry().create(TYPE, parameters=parameters)
    with pytest.raises(ValueError):
        block.validate_parameters()
    assert block.last_result is None


def test_region_selects_only_current_grid_fill_centers():
    values, masks, contexts, _ = fixture()
    shape = builtin_registry().create("region.shape", parameters={"rectangle": [34, 4, 3, 12]})
    region = shape.execute({"image": ValuePacket(values[0], contexts[0])})
    assert region is not None, shape.error
    block = builtin_registry().create(TYPE)
    result = block.execute(
        {
            "frames": group(values, contexts),
            "coverage": group(masks, contexts),
            "region": ValuePacket(region.outputs["region"], region.payloads["region"]),
        }
    )
    assert result is not None and result.passed, block.error
    expected = np.zeros((H, W), bool)
    expected[4:16, 34:37] = True
    np.testing.assert_array_equal(result.outputs["filled_mask"] != 0, expected)


def test_transparency_does_not_become_geometric_fill_permission_or_donor_support():
    values, masks, contexts, _ = fixture(np.uint8, "rgba")
    values[0][10, 10, 3] = 0
    values[1][:, 34, 3] = 0  # maps to target x=36
    _, result = execute(values, masks, contexts, interpolation="nearest")
    assert result.passed
    assert not result.outputs["coverage"][10, 10]
    assert not result.outputs["filled_mask"][10, 10]
    assert not result.outputs["filled_mask"][:, 36].any()
    assert np.all(result.outputs["remaining_mask"][:, 36] == 255)
    assert np.all(result.outputs["filled_mask"][:, 35] == 255)


def test_workspace_precedes_conversion_and_native_error_is_contained_then_recovers(monkeypatch):
    from opencv_dataflow import stabilization_fill

    values, masks, contexts, _ = fixture()
    inputs = {"frames": group(values, contexts), "coverage": group(masks, contexts)}
    original = stabilization_fill._frames
    monkeypatch.setattr(
        stabilization_fill, "_frames", lambda *a: (_ for _ in ()).throw(AssertionError("converted"))
    )
    limited = builtin_registry().create(TYPE, parameters={"maximum_working_bytes": 1})
    assert limited.execute(inputs) is None and "converted" not in limited.error
    monkeypatch.setattr(stabilization_fill, "_frames", original)
    original_remap = cv2.remap
    monkeypatch.setattr(
        cv2, "remap", lambda *a, **k: (_ for _ in ()).throw(cv2.error("test remap"))
    )
    block = builtin_registry().create(TYPE)
    assert block.execute(inputs) is None and "test remap" in block.error
    assert not block.current_emissions.get("display")
    monkeypatch.setattr(cv2, "remap", original_remap)
    assert block.execute(inputs).passed


def test_graphics_off_preserves_numerical_outputs_and_current_pyramid_identity():
    values, masks, contexts, _ = fixture()
    start = mark()
    contexts[0]["processing_time"] = advance_timing(None, start, start, source=True)
    contexts[0]["pyramid"] = {
        "level": 1,
        "largest_size": [W * 2, H * 2],
        "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
    }
    block = builtin_registry().create(TYPE)
    block.set_graphics_mode("off")
    outer = TicketClock().issue("batch", "external", "outer")
    result = block.execute(
        {
            "frames": ValuePacket(group(values, contexts), {"ticket": outer}),
            "coverage": group(masks, contexts),
        }
    )
    assert result is not None and result.passed, block.error
    assert result.outputs["preview"] is NO_VALUE and not result.drawings
    assert result.payloads["display"]["pyramid"] == contexts[0]["pyramid"]
    assert result.payloads["display"]["ticket"] == contexts[0]["ticket"] != outer
    assert result.payloads["display"]["processing_time"]["started_ns"] == start["ns"]


def test_offscreen_editor_and_xml_expose_choices_without_filling(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.project_xml import export_project_xml, read_project_xml
    from opencv_dataflow.toolbox import block_entry

    graph = Graph()
    block = graph.add(builtin_registry().create(TYPE, parameters={"blend": "mean"}))
    saved = graph.to_dict()
    path = tmp_path / "review.xml"
    export_project_xml(saved, path)
    restored = Graph.from_dict(read_project_xml(path), builtin_registry())
    assert restored.to_dict() == saved
    restored.close()
    assert "video" in block_entry(block).path[1].lower()
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        for field in ("alignment", "interpolation", "consistency", "holes", "blend", "unfilled"):
            assert isinstance(window.inspector.fields[field], QComboBox)
        assert block.last_result is None
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
