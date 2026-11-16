"""Known blur truth, changed-object protection, support and public app contracts."""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock

TYPE = "sequence.restore_neighbors"


def batch(images, payloads=None):
    contexts = payloads or [
        {"spatial": frame(list(images[0].shape[1::-1]), "registered")} for _ in images
    ]
    return FlowGroup(
        tuple(ValuePacket(a, p) for a, p in zip(images, contexts, strict=True)),
        tuple(f"frame-{i}" for i in range(len(images))),
    )


def run(images, **parameters):
    block = builtin_registry().create(TYPE, parameters=parameters)
    result = block.execute({"frames": batch(images)})
    assert result is not None, block.error
    return block, result


def truth(dtype=np.uint8, encoding="gray", scale=1):
    y, x = np.mgrid[:80, :128]
    # Non-native independent analytic signal and horizontal finite box average.
    sharp = 0.5 + 0.22 * np.sin(x * 0.6) + 0.16 * np.cos(y * 0.43)
    if encoding != "gray":
        sharp = np.dstack((sharp, np.roll(sharp, 3, 1), np.roll(sharp, 2, 0)))
        if encoding in ("rgba", "bgra"):
            sharp = np.dstack((sharp, np.full((80, 128), 0.8)))
    if np.dtype(dtype).kind == "u":
        sharp = np.rint(sharp * np.iinfo(dtype).max).astype(dtype)
    else:
        sharp = (sharp * scale).astype(dtype)
    blurred = sharp.copy()
    values = sharp[..., :3] if encoding in ("rgba", "bgra") else sharp
    padded = np.pad(
        values, ((0, 0), (3, 3), *(((0, 0),) if values.ndim == 3 else ())), mode="reflect"
    )
    # Divide before summing so extreme floating references stay finite.
    box = sum(padded[:, i : i + 128].astype(np.float64) / 7 for i in range(7))
    if np.dtype(dtype).kind == "u":
        box = np.rint(box)
    if encoding in ("rgba", "bgra"):
        blurred[..., :3] = box.astype(dtype)
    else:
        blurred = box.astype(dtype)
    return ImageArray(sharp, encoding), ImageArray(blurred, encoding)


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32, np.float64))
@pytest.mark.parametrize("encoding", ("gray", "rgb", "bgr", "rgba", "bgra"))
def test_known_box_blur_recovers_independent_sharp_truth_preserving_depth_and_alpha(
    dtype, encoding
):
    sharp, blurred = truth(dtype, encoding)
    _, result = run([sharp, blurred, sharp], target_index=1)
    restored = result.outputs["restored_mask"] != 0
    assert result.passed and restored.mean() > 0.55
    np.testing.assert_allclose(
        result.outputs["image"][restored], sharp[restored], rtol=1e-7, atol=0
    )
    np.testing.assert_array_equal(result.outputs["image"][~restored], blurred[~restored])
    assert result.outputs["image"].dtype == dtype
    assert result.outputs["image"].color_format == encoding
    if encoding in ("rgba", "bgra"):
        np.testing.assert_array_equal(result.outputs["image"][..., 3], blurred[..., 3])
    error_before = np.mean((sharp.astype(float) - blurred.astype(float)) ** 2)
    error_after = np.mean((sharp.astype(float) - result.outputs["image"].astype(float)) ** 2)
    assert error_after < error_before * 0.4
    assert np.all(result.outputs["donor_index"][restored] == 0)
    assert result.outputs["info"]["neighbors"][2]["contributed_pixels"] == 0


@pytest.mark.parametrize("scale", (1e-300, 1.0, 1e300))
@pytest.mark.parametrize("blend", ("best", "weighted"))
def test_shared_scale_and_convex_mix_are_finite_for_tiny_and_hdr_samples(scale, blend):
    sharp, blurred = truth(np.float64, scale=scale)
    _, result = run([blurred, sharp, sharp], blend=blend, mix=0.4)
    selected = result.outputs["restored_mask"] != 0
    assert selected.any()
    expected = 0.6 * (blurred / scale) + 0.4 * (sharp / scale)
    np.testing.assert_allclose(
        result.outputs["image"][selected] / scale, expected[selected], rtol=1e-12, atol=0
    )
    assert np.isfinite(result.outputs["image"]).all()
    counts = result.outputs["contributor_count"]
    assert np.all(counts[selected] == 2)
    for entry in result.outputs["contributions"].items[1:]:
        assert np.any(entry.value)  # both donors contribute in weighted, earliest in best
        if blend == "best":
            break


@pytest.mark.parametrize("change", ("moved", "appeared", "disappeared"))
def test_changed_foreground_is_preserved_without_duplicate_parts(change):
    sharp, _ = truth(np.float64)
    target_truth, donor = sharp.copy(), sharp.copy()
    if change != "disappeared":
        target_truth[28:45, 32:48] = 1
    if change != "appeared":
        donor[28:45, 76:92] = 1
    blurred = cv2.blur(target_truth, (7, 1), borderType=cv2.BORDER_REFLECT_101)
    _, result = run([blurred, donor], maximum_model_error=0.002, change_radius=5)
    output = result.outputs["image"]
    for x in (32, 76) if change == "moved" else (32,) if change == "appeared" else (76,):
        np.testing.assert_array_equal(output[23:50, x - 5 : x + 21], blurred[23:50, x - 5 : x + 21])
    # Far static detail improves; no target foreground is erased and no donor-only part is copied.
    assert np.mean((output[10:20, 10:100] - target_truth[10:20, 10:100]) ** 2) < np.mean(
        (blurred[10:20, 10:100] - target_truth[10:20, 10:100]) ** 2
    )
    assert result.outputs["info"]["neighbors"][1]["changed_pixels"] > 0


def test_coverage_and_selection_keep_original_target_support_and_copy_every_other_pixel():
    sharp, blurred = truth()
    block = builtin_registry().create(TYPE)
    masks = [np.full(sharp.shape, 255, np.uint8) for _ in range(2)]
    masks[0][:, :8] = 0
    masks[1][30:40, 40:50] = 0
    mask = np.zeros(sharp.shape, np.uint8)
    mask[15:65, 15:100] = 255
    result = block.execute(
        {"frames": batch([blurred, sharp]), "coverage": batch(masks), "mask": mask}
    )
    assert result is not None, block.error
    selected = result.outputs["restored_mask"] != 0
    assert selected.any() and not selected[:, :15].any()
    assert not selected[27:43, 37:53].any()
    np.testing.assert_array_equal(result.outputs["valid_mask"], masks[0])
    np.testing.assert_array_equal(result.outputs["image"][~selected], blurred[~selected])
    np.testing.assert_array_equal(masks[1][30:40, 40:50], 0)


@pytest.mark.parametrize("policy,passed", (("pass_through", True), ("fail", False)))
def test_unchanged_and_failed_no_donor_are_explicit(policy, passed):
    sharp, _ = truth()
    _, result = run([sharp, sharp], no_neighbor=policy)
    assert result.passed is passed
    assert result.outputs["info"]["state"] == "unchanged"
    assert not np.any(result.outputs["restored_mask"])
    np.testing.assert_array_equal(result.outputs["image"], sharp)
    _, minimum = run([sharp, sharp], minimum_restored_fraction=0.1)
    assert not minimum.passed


def test_temporal_radius_and_target_ticket_do_not_substitute_neighbor_identity_or_outer_ticket():
    sharp, blurred = truth()
    clock = TicketClock()
    contexts = [
        {
            "spatial": frame([128, 80], "root"),
            "ticket": clock.issue("camera", "external", str(i)),
            "custom": i,
        }
        for i in range(4)
    ]
    contexts[2]["pyramid"] = {
        "level": 2,
        "largest_size": [512, 320],
        "to_largest": [[4, 0, 0], [0, 4, 0], [0, 0, 1]],
    }
    block = builtin_registry().create(TYPE, parameters={"target_index": 2, "temporal_radius": 1})
    group = batch([sharp, blurred, blurred, sharp], contexts)
    result = block.execute(
        {"frames": ValuePacket(group, {"ticket": clock.issue("batch", "external", "outer")})}
    )
    assert result is not None, block.error
    assert result.payloads["image"]["ticket"] == contexts[2]["ticket"]
    assert result.payloads["image"]["pyramid"] == contexts[2]["pyramid"]
    assert result.payloads["image"]["custom"] == 2
    assert result.outputs["info"]["neighbors"][0]["state"] == "outside_radius"
    assert np.any(result.outputs["donor_index"] == 3)
    for entry in result.outputs["contributions"].items:
        assert entry.payload["ticket"] == contexts[2]["ticket"]
    assert result.outputs["info"]["neighbors"][3]["ticket"] == contexts[3]["ticket"]


@pytest.mark.parametrize("model", ("vertical_box", "gaussian"))
def test_other_native_models_recover_sharp_pixels(model):
    sharp, _ = truth(np.float64)
    blurred = (
        cv2.GaussianBlur(sharp, (7, 7), 1.5) if model == "gaussian" else cv2.blur(sharp, (1, 7))
    )
    _, result = run([blurred, sharp], blur_model=model, maximum_model_error=1e-12)
    restored = result.outputs["restored_mask"] != 0
    assert restored.mean() > 0.5
    np.testing.assert_array_equal(result.outputs["image"][restored], sharp[restored])


@pytest.mark.parametrize(
    "case",
    (
        "depth",
        "encoding",
        "nonfinite",
        "missing_grid",
        "grid",
        "coverage",
        "target",
        "empty_selection",
    ),
)
def test_invalid_inputs_are_contained_and_recover_on_next_activation(case):
    sharp, blurred = truth()
    values = {"frames": batch([blurred, sharp])}
    if case == "depth":
        values["frames"] = batch([blurred, sharp.astype(np.float32)])
    elif case == "encoding":
        invalid = ImageArray(np.zeros((80, 128, 3), np.uint8), "hsv")
        values["frames"] = batch([invalid, invalid])
    elif case == "nonfinite":
        invalid = blurred.astype(float)
        invalid[5, 5] = np.nan
        values["frames"] = batch([invalid, invalid])
    elif case == "missing_grid":
        values["frames"] = batch([blurred, sharp], [{}, {}])
    elif case == "grid":
        values["frames"] = batch(
            [blurred, sharp],
            [{"spatial": frame([128, 80], "a")}, {"spatial": frame([128, 80], "b")}],
        )
    elif case == "coverage":
        values["coverage"] = batch([np.ones((80, 128), np.float32)] * 2)
    elif case == "target":
        pass
    else:
        values["mask"] = np.zeros(sharp.shape, np.uint8)
    block = builtin_registry().create(
        TYPE, parameters={"target_index": 5} if case == "target" else {}
    )
    assert block.execute(values) is None
    assert block.error_kind == "invalid"
    block.parameters["target_index"] = 0
    assert block.execute({"frames": batch([blurred, sharp])}).passed


def test_workspace_guard_runs_before_filtering_and_state_or_global_threads_are_not_changed(
    monkeypatch,
):
    import opencv_dataflow.neighbor_restoration as module

    def unexpected(*args, **kwargs):
        raise AssertionError("Oversized inputs reached native filtering")

    monkeypatch.setattr(module, "focus_energy", unexpected)
    image = np.zeros((1024, 1024), np.uint8)
    block = builtin_registry().create(TYPE)
    assert block.execute({"frames": batch([image, image])}) is None
    assert block.error_kind == "invalid" and "256 MiB" in block.error


def test_weighted_relative_energy_avoids_underflow_within_a_high_dynamic_range_capture():
    sharp, blurred = truth(np.float64, scale=1e-160)
    sharp[0, 0] = blurred[0, 0] = 1  # Shared scale stays one; interior energy is subnormal.
    _, result = run([blurred, sharp, sharp], blend="weighted", minimum_energy=0)
    restored = result.outputs["restored_mask"] != 0
    assert restored[20:60, 20:100].any()
    np.testing.assert_allclose(
        result.outputs["image"][restored], sharp[restored], rtol=1e-12, atol=0
    )


def test_mask_region_intersection_selects_centers_without_changing_filter_context():
    sharp, blurred = truth()
    _, full = run([blurred, sharp])
    block = builtin_registry().create(TYPE)
    mask = np.zeros(sharp.shape, np.uint8)
    mask[10:65, 10:75] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [128, 80],
        "shape": "rectangle",
        "rectangle": [35, 20, 60, 35],
    }
    result = block.execute({"frames": batch([blurred, sharp]), "mask": mask, "region": region})
    assert result is not None, block.error
    selected = np.zeros(sharp.shape, bool)
    selected[20:55, 35:75] = True
    expected = np.where(selected, full.outputs["restored_mask"], 0)
    np.testing.assert_array_equal(result.outputs["restored_mask"], expected)
    np.testing.assert_array_equal(result.outputs["image"][~selected], blurred[~selected])


def test_readonly_parallel_execution_graph_xml_and_graphics_off(tmp_path):
    sharp, blurred = truth()
    sharp.setflags(write=False)
    blurred.setflags(write=False)
    count = cv2.getNumThreads()
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda _: run([blurred, sharp])[1], range(4)))
    for result in results[1:]:
        np.testing.assert_array_equal(result.outputs["image"], results[0].outputs["image"])
    assert cv2.getNumThreads() == count
    registry = builtin_registry()
    graph = Graph()
    try:
        block = graph.add(
            registry.create(TYPE, parameters={"blend": "weighted", "blur_model": "gaussian"})
        )
        expected = graph.to_dict()
        path = tmp_path / "restoration.xml"
        export_project_xml(expected, path)
        reopened = Graph.from_dict(read_project_xml(path), registry)
        try:
            assert reopened.to_dict() == expected
        finally:
            reopened.close()
        block.set_graphics_mode("off")
        result = block.execute({"frames": batch([blurred, sharp])})
        assert result is not None, block.error
        assert result.outputs["preview"] is NO_VALUE and not result.graphics and not result.drawings
    finally:
        graph.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("blur_size", 4),
        ("energy_window", 0),
        ("mix", 0),
        ("maximum_model_error", -1),
        ("blur_model", "unknown"),
    ],
)
def test_parameter_constraints_fail_before_processing(field, value):
    sharp, blurred = truth()
    block = builtin_registry().create(TYPE, parameters={field: value})
    assert block.execute({"frames": batch([blurred, sharp])}) is None
    assert block.error_kind == "invalid"


def test_validated_numeric_values_are_typed_locally_without_changing_saved_parameters():
    sharp, blurred = truth()
    parameters = {
        "target_index": 1.0,
        "blur_size": 7.0,
        "change_radius": 3.0,
        "energy_window": 5.0,
        "mix": "1.0",
        "minimum_energy": "0.000001",
    }
    block, result = run([sharp, blurred, sharp], **parameters)
    assert result.passed and result.outputs["info"]["target_index"] == 1
    for name, value in parameters.items():
        assert block.parameters[name] == value and type(block.parameters[name]) is type(value)


def test_native_exception_is_contained_and_later_execution_recovers(monkeypatch):
    import opencv_dataflow.neighbor_restoration as module

    sharp, blurred = truth()
    block = builtin_registry().create(TYPE)
    native = module._model

    def broken(*args):
        raise cv2.error("Controlled model filter failure")

    monkeypatch.setattr(module, "_model", broken)
    assert block.execute({"frames": batch([blurred, sharp])}) is None
    assert block.error_kind == "exception" and not any(block.current_emissions.values())
    monkeypatch.setattr(module, "_model", native)
    assert block.execute({"frames": batch([blurred, sharp])}).passed


def test_explicit_grid_assertion_and_changed_alpha_are_conservative():
    sharp, blurred = truth(np.uint16, "rgba")
    sharp[25:40, 50:65, 3] = 0
    block = builtin_registry().create(TYPE, parameters={"alignment": "pixel_grid"})
    result = block.execute({"frames": batch([blurred, sharp], [{}, {}])})
    assert result is not None, block.error
    assert not result.outputs["restored_mask"][22:43, 47:68].any()
    np.testing.assert_array_equal(result.outputs["image"][..., 3], blurred[..., 3])


def test_offscreen_editor_exposes_blur_and_blend_dropdowns(app):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    window = MainWindow()
    graph = Graph()
    block = graph.add(builtin_registry().create(TYPE))
    window.replace_graph(graph)
    try:
        window.inspector.select(block)
        app.processEvents()
        combos = window.inspector.findChildren(QComboBox)
        choices = [{combo.itemText(i) for i in range(combo.count())} for combo in combos]
        assert any({"horizontal_box", "vertical_box", "gaussian"} <= values for values in choices)
        assert any({"best", "weighted"} <= values for values in choices)
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
