"""Validate native fuzzy filters and bounded, selected repair."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.fuzzy_restoration import FuzzyFilter, FuzzyInpaint
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/fuzzy_restoration"


def rmse(actual, expected):
    return float(np.sqrt(np.mean((actual.astype(float) - expected) ** 2)))


@pytest.fixture
def ramp():
    return tuple(
        cv2.imread(str(EXAMPLE / f"assets/{name}.png"), cv2.IMREAD_GRAYSCALE)
        for name in ("clean", "noisy", "damaged")
    )


@pytest.mark.parametrize(("profile", "workers"), [("debug", 1), ("lean", 3)])
def test_saved_example_reduces_noise_and_repairs_known_hole(profile, workers, ramp):
    clean, noisy, damaged = ramp
    graph = Graph.load(EXAMPLE / "fuzzy_restoration.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(workers)
        graph.run()
        block = graph.blocks["smooth"]
        assert block_entry(block).path[:2] == ("OpenCV", "Fuzzy Image Processing (fuzzy)")
        result = block.last_result
        assert result and result.passed, block.error
        assert rmse(noisy, clean) == pytest.approx(7.05563, abs=1e-5)
        assert rmse(result.outputs["image"], clean) == pytest.approx(1.41416, abs=1e-5)
        keep = damaged == 0
        assert keep.sum() == 49 and rmse(damaged[keep], clean[keep]) > 99
        for name, passes, error in (
            ("one_step", 1, 1.23718),
            ("multi_step", 3, 1.23718),
            ("iterative", 3, 0.93678),
        ):
            node = graph.blocks[name]
            result = node.last_result
            assert result and result.passed, node.error
            assert result.outputs["report"]["used_radius"] == 4
            assert result.outputs["report"]["passes"] == passes
            np.testing.assert_array_equal(result.outputs["repair_mask"] != 0, keep)
            recovered = result.outputs["image"]
            np.testing.assert_array_equal(recovered[~keep], damaged[~keep])
            assert rmse(recovered[keep], clean[keep]) == pytest.approx(error, abs=1e-5)
            assert result.payloads["image"]["spatial"]["root_id"] == "damaged:image"
            assert result.payloads["image"]["ticket"]["value"] == "damaged.png"
        assert graph.blocks["filmstrip"].last_result.passed
    finally:
        graph.close()


@pytest.mark.parametrize(
    ("algorithm", "native", "radius"),
    [
        ("one_step", cv2.ft.ONE_STEP, 4),
        ("multi_step", cv2.ft.MULTI_STEP, 2),
        ("iterative", cv2.ft.ITERATIVE, 2),
    ],
)
def test_bounded_methods_match_native_on_aligned_crop(ramp, algorithm, native, radius):
    clean, _, damaged = ramp
    missing = clean != damaged
    # Both canvas axes are multiples of each visited radius (2, 3 and 4).
    expected = cv2.ft.inpaint(damaged, (~missing).astype(np.uint8), radius, cv2.ft.LINEAR, native)
    block = FuzzyInpaint(parameters={"algorithm": algorithm, "radius": radius, "maximum_radius": 4})
    result = block.execute({"image": damaged, "mask": missing.astype(np.uint8) * 255})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"][missing], np.rint(expected[missing]))


@pytest.mark.parametrize("radius", [1, 3, 6])
def test_filter_keeps_flat_odd_sized_color_and_alpha(radius):
    pixels = np.full((23, 27, 4), (3105, 5820, 10660, 32500), np.uint16)
    source = ImageArray(pixels, "rgba")
    block = FuzzyFilter(parameters={"radius": radius})
    output = block.execute({"image": source})
    assert output and output.passed, block.error
    actual = output.outputs["image"]
    assert actual.color_format == "rgba" and actual.dtype == np.uint16
    np.testing.assert_array_equal(actual, source)


def test_identity_filter_does_not_consume_native_work_budget(monkeypatch):
    monkeypatch.setattr("opencv_dataflow.fuzzy_restoration._TAP_BUDGET", 0)
    source = np.arange(63, dtype=np.uint8).reshape(7, 9)
    block = FuzzyFilter(parameters={"radius": 1})
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], source)


def test_radius_one_axis_keeps_nonuniform_values_in_that_direction():
    columns = np.linspace(1e32, 3e32, 29, dtype=np.float32)
    source = np.repeat(columns[None, :], 21, axis=0)
    block = FuzzyFilter(parameters={"kernel": "custom", "horizontal_weights": [0, 1, 0]})
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    np.testing.assert_allclose(result.outputs["image"], source, rtol=1e-6)


def test_custom_kernel_matches_native_and_selection_preserves_frame(ramp):
    _, noisy, _ = ramp
    x = [0, 0.2, 1, 0.8, 0]
    y = [0, 0.5, 0.75, 1, 0.5, 0.25, 0]
    kernel = cv2.ft.createKernel1(np.float32(x)[None, :], np.float32(y)[:, None], 1)
    expected = np.rint(cv2.ft.filter(noisy, kernel)).astype(np.uint8)
    mask = np.zeros(noisy.shape, np.uint8)
    mask[8:52, 10:65] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [96, 72],
        "shape": "rectangle",
        "rectangle": [40, 20, 35, 43],
    }
    context = frame([96, 72], "camera")
    block = FuzzyFilter(
        parameters={
            "kernel": "custom",
            "horizontal_weights": x,
            "vertical_weights": y,
        }
    )
    result = block.execute(
        {
            "image": ValuePacket(noisy, {"spatial": context}),
            "mask": ValuePacket(mask, {"spatial": context}),
            "region": region,
        }
    )
    assert result and result.passed, block.error
    keep = np.zeros_like(mask, dtype=bool)
    keep[20:52, 40:65] = True
    np.testing.assert_array_equal(result.outputs["image"], np.where(keep, expected, noisy))
    assert result.payloads["image"]["spatial"] == context


def test_fuzzy_inpaint_reports_unsupported_and_bounded_cases(ramp, monkeypatch):
    _, _, image = ramp
    mask = (image == 0).astype(np.uint8)
    one_pass = FuzzyInpaint(parameters={"algorithm": "one_step", "radius": 2})
    assert one_pass.execute({"image": image, "mask": mask}) is None
    assert "support is incomplete" in one_pass.error
    inpaint = FuzzyInpaint(parameters={"algorithm": "iterative", "radius": 2, "maximum_radius": 3})
    assert inpaint.execute({"image": image, "mask": mask}) is None
    assert "maximum radius" in inpaint.error
    assert inpaint.execute({"image": image, "mask": np.ones_like(mask)}) is None
    assert "known pixels" in inpaint.error
    monkeypatch.setattr(
        cv2.ft, "FT02D_iteration", lambda *_: pytest.fail("Empty repair invoked native processing")
    )
    result = inpaint.execute({"image": image, "mask": np.zeros_like(mask)})
    assert result and result.passed and result.outputs["report"]["passes"] == 0
    np.testing.assert_array_equal(result.outputs["image"], image)


def test_rejects_nonpartitioning_membership_profile(ramp):
    tool = FuzzyFilter(parameters={"kernel": "custom", "horizontal_weights": [0, 0.2, 1, 0.3, 0]})
    assert tool.execute({"image": ramp[0]}) is None
    assert "overlaps" in tool.error
