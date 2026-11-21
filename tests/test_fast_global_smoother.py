"""Check native fast global smoothing, typed outputs and registered guidance."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

TYPE = "cv.fast_global_smoother"
EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/fast_global_smoother"


def smoother(**parameters):
    return builtin_registry().create(TYPE, parameters=parameters)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
def test_native_depth_and_all_solver_parameters(dtype):
    noise = np.random.default_rng(4385).integers(0, 250, (31, 57), dtype=np.uint8)
    guide = noise.copy()
    guide[:, :29] = 60
    guide[:, 29:] = 180
    if dtype == np.uint8:
        sample = noise
    elif dtype == np.int16:
        sample = noise.astype(dtype) * 101 - 13000
    else:
        sample = noise.astype(dtype) * 37.5 - 5000
    block = smoother(lambda_weight=287, sigma_color=17, lambda_attenuation=0.6, iterations=5)
    result = block.execute({"image": sample, "guide": guide})
    assert result and result.passed, block.error
    expected = cv2.ximgproc.createFastGlobalSmootherFilter(guide, 287, 17, 0.6, 5).filter(sample)
    image = result.outputs["image"]
    assert image.dtype == dtype and image.shape == sample.shape
    np.testing.assert_array_equal(image, expected)


def test_rgb16_preserves_alpha_and_pixels_outside_mask_region_intersection():
    rng = np.random.default_rng(5446)
    original = rng.integers(0, 65536, (32, 40, 4), dtype=np.uint16)
    source = ImageArray(original, "rgba")
    reference = np.zeros_like(source)
    reference[:, 20:, :3] = 65535
    reference[..., 3] = original[..., 3]
    guide = ImageArray(reference, "rgba")
    mask = np.zeros(original.shape[:2], np.uint8)
    mask[5:22, 8:26] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [40, 32],
        "shape": "rectangle",
        "rectangle": [13, 9, 20, 12],
    }
    block = smoother()
    result = block.execute({"image": source, "guide": guide, "mask": mask, "region": region})
    assert result and result.passed, block.error
    out = result.outputs["image"]
    ref8 = np.where(np.asarray(reference[..., :3]) > 0, 255, 0).astype(np.uint8)
    expected = cv2.ximgproc.createFastGlobalSmootherFilter(ref8, 500, 5, 0.25, 3).filter(
        original[..., :3].astype(np.float32)
    )
    expected = np.rint(np.clip(expected, 0, 65535)).astype(np.uint16)
    selected = np.zeros_like(mask, dtype=bool)
    selected[9:21, 13:26] = True
    assert image_format(out) == "rgba" and out.dtype == np.uint16
    np.testing.assert_array_equal(out[..., :3][selected], expected[selected])
    np.testing.assert_array_equal(out[~selected], original[~selected])
    np.testing.assert_array_equal(out[..., 3], original[..., 3])
    np.testing.assert_array_equal(source, original)


def test_signed_self_guidance_and_disabled_smoothing():
    img = np.random.default_rng(77).integers(-32768, 32768, (11, 23), dtype=np.int16)
    block = smoother(sigma_color=28)
    result = block.execute({"image": img})
    assert result and result.passed, block.error
    guide = np.rint((img.astype(np.int32) + 32768) / 257).astype(np.uint8)
    native = cv2.ximgproc.createFastGlobalSmootherFilter(guide, 500, 28, 0.25, 3).filter(img)
    np.testing.assert_array_equal(result.outputs["image"], native)
    empty = block.execute({"image": img, "mask": np.zeros_like(img, np.uint8)})
    assert empty and empty.passed, block.error
    np.testing.assert_array_equal(empty.outputs["image"], img)
    block.parameters["lambda_weight"] = 0
    result = block.execute({"image": img})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], img)


def test_guide_frame_validation_and_source_ticket():
    source = np.arange(256, dtype=np.uint8).reshape(16, 16)
    clock = TicketClock("filter")
    sample_ticket = clock.issue("camera", "fixed", "current-part")
    guide_ticket = clock.issue("reference", "fixed", "reference-part")
    model = frame([16, 16], "part")
    ports = {
        "image": ValuePacket(source, {"spatial": model, "ticket": sample_ticket}),
        "guide": ValuePacket(
            source, {"spatial": frame([16, 16], "reference"), "ticket": guide_ticket}
        ),
    }
    strict = smoother()
    assert strict.execute(ports) is None and "share the source pixel frame" in strict.error
    registered = smoother(guide_alignment="pixel_grid")
    result = registered.execute(ports)
    assert result and result.passed, registered.error
    assert result.payloads["image"]["ticket"] == sample_ticket
    assert result.payloads["image"]["spatial"] == model
    assert registered.execute({"image": source, "guide": source[:, :-1]}) is None
    assert "same dimensions" in registered.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_recovers_small_structure_and_preserves_output_selection(profile):
    graph = Graph.load(EXAMPLE / "fast_global_smoother.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        graph.run()
        output = {
            name: graph.blocks[name].last_result.outputs["image"]
            for name in ("noisy", "self_guided", "guided", "selected")
        }
        truth = cv2.imread(str(EXAMPLE / "assets/ground_truth.png"), cv2.IMREAD_UNCHANGED)
        noisy_rms = np.sqrt(np.mean((output["noisy"].astype(float) - truth) ** 2))
        self_rms = np.sqrt(np.mean((output["self_guided"].astype(float) - truth) ** 2))
        guide_rms = np.sqrt(np.mean((output["guided"].astype(float) - truth) ** 2))
        assert guide_rms < 0.5 < self_rms < noisy_rms * 0.6
        edge = output["guided"][10:80, 96].astype(float) - output["guided"][10:80, 95]
        assert np.mean(edge) == pytest.approx(130, abs=1)
        write = np.zeros_like(truth, dtype=bool)
        write[8:72, 16:176] = True
        np.testing.assert_array_equal(output["selected"][write], output["guided"][write])
        np.testing.assert_array_equal(output["selected"][~write], output["noisy"][~write])
    finally:
        graph.close()
