"""Qualify BM3D stages, selected writes, and native build availability."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from opencv_dataflow.bm3d_denoising import Bm3dDenoise
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/bm3d_denoising"
TYPE = "cv.bm3d_denoise"
_NATIVE = any(
    line.strip().startswith("Non-free algorithms:") and line.rstrip().endswith("YES")
    for line in cv2.getBuildInformation().splitlines()
)


@pytest.fixture(scope="module")
def registry():
    from opencv_dataflow.blocks import builtin_registry

    return builtin_registry()


@pytest.fixture(scope="module")
def fixture_pair():
    # Pillow also works in the isolated native OpenCV build without imgcodecs.
    return tuple(
        np.asarray(Image.open(EXAMPLE / f"assets/{name}.png")) for name in ("clean", "noisy")
    )


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_portable_graph_reduces_noise_and_retains_step(registry, fixture_pair, profile):
    clean, noisy = fixture_pair
    graph = Graph.load(EXAMPLE / "bm3d_denoising.json", registry)
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["denoise"]
        assert block_entry(block).path[:2] == (
            "OpenCV",
            "Extended Computational Photography (xphoto)",
        )
        result = block.last_result
        assert result and result.passed, block.error
        report = result.outputs["report"]
        assert report["backend"] == "portable" and report["stage"] == "both"
        final, basic = result.outputs["image"], result.outputs["basic"]
        assert final.shape == noisy.shape and final.dtype == noisy.dtype
        errors = [np.sqrt(np.mean((a.astype(float) - clean) ** 2)) for a in (noisy, basic, final)]
        assert 13.0 < errors[0] < 13.2
        assert errors[2] < errors[1] < 3.0
        step = float(final[10:20, 59:65].mean() - final[10:20, 48:54].mean())
        assert 90.0 < step < 92.5
        assert result.payloads["image"]["spatial"]["root_id"] == "noisy:image"
        assert graph.blocks["filmstrip"].last_result.passed
    finally:
        graph.close()


def test_portable_separate_stages_with_float32_basic(registry, fixture_pair):
    image = fixture_pair[1][20:49, 43:76].astype(np.float32) / 255
    settings = {"noise_sigma": 13, "search_window": 13, "group_size": 8}
    both = registry.create(TYPE, parameters=settings).execute({"image": image})
    generator = registry.create(TYPE, parameters={**settings, "stage": "hard_threshold"})
    hard = generator.execute({"image": image})
    assert hard and hard.passed, generator.error
    np.testing.assert_array_equal(hard.outputs["basic"], both.outputs["basic"])
    refiner = registry.create(TYPE, parameters={**settings, "stage": "wiener"})
    refined = refiner.execute({"image": image, "basic": hard.outputs["image"]})
    assert refined and refined.passed, refiner.error
    assert refined.outputs["image"].dtype == np.float32
    np.testing.assert_allclose(refined.outputs["image"], both.outputs["image"], atol=2e-6, rtol=0)


def test_uint16_selected_writes_and_context(registry, fixture_pair):
    image = fixture_pair[1][0:30, 42:78].astype(np.uint16) * 257
    mask = np.zeros(image.shape, np.uint8)
    mask[5:21, 4:31] = 255
    spatial = frame([36, 30], "camera")
    ticket = TicketClock("bm3d").issue("camera", "fixed", "noisy")
    tool = registry.create(TYPE, parameters={"noise_sigma": 13, "search_window": 13})
    result = tool.execute(
        {
            "image": ValuePacket(image, {"spatial": spatial, "ticket": ticket}),
            "mask": ValuePacket(mask, {"spatial": spatial}),
        }
    )
    assert result and result.passed, tool.error
    selected = mask != 0
    for output in ("image", "basic"):
        value = result.outputs[output]
        np.testing.assert_array_equal(value[~selected], image[~selected])
        assert value.dtype == np.uint16 and np.any(value[selected] != image[selected])
        assert result.payloads[output]["ticket"] == ticket
        assert result.payloads[output]["spatial"] == spatial


@pytest.mark.parametrize(
    "params,shape,error",
    [
        ({"patch_size": 6}, (32, 32), "power of two"),
        ({"patch_size": 16}, (12, 32), "dimensions must fit"),
        ({"patch_size": 8, "sliding_step": 8}, (32, 32), "smaller than the patch"),
        ({"search_window": 8}, (32, 32), "must exceed"),
        ({"stage": "wiener"}, (32, 32), "Connect Basic"),
        ({"search_window": 127, "sliding_step": 1}, (128, 128), "comparison budget"),
    ],
)
def test_validation_guards_before_filtering(registry, params, shape, error):
    tool = registry.create(TYPE, parameters=params)
    assert tool.execute({"image": np.full(shape, 140, np.uint8)}) is None
    assert error in tool.error


@pytest.mark.skipif(_NATIVE, reason="Stock-wheel diagnostic requires BM3D disabled")
def test_disabled_native_backend_has_actionable_error(registry, fixture_pair):
    tool = registry.create(TYPE, parameters={"backend": "opencv"})
    assert tool.execute({"image": fixture_pair[1]}) is None
    assert "portable backend" in tool.error


@pytest.mark.skipif(not _NATIVE, reason="Requires a locally enabled native OpenCV build")
@pytest.mark.parametrize("norm,scale", (("l2", 1), ("l1", 257)))
def test_native_enabled_quality_and_split_stages(fixture_pair, norm, scale):
    clean, noisy = fixture_pair
    original = noisy if scale == 1 else noisy.astype(np.uint16) * scale
    settings = {"backend": "opencv", "noise_sigma": 13, "norm": norm}
    before = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        both = Bm3dDenoise(parameters=settings).execute({"image": original})
        hard = Bm3dDenoise(parameters={**settings, "stage": "hard_threshold"})
        basic = hard.execute({"image": original})
        assert basic and basic.passed, hard.error
        wiener = Bm3dDenoise(parameters={**settings, "stage": "wiener"})
        refined = wiener.execute({"image": original, "basic": basic.outputs["image"]})
        assert both and both.passed and refined and refined.passed, wiener.error
        np.testing.assert_array_equal(both.outputs["basic"], basic.outputs["image"])
        np.testing.assert_array_equal(both.outputs["image"], refined.outputs["image"])
        native_basic, native_final = cv2.xphoto.bm3dDenoising(
            original,
            np.empty_like(original),
            np.empty_like(original),
            h=13.0 * scale,
            templateWindowSize=8,
            searchWindowSize=19,
            blockMatchingStep1=2500 * scale**2,
            blockMatchingStep2=400 * scale**2,
            groupSize=16,
            slidingStep=3,
            beta=2.0,
            normType=cv2.NORM_L2 if norm == "l2" else cv2.NORM_L1,
            step=cv2.xphoto.BM3D_STEPALL,
            transformType=cv2.xphoto.HAAR,
        )
        np.testing.assert_array_equal(both.outputs["basic"], native_basic)
        np.testing.assert_array_equal(both.outputs["image"], native_final)
        final = both.outputs["image"].astype(np.float64) / scale
        rms = np.sqrt(np.mean((final - clean) ** 2))
        contrast = final[10:20, 59:65].mean() - final[10:20, 48:54].mean()
        assert rms < 3.6 and contrast > 89
    finally:
        cv2.setNumThreads(before)
