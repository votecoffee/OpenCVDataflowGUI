"""Exercise segmentation labels, selection, and quality metrics through the registry."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame


def test_flood_fill_mask_is_a_barrier():
    image = np.full((9, 11), 72, np.uint8)
    selection = np.full_like(image, 255)
    selection[:, 4] = 0
    fill = builtin_registry().create("inspect.flood_fill", parameters={"seed_y": 4})
    result = fill.execute({"image": image, "mask": selection})
    assert result is not None, fill.error
    assert result.outputs["area"] == 36
    assert result.outputs["bounds"] == [0, 0, 4, 9]
    assert np.all(result.outputs["mask"][:, :4] == 255)
    assert not result.outputs["mask"][:, 4:].any()


@pytest.mark.parametrize("mode", ["binary_components", "labels"])
def test_watershed_maintains_seed_ids_and_input(mode):
    image = np.zeros((30, 40), np.uint8)
    image[:, 20:] = 255
    markers = np.zeros_like(image, dtype=np.uint8 if mode == "binary_components" else np.int32)
    values = (1, 2) if mode == "binary_components" else (7, 24)
    markers[12, 10], markers[12, 30] = values
    original = markers.copy()
    block = builtin_registry().create("inspect.watershed", parameters={"marker_mode": mode})
    result = block.execute({"image": image, "markers": markers})
    assert result is not None, block.error
    assert result.outputs["count"] == 2
    assert tuple(result.outputs["labels"][12, [10, 30]]) == values
    assert result.outputs["labels"].dtype == np.int32
    assert result.outputs["boundaries"][:, 19:22].any()
    np.testing.assert_array_equal(markers, original)


@pytest.mark.parametrize("metric", ["ssim", "mse", "psnr"])
def test_quality_defect_and_selection(metric):
    reference = np.full((40, 45), 120, np.uint8)
    sample = reference.copy()
    sample[3:10, 2:12] = 20
    selection = np.zeros_like(reference)
    selection[22:, :] = 255
    threshold = {"ssim": 0.95, "mse": 20, "psnr": 45}[metric]
    block = builtin_registry().create(
        "inspect.image_quality",
        parameters={"metric": metric, "check_threshold": True, "threshold": threshold},
    )
    full = block.execute({"image_a": reference, "image_b": sample})
    assert full is not None, block.error
    assert not full.passed
    assert full.fail_reasons[0]["code"] == "image_quality_out_of_tolerance"
    if metric == "psnr":
        assert full.outputs["map"] is NO_VALUE
    else:
        assert full.outputs["map"].shape == reference.shape
    masked = block.execute({"image_a": reference, "image_b": sample, "mask": selection})
    assert masked is not None, block.error
    assert masked.passed


@pytest.mark.parametrize("mode", ["rectangle", "binary_mask", "labels"])
def test_grabcut_modes(mode):
    image = np.full((64, 64), 20, np.uint8)
    image[24:40, 24:40] = 220
    markers = np.zeros_like(image)
    markers[16:48, 16:48] = 3 if mode == "labels" else 255
    block = builtin_registry().create(
        "inspect.grabcut", parameters={"mode": mode, "rectangle": [16, 16, 32, 32]}
    )
    result = block.execute({"image": image, "markers": markers})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["mask"] != 0, image == 220)


def test_grabcut_seed_masks_override_probable_labels_and_keep_hard_seeds():
    registry = builtin_registry()
    image = np.full((64, 64), 20, np.uint8)
    image[22:42, 20:40] = 220
    expected = image == 220
    likely = np.zeros_like(image)
    likely[16:48, 14:46] = 255
    core = np.zeros_like(image)
    core[24:30, 22:28] = 255
    hard_background = np.zeros_like(image)
    hard_background[17:20, 16:19] = 255
    seeds = registry.create("inspect.grabcut_mask_seeds")
    result = seeds.execute({
        "image": image, "probable_foreground": likely, "foreground": core,
        "background": hard_background,
    })
    assert result is not None, seeds.error
    markers = result.outputs["markers"]
    assert result.outputs["counts"] == {
        "background": 9, "foreground": 36, "probable_background": 3072,
        "probable_foreground": 979,
    }
    assert np.all(markers[core != 0] == cv2.GC_FGD)
    assert np.all(markers[hard_background != 0] == cv2.GC_BGD)
    assert result.outputs["preview"].shape == (64, 64, 3)

    block = registry.create("inspect.grabcut", parameters={"mode": "labels"})
    original = markers.copy()
    cut = block.execute({"image": image, "markers": markers})
    assert cut is not None, block.error
    np.testing.assert_array_equal(cut.outputs["mask"] != 0, expected)
    assert np.all(cut.outputs["labels"][core != 0] == cv2.GC_FGD)
    assert np.all(cut.outputs["labels"][hard_background != 0] == cv2.GC_BGD)
    np.testing.assert_array_equal(markers, original)


@pytest.mark.parametrize("prefix", ["", "probable_"])
def test_grabcut_seed_masks_reject_equal_certainty_conflicts(prefix):
    image = np.zeros((12, 15), np.uint8)
    mask = image.copy()
    mask[3:7, 3:7] = 255
    seeds = builtin_registry().create("inspect.grabcut_mask_seeds")
    assert seeds.execute({"image": image, prefix + "foreground": mask,
                          prefix + "background": mask}) is None
    assert "equal certainty overlap" in seeds.error


def test_grabcut_seed_masks_require_same_frame_and_both_training_classes():
    image = np.zeros((12, 15), np.uint8)
    mask = image.copy()
    mask[3:7, 3:7] = 255
    seeds = builtin_registry().create("inspect.grabcut_mask_seeds")
    assert seeds.execute({"image": image}) is None
    assert "both foreground and background samples" in seeds.error
    assert seeds.execute({
        "image": ValuePacket(image, {"spatial": frame([15, 12], "camera_a")}),
        "foreground": ValuePacket(mask, {"spatial": frame([15, 12], "camera_b")}),
    }) is None
    assert "image coordinate frame" in seeds.error


@pytest.mark.parametrize("method", ["zhang_suen", "guo_hall"])
def test_thinning_removes_width_and_respects_selection(method):
    image = np.zeros((24, 28), np.uint8)
    image[8:15, 2:26] = 255
    selection = np.zeros_like(image)
    selection[:, :20] = 255
    block = builtin_registry().create("inspect.thin_mask", parameters={"method": method})
    result = block.execute({"image": image, "mask": selection})
    assert result is not None, block.error
    skeleton = result.outputs["mask"]
    assert 5 < np.count_nonzero(skeleton) < 20
    assert not skeleton[:, 20:].any()


def test_fill_holes_leaves_edge_connected_background_and_large_holes():
    image = np.zeros((30, 30), np.uint8)
    image[2:28, 2:28] = 255
    image[2:12, 10] = 0
    image[6:9, 14:17] = 0
    image[16:22, 16:22] = 0
    block = builtin_registry().create("inspect.fill_mask_holes", parameters={"maximum_area": 10})
    result = block.execute({"image": image})
    assert result is not None, block.error
    assert result.outputs["count"] == 1
    assert np.count_nonzero(result.outputs["holes"]) == 9
    assert result.outputs["mask"][7, 15] == 255
    assert result.outputs["mask"][7, 10] == result.outputs["mask"][17, 17] == 0


def test_distance_markers_split_touching_parts_and_exclude_background():
    image = np.zeros((64, 88), np.uint8)
    cv2.circle(image, (29, 32), 18, 255, -1)
    cv2.circle(image, (57, 32), 18, 255, -1)
    registry = builtin_registry()
    seeds = registry.create("inspect.distance_markers")
    result = seeds.execute({"image": image})
    assert result is not None, seeds.error
    assert result.outputs["count"] == 2
    markers = result.outputs["markers"]
    assert markers.dtype == np.int32
    assert tuple(markers[32, [29, 57]]) == (2, 3)
    assert markers[32, 43] == 0
    assert markers[5, 5] == 1
    watershed = registry.create(
        "inspect.watershed", parameters={"marker_mode": "labels", "background_label": 1}
    )
    segmented = watershed.execute({"image": image, "markers": markers})
    assert segmented is not None, watershed.error
    assert segmented.outputs["count"] == 2
    assert segmented.outputs["mask"][5, 5] == 0
    assert segmented.outputs["boundaries"][32, 43] == 255
    empty = seeds.execute({"image": np.zeros_like(image)})
    assert empty.outputs["count"] == 0
    assert (
        watershed.execute({"image": image, "markers": empty.outputs["markers"]}).outputs["count"]
        == 0
    )


@pytest.mark.parametrize("method", ["dilation", "erosion"])
def test_reconstruction_respects_barriers_and_reports_limit(method):
    mask = np.zeros((8, 20), np.uint16)
    mask[2:5, 2:10] = 300
    mask[2:5, 11:17] = 200
    marker = np.zeros_like(mask)
    marker[3, 2] = 250
    expected = np.zeros_like(mask)
    expected[2:5, 2:10] = 250
    if method == "erosion":
        mask, marker, expected = (1000 - image for image in (mask, marker, expected))
    block = builtin_registry().create(
        "cv.morph_reconstruct", parameters={"method": method, "connectivity": "4"}
    )
    result = block.execute({"image": mask, "marker": marker})
    assert result is not None, block.error
    assert result.outputs["info"]["converged"]
    assert result.outputs["iterations"] == 8
    np.testing.assert_array_equal(result.outputs["image"], expected)
    block.parameters["maximum_iterations"] = 1
    limited = block.execute({"image": mask, "marker": marker})
    assert limited is not None, block.error
    assert not limited.passed
    assert limited.outputs["image"] is NO_VALUE
    assert limited.fail_reasons[0]["code"] == "reconstruction_iteration_limit"


@pytest.mark.parametrize(
    "metric,full_value,selected_value",
    [
        ("mae", 22.5, 20),
        ("rmse", np.sqrt(875), np.sqrt(500)),
        ("max_absolute_error", 50, 30),
    ],
)
def test_absolute_quality_metrics_reduce_only_selected_pixels(metric, full_value, selected_value):
    a = np.array([[100, 100], [100, 100]], dtype=np.uint16)
    b = np.array([[90, 50], [130, 100]], dtype=np.uint16)
    mask = np.array([[255, 0], [255, 0]], np.uint8)
    block = builtin_registry().create(
        "inspect.image_quality",
        parameters={
            "metric": metric,
            "check_threshold": True,
            "threshold": (full_value + selected_value) / 2,
        },
    )
    full = block.execute({"image_a": a, "image_b": b})
    assert full is not None, block.error
    assert full.outputs["value"] == pytest.approx(full_value)
    assert full.outputs["measurements"]["units"] == "sample"
    assert not full.passed
    masked = block.execute({"image_a": a, "image_b": b, "mask": mask})
    assert masked is not None, block.error
    assert masked.outputs["value"] == pytest.approx(selected_value)
    assert masked.passed
    np.testing.assert_array_equal(masked.outputs["map"], [[10, 50], [30, 0]])
    assert masked.outputs["map"].dtype == np.float64
    assert masked.graphics["absolute error"][0, 1] == 255


@pytest.mark.parametrize("metric", ["mae", "rmse"])
def test_absolute_quality_reductions_handle_high_range_float_images(metric):
    image = np.full((2, 3), 1e308)
    block = builtin_registry().create("inspect.image_quality", parameters={"metric": metric})
    result = block.execute({"image_a": image, "image_b": np.zeros_like(image)})
    assert result is not None, block.error
    assert result.outputs["value"] == pytest.approx(1e308)
    assert block.execute({"image_a": image, "image_b": -image}) is None
    assert block.error_kind == "invalid" and "float64 range" in block.error


def test_mean_absolute_error_passes_at_the_exact_threshold():
    clean = np.zeros((96, 160), np.uint8)
    sample = clean.copy()
    sample[20:44, 24:56] = 40
    sample[50:66, 96:112] = 60
    block = builtin_registry().create(
        "inspect.image_quality",
        parameters={"metric": "mae", "check_threshold": True, "threshold": 3},
    )
    result = block.execute({"image_a": clean, "image_b": sample})
    assert result is not None, block.error
    assert result.outputs["value"] == 3
    assert result.passed
