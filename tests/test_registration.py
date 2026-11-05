import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format


def run(type_id, inputs, parameters=None):
    block = builtin_registry().create(type_id, parameters=parameters)
    block.execute(inputs)
    assert block.state in ("pass", "fail"), block.error
    return block


@pytest.mark.parametrize("kind", ["cv.warp_affine", "cv.warp_perspective"])
def test_warp_translation_preserves_encoding_and_inverse_geometry(kind):
    image = ImageArray(np.arange(12 * 15 * 3, dtype=np.uint8).reshape(12, 15, 3), "rgb")
    matrix = [[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]]
    if kind == "cv.warp_perspective":
        matrix.append([0.0, 0.0, 1.0])
    block = run(kind, {"image": image}, {"matrix": matrix, "interpolation": "nearest"})
    output = block.last_output_images["image"]
    np.testing.assert_array_equal(output[2:, 3:], image[:-2, :-3])
    np.testing.assert_array_equal(output[:2], 0)
    assert image_format(output) == "rgb"
    inverse = block.last_result.outputs["geometry"]["output_to_input"]
    points = run(
        "cv.transform_points",
        {"transform": block.last_result.outputs["geometry"], "points": [[3.0, 2.0], [6.0, 8.0]]},
    )
    np.testing.assert_allclose(points.last_result.outputs["points"], [[0, 0], [3, 6]])
    np.testing.assert_allclose(np.array(inverse) @ [3, 2, 1], [0, 0, 1])


def test_rotation_zero_identity_and_border_geometry():
    image = np.arange(15 * 20, dtype=np.uint16).reshape(15, 20)
    rotated = run("cv.rotate", {"image": image}, {"angle_degrees": 0.0})
    np.testing.assert_array_equal(rotated.last_output_images["image"], image)
    bordered = run(
        "cv.copy_border",
        {"image": image},
        {"left": 3, "top": 2, "right": 4, "bottom": 1, "value": 90.0},
    )
    output = bordered.last_output_images["image"]
    assert output.shape == (18, 27)
    np.testing.assert_array_equal(output[2:17, 3:23], image)
    assert output[0, 0] == 90
    np.testing.assert_allclose(
        np.array(bordered.last_result.outputs["geometry"]["output_to_input"]) @ [3, 2, 1], [0, 0, 1]
    )


def test_pyramids_filter_and_unsharp_native_results():
    image = np.random.default_rng(8).integers(0, 256, (32, 40), dtype=np.uint8)
    down = run("cv.pyramid", {"image": image})
    np.testing.assert_array_equal(down.last_output_images["image"], cv2.pyrDown(image))
    up = run("cv.pyramid", {"image": image}, {"direction": "up"})
    np.testing.assert_array_equal(up.last_output_images["image"], cv2.pyrUp(image))
    filtered = run("cv.filter2d", {"image": image}, {"kernel": [[1.0]]})
    np.testing.assert_array_equal(filtered.last_output_images["image"], image)
    unsharp = run("cv.unsharp", {"image": image}, {"amount": 0.0})
    np.testing.assert_array_equal(unsharp.last_output_images["image"], image)


def test_fast_and_orb_empty_features_are_valid():
    image = np.zeros((128, 128), np.uint8)
    for kind in ("inspect.fast", "inspect.orb"):
        block = run(kind, {"image": image})
        assert block.state == "pass" and block.last_result.outputs["count"] == 0
    features = run("inspect.orb", {"image": image}).last_result.outputs["features"]
    matched = run("inspect.match_orb", {"query": features, "train": features})
    assert matched.last_result.outputs["count"] == 0


def test_feature_registration_recovers_known_translation():
    image = np.random.default_rng(31).integers(0, 256, (320, 384), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.6)
    transform = np.array([[1.0, 0.0, 9.0], [0.0, 1.0, 13.0]])
    moved = cv2.warpAffine(image, transform, (384, 320))
    query = run("inspect.orb", {"image": image})
    train = run("inspect.orb", {"image": moved})
    assert query.last_result.outputs["count"] > 50
    matching = run(
        "inspect.match_orb",
        {
            "query": query.last_result.outputs["features"],
            "train": train.last_result.outputs["features"],
        },
    )
    assert matching.last_result.outputs["count"] > 20
    estimated = run("inspect.homography", {"matches": matching.last_result.outputs["matches"]})
    assert estimated.state == "pass"
    mapped = run(
        "cv.transform_points",
        {
            "transform": estimated.last_result.outputs["transform"],
            "points": [[100.0, 100.0], [200.0, 200.0]],
        },
    )
    np.testing.assert_allclose(
        mapped.last_result.outputs["points"], [[109, 113], [209, 213]], atol=2.0
    )


def test_ransac_outlier_rejection_direction_and_quality_status():
    query = np.array(
        [[0, 0], [100, 0], [0, 100], [100, 100], [50, 30], [70, 80], [20, 70], [80, 20]], float
    )
    train = query + [12, 27]
    train[-1] = [2000, -1000]
    matched = {"query_points": query.tolist(), "train_points": train.tolist()}
    block = run("inspect.homography", {"matches": matched}, {"ransac_pixels": 0.5})
    assert block.last_result.outputs["inliers"] == 7
    assert block.last_result.score == pytest.approx(7 / 8)
    matrix = block.last_result.outputs["transform"]
    reverse = run(
        "cv.transform_points", {"transform": matrix, "points": [[12.0, 27.0]]}, {"invert": True}
    )
    np.testing.assert_allclose(reverse.last_result.outputs["points"], [[0, 0]], atol=1e-4)
    block.parameters["min_inliers"] = 8
    block.execute({"matches": matched})
    assert block.state == "fail" and "transform" in block.last_result.outputs


def test_descriptor_contract_rejects_out_of_range_bytes():
    image = np.random.default_rng(1).integers(0, 256, (128, 128), dtype=np.uint8)
    features = run("inspect.orb", {"image": image}).last_result.outputs["features"]
    assert features["descriptors"]
    features["descriptors"][0][0] = 300
    block = builtin_registry().create("inspect.match_orb")
    block.execute({"query": features, "train": features})
    assert block.state == "error" and "descriptor" in block.error


@pytest.mark.parametrize(
    "kind,parameters",
    [
        ("cv.warp_affine", {"matrix": [[0, 0, 0], [0, 0, 0]]}),
        ("cv.warp_perspective", {"matrix": [[1, 0], [0, 1]]}),
        ("cv.rotate", {"scale": 0.0}),
        ("cv.filter2d", {"kernel": []}),
        ("cv.pyramid", {"levels": 0}),
        ("cv.unsharp", {"kernel": 4}),
        ("inspect.orb", {"scale_factor": 1.0}),
    ],
)
def test_invalid_parameters_clear_previous_images(kind, parameters):
    image = np.zeros((100, 120), np.uint8)
    block = run(kind, {"image": image})
    block.parameters.update(parameters)
    block.execute({"image": image})
    assert block.state == "error" and block.last_result is None and not block.last_output_images


def test_degenerate_homography_and_points_at_infinity():
    block = builtin_registry().create("inspect.homography")
    block.execute(
        {
            "matches": {
                "query_points": [[0, 0], [1, 1], [2, 2], [3, 3]],
                "train_points": [[0, 0], [1, 1], [2, 2], [3, 3]],
            }
        }
    )
    assert block.state == "error"
    points = builtin_registry().create(
        "cv.transform_points",
        parameters={"matrix": [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]},
    )
    points.execute({"points": [[0.0, 1.0]]})
    assert points.state == "error" and "infinity" in points.error


@pytest.mark.parametrize("descriptor_depth,mode", [("float32", "ratio"), ("uint8", "cross_check")])
def test_sift_features_match_a_known_translation_and_respect_roi(descriptor_depth, mode):
    image = np.random.default_rng(42).integers(0, 256, (192, 240), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.7)
    transform = np.float32([[1, 0, 7], [0, 1, 11]])
    shifted = cv2.warpAffine(image, transform, image.shape[::-1])
    mask = np.zeros_like(image)
    mask[20:172, 20:220] = 255
    query = run(
        "inspect.sift",
        {"image": image, "mask": mask},
        {"descriptor_depth": descriptor_depth, "precise_upscale": True},
    ).last_result
    train = run("inspect.sift", {"image": shifted}).last_result
    packet = query.outputs["features"]
    assert packet["algorithm"] == "SIFT" and packet["descriptor_depth"] == descriptor_depth
    assert np.asarray(packet["descriptors"]).shape == (query.outputs["count"], 128)
    for x, y in np.rint(query.outputs["points"]).astype(int):
        assert mask[y, x]
    assert len(query.drawings["SIFT"].objects) == query.outputs["count"]
    result = run(
        "inspect.match_sift",
        {"query": packet, "train": train.outputs["features"]},
        {"mode": mode},
    ).last_result
    assert result.outputs["count"] > 20
    estimate = run("inspect.homography", {"matches": result.outputs["matches"]})
    assert estimate.state == "pass"
    matrix = estimate.last_result.outputs["transform"]["source_to_target"]
    np.testing.assert_allclose(np.asarray(matrix)[:2, 2], [7, 11], atol=0.5)


def test_sift_matcher_handles_empty_features_and_rejects_other_descriptors():
    image = np.zeros((64, 80), np.uint8)
    blank = run("inspect.sift", {"image": image}).last_result.outputs["features"]
    result = run("inspect.match_sift", {"query": blank, "train": blank}).last_result
    assert result.outputs["count"] == 0
    orb = run("inspect.orb", {"image": image}).last_result.outputs["features"]
    block = builtin_registry().create("inspect.match_sift")
    assert block.execute({"query": blank, "train": orb}) is None
    assert block.error_kind == "invalid"
    assert "Connect SIFT" in block.error


@pytest.mark.parametrize(
    "descriptor_mode,bits,channels,width",
    [
        ("mldb", 0, "3", 61),
        ("mldb", 64, "3", 8),
        ("mldb_upright", 64, "3", 8),
        ("kaze", 0, "3", 64),
    ],
)
def test_akaze_registration_uses_matching_descriptor_settings(
    descriptor_mode, bits, channels, width
):
    image = np.random.default_rng(27).integers(0, 256, (192, 240), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.8)
    moved = cv2.warpAffine(image, np.float32([[1, 0, 7], [0, 1, 11]]), image.shape[::-1])
    mask = np.zeros_like(image)
    mask[20:170, 20:210] = 255
    p = {
        "descriptor_mode": descriptor_mode,
        "descriptor_bits": bits,
        "descriptor_channels": channels,
    }
    query = run("inspect.akaze", {"image": image, "mask": mask}, p).last_result
    train = run("inspect.akaze", {"image": moved}, p).last_result
    packet = query.outputs["features"]
    assert packet["algorithm"] == "AKAZE"
    assert packet["descriptor_size"] == width
    assert np.asarray(packet["descriptors"]).shape == (query.outputs["count"], width)
    for x, y in np.rint(query.outputs["points"]).astype(int):
        assert mask[y, x]
    matched = run(
        "inspect.match_akaze", {"query": packet, "train": train.outputs["features"]}
    ).last_result
    assert matched.outputs["count"] > 10
    homography = run("inspect.homography", {"matches": matched.outputs["matches"]})
    assert homography.state == "pass"
    transform = homography.last_result.outputs["transform"]["source_to_target"]
    np.testing.assert_allclose(np.asarray(transform)[:2, 2], [7, 11], atol=0.8)


@pytest.mark.parametrize("channels", ["1", "2"])
@pytest.mark.parametrize(
    "mode,bits", [("mldb", 0), ("mldb", 64), ("mldb_upright", 0), ("mldb_upright", 64)]
)
def test_akaze_rejects_unsafe_native_settings(channels, mode, bits):
    block = builtin_registry().create(
        "inspect.akaze",
        parameters={
            "descriptor_mode": mode,
            "descriptor_bits": bits,
            "descriptor_channels": channels,
        },
    )
    image = np.zeros((80, 120), np.uint8)
    assert block.execute({"image": image}) is None
    assert block.error_kind == "invalid"
    assert "MLDB requires three channels" in block.error


def test_akaze_matcher_rejects_incompatible_layouts_of_the_same_byte_width():
    image = np.zeros((80, 120), np.uint8)
    first = run("inspect.akaze", {"image": image}, {"descriptor_bits": 64}).last_result.outputs[
        "features"
    ]
    second = run("inspect.akaze", {"image": image}, {"descriptor_bits": 63}).last_result.outputs[
        "features"
    ]
    assert first["descriptor_size"] == second["descriptor_size"] == 8
    assert (
        run("inspect.match_akaze", {"query": first, "train": first}).last_result.outputs["count"]
        == 0
    )
    block = builtin_registry().create("inspect.match_akaze")
    assert block.execute({"query": first, "train": second}) is None
    assert block.error_kind == "invalid"
    assert "settings" in block.error or "bit counts" in block.error


@pytest.mark.parametrize(
    "detector,mode",
    [("inspect.sift", "ratio"), ("inspect.sift", "cross_check"), ("inspect.akaze", "ratio")],
)
def test_flann_recovers_known_shift_for_numeric_descriptors(detector, mode):
    image = np.random.default_rng(37).integers(0, 256, (180, 224), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.8)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 8], [0, 1, 12]]), image.shape[::-1])
    settings = {"descriptor_mode": "kaze"} if detector == "inspect.akaze" else {}
    query, train = [
        run(detector, {"image": view}, settings).last_result.outputs["features"]
        for view in (image, shifted)
    ]
    matched = run(
        "inspect.match_flann",
        {"query": query, "train": train},
        {"mode": mode, "trees": 4, "checks": 100},
    ).last_result
    assert matched.outputs["count"] > 15
    estimate = run("inspect.homography", {"matches": matched.outputs["matches"]})
    assert estimate.state == "pass"
    matrix = np.asarray(estimate.last_result.outputs["transform"]["source_to_target"])
    np.testing.assert_allclose(matrix[:2, 2], [8, 12], atol=0.5)


def test_flann_distance_and_single_neighbor_behavior():
    def packet(first_components):
        return {
            "algorithm": "SIFT",
            "descriptor_depth": "float32",
            "descriptor_size": 128,
            "image_size": [64, 64],
            "keypoints": [{"point": [5 + index, 10]} for index, _ in enumerate(first_components)],
            "descriptors": [[value] + [0] * 127 for value in first_components],
        }

    inputs = {"query": packet([1]), "train": packet([4])}
    block = run("inspect.match_flann", inputs)
    assert block.last_result.outputs["count"] == 0
    block.parameters.update(mode="cross_check", max_distance=3)
    result = block.execute(inputs)
    assert result is not None, block.error
    assert result.outputs["matches"]["matches"] == [
        {"query_index": 0, "train_index": 0, "distance": 3.0}
    ]
    block.parameters["max_distance"] = 2.9
    assert block.execute(inputs).outputs["count"] == 0


def test_flann_rejects_binary_akaze():
    image = np.zeros((64, 80), np.uint8)
    binary = run("inspect.akaze", {"image": image}).last_result.outputs["features"]
    block = builtin_registry().create("inspect.match_flann")
    assert block.execute({"query": binary, "train": binary}) is None
    assert block.error_kind == "invalid"
    assert "float AKAZE" in block.error
