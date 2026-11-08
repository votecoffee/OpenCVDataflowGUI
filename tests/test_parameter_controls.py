import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import preview_rgb


def run(kind, image, parameters, **extra):
    block = builtin_registry().create(kind, parameters=parameters)
    block.execute({"image": image, **extra})
    assert block.state == "pass", block.error
    return block


def test_native_gapi_parameters_match_opencv():
    image = np.random.default_rng(6).integers(0, 256, (40, 60), dtype=np.uint8)
    blur = run(
        "gapi.blur",
        image,
        {"kernel": 3, "kernel_y": 7, "sigma": 0.8, "sigma_y": 2.1, "border": "replicate"},
    )
    np.testing.assert_array_equal(
        blur.last_output_images["image"],
        cv2.GaussianBlur(image, (3, 7), 0.8, sigmaY=2.1, borderType=cv2.BORDER_REPLICATE),
    )
    edges = run(
        "gapi.canny", image, {"aperture": "5", "l2_gradient": True, "low": 40.0, "high": 120.0}
    )
    np.testing.assert_array_equal(
        edges.last_output_images["image"],
        cv2.Canny(image, 40, 120, apertureSize=5, L2gradient=True),
    )


def test_filter_anchor_depth_border_and_signed_preview():
    image = np.arange(80, dtype=np.uint8).reshape(8, 10)
    params = {
        "kernel": [[-1.0, 0.0, 1.0]],
        "anchor": [0, 0],
        "delta": 2.0,
        "border": "replicate",
        "output_depth": "int16",
    }
    block = run("cv.filter2d", image, params)
    expected = cv2.filter2D(
        image,
        cv2.CV_16S,
        np.array(params["kernel"]),
        anchor=(0, 0),
        delta=2,
        borderType=cv2.BORDER_REPLICATE,
    )
    np.testing.assert_array_equal(block.last_output_images["image"], expected)
    assert preview_rgb(block.last_output_images["image"]).dtype == np.uint8
    box = run(
        "cv.box_blur", image, {"kernel": 3, "kernel_y": 5, "anchor": [0, 1], "border": "replicate"}
    )
    np.testing.assert_array_equal(
        box.last_output_images["image"],
        cv2.blur(image, (3, 5), anchor=(0, 1), borderType=cv2.BORDER_REPLICATE),
    )


@pytest.mark.parametrize(
    "kind,kernel",
    [
        ("cv.filter2d", {"kernel": [[-1, -2], [3, 6]]}),
        ("cv.separable_filter", {"kernel_x": [1, 2], "kernel_y": [-1, 3]}),
    ],
)
def test_convolution_reverses_even_kernel_and_preserves_its_original_anchor(kind, kernel):
    image = np.array([[6, 17, 3], [10, 1, 14], [2, 8, 5]], np.uint8)
    block = run(
        kind,
        image,
        {
            **kernel,
            "operation": "convolution",
            "anchor": [0, -1],
            "border": "replicate",
            "output_depth": "int16",
            "delta": 2,
        },
    )
    # At each pixel, x weights the current and left samples, and y weights
    # the row below and the current row. The image boundary repeats its edge.
    signed = image.astype(np.int16)
    horizontal = signed + 2 * signed[:, [0, 0, 1]]
    expected = 3 * horizontal - horizontal[[1, 2, 2]] + 2
    np.testing.assert_array_equal(block.last_output_images["image"], expected)


@pytest.mark.parametrize(
    "mode,flag",
    [
        ("binary_inv", cv2.THRESH_BINARY_INV),
        ("trunc", cv2.THRESH_TRUNC),
        ("tozero", cv2.THRESH_TOZERO),
        ("tozero_inv", cv2.THRESH_TOZERO_INV),
    ],
)
def test_threshold_modes_and_physical_numeric_range(mode, flag):
    image = np.array([[0.0, 2.0, 4.0, 8.0]], np.float32)
    block = run("cv.threshold", image, {"value": 3.0, "max_value": 5.0, "mode": mode})
    np.testing.assert_array_equal(
        block.last_output_images["image"], cv2.threshold(image, 3, 5, flag)[1]
    )


def test_morphology_rectangular_anchor_and_constant_border():
    image = np.zeros((12, 16), np.uint8)
    image[0:6, 0:8] = 255
    parameters = {
        "operation": "erode",
        "kernel": 3,
        "kernel_y": 5,
        "shape": "cross",
        "shape_anchor": [0, 0],
        "anchor": [1, 2],
        "border": "constant",
        "border_value": [255.0],
    }
    block = run("cv.morphology", image, parameters)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 5), (0, 0))
    np.testing.assert_array_equal(
        block.last_output_images["image"],
        cv2.erode(image, kernel, anchor=(1, 2), borderType=cv2.BORDER_CONSTANT, borderValue=255),
    )


def test_contour_modes_and_offset_coordinates():
    image = np.zeros((30, 40), np.uint8)
    image[4:25, 5:30] = 255
    image[10:17, 12:20] = 0
    block = run(
        "inspect.contours",
        image,
        {
            "retrieval": "tree",
            "approximation": "none",
            "offset": [100, 200],
            "min_area": 1.0,
            "expected_count": 2,
        },
    )
    assert block.last_result.outputs["count"] == 2
    points = block.last_result.outputs["data"]["contours"][0]
    assert all(x >= 100 and y >= 200 for x, y in points)
    assert block.last_output_images["mask"].any()


def test_orb_wta_norm_and_mask_handling():
    image = np.random.default_rng(41).integers(0, 256, (180, 200), dtype=np.uint8)
    params = {
        "wta_k": "4",
        "score_type": "fast",
        "first_level": 1,
        "patch_size": 21,
        "edge_threshold": 21,
    }
    block = run("inspect.orb", image, params)
    features = block.last_result.outputs["features"]
    assert features["wta_k"] == 4 and features["descriptors"]
    matcher = builtin_registry().create("inspect.match_orb", parameters={"mode": "cross_check"})
    matcher.execute({"query": features, "train": features})
    assert matcher.state == "pass" and matcher.last_result.outputs["count"] > 0
    matcher.parameters["norm"] = "hamming"
    matcher.execute({"query": features, "train": features})
    assert "incompatible" in matcher.error
    masked = run("inspect.orb", image, params, mask=np.zeros_like(image))
    assert masked.last_result.outputs["count"] == 0
    corners = run("inspect.corners", image, {"gradient_size": "5"}, mask=np.zeros_like(image))
    assert corners.last_result.outputs["count"] == 0


def test_normalization_mask_distance_types_and_inverse_warp():
    image = np.array([[1.0, 2.0, 3.0, 4.0]], np.float32)
    normalized = run(
        "cv.normalize", image, {"norm_type": "l2", "norm_value": 2.0, "pixel_depth": "float32"}
    )
    assert np.linalg.norm(normalized.last_output_images["image"]) == pytest.approx(2.0)
    mask = np.zeros((9, 9), np.uint8)
    mask[2:7, 2:7] = 255
    l1 = run("cv.distance_transform", mask, {"distance_type": "l1", "output_depth": "uint8"})
    assert l1.last_output_images["distance"].dtype == np.uint8
    assert l1.last_output_images["distance"][4, 4] == 3
    source = np.arange(100, dtype=np.uint8).reshape(10, 10)
    matrix = [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]
    warped = run(
        "cv.warp_affine",
        source,
        {"matrix": matrix, "inverse_map": True, "interpolation": "nearest"},
    )
    np.testing.assert_array_equal(warped.last_output_images["image"][:-1, :-2], source[1:, 2:])
    np.testing.assert_allclose(
        warped.last_result.outputs["geometry"]["output_to_input"], matrix + [[0, 0, 1]]
    )


@pytest.mark.parametrize("method", ["direct", "lmeds", "rho"])
def test_homography_solver_dropdowns(method):
    points = np.random.default_rng(19).uniform(0, 200, (40, 2)).tolist()
    block = builtin_registry().create(
        "inspect.homography",
        parameters={
            "method": method,
            "max_iterations": 2000,
            "confidence": 0.99,
            "ransac_pixels": 0.1,
        },
    )
    block.execute(
        {"matches": {"query_points": points, "train_points": (np.array(points) + [7, 9]).tolist()}}
    )
    assert block.state == "pass", block.error
    np.testing.assert_allclose(
        np.array(block.last_result.outputs["transform"]["source_to_target"]),
        [[1, 0, 7], [0, 1, 9], [0, 0, 1]],
        atol=0.01,
    )


def test_every_declared_enum_has_valid_default_and_roundtrips():
    registry = builtin_registry()
    graph = Graph()
    for kind, cls in registry.types.items():
        block = graph.add(registry.create(kind))
        assert cls.parameter_choices.keys() <= cls.defaults.keys()
        block.validate_parameters()
    restored = Graph.from_dict(graph.to_dict(), registry)
    assert restored.to_dict() == graph.to_dict()


def test_png_encoder_controls_and_preservation(tmp_path):
    path = tmp_path / "image.png"
    image = np.random.default_rng(11).integers(0, 256, (30, 40, 3), dtype=np.uint8)
    block = builtin_registry().create(
        "output.image",
        parameters={"path": str(path), "png_compression": 9, "png_strategy": "filtered"},
    )
    block.execute({"image": image})
    assert block.state == "pass", block.error
    np.testing.assert_array_equal(
        cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), 1), image
    )
    original = path.read_bytes()
    block.execute({"image": image})
    assert block.state == "error" and path.read_bytes() == original


def test_blend_and_normalization_output_depths():
    registry = builtin_registry()
    image = np.array([[100, 200]], np.uint8)
    blend = registry.create(
        "cv.blend", parameters={"alpha": 1.0, "beta": 1.0, "output_depth": "float32"}
    )
    blend.execute({"image_a": image, "image_b": image})
    assert blend.state == "pass", blend.error
    np.testing.assert_array_equal(blend.last_output_images["image"], [[200.0, 400.0]])
    assert blend.last_output_images["image"].dtype == np.float32
    for depth, dtype in (("int16", np.int16), ("float64", np.float64)):
        normalized = run(
            "cv.normalize", image, {"minimum": -10.0, "maximum": 10.0, "pixel_depth": depth}
        )
        output = normalized.last_output_images["image"]
        assert output.dtype == dtype
        np.testing.assert_allclose(output, [[-10, 10]], atol=1e-12)
