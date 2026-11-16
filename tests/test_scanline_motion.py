"""Independent row geometry, native track estimation and paired dense-flow correction."""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.remap_maps import COORDINATE_MAP_TAG
from opencv_dataflow.scanline_motion import EstimateScanlineMotion, fit_rows
from opencv_dataflow.tickets import TicketClock

TYPE = "inspect.scanline_motion"


def displacement(y, height):
    u = 2 * y / (height - 1) - 1
    return np.stack((1.3 + 2.1 * u - 0.7 * u**2 + 0.9 * u**3, 0.4 * u**2 - 0.25), axis=-1)


def points(height=240, width=320):
    y, x = np.meshgrid(
        np.linspace(6, height - 7, 20), np.linspace(12, width - 13, 12), indexing="ij"
    )
    reference = np.stack((x.ravel(), y.ravel()), axis=1)
    return reference, reference + displacement(reference[:, 1], height)


def bilinear(image, x, y):
    left, top = np.floor(x).astype(int), np.floor(y).astype(int)
    result = np.zeros(x.shape)
    for dy in (0, 1):
        for dx in (0, 1):
            xx, yy = left + dx, top + dy
            good = (xx >= 0) & (xx < image.shape[1]) & (yy >= 0) & (yy < image.shape[0])
            weight = (x - left if dx else 1 - (x - left)) * (y - top if dy else 1 - (y - top))
            result[good] += image[yy[good], xx[good]] * weight[good]
    return result


def fixture(dtype=np.uint8, encoding="gray", scale=1):
    h, w = 240, 320
    rng = np.random.default_rng(169)
    reference = cv2.GaussianBlur(rng.uniform(0.15, 0.85, (h, w)), (5, 5), 0.8)
    # Independent inverse of monotone row mapping, followed by array bilinear sampling.
    yy, xx = np.mgrid[:h, :w].astype(float)
    low, high = np.full((h, w), -16.0), np.full((h, w), h + 16.0)
    for _ in range(40):
        middle = (low + high) / 2
        before = middle + displacement(middle, h)[..., 1] < yy
        low = np.where(before, middle, low)
        high = np.where(before, high, middle)
    source_y = (low + high) / 2
    source_x = xx - displacement(source_y, h)[..., 0]
    moving = bilinear(reference, source_x, source_y)
    if encoding != "gray":
        reference = np.dstack((reference, reference * 0.8, reference * 0.6))
        moving = np.dstack((moving, moving * 0.8, moving * 0.6))
        if encoding in ("rgba", "bgra"):
            reference = np.dstack((reference, np.ones((h, w))))
            moving = np.dstack((moving, np.ones((h, w))))
    if np.dtype(dtype).kind == "u":
        maximum = np.iinfo(dtype).max
        reference, moving = (
            np.rint(value * maximum).astype(dtype) for value in (reference, moving)
        )
    else:
        reference, moving = ((value * scale).astype(dtype) for value in (reference, moving))
        if encoding in ("rgba", "bgra"):
            reference[..., 3] = moving[..., 3] = 1
    clock = TicketClock()
    target_payload = {"spatial": frame([w, h], "reference"), "ticket": clock.issue("reference")}
    source_payload = {
        "spatial": frame([w, h], "camera"),
        "ticket": clock.issue("camera"),
        "custom": {"encoder": 17},
    }
    return ImageArray(moving, encoding, source_payload), ImageArray(
        reference, encoding, target_payload
    )


def estimate(image, reference, **parameters):
    block = builtin_registry().create(TYPE, parameters=parameters)
    result = block.execute({"image": image, "reference": reference})
    assert result is not None, block.error
    return block, result


def warp(image, reference, result):
    block = builtin_registry().create("cv.warp_flow", parameters={"consistency": "off"})
    output = block.execute(
        {
            "image": image,
            "target": reference,
            "flow": result.outputs["flow"],
            "flow_support": ImageArray(
                result.outputs["support"], "gray", result.payloads["support"]
            ),
        }
    )
    assert output is not None, block.error
    return output


def test_robust_polynomial_fit_has_independent_coefficients_holdout_and_edge_predictions():
    reference, moving = points()
    moving[3::13] += [8, -5]
    fitted, report = fit_rows(reference, moving, 240, EstimateScanlineMotion.defaults)
    assert fitted is not None, report["reason"]
    values, support = fitted
    np.testing.assert_allclose(values, displacement(np.arange(240), 240), atol=1e-10)
    assert support.all()
    np.testing.assert_allclose(
        report["coefficients_xy"], [[1.3, -0.25], [2.1, 0], [-0.7, 0.4], [0.9, 0]], atol=1e-10
    )
    assert report["training_inlier_ratio"] > 0.8 and report["validation_inlier_ratio"] > 0.8


@pytest.mark.parametrize(
    "case,reason",
    [
        ("few", "insufficient_tracks"),
        ("degenerate", "degenerate_rows"),
        ("holdout", "validation_rejected"),
        ("span", "insufficient_row_coverage"),
        ("gap", "insufficient_row_coverage"),
        ("motion", "excess_motion"),
        ("fold", "unsafe_row_geometry"),
    ],
)
def test_unusable_or_unsafe_motion_never_produces_a_field(case, reason):
    reference, moving = points()
    p = dict(EstimateScanlineMotion.defaults)
    if case == "few":
        reference, moving = reference[:4], moving[:4]
    elif case == "degenerate":
        reference[:, 1] = moving[:, 1] = 100
    elif case == "holdout":
        moving[::4] += 5
    elif case == "span":
        reference[:, 1] = reference[:, 1] * 0.2 + 80
        moving = reference + displacement(reference[:, 1], 240)
    elif case == "gap":
        keep = (reference[:, 1] < 60) | (reference[:, 1] > 180)
        reference, moving = reference[keep], moving[keep]
    elif case == "motion":
        moving += [50, 0]
    elif case == "fold":
        moving[:, 1] = -reference[:, 1]
        p.update(maximum_motion=1000, maximum_row_step=5)
    fitted, report = fit_rows(reference, moving, 240, p)
    assert fitted is None and report["reason"] == reason


@pytest.mark.parametrize(
    "dtype,encoding",
    [(np.uint8, "gray"), (np.uint16, "rgb"), (np.float32, "bgra"), (np.float64, "gray")],
)
def test_native_tracks_reduce_known_wobble_and_preserve_current_type_ticket_and_reference_grid(
    dtype, encoding
):
    moving, reference = fixture(dtype, encoding)
    _, result = estimate(moving, reference)
    assert result.passed, result.outputs["report"]
    flow = result.outputs["flow"].vectors
    truth = displacement(np.arange(240), 240)
    good = result.outputs["support"][:, 0] != 0
    errors = np.linalg.norm(flow[good, 0] - truth[good], axis=1)
    assert np.quantile(errors, 0.99) < 0.35
    assert np.linalg.norm(flow[8, 8] - truth[8]) < 0.35
    assert np.linalg.norm(flow[-9, -9] - truth[-9]) < 0.35
    corrected = warp(moving, reference, result)
    supported = corrected.outputs["coverage"] != 0
    assert supported.mean() > 0.9
    assert not supported[:, -1].all()  # Positive shifts cannot invent right-edge donors.
    output = corrected.outputs["image"]
    divisor = np.iinfo(dtype).max if np.dtype(dtype).kind == "u" else 1
    a, b, c = (np.asarray(value, float) / divisor for value in (moving, reference, output))
    before, after = (
        np.mean((a[supported] - b[supported]) ** 2),
        np.mean((c[supported] - b[supported]) ** 2),
    )
    assert after < before * 0.35
    assert output.dtype == dtype and output.color_format == encoding
    assert corrected.payloads["image"]["ticket"] == moving.metadata["ticket"]
    assert corrected.payloads["image"]["custom"] == moving.metadata["custom"]
    assert corrected.payloads["image"]["spatial"] == reference.metadata["spatial"]
    assert (
        corrected.payloads["map_x"][COORDINATE_MAP_TAG]["source_spatial"]
        == moving.metadata["spatial"]
    )
    assert corrected.outputs["report"]["file_association"]["checked_images"] == ["source", "target"]


@pytest.mark.parametrize("scale", (1e-300, 1e300))
def test_lk_working_conversion_is_shared_and_stable_at_extreme_floating_scales(scale):
    moving, reference = fixture(np.float64, scale=scale)
    _, result = estimate(moving, reference)
    assert result.passed, result.outputs["report"]["reason"]
    assert np.isfinite(result.outputs["flow"].vectors).all()


def test_no_texture_is_a_valid_failure_and_does_not_reemit_old_flow():
    moving, reference = fixture()
    block, result = estimate(moving, reference)
    assert result.passed
    result = block.execute({"image": np.zeros_like(moving), "reference": np.zeros_like(reference)})
    assert result is not None and not result.passed
    assert result.outputs["flow"] is NO_VALUE and not block.current_emissions["flow"]
    assert result.outputs["report"]["reason"] == "insufficient_tracks"
    assert not result.outputs["support"].any()


def test_zero_extrapolation_reports_unsupported_rows_and_selection_excludes_moving_parts(
    monkeypatch,
):
    import opencv_dataflow.scanline_motion as module

    moving, reference = fixture()
    original, found = points()
    seen = {}

    def features(gray, mask, parameters, limit):
        seen["mask"] = mask
        return original.astype(np.float32)[:, None, :]

    monkeypatch.setattr(module, "_features", features)
    monkeypatch.setattr(module, "_correspondences", lambda *args: (np.arange(len(found)), found))
    block = builtin_registry().create(TYPE, parameters={"extrapolation_rows": 0})
    mask = np.zeros((240, 320), np.uint8)
    mask[:, :200] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [320, 240],
        "shape": "rectangle",
        "rectangle": [10, 0, 290, 240],
    }
    result = block.execute(
        {
            "image": moving,
            "reference": reference,
            "reference_mask": mask,
            "reference_region": region,
        }
    )
    assert result is not None and result.passed
    assert seen["mask"][:, 10:200].all() and not seen["mask"][:, :10].any()
    assert not result.outputs["support"][:6].any() and not result.outputs["support"][-6:].any()
    np.testing.assert_array_equal(result.outputs["flow"].vectors[:6], 0)


@pytest.mark.parametrize("case", ("source", "reference", "vectors", "direction"))
def test_warper_rejects_changed_pair_or_flow_direction(case):
    moving, reference = fixture()
    _, result = estimate(moving, reference)
    if case == "source":
        moving = moving.copy()
        moving[10, 10] ^= 1
    elif case == "reference":
        reference = reference.copy()
        reference[10, 10] ^= 1
    elif case == "vectors":
        result.outputs["flow"].vectors[20, 20, 0] += 1
    block = builtin_registry().create(
        "cv.warp_flow",
        parameters={"direction": "source_to_target" if case == "direction" else "target_to_source"},
    )
    assert (
        block.execute({"image": moving, "target": reference, "flow": result.outputs["flow"]})
        is None
    )
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())


def test_readonly_parallel_reset_xml_graphics_off_and_no_global_thread_change(tmp_path):
    moving, reference = fixture()
    moving.setflags(write=False)
    reference.setflags(write=False)
    count = cv2.getNumThreads()
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: estimate(moving, reference)[1], range(3)))
    for result in results[1:]:
        np.testing.assert_array_equal(
            result.outputs["flow"].vectors, results[0].outputs["flow"].vectors
        )
    assert count == cv2.getNumThreads()
    registry, graph = builtin_registry(), Graph()
    block = graph.add(registry.create(TYPE))
    try:
        path = tmp_path / "scanline.xml"
        export_project_xml(graph.to_dict(), path)
        restored = Graph.from_dict(read_project_xml(path), registry)
        assert restored.to_dict() == graph.to_dict()
        restored.close()
        block.reset_runtime()
        block.set_graphics_mode("off")
        result = block.execute({"image": moving, "reference": reference})
        assert result.passed and not result.drawings and not result.graphics
    finally:
        graph.close()


def test_resource_guards_precede_native_work_and_native_failure_recovers(monkeypatch):
    import opencv_dataflow.scanline_motion as module

    native = module._features

    def broken(*args):
        raise cv2.error("Controlled feature failure")

    monkeypatch.setattr(module, "_features", broken)
    moving, reference = fixture()
    block = builtin_registry().create(TYPE, parameters={"maximum_work": 1})
    assert block.execute({"image": moving, "reference": reference}) is None
    assert block.error_kind == "invalid"
    block.parameters["maximum_work"] = 64_000_000
    assert block.execute({"image": moving, "reference": reference}) is None
    assert block.error_kind == "exception" and not any(block.current_emissions.values())
    monkeypatch.setattr(module, "_features", native)
    assert block.execute({"image": moving, "reference": reference}).passed


@pytest.mark.parametrize("degree", ("1", "2", "4", "5"))
def test_other_degrees_reproduce_independently_supplied_row_motion(degree):
    reference, _ = points()
    p = {**EstimateScanlineMotion.defaults, "degree": degree}
    u = 2 * reference[:, 1] / 239 - 1
    delta = np.stack((1 + u + 0.05 * u ** int(degree), 0.1 * u ** int(degree)), axis=1)
    fitted, report = fit_rows(reference, reference + delta, 240, p)
    assert fitted is not None, report["reason"]
    u = 2 * np.arange(240) / 239 - 1
    expected = np.stack((1 + u + 0.05 * u ** int(degree), 0.1 * u ** int(degree)), axis=1)
    np.testing.assert_allclose(fitted[0], expected, atol=1e-11)


@pytest.mark.parametrize("case", ("vibration", "rotation"))
def test_strong_motion_outside_the_row_translation_model_is_not_accepted(case):
    reference, moving = points()
    if case == "vibration":
        moving[:, 0] += 6 * np.sin(reference[:, 1] * 0.4)
    else:
        moving[:, 0] += 0.1 * (reference[:, 0] - 160)
    fitted, report = fit_rows(reference, moving, 240, EstimateScanlineMotion.defaults)
    assert fitted is None and report["reason"] in ("insufficient_consensus", "validation_rejected")


@pytest.mark.parametrize(
    "parameters",
    (
        {"degree": "0"},
        {"tracking_window": 4},
        {"minimum_tracks": 900},
        {"feature_quality": 0},
        {"max_pyramid_level": 6},
        {"minimum_validation_tracks": 1000},
        {"inlier_pixels": 0},
    ),
)
def test_invalid_parameter_ranges_and_coupled_counts_are_contained(parameters):
    image = np.zeros((32, 48), np.uint8)
    block = builtin_registry().create(TYPE, parameters=parameters)
    assert block.execute({"image": image, "reference": image}) is None
    assert block.error_kind == "invalid"


@pytest.mark.parametrize(
    "case", ("shape", "depth", "encoding", "nonfinite", "mask_grid", "workspace", "dimension")
)
def test_invalid_image_or_selection_data_never_reaches_estimation(case, monkeypatch):
    import opencv_dataflow.scanline_motion as module

    def unexpected(*args):
        raise AssertionError("Invalid inputs reached native features")

    monkeypatch.setattr(module, "_features", unexpected)
    image = ImageArray(np.zeros((32, 48), np.uint8), "gray", {"spatial": frame([48, 32], "camera")})
    inputs = {"image": image, "reference": image}
    if case == "shape":
        inputs["reference"] = np.zeros((33, 48), np.uint8)
    elif case == "depth":
        inputs["reference"] = image.astype(np.uint16)
    elif case == "encoding":
        invalid = ImageArray(np.zeros((32, 48, 3), np.uint8), "hsv")
        inputs = {"image": invalid, "reference": invalid}
    elif case == "nonfinite":
        invalid = image.astype(float)
        invalid[0, 0] = np.inf
        inputs = {"image": invalid, "reference": invalid}
    elif case == "mask_grid":
        inputs["reference_mask"] = ImageArray(
            np.ones((32, 48), np.uint8), "gray", {"spatial": frame([48, 32], "other")}
        )
    elif case == "workspace":
        invalid = np.zeros((1024, 2048), np.uint8)
        inputs = {"image": invalid, "reference": invalid}
    else:
        invalid = np.zeros((24, 32767), np.uint8)
        inputs = {"image": invalid, "reference": invalid}
    block = builtin_registry().create(TYPE)
    assert block.execute(inputs) is None
    assert block.error_kind == "invalid", block.error


def test_current_selection_is_used_and_integral_float_settings_keep_saved_values():
    moving, reference = fixture()
    block = builtin_registry().create(
        TYPE, parameters={"tracking_window": 21.0, "fit_iterations": 10.0, "maximum_motion": "32"}
    )
    result = block.execute(
        {"image": moving, "reference": reference, "mask": np.zeros((240, 320), np.uint8)}
    )
    assert result is not None and not result.passed and result.outputs["report"]["tracked"] == 0
    result = block.execute({"image": moving, "reference": reference})
    assert result.passed and block.parameters["maximum_motion"] == "32"


def test_offscreen_editor_exposes_model_dropdown(app):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    window, graph = MainWindow(), Graph()
    block = graph.add(builtin_registry().create(TYPE))
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        combo = window.inspector.fields["degree"]
        assert isinstance(combo, QComboBox)
        assert [combo.itemText(i) for i in range(combo.count())] == ["1", "2", "3", "4", "5"]
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
