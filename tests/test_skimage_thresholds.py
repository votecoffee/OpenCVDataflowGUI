"""Reference parity, native units, selected fitting and failure containment."""

import json

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock, ticket_key

filters = pytest.importorskip("skimage.filters")


def run(kind, image, **parameters):
    block = builtin_registry().create("skimage." + kind, parameters=parameters)
    result = block.execute(image if isinstance(image, dict) else {"image": image})
    assert result is not None, block.error
    json.dumps(result.outputs["statistics"], allow_nan=False)
    return result


def parts():
    y, x = np.indices((40, 60))
    return (500 + x * 25 + y * 10 + (x > 30) * 1800).astype(np.uint16)


def test_multi_otsu_has_exact_integer_cutoffs_and_class_counts():
    image = np.repeat(np.arange(12, dtype=np.uint8), 10).reshape(10, 12)
    result = run("multi_otsu", image, classes=3, selected_classes=[0, 2])
    native = filters.threshold_multiotsu(image, classes=3)
    np.testing.assert_equal(result.outputs["thresholds"], native)
    expected = np.digitize(image, native)
    np.testing.assert_equal(result.outputs["labels"], expected)
    np.testing.assert_equal(result.outputs["mask"], np.isin(expected, [0, 2]) * 255)
    assert sum(result.outputs["statistics"]["class_counts"]) == image.size
    # A value exactly on the cutoff belongs to the upper class, matching native examples.
    assert result.outputs["labels"][image == native[0]].min() == 1


def test_multi_otsu_raw_16_bit_binned_scope_and_label_zero_regionprops():
    image = parts()
    mask = np.full(image.shape, 255, np.uint8)
    mask[:, 35:] = 0
    result = run(
        "multi_otsu", {"image": image, "mask": mask}, classes=4, bins=32, selected_classes=[3]
    )
    stats = result.outputs["statistics"]
    hist = (np.array(stats["histogram"]["counts"]), np.array(stats["histogram"]["centers"]))
    expected = filters.threshold_multiotsu(classes=4, hist=hist)
    np.testing.assert_equal(result.outputs["thresholds"], expected)
    assert max(expected) > 255
    assert (result.outputs["labels"][:, 35:] == -1).all()
    assert not result.outputs["mask"][:, 35:].any()
    assert sum(stats["class_counts"]) == 40 * 35
    block = builtin_registry().create(
        "skimage.region_properties", parameters={"ignored_labels": [-1]}
    )
    measured = block.execute({"labels": result.outputs["labels"]})
    assert measured is not None, block.error
    assert [row["label"] for row in measured.outputs["measurements"]["rows"]] == [0, 1, 2, 3]


@pytest.mark.parametrize("method", ["otsu", "yen", "li", "isodata", "minimum", "triangle", "mean"])
def test_global_estimators_match_native_with_raw_units_and_complementary_polarity(method):
    # Broad, overlapping modes give Minimum an unambiguous two-peak histogram.
    rng = np.random.default_rng(37)
    image = np.r_[rng.normal(1000, 180, 4000), rng.normal(4000, 220, 4000)].reshape(80, 100)
    result = run("global_threshold", image, method=method)
    function = getattr(filters, "threshold_" + method)
    threshold = function(image)
    assert result.outputs["threshold"] == pytest.approx(threshold)
    np.testing.assert_equal(result.outputs["mask"], (image > threshold) * 255)
    inverse = run("global_threshold", image, method=method, polarity="dark")
    assert np.all(result.outputs["mask"] + inverse.outputs["mask"] == 255)


def test_selected_global_fit_rgb_luminance_payload_and_source_immutability():
    rgb = ImageArray(np.tile([10000, 20000, 40000], (20, 30, 1)).astype(np.uint16), "rgb")
    rgb[10:] = [20000, 30000, 50000]
    original = rgb.copy()
    scope = np.zeros(rgb.shape[:2], np.uint8)
    scope[:10] = 255
    ticket = TicketClock().issue("camera")
    payload = {"ticket": ticket, "spatial": frame([30, 20], "camera"), "recipe": "A"}
    result = run(
        "global_threshold", {"image": ValuePacket(rgb, payload), "mask": scope}, method="mean"
    )
    assert result.outputs["threshold"] == pytest.approx(19290)
    assert not result.outputs["mask"].any()
    assert ticket_key(result.payloads["mask"]["ticket"]) == ticket_key(ticket)
    assert result.payloads["mask"]["recipe"] == "A"
    np.testing.assert_equal(original, rgb)


@pytest.mark.parametrize("method", ["gaussian", "mean", "median"])
@pytest.mark.parametrize("border", ["reflect", "constant", "nearest", "mirror", "wrap"])
def test_local_estimators_window_offset_sigma_border_and_scope(method, border):
    image = parts().astype(np.float64)
    scope = np.zeros(image.shape, np.uint8)
    scope[4:35, 6:50] = 255
    result = run(
        "local_threshold",
        {"image": image, "mask": scope},
        method=method,
        window_yx=[5, 7],
        sigma_yx=[1.2, 2.3],
        offset=85,
        border=border,
        border_value=300,
    )
    expected = filters.threshold_local(
        image, (5, 7), method=method, offset=85, mode=border, cval=300, param=(1.2, 2.3)
    )
    np.testing.assert_allclose(result.outputs["threshold_map"][scope != 0], expected[scope != 0])
    assert not result.outputs["threshold_map"][scope == 0].any()
    np.testing.assert_equal(result.outputs["mask"], ((image > expected) & (scope != 0)) * 255)


@pytest.mark.parametrize("method", ["niblack", "sauvola"])
def test_niblack_sauvola_float_range_and_sign_convention(method):
    image = parts().astype(float) / 65535
    kwargs = {"window_size": (5, 9), "k": 0.34}
    if method == "sauvola":
        kwargs["r"] = 0.5
    expected = getattr(filters, "threshold_" + method)(image, **kwargs) - 0.01
    result = run(
        "local_threshold",
        image,
        method=method,
        window_yx=[5, 9],
        k=0.34,
        sauvola_range=0.5,
        offset=0.01,
    )
    np.testing.assert_allclose(result.outputs["threshold_map"], expected)
    np.testing.assert_equal(result.outputs["mask"], (image > expected) * 255)


@pytest.mark.parametrize("initial", ["mean", "median", "quantile", "value"])
def test_li_initial_guess_controls(initial):
    image = parts().astype(float)
    result = run(
        "global_threshold",
        image,
        method="li",
        li_initial=initial,
        li_initial_value=1700,
        li_quantile=0.6,
        li_tolerance=0.1,
    )
    guess = {
        "mean": None,
        "median": np.median,
        "quantile": lambda a: np.quantile(a, 0.6),
        "value": 1700,
    }[initial]
    assert result.outputs["threshold"] == pytest.approx(
        filters.threshold_li(image, tolerance=0.1, initial_guess=guess)
    )
    assert len(result.outputs["statistics"]["li_iterations"]) > 1


@pytest.mark.parametrize("selection", ["lowest", "middle", "highest"])
def test_isodata_reports_all_candidates(selection):
    image = np.tile(np.arange(80, dtype=np.uint8), (10, 1))
    candidates = filters.threshold_isodata(image, return_all=True).tolist()
    result = run("global_threshold", image, method="isodata", isodata_threshold=selection)
    assert result.outputs["statistics"]["candidates"] == candidates
    index = {"lowest": 0, "middle": len(candidates) // 2, "highest": -1}[selection]
    assert result.outputs["threshold"] == candidates[index]


@pytest.mark.parametrize(
    "kind, parameters, image, fragment",
    [
        ("multi_otsu", {"classes": 6}, parts(), "budget"),
        ("multi_otsu", {}, np.ones((8, 8), np.uint8), "occupied"),
        ("multi_otsu", {"selected_classes": [5]}, parts(), "class IDs"),
        ("global_threshold", {"method": "li", "maximum_iterations": 1}, parts(), "converge"),
        ("global_threshold", {"method": "minimum", "maximum_iterations": 1}, parts(), "maxima"),
        ("local_threshold", {"window_yx": [2, 7]}, parts(), "odd"),
        ("local_threshold", {"sigma_yx": [0, 2]}, parts(), "sigmas"),
        ("local_threshold", {"method": "median", "window_yx": [255, 255]}, parts(), "work"),
        ("global_threshold", {}, np.full((8, 8), np.nan), "finite"),
        ("global_threshold", {}, np.full((8, 8), 1e13), "1e12"),
    ],
)
def test_errors_are_contained_and_do_not_emit_outputs(kind, parameters, image, fragment):
    block = builtin_registry().create("skimage." + kind, parameters=parameters)
    assert block.execute({"image": image}) is None
    assert fragment.lower() in block.error.lower()


def test_empty_selection_is_explicit_not_a_uniform_black_result():
    block = builtin_registry().create("skimage.global_threshold")
    assert block.execute({"image": parts(), "mask": np.zeros((40, 60), np.uint8)}) is None
    assert "empty" in block.error


def test_constant_float_otsu_keeps_exact_sample_value():
    image = np.full((12, 20), 0.421, np.float64)
    result = run("global_threshold", image, method="otsu")
    assert result.outputs["threshold"] == 0.421
    assert not result.outputs["mask"].any()
