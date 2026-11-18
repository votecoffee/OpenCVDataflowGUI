"""Native HOG controls, linear-SVM margins, bounded models and detection contracts."""

import json

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.classifier_common import model_revision
from opencv_dataflow.hog import HOGFeatures
from opencv_dataflow.hog_detection import make_model, suppress
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame
from opencv_dataflow.svm_blocks import read_svm
from opencv_dataflow.tickets import TicketClock


@pytest.fixture
def make():
    registry, blocks = builtin_registry(), []

    def create(kind, **parameters):
        block = registry.create(kind, parameters=parameters)
        blocks.append(block)
        return block

    yield create
    for block in blocks:
        block.close()


def constant_model():
    g = {**HOGFeatures.defaults, "window_size": [32, 32], "working_encoding": "gray"}
    return make_model(g, [0.0] * 324 + [1.0], "fixture", {"test": True})


def native(model, levels=3):
    g = model["geometry"]
    hog = cv2.HOGDescriptor(
        tuple(g["window_size"]),
        tuple(g["block_size"]),
        tuple(g["block_stride"]),
        tuple(g["cell_size"]),
        g["bins"],
        _winSigma=g["win_sigma"],
        _L2HysThreshold=g["l2_hys_threshold"],
        _gammaCorrection=g["gamma_correction"],
        _signedGradient=g["signed_gradient"],
        _nlevels=levels,
    )
    hog.setSVMDetector(np.array(model["coefficients"], np.float32))
    return hog


@pytest.mark.parametrize(
    "grouping,threshold", [("none", 0.0), ("rectangles", 1.0), ("meanshift", 0.5)]
)
@pytest.mark.parametrize("padding", [[0, 0], [3, 5]])
def test_native_scale_controls_boxes_weights_and_padding_match_direct_calls(
    make, grouping, threshold, padding
):
    model = constant_model()
    image = np.random.default_rng(15).integers(0, 256, (72, 80), np.uint8)
    block = make(
        "ml.hog_detect",
        grouping=grouping,
        group_threshold=threshold,
        padding=padding,
        scale_factor=1.25,
        pyramid_levels=3,
        hit_threshold=1.0,
        nms_enabled=False,
        maximum_results=2048,
    )
    result = block.execute({"image": image, "model": model})
    assert result is not None, block.error
    boxes, weights = native(model).detectMultiScale(
        image,
        hitThreshold=1.0,
        winStride=(8, 8),
        padding=tuple(padding),
        scale=1.25,
        groupThreshold=0 if grouping == "none" else threshold,
        useMeanshiftGrouping=grouping == "meanshift",
    )
    expected = sorted(
        [
            tuple(map(float, box)) + (float(score),)
            for box, score in zip(boxes, np.asarray(weights).reshape(-1), strict=True)
        ]
    )
    observed = sorted(
        [
            tuple(item["geometry"]["bbox"]) + (item["native_weight"],)
            for item in result.outputs["collection"]["items"]
        ]
    )
    np.testing.assert_allclose(observed, expected, atol=1e-4 if grouping == "meanshift" else 1e-6)
    assert result.outputs["report"]["plan"]["effective_padding"] == (
        [0, 0] if padding == [0, 0] else [8, 8]
    )
    assert result.score is None and result.final_judgment is None


def test_independent_window_enumeration_and_descriptor_dot_products_match_native_margins(make):
    model = constant_model()
    model["coefficients"][0] = float(np.float32(0.7))
    model["revision"] = model_revision(model)
    image = np.random.default_rng(74).integers(0, 256, (64, 72), np.uint8)
    detector = make("ml.hog_detect", hit_threshold=0.5, nms_enabled=False, pyramid_levels=1)
    result = detector.execute({"model": model, "image": image})
    assert result is not None, detector.error
    hog = native(model, 1)
    expected = {}
    for y in range(0, 64 - 32 + 1, 8):
        for x in range(0, 72 - 32 + 1, 8):
            descriptor = hog.compute(image, locations=[(x, y)]).reshape(-1).astype(float)
            expected[(x, y, 32, 32)] = (
                descriptor @ np.array(model["coefficients"][:-1]) + model["coefficients"][-1]
            )
    assert (
        len(expected)
        == result.outputs["report"]["raw_candidates"]
        == result.outputs["report"]["plan"]["windows"]
    )
    for item in result.outputs["collection"]["items"]:
        assert item["native_weight"] == pytest.approx(
            expected[tuple(item["geometry"]["bbox"])], abs=1e-7
        )


def test_nms_has_stable_spatial_ties_and_an_inclusive_iou_boundary():
    boxes = np.array([[0, 0, 3, 2], [1, 0, 3, 2], [20, 0, 3, 2]], float)
    assert suppress(boxes, np.ones(3), 0.5) == [0, 1, 2]
    assert suppress(boxes, np.ones(3), 0.49) == [0, 2]


@pytest.mark.parametrize("positive", ["a", "z"])
def test_linear_image_svm_export_preserves_both_class_signs_and_native_raw_decisions(
    make, positive
):
    samples, labels = [], []
    extractor = make("inspect.hog")
    for index in range(8):
        image = np.full((32, 32), 30, np.uint8)
        if index < 4:
            cv2.rectangle(image, (8, 5), (23, 26), 220, 2)
        else:
            cv2.line(image, (5, 5), (26, 26), 220, 2)
        image = np.uint8(
            np.clip(
                image.astype(int) + np.random.default_rng(index).integers(-2, 3, image.shape),
                0,
                255,
            )
        )
        result = extractor.execute({"image": image})
        samples.append(ValuePacket(result.outputs["descriptor"], result.payloads["descriptor"]))
        labels.append("a" if index < 4 else "z")
    svm = (
        make("ml.svm_train", kernel="linear", labels=labels)
        .execute({"samples": FlowGroup(tuple(samples), tuple(map(str, range(8))))})
        .outputs["model"]
    )
    model = (
        make("ml.hog_detector_model", preset="linear_svm", positive_label=positive)
        .execute({"svm": svm})
        .outputs["model"]
    )
    table = np.array([packet.value for packet in samples], np.float32)
    expected = read_svm(svm).predict(table, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1].reshape(-1)
    if positive == "z":
        expected = -expected
    observed = (
        table.astype(float) @ np.array(model["coefficients"][:-1]) + model["coefficients"][-1]
    )
    np.testing.assert_allclose(observed, expected, atol=2e-6)
    assert model["label"] == positive and model["source"]["svm_revision"] == svm["revision"]
    assert model["geometry"]["window_size"] == [32, 32]


@pytest.mark.parametrize(
    "preset,size,length", [("default_people", [64, 128], 3781), ("daimler_people", [48, 96], 1981)]
)
def test_native_people_models_have_matching_geometry_and_blank_image_runs(
    make, preset, size, length
):
    provider = make("ml.hog_detector_model", preset=preset)
    model = provider.execute({}).outputs["model"]
    assert model["geometry"]["window_size"] == size and len(model["coefficients"]) == length
    detector = make("ml.hog_detect", detector=preset, minimum_count=0)
    result = detector.execute({"image": np.zeros((160, 160), np.uint8)})
    assert result is not None and result.passed and result.outputs["count"] == 0, detector.error


@pytest.mark.parametrize("policy", ["center", "coverage", "center_and_coverage"])
def test_selection_filters_visible_boxes_before_nms_and_preserves_current_image_context(
    make, policy
):
    context = {
        "ticket": TicketClock("one").issue("camera"),
        "spatial": frame([64, 64], "one"),
        "pyramid": {"level": 0, "largest_size": [64, 64], "to_largest": np.eye(3).tolist()},
        "custom": {"kept": True},
    }
    gray = np.full((64, 64), 200, np.uint16)
    image = ImageArray(np.repeat(gray[..., None], 3, axis=2), "rgb")
    mask = np.zeros((64, 64), np.uint8)
    mask[:, :40] = 255
    block = make(
        "ml.hog_detect",
        pyramid_levels=1,
        nms_enabled=False,
        selection_policy=policy,
        minimum_coverage=1.0,
    )
    result = block.execute(
        {"image": ValuePacket(image, context), "mask": mask, "model": constant_model()}
    )
    assert result is not None and result.outputs["count"] > 0, block.error
    for item in result.outputs["collection"]["items"]:
        x, y, w, h = map(int, item["geometry"]["bbox"])
        if policy != "center":
            assert x + w <= 40
        else:
            assert round(x + (w - 1) / 2) < 40
    np.testing.assert_array_equal(result.outputs["image"], image)
    assert result.outputs["image"].color_format == "rgb"
    assert result.payloads["collection"]["custom"] == {"kept": True}
    assert result.payloads["collection"]["ticket"] == context["ticket"]
    had_drawings = bool(result.drawings)
    block.set_graphics_mode("off")
    quiet = block.execute(
        {"image": ValuePacket(image, context), "mask": mask, "model": constant_model()}
    )
    assert quiet.outputs["collection"] == result.outputs["collection"]
    np.testing.assert_array_equal(quiet.outputs["image"], result.outputs["image"])
    assert not quiet.drawings and had_drawings


def test_small_or_empty_selected_images_do_not_call_native_search_and_emit_valid_empty_results(
    make, monkeypatch
):
    model = constant_model()
    monkeypatch.setattr(
        cv2, "HOGDescriptor", lambda *a, **k: pytest.fail("Unsearchable image reached native HOG")
    )
    block = make("ml.hog_detect")
    result = block.execute({"image": np.zeros((16, 16), np.uint8), "model": model})
    assert result is not None and not result.passed and result.outputs["count"] == 0
    result = block.execute(
        {
            "image": np.zeros((64, 64), np.uint8),
            "mask": np.zeros((64, 64), np.uint8),
            "model": model,
        }
    )
    assert result is not None and result.outputs["count"] == 0


@pytest.mark.parametrize(
    "setting", [{"maximum_windows": 1}, {"max_work": 1}, {"max_bytes_mb": 0.01}]
)
def test_search_budget_guards_run_before_native_and_recover(make, monkeypatch, setting):
    block = make("ml.hog_detect", **setting)
    model = constant_model()
    original = cv2.HOGDescriptor
    monkeypatch.setattr(
        cv2, "HOGDescriptor", lambda *a, **k: pytest.fail("Budget reached native search")
    )
    inputs = {"image": np.zeros((64, 64), np.uint8), "model": model}
    assert block.execute(inputs) is None and not block.current_emissions
    monkeypatch.setattr(cv2, "HOGDescriptor", original)
    block.parameters.update(maximum_windows=100000, max_work=256_000_000, max_bytes_mb=128)
    assert block.execute(inputs) is not None, block.error


def test_strict_model_json_preserves_existing_files_and_rejects_changes(make, tmp_path):
    model = constant_model()
    path = tmp_path / "detector.json"
    save, load = (
        make("ml.hog_detector_save", path=str(path)),
        make("ml.hog_detector_load", path=str(path)),
    )
    assert save.execute({"model": model}) is not None
    raw = path.read_bytes()
    assert save.execute({"model": model}) is None and path.read_bytes() == raw
    assert load.execute({}).outputs["model"] == model
    for invalid in (
        '{"model":1,"model":2}',
        '{"model":NaN}',
        json.dumps({**model, "revision": "changed"}),
    ):
        path.write_text(invalid, encoding="utf-8", newline="\n")
        assert load.execute({}) is None and not load.current_emissions
    path.write_bytes(raw)
    assert load.execute({}).outputs["model"] == model


@pytest.mark.parametrize(
    "field,value",
    [("opencv_version", "4.12"), ("coefficients", [0]), ("geometry", {}), ("label", "")],
)
def test_invalid_rehashed_models_never_reach_native_detector(make, monkeypatch, field, value):
    model = constant_model()
    model[field] = value
    model["revision"] = model_revision(model)
    monkeypatch.setattr(
        cv2, "HOGDescriptor", lambda *a, **k: pytest.fail("Invalid model reached native")
    )
    block = make("ml.hog_detect")
    assert block.execute({"image": np.zeros((64, 64), np.uint8), "model": model}) is None
    assert not block.current_emissions


def test_margin_mapping_preserves_raw_weights_and_count_failure_is_not_exception(make):
    block = make(
        "ml.hog_detect", pyramid_levels=1, maximum_count=1, score_mode="sigmoid", nms_enabled=False
    )
    result = block.execute({"image": np.zeros((48, 48), np.uint8), "model": constant_model()})
    assert result is not None and not result.passed and not block.error
    assert result.fail_reasons[0]["code"] == "hog.count" and result.score is None
    assert all(
        item["score"] == pytest.approx(1 / (1 + np.exp(-1))) and item["native_weight"] == 1
        for item in result.outputs["collection"]["items"]
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"win_stride": [0, 8]},
        {"padding": [-1, 0]},
        {"grouping": "rectangles", "group_threshold": 1.5},
        {"nms_enabled": 1},
        {"scale_factor": 1.0},
        {"pyramid_levels": 1.0},
        {"minimum_count": 2, "maximum_count": 1},
    ],
)
def test_search_configuration_validation_has_actionable_fields(make, parameters):
    with pytest.raises(ValueError):
        make("ml.hog_detect", **parameters).validate_parameters()


def test_native_exception_is_contained_without_stale_outputs_and_later_work_recovers(
    make, monkeypatch
):
    block = make("ml.hog_detect")
    inputs = {"image": np.zeros((64, 64), np.uint8), "model": constant_model()}
    assert block.execute(inputs) is not None
    original = cv2.HOGDescriptor

    def broken(*args, **kwargs):
        raise cv2.error("controlled native HOG failure")

    monkeypatch.setattr(cv2, "HOGDescriptor", broken)
    assert block.execute(inputs) is None and not block.current_emissions and block.state == "error"
    monkeypatch.setattr(cv2, "HOGDescriptor", original)
    assert block.execute(inputs) is not None


@pytest.mark.parametrize("parameters", [{"bins": 9.0}, {"block_stride": [8.0, 8]}])
def test_model_geometry_requires_native_integer_types_before_construction(make, parameters):
    with pytest.raises(ValueError, match="integer"):
        make("ml.hog_detector_model", **parameters).validate_parameters()
