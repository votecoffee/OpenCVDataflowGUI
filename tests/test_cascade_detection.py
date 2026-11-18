"""Native cascade parity, selection, evidence, bounds and failure recovery."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cascade_detection import model_metadata, search_plan
from opencv_dataflow.images import ImageArray
from opencv_dataflow.objects import validate_collection
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/cascade_detection/assets"
pytestmark = pytest.mark.local_examples


@pytest.fixture
def make():
    blocks = []

    def create(**p):
        block = builtin_registry().create(
            "ml.cascade_detect",
            parameters={
                "path": str(BASE / "haar.xml"),
                "min_size": [70, 70],
                "max_size": [150, 150],
                **p,
            },
        )
        blocks.append(block)
        return block

    yield create
    for block in blocks:
        block.close()


def portrait(name="portrait.png"):
    return cv2.imread(str(BASE / name), cv2.IMREAD_UNCHANGED)


@pytest.mark.parametrize("model", ["haar", "lbp"])
@pytest.mark.parametrize("mode", ["weights", "neighbors", "boxes"])
def test_matches_native_boxes_and_distinct_evidence(make, model, mode):
    image = portrait()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    native = cv2.CascadeClassifier(str(BASE / (model + ".xml")))
    kwargs = dict(scaleFactor=1.1, minNeighbors=3, flags=0, minSize=(70, 70), maxSize=(150, 150))
    if mode == "weights":
        expected, rejects, scores = native.detectMultiScale3(
            gray, outputRejectLevels=True, **kwargs
        )
    elif mode == "neighbors":
        expected, scores = native.detectMultiScale2(gray, **kwargs)
        rejects = None
    else:
        expected = native.detectMultiScale(gray, **kwargs)
        rejects = scores = None
    block = make(path=str(BASE / (model + ".xml")), result_mode=mode)
    result = block.execute({"image": image})
    assert result is not None, block.error
    items = result.outputs["collection"]["items"]
    validate_collection(result.outputs["collection"])
    assert len(items) == len(expected) == 1
    np.testing.assert_array_equal(items[0]["geometry"]["bbox"], expected[0])
    assert result.score is None and not result.outputs["report"]["score_is_probability"]
    if mode == "boxes":
        assert items[0]["score"] is None and "native_weight" not in items[0]
    else:
        assert items[0]["score"] == pytest.approx(float(scores[0]), abs=1e-5)
    if mode == "weights":
        assert items[0]["reject_level"] == int(rejects[0]) and "neighbor_count" not in items[0]
    if mode == "neighbors":
        assert items[0]["neighbor_count"] == int(scores[0]) and "reject_level" not in items[0]


@pytest.mark.parametrize("policy", ["center", "coverage", "center_and_coverage"])
def test_selection_depth_encoding_payload_and_graphics_policy(make, policy):
    image = portrait("portrait16.png")
    image = ImageArray(image[..., ::-1].copy(), "rgb")
    context = {
        "ticket": TicketClock("camera").issue("camera"),
        "spatial": frame([512, 512], "camera"),
        "pyramid": {
            "level": 1,
            "largest_size": [1024, 1024],
            "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        },
        "custom": {"ok": True},
    }
    mask = np.zeros((512, 512), np.uint8)
    mask[50:180, 150:300] = 255
    block = make(selection_policy=policy)
    result = block.execute(
        {"image": ValuePacket(image, context), "mask": ValuePacket(mask, context)}
    )
    assert result is not None and result.outputs["count"] == 1, block.error
    np.testing.assert_array_equal(result.outputs["image"], image)
    assert (
        result.outputs["image"].dtype == np.uint16 and result.outputs["image"].color_format == "rgb"
    )
    assert all(
        result.payloads[name][key] == context[key]
        for name in result.outputs
        for key in ("ticket", "pyramid", "custom")
    )
    assert result.drawings
    block.set_graphics_mode("off")
    lean = block.execute({"image": ValuePacket(image, context), "mask": ValuePacket(mask, context)})
    assert lean.outputs["collection"] == result.outputs["collection"] and not lean.drawings
    mask[:] = 0
    assert block.execute({"image": image, "mask": mask}).outputs["count"] == 0


def test_weight_boundary_and_count_failure_keep_negative_native_margins(make):
    image = portrait()
    block = make(min_neighbors=0)
    raw = block.execute({"image": image})
    assert raw is not None, block.error
    weights = [i["native_weight"] for i in raw.outputs["collection"]["items"]]
    assert min(weights) < 0 and len(weights) > 10
    threshold = 5.0
    filtered = make(min_neighbors=0, minimum_weight_enabled=True, minimum_weight=threshold)
    result = filtered.execute({"image": image})
    assert result.outputs["count"] == sum(w >= threshold for w in weights)
    failed = make(maximum_count=1, min_neighbors=0).execute({"image": image})
    assert not failed.passed and failed.fail_reasons[0]["code"] == "cascade.count"


def test_nearest_native_scale_is_reported_and_strict_size_filters_it(make):
    loose = make(min_size=[97, 97], max_size=[97, 97], strict_size=False).execute(
        {"image": portrait()}
    )
    assert (
        loose.outputs["count"] == 1
        and loose.outputs["report"]["plan"]["native_nearest_scale_fallback"]
    )
    assert loose.outputs["collection"]["items"][0]["geometry"]["bbox"][2:] == [100, 100]
    strict = make(min_size=[97, 97], max_size=[97, 97]).execute({"image": portrait()})
    assert strict.outputs["count"] == 0 and not strict.passed


@pytest.mark.parametrize(
    "settings",
    [
        {"scale_factor": 1},
        {"scale_factor": "1.1"},
        {"maximum_model_mb": True},
        {"min_neighbors": 1.0},
        {"min_size": [1, 0]},
        {"max_size": [20, 20]},
        {"result_mode": "unknown"},
        {"result_mode": "boxes", "minimum_weight_enabled": True},
        {"strict_size": 1},
        {"model_sha256": "bad"},
        {"label": ""},
        {"maximum_count": 1, "minimum_count": 2},
    ],
)
def test_invalid_configuration_has_actionable_fields(make, settings):
    with pytest.raises(ValueError):
        make(**settings).validate_parameters()


@pytest.mark.parametrize(
    "settings",
    [
        {"maximum_image_pixels": 10},
        {"maximum_model_mb": 0.001},
        {"maximum_levels": 1},
        {"maximum_search_windows": 1},
        {"max_work": 1},
        {"max_bytes_mb": 0.01},
    ],
)
def test_resource_rejection_precedes_native_model_load(make, monkeypatch, settings):
    monkeypatch.setattr(
        cv2, "CascadeClassifier", lambda *a, **k: pytest.fail("Native load before bounds")
    )
    block = make(**settings)
    assert block.execute({"image": portrait()}) is None
    assert not block.current_emissions and block.state == "error"


@pytest.mark.parametrize(
    "raw",
    [
        b"<invalid/>",
        b"<!DOCTYPE x><opencv_storage/>",
        b"<opencv_storage><cascade/>",
        (BASE / "haar.xml").read_bytes().replace(b"<stageNum>25", b"<stageNum>24"),
    ],
)
def test_invalid_xml_is_rejected_before_native(make, monkeypatch, tmp_path, raw):
    p = tmp_path / "bad.xml"
    p.write_bytes(raw)
    monkeypatch.setattr(
        cv2, "CascadeClassifier", lambda *a, **k: pytest.fail("Invalid XML reached native")
    )
    block = make(path=str(p))
    assert block.execute({"image": portrait()}) is None


def test_pin_detects_file_replacement_and_unpinned_reads_current_bytes(make, tmp_path):
    p = tmp_path / "current.xml"
    p.write_bytes((BASE / "haar.xml").read_bytes())
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    pinned = make(path=str(p), model_sha256=digest)
    assert pinned.execute({"image": portrait()}).outputs["report"]["model_sha256"] == digest
    p.write_bytes((BASE / "lbp.xml").read_bytes())
    assert pinned.execute({"image": portrait()}) is None and not pinned.current_emissions
    changed = make(path=str(p)).execute({"image": portrait()})
    assert changed.outputs["report"]["model"]["feature_type"] == "LBP"


def test_small_image_empty_selection_and_native_exception_recovery(make, monkeypatch):
    block = make()
    small = np.zeros((12, 12), np.uint8)
    assert block.execute({"image": small}).outputs["count"] == 0
    original = cv2.CascadeClassifier

    def broken():
        raise cv2.error("controlled native load failure")

    monkeypatch.setattr(cv2, "CascadeClassifier", broken)
    assert block.execute({"image": portrait()}) is None and not block.current_emissions
    monkeypatch.setattr(cv2, "CascadeClassifier", original)
    assert block.execute({"image": portrait()}).outputs["count"] == 1


def test_float_range_and_selection_frame_are_checked(make):
    block = make()
    image = portrait().astype(np.float32) / 255
    assert block.execute({"image": image}) is not None, block.error
    assert block.execute({"image": image * 2}) is None
    a = ValuePacket(portrait(), {"spatial": frame([512, 512], "a")})
    b = ValuePacket(np.ones((512, 512), np.uint8), {"spatial": frame([512, 512], "b")})
    assert block.execute({"image": a, "mask": b}) is None and "differs" in block.error


def test_scale_plan_includes_native_fallback_and_bounds_original_levels():
    model = model_metadata((BASE / "haar.xml").read_bytes())
    p = {
        **builtin_registry().types["ml.cascade_detect"].defaults,
        "min_size": [97, 97],
        "max_size": [97, 97],
    }
    plan = search_plan((512, 512), model, p)
    assert plan["native_nearest_scale_fallback"] and len(plan["levels"]) == 1
    assert plan["levels"][0]["window_size"] == [100, 100]
    assert plan["node_work_upper_bound"] == plan["window_upper_bound"] * model["tree_nodes"]


@pytest.mark.parametrize("setting", ["maximum_candidates", "maximum_results"])
def test_detection_limits_reject_without_silent_truncation(make, setting):
    block = make(min_neighbors=0, **{setting: 1})
    assert block.execute({"image": portrait()}) is None
    assert "allowance" in block.error and not block.current_emissions
