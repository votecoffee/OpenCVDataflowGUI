"""Real local-weight DISK extraction, independent geometry and retained image contracts."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph, Result
from opencv_dataflow.disk_features import (
    DISKFeatures,
    DISKLightGlueMatch,
    disk_arrays,
    prepare_image,
)
from opencv_dataflow.images import ImageArray
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.timing import advance_timing, mark

BASE = Path(__file__).resolve().parents[1] / "examples/custom tools/disk_matching"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
DEPTH = Path(".local/models/disk_depth.pth")
EPIPOLAR = Path(".local/models/disk_epipolar.pth")
MATCHER = Path(".local/models/disk_lightglue.pth")


def texture():
    return cv2.GaussianBlur(
        np.random.default_rng(4).integers(0, 255, (160, 200), np.uint8), (3, 3), 0
    )


@pytest.fixture
def native(monkeypatch):
    if not all(p.exists() for p in (DEPTH, EPIPOLAR, MATCHER)):
        pytest.skip("Explicit optional DISK assets not prepared")
    torch = pytest.importorskip("torch")
    pytest.importorskip("kornia")
    pytest.importorskip("lightglue")
    before = torch.get_num_threads()
    torch.set_num_threads(2)

    def forbidden(*a, **k):
        pytest.fail("Runtime must never fetch models")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    import kornia.core.download

    monkeypatch.setattr(kornia.core.download, "load_state_dict_from_url", forbidden)
    try:
        yield torch
    finally:
        torch.set_num_threads(before)


@pytest.mark.parametrize(
    "settings",
    [
        {"max_features": 0},
        {"max_features": 4097},
        {"nms_window_size": 4},
        {"resize_long_edge": 10},
        {"maximum_network_pixels": 100},
        {"detection_threshold": float("inf")},
        {"pad_if_not_divisible": 1},
        {"minimum_count": 5, "max_features": 4},
        {"weights_sha256": "bad"},
        {"graphics_maximum_features": 1.0},
    ],
)
def test_field_validation_without_optional_runtime(settings):
    with pytest.raises(ValidationError):
        DISKFeatures(parameters=settings).validate_parameters()


@pytest.mark.parametrize("depth", ["uint8", "uint16", "float32"])
@pytest.mark.parametrize("encoding", ["rgb", "bgr", "rgba", "bgra", "gray"])
def test_normalized_channel_conversion_and_half_pixel_scale(depth, encoding):
    a = texture()
    rgb = np.stack([a, np.roll(a, 2, axis=1), np.roll(a, 3, axis=0)], axis=-1)
    values = rgb if encoding.startswith("rgb") else rgb[..., ::-1]
    if encoding == "gray":
        values = a
    if encoding.endswith("a"):
        values = np.dstack([values, np.full(a.shape, 255, np.uint8)])
    if depth == "uint16":
        values = values.astype(np.uint16) * 257
    elif depth == "float32":
        values = values.astype(np.float32) / 255
    image = ImageArray(values, encoding)
    block = DISKFeatures(parameters={"resize_long_edge": 100})
    converted, selected, scale, padded = prepare_image({"image": image}, block.parameters)
    expected = np.repeat(a[..., None], 3, axis=2) if encoding == "gray" else rgb
    expected = cv2.resize(
        expected.astype(np.float32) / 255, (100, 80), interpolation=cv2.INTER_AREA
    )
    np.testing.assert_allclose(converted, expected.transpose(2, 0, 1), atol=1e-7)
    np.testing.assert_array_equal(scale, [0.5, 0.5])
    assert padded == [112, 80]
    assert selected is None or np.all(selected == 255)


def test_alpha_selection_and_early_geometry_range_budgets():
    a = np.repeat(texture()[..., None], 4, axis=2)
    a[..., 3] = 255
    a[:, :100, 3] = 0
    p = DISKFeatures.defaults
    _, selection, _, _ = prepare_image({"image": ImageArray(a, "rgba")}, p)
    assert np.all(selection[:, :100] == 0) and np.all(selection[:, 100:] != 0)
    assert (
        prepare_image({"image": ImageArray(a, "rgba")}, {**p, "alpha_policy": "ignore"})[1] is None
    )
    for image, settings in [
        (np.ones((160, 200), np.float32) * 2, {}),
        (np.ones((16, 16), np.uint8), {}),
        (texture(), {"pad_if_not_divisible": False}),
        (texture(), {"maximum_image_pixels": 1}),
        (texture(), {"maximum_network_pixels": 1024}),
    ]:
        with pytest.raises(ValidationError):
            prepare_image({"image": image}, {**p, **settings})


def test_native_rows_descriptors_scores_match_public_kornia_api_and_training(native):
    from kornia.feature import DISK

    image = texture()
    block = DISKFeatures(parameters={"weights_path": str(DEPTH), "max_features": 64})
    result = block.execute({"image": image})
    assert result is not None and result.passed, block.error
    direct = DISK(desc_dim=128)
    direct.load_state_dict(native.load(DEPTH, weights_only=True, map_location="cpu")["extractor"])
    direct.eval()
    rgb = np.repeat(image[None, None], 3, axis=1).astype(np.float32) / 255
    with native.inference_mode():
        expected = direct(
            native.from_numpy(rgb),
            n=64,
            window_size=5,
            score_threshold=0.0,
            pad_if_not_divisible=True,
        )[0]
    data = result.outputs["features"]
    np.testing.assert_array_equal(data["points"], expected.keypoints.numpy())
    np.testing.assert_allclose(data["descriptors"], expected.descriptors.numpy(), atol=1e-7)
    np.testing.assert_array_equal(data["scores"], expected.detection_scores.numpy())
    assert any(v > 1 for v in data["scores"])
    np.testing.assert_allclose(np.linalg.norm(data["descriptors"], axis=1), 1, atol=1e-6)
    captured = block.train({"image": image})
    assert captured["coordinate_context"] == {}
    assert all(captured[k] == data[k] for k in ("points", "descriptors", "scores", "extractor"))
    assert json.loads(json.dumps(data)) == data
    assert result.outputs["report"]["detector_executed"]
    assert result.outputs["report"]["weights_sha256"] == TRUTH["weights_sha256"]["depth"]
    assert block._model is not None
    cached = block._model
    block.execute({"image": image})
    assert block._model is cached
    block.close()
    assert block._model is None


def test_post_top_n_filters_current_provenance_lean_and_readonly(native):
    source = ImageArray(np.repeat(texture()[..., None], 3, axis=2).astype(np.uint16) * 257, "rgb")
    before = source.copy()
    payload = {
        "ticket": TicketClock().issue("camera"),
        "spatial": frame([200, 160], "camera"),
        "recipe_ref": {"id": "part", "revision": 1, "digest": "1" * 64},
        "processing_time": advance_timing(None, mark(), mark(), source=True),
    }
    source.metadata = copy.deepcopy(payload)
    mask = np.zeros(source.shape[:2], np.uint8)
    mask[:, :100] = 255
    block = DISKFeatures(
        parameters={
            "weights_path": str(DEPTH),
            "max_features": 128,
            "minimum_count": 1,
            "graphics_maximum_features": 1,
        }
    )
    result = block.execute({"image": source, "mask": ValuePacket(mask, payload)})
    assert result is not None and result.passed, block.error
    assert all(xy[0] < 100 for xy in result.outputs["features"]["points"])
    assert result.outputs["features"]["coordinate_context"]["ticket"] == payload["ticket"]
    assert result.outputs["features"]["extractor"]["family"] == "DISK"
    assert (
        result.payloads["features"]["processing_time"]["started_at"]
        == payload["processing_time"]["started_at"]
    )
    assert result.outputs["report"]["graphics_truncated"]
    assert (
        result.outputs["image"].color_format == "rgb" and result.outputs["image"].dtype == np.uint16
    )
    np.testing.assert_array_equal(source, before)
    block.run_profile = profile("lean")
    lean = block.execute({"image": source})
    assert lean is not None and not lean.drawings
    np.testing.assert_array_equal(lean.outputs["image"], before)


@pytest.mark.parametrize("case", list(TRUTH["cases"]))
@pytest.mark.parametrize("run_profile", ["debug", "lean"])
def test_native_projects_known_geometry_and_filmstrip(case, run_profile, native):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(run_profile)
    try:
        graph.run()
        assert not any(b.error for b in graph.blocks.values()), {
            b.id: b.error for b in graph.blocks.values() if b.error
        }
        result = graph.blocks["match"].last_result
        assert result is not None and result.passed is TRUTH["cases"][case]["passed"]
        for side in ("reference", "query"):
            features = graph.blocks[side + "_features"].last_result
            assert features is not None and features.passed
            disk_arrays(features.outputs["features"])
            assert features.outputs["report"]["detector_executed"]
            np.testing.assert_array_equal(
                features.outputs["image"], graph.blocks[side].last_result.outputs["image"]
            )
        if result.passed:
            matrix = np.array(result.outputs["pose"]["source_to_target"])
            truth = TRUTH["cases"][case]
            if truth["reference_landmarks"]:
                projected = cv2.perspectiveTransform(
                    np.float64(truth["reference_landmarks"])[None], matrix
                )[0]
                assert (
                    np.linalg.norm(projected - np.array(truth["query_landmarks"]), axis=1).max()
                    <= truth["landmark_tolerance"]
                )
            else:
                np.testing.assert_allclose(
                    result.outputs["pose"]["origin"],
                    truth["translation"],
                    atol=truth["pose_origin_tolerance"],
                )
                matches = result.outputs["matches"]
                delta = np.array(matches["query_points"]) - np.array(matches["reference_points"])
                inliers = np.array(matches["inliers"], bool)
                assert (
                    np.median(np.linalg.norm(delta[inliers] - truth["translation"], axis=1)) <= 0.8
                )
        else:
            assert result.outputs["pose"] is NO_VALUE and result.outputs["transform"] is NO_VALUE
        np.testing.assert_array_equal(
            result.outputs["image"], graph.blocks["query"].last_result.outputs["image"]
        )
        overlay = graph.blocks["overlay"].last_result.outputs["image"]
        np.testing.assert_array_equal(
            overlay, graph.blocks["filmstrip"].last_result.outputs["image"]
        )
        expected_review = result.outputs["image"]
        if expected_review.ndim == 2:
            expected_review = np.repeat(expected_review[..., None], 3, axis=2)
        if run_profile == "lean":
            assert not result.drawings and not result.graphics
            np.testing.assert_array_equal(overlay, expected_review)
        elif result.outputs["count"]:
            assert result.drawings and np.any(overlay != expected_review)
        if case == "pose_region":
            assert graph.blocks["inspection_region"].last_result is not None
        if case == "comparison":
            assert graph.blocks["sift"].last_result.passed
            np.testing.assert_allclose(
                graph.blocks["sift"].last_result.outputs["pose"]["origin"], [8, 6], atol=0.2
            )
        if case == "rgb16":
            assert result.outputs["image"].dtype == np.uint16
        if case == "empty":
            assert graph.blocks["match"]._matcher is None
    finally:
        graph.close()


def test_sha_native_error_recovery_cache_and_independent_frames(native, monkeypatch):
    image = texture()
    block = DISKFeatures(
        parameters={"weights_path": str(DEPTH), "max_features": 64, "weights_sha256": "0" * 64}
    )
    assert block.execute({"image": image}) is None and not block.current_emissions
    block.parameters["weights_sha256"] = TRUTH["weights_sha256"]["depth"]
    assert block.execute({"image": image}).passed
    forward = block._model.forward

    def broken(*a, **k):
        raise RuntimeError("Controlled DISK error")

    monkeypatch.setattr(block._model, "forward", broken)
    assert block.execute({"image": image}) is None and "Controlled DISK" in block.error
    monkeypatch.setattr(block._model, "forward", forward)
    assert block.execute({"image": image}).passed
    block.parameters["weights_path"] = str(EPIPOLAR)
    block.parameters["weights_sha256"] = TRUTH["weights_sha256"]["epipolar"]
    assert block.execute({"image": image}).passed
    assert block._digest == TRUTH["weights_sha256"]["epipolar"]
    block.close()


def test_missing_wired_selection_has_no_new_work(native, monkeypatch):
    graph = Graph.load(BASE / "masked.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["query_features"]
        before = block.timing.count
        cached = block._model
        monkeypatch.setattr(
            graph.blocks["query_region"],
            "process",
            lambda inputs: Result({"mask": NO_VALUE, "region": NO_VALUE, "overlay": NO_VALUE}),
        )
        graph.run()
        assert block.timing.count == before and not block.current_emissions
        assert block._model is cached
    finally:
        graph.close()


def test_json_xml_save_as_deterministic_generation_and_source_integrity(tmp_path, native):
    before = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    graph = Graph.load(BASE / "translation.json", builtin_registry())
    try:
        graph.save(tmp_path / "saved.json")
        restored = Graph.load(tmp_path / "saved.json", builtin_registry())
        try:
            restored.run()
            assert restored.blocks["match"].last_result.passed
        finally:
            restored.close()
        export_project_xml(graph.to_dict(), tmp_path / "saved.xml")
        restored = Graph.from_dict(
            read_project_xml(tmp_path / "saved.xml"), builtin_registry(), project_dir=tmp_path
        )
        try:
            restored.run()
            assert restored.blocks["match"].last_result.passed
        finally:
            restored.close()
    finally:
        graph.close()
    spec = importlib.util.spec_from_file_location("disk_examples", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path / "regenerated"
    module.build()
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((module.HERE / "assets" / name).read_bytes()).hexdigest() == digest
    for path in BASE.glob("*.json"):
        assert json.loads(path.read_bytes()) == json.loads((module.HERE / path.name).read_bytes())
    assert all(
        hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
        for name, digest in before.items()
    )


def test_lazy_registration_and_optional_dependency_guidance(tmp_path, monkeypatch):
    import builtins

    original = builtins.__import__

    def without_torch(name, *a, **k):
        if name in ("torch", "kornia"):
            raise ImportError("Optional runtime unavailable")
        return original(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    registry = builtin_registry()
    for kind in ("ml.disk_features", "ml.lightglue_disk"):
        block = registry.create(kind)
        block.validate_parameters()
        assert block.defaults.keys() == block.parameter_help.keys()
    path = tmp_path / "inert.pth"
    path.write_bytes(b"inert fixture")
    block = DISKFeatures(parameters={"weights_path": str(path)})
    assert block.execute({"image": texture()}) is None
    assert "Install optional" in block.error


def test_disk_raw_scores_and_family_are_distinct_from_other_128_descriptors():
    data = {
        "model": "lightglue_disk_v1",
        "points": [[1.0, 1.0]],
        "descriptors": [[1.0] + [0.0] * 127],
        "scores": [-12.0],
        "image_size": [10, 10],
        "coordinate_context": {},
    }
    assert disk_arrays(data)["descriptors"].shape == (1, 128)
    matcher = DISKLightGlueMatch()
    matcher.validate_parameters()
    assert matcher.parameter_help.keys() == matcher.defaults.keys()
    data["model"] = "lightglue_sift_v1"
    with pytest.raises(ValidationError):
        matcher.feature_arrays(data)


def test_failed_input_gate_preserves_last_image_and_does_not_run(native, monkeypatch):
    graph = Graph.load(BASE / "translation.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["query_features"]
        block.input_failure_policies = {"image": "reject_failed"}
        before = block.timing.count
        previous = block.last_inputs["image"].copy()
        process = graph.blocks["query"].process

        def rejected(inputs):
            result = process(inputs)
            result.outputs["image"] = np.zeros_like(result.outputs["image"])
            result.passed = False
            return result

        monkeypatch.setattr(graph.blocks["query"], "process", rejected)
        graph.run()
        assert block.input_rejected and block.timing.count == before
        assert not block.current_emissions
        np.testing.assert_array_equal(block.last_inputs["image"], previous)
    finally:
        graph.close()
