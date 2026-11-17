"""Real ALIKED models, public native parity and image-space inspection qualification."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.aliked_features import ALIKEDFeatures, ALIKEDLightGlueMatch, aliked_arrays
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph, Result
from opencv_dataflow.images import ImageArray
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock

BASE = Path(__file__).resolve().parents[1] / "examples/custom tools/aliked_matching"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
WEIGHTS = {
    name: Path(".local/models") / f"aliked_{name}.pth"
    for name in ("n16", "n16rot", "n32", "lightglue")
}


def texture():
    return cv2.GaussianBlur(
        np.random.default_rng(4).integers(0, 255, (160, 200), np.uint8), (3, 3), 0
    )


@pytest.fixture
def native(monkeypatch):
    if not all(p.exists() for p in WEIGHTS.values()):
        pytest.skip("Optional pinned ALIKED models not prepared")
    torch = pytest.importorskip("torch")
    kornia = pytest.importorskip("kornia")
    if not hasattr(kornia.feature, "ALIKED"):
        pytest.skip("Kornia ALIKED unavailable")
    pytest.importorskip("lightglue")
    before = torch.get_num_threads()
    torch.set_num_threads(2)

    def forbidden(*a, **k):
        pytest.fail("Execution must not fetch weights")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    import kornia.core.download

    monkeypatch.setattr(kornia.core.download, "load_state_dict_from_url", forbidden)
    try:
        yield torch, kornia
    finally:
        torch.set_num_threads(before)


@pytest.mark.parametrize(
    "settings",
    [
        {"model_name": "aliked-t16"},
        {"detection_mode": "invalid"},
        {"nms_radius": 0},
        {"nms_radius": 2.5},
        {"detection_mode": "threshold", "detection_threshold": 0},
        {"minimum_score": 1.01},
        {"minimum_count": 129, "max_features": 128},
        {"maximum_network_pixels": 10},
        {"weights_sha256": "bad"},
        {"max_features": True},
    ],
)
def test_configuration_rules_and_enums(settings):
    with pytest.raises(ValidationError):
        ALIKEDFeatures(parameters=settings).validate_parameters()


@pytest.mark.parametrize("variant", ["n16", "n16rot", "n32"])
def test_native_public_kornia_parity_and_portable_train_capture(variant, native):
    torch, kornia = native
    a = texture()
    block = ALIKEDFeatures(
        parameters={
            "weights_path": str(WEIGHTS[variant]),
            "model_name": "aliked-" + variant,
            "max_features": 64,
        }
    )
    result = block.execute({"image": a})
    assert result is not None and result.passed, block.error
    direct = kornia.feature.ALIKED(
        model_name="aliked-" + variant, max_num_keypoints=64, detection_threshold=-1.0, nms_radius=2
    )
    state = torch.load(WEIGHTS[variant], map_location="cpu", weights_only=True)
    missing = direct.load_state_dict(state, strict=False)
    assert set(missing.missing_keys) == {"dkd.hw_grid"} and not missing.unexpected_keys
    direct.eval()
    tensor = torch.from_numpy(np.repeat(a[None, None], 3, axis=1).astype(np.float32) / 255)
    with torch.inference_mode():
        expected = direct(tensor)[0]
    features = result.outputs["features"]
    np.testing.assert_allclose(features["points"], expected.keypoints.numpy(), atol=1e-6)
    np.testing.assert_allclose(features["descriptors"], expected.descriptors.numpy(), atol=1e-6)
    np.testing.assert_array_equal(features["scores"], expected.keypoint_scores.numpy())
    assert np.all(np.array(features["scores"]) >= 0) and np.all(np.array(features["scores"]) <= 1)
    np.testing.assert_allclose(np.linalg.norm(features["descriptors"], axis=1), 1, atol=1e-6)
    captured = block.train({"image": a})
    assert all(captured[k] == features[k] for k in ("points", "descriptors", "scores", "extractor"))
    assert captured["coordinate_context"] == {} and json.loads(json.dumps(features)) == features
    assert features["extractor"]["settings"]["model_name"] == "aliked-" + variant


@pytest.mark.parametrize("case", list(TRUTH["cases"]))
@pytest.mark.parametrize("run_profile", ["debug", "lean"])
def test_native_projects_source_geometry_failure_and_filmstrip(case, run_profile, native):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(run_profile)
    try:
        graph.run()
        assert not any(b.error for b in graph.blocks.values()), {
            b.id: b.error for b in graph.blocks.values() if b.error
        }
        result = graph.blocks["match"].last_result
        truth = TRUTH["cases"][case]
        assert result is not None and result.passed is truth["passed"]
        for side in ("reference", "query"):
            got = graph.blocks[side + "_features"].last_result
            assert got is not None and got.passed
            aliked_arrays(got.outputs["features"])
            assert got.outputs["report"]["detector_executed"]
            np.testing.assert_array_equal(
                got.outputs["image"], graph.blocks[side].last_result.outputs["image"]
            )
            assert (
                got.payloads["features"]["ticket"]
                == graph.blocks[side].last_result.outputs["ticket"]
            )
        if result.passed:
            if truth["reference_landmarks"]:
                xy = cv2.perspectiveTransform(
                    np.float64(truth["reference_landmarks"])[None],
                    np.array(result.outputs["pose"]["source_to_target"]),
                )[0]
                assert (
                    np.linalg.norm(xy - np.array(truth["query_landmarks"]), axis=1).max()
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
        plain = result.outputs["image"]
        if plain.ndim == 2:
            plain = np.repeat(plain[..., None], 3, axis=2)
        if run_profile == "lean":
            assert not result.drawings and not result.graphics
            np.testing.assert_array_equal(overlay, plain)
        elif result.outputs["count"]:
            assert result.drawings and np.any(overlay != plain)
        if case == "pose_region":
            assert graph.blocks["inspection_region"].last_result is not None
        if case == "comparison":
            sift = graph.blocks["sift"].last_result
            assert sift is not None and sift.passed
            np.testing.assert_allclose(sift.outputs["pose"]["origin"], [8, 6], atol=0.2)
        if case == "threshold_fallback":
            assert graph.blocks["query_features"].last_result.outputs["count"] > 0
        if case in ("empty", "strict_score"):
            assert graph.blocks["query_features"].last_result.outputs["count"] == 0
            assert graph.blocks["match"]._matcher is None
        if case == "rgb16":
            assert result.outputs["image"].dtype == np.uint16
            assert result.outputs["image"].color_format == "rgb"
    finally:
        graph.close()


def test_owned_cache_configuration_hash_missing_parameter_and_native_error_recovery(
    tmp_path, native, monkeypatch
):
    torch, _ = native
    a = texture()
    block = ALIKEDFeatures(
        parameters={
            "weights_path": str(WEIGHTS["n16"]),
            "max_features": 64,
            "weights_sha256": TRUTH["weights_sha256"]["n16"],
        }
    )
    assert block.execute({"image": a}).passed
    cached = block._model
    assert block.execute({"image": a}).passed and block._model is cached
    block.parameters["nms_radius"] = 3
    assert block.execute({"image": a}).passed and block._model is not cached
    forward = block._model.forward

    def broken(*a, **k):
        raise RuntimeError("Controlled ALIKED failure")

    monkeypatch.setattr(block._model, "forward", broken)
    assert block.execute({"image": a}) is None and not block.current_emissions
    monkeypatch.setattr(block._model, "forward", forward)
    assert block.execute({"image": a}).passed
    block.parameters["weights_sha256"] = "0" * 64
    assert block.execute({"image": a}) is None and "SHA-256" in block.error
    state = torch.load(WEIGHTS["n16"], map_location="cpu", weights_only=True)
    state.pop(next(k for k in state if k.endswith("weight")))
    bad = tmp_path / "missing_parameter.pth"
    torch.save(state, bad)
    block.parameters.update(weights_path=str(bad), weights_sha256="")
    assert block.execute({"image": a}) is None and "required model" in block.error
    block.parameters.update(
        weights_path=str(WEIGHTS["n16"]), weights_sha256=TRUTH["weights_sha256"]["n16"]
    )
    assert block.execute({"image": a}).passed
    block.close()
    assert block._model is None and block._key is None


def test_alpha_source_metadata_readonly_and_graphics_off(native):
    image = ImageArray(np.repeat(texture()[..., None], 4, axis=2).astype(np.uint16) * 257, "rgba")
    image[..., 3] = 65535
    image[:, :100, 3] = 0
    ticket = TicketClock().issue("camera")
    image.metadata = {"ticket": ticket, "spatial": frame([200, 160], "camera")}
    original = image.copy()
    block = ALIKEDFeatures(
        parameters={
            "weights_path": str(WEIGHTS["n16"]),
            "max_features": 128,
            "minimum_count": 1,
            "graphics_maximum_features": 1,
        }
    )
    result = block.execute({"image": image})
    assert result is not None and result.passed, block.error
    assert all(xy[0] >= 100 for xy in result.outputs["features"]["points"])
    assert result.outputs["features"]["coordinate_context"]["ticket"] == ticket
    assert result.outputs["report"]["graphics_truncated"]
    np.testing.assert_array_equal(result.outputs["image"], original)
    assert (
        result.outputs["image"].dtype == np.uint16
        and result.outputs["image"].color_format == "rgba"
    )
    block.run_profile = profile("lean")
    assert not block.execute({"image": image}).drawings
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("field", ["ticket", "spatial", "recipe_ref"])
def test_matcher_context_conflicts_reject_before_loading(field, native):
    graph = Graph.load(BASE / "translation.json", builtin_registry())
    try:
        graph.run()
        data = copy.deepcopy(graph.blocks["query_features"].last_result.outputs["features"])
        bad = copy.deepcopy(data["coordinate_context"])
        if field == "ticket":
            bad[field] = TicketClock().issue("other")
        elif field == "spatial":
            bad[field]["root_id"] = "other"
        else:
            data["coordinate_context"][field] = {"id": "p", "revision": 1, "digest": "1" * 64}
            bad[field] = {"id": "p", "revision": 2, "digest": "1" * 64}
        block = ALIKEDLightGlueMatch()
        assert (
            block.execute(
                {
                    "image": graph.blocks["query"].last_result.outputs["image"],
                    "query_features": ValuePacket(data, bad),
                    "reference_features": graph.blocks["reference_features"].last_result.outputs[
                        "features"
                    ],
                }
            )
            is None
        )
        assert block._matcher is None and not block.current_emissions
        assert "conflict" in block.error.lower()
    finally:
        graph.close()


def test_missing_selection_and_failed_input_admission_prevent_new_work(native, monkeypatch):
    graph = Graph.load(BASE / "masked.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["query_features"]
        count = block.timing.count
        previous = block.last_inputs["image"].copy()
        block.input_failure_policies = {"image": "reject_failed"}
        process = graph.blocks["query"].process

        def rejected(inputs):
            result = process(inputs)
            result.outputs["image"] = np.zeros_like(result.outputs["image"])
            result.passed = False
            return result

        monkeypatch.setattr(graph.blocks["query"], "process", rejected)
        graph.run()
        assert block.input_rejected and block.timing.count == count
        assert not block.current_emissions
        np.testing.assert_array_equal(block.last_inputs["image"], previous)
        monkeypatch.setattr(graph.blocks["query"], "process", process)
        monkeypatch.setattr(
            graph.blocks["query_region"],
            "process",
            lambda inputs: Result({"mask": NO_VALUE, "region": NO_VALUE, "overlay": NO_VALUE}),
        )
        graph.run()
        assert block.timing.count == count and not block.current_emissions
        assert not graph.blocks["query_region"].error
    finally:
        graph.close()


def test_project_json_xml_save_as_and_exact_source_preserving_generation(tmp_path, native):
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
    spec = importlib.util.spec_from_file_location("aliked_examples", BASE / "build_example.py")
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
    with pytest.raises(FileExistsError):
        module.build()


def test_lazy_catalog_help_and_dependency_failure(tmp_path, monkeypatch):
    import builtins

    original = builtins.__import__

    def no_ml(name, *a, **k):
        if name in ("torch", "kornia"):
            raise ImportError("Optional ML unavailable")
        return original(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_ml)
    registry = builtin_registry()
    for kind in ("ml.aliked_features", "ml.lightglue_aliked"):
        block = registry.create(kind)
        block.validate_parameters()
        assert block.defaults.keys() == block.parameter_help.keys()
    path = tmp_path / "inert.pth"
    path.write_bytes(b"local fixture")
    block = ALIKEDFeatures(parameters={"weights_path": str(path)})
    assert block.execute({"image": texture()}) is None and "Install optional" in block.error
