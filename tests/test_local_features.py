"""External feature interface qualification; no SuperPoint detector or weights are used."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.dnn_blocks import DNNInference, preprocess
from opencv_dataflow.local_features import (
    PackSuperPointFeatures,
    SuperPointLightGlueMatch,
    feature_arrays,
)
from opencv_dataflow.ml_common import tensor_record
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock

BASE = Path(__file__).resolve().parents[1] / "examples/custom tools/local_features"
WEIGHTS = Path(".local/models/superpoint_lightglue.pth")
TRUTH = json.loads((BASE / "truth.json").read_bytes())


def arguments(*, letterbox=False, channels_first=False, batch=True, normalized=False, edges=False):
    image = np.zeros((80, 160 if letterbox else 120, 3), np.uint16)
    _, geometry = preprocess(
        {"image": image},
        {
            **DNNInference.defaults,
            "width": 24,
            "height": 16,
            "resize_mode": "letterbox" if letterbox else "stretch",
        },
    )
    xy = np.array([[4, 5], [10, 9], [15, 6], [20, 11]], np.float32)
    if edges:
        xy += 0.5
    if normalized:
        xy /= [24, 16]
    desc = np.random.default_rng(0).normal(size=(4, 256)).astype(np.float32)
    if channels_first:
        desc = desc.T
    values = {
        "keypoints": xy,
        "descriptors": desc,
        "scores": np.array([0.2, 0.5, 0.7, 0.9], np.float32),
    }
    tensors = {
        k: tensor_record(v[None] if batch else v, 2_000_000, allow_empty=True)
        for k, v in values.items()
    }
    return {"image": image, "tensors": tensors, "preprocessing": geometry}


def pack(args=None, **params):
    block = PackSuperPointFeatures(parameters=params)
    result = block.execute(arguments() if args is None else args)
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("batch", [True, False])
@pytest.mark.parametrize("channels_first", [True, False])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("edges", [True, False])
def test_explicit_tensor_layouts_and_source_center_coordinates(
    batch, channels_first, normalized, edges
):
    args = arguments(batch=batch, channels_first=channels_first, normalized=normalized, edges=edges)
    before = copy.deepcopy(args["tensors"])
    _, result = pack(
        args,
        descriptor_layout="channels_first" if channels_first else "rows",
        coordinate_units="normalized" if normalized else "network_pixels",
        coordinate_origin="pixel_edges" if edges else "pixel_centers",
    )
    data = result.outputs["features"]
    np.testing.assert_allclose(
        data["points"], (np.array([[4, 5], [10, 9], [15, 6], [20, 11]]) + 0.5) * 5 - 0.5, atol=1e-5
    )
    np.testing.assert_allclose(np.linalg.norm(data["descriptors"], axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(result.outputs["image"], args["image"])
    assert result.outputs["image"].dtype == np.uint16 and args["tensors"] == before


def test_letterbox_score_selection_row_identity_and_no_score_mode():
    args = arguments(letterbox=True)
    mask = np.zeros(args["image"].shape[:2], np.uint8)
    mask[:, :100] = 255
    args["mask"] = mask
    _, result = pack(args, minimum_score=0.5, minimum_count=1)
    assert result.passed and result.outputs["indices"] == [1]
    np.testing.assert_allclose(result.outputs["features"]["points"], [[69.5, 49.5]], atol=1e-5)
    assert result.outputs["features"]["raw_indices"] == [1]
    _, result = pack(scores_output="")
    assert result.outputs["features"]["scores"] == [1] * 4


@pytest.mark.parametrize(
    "problem",
    [
        "shape",
        "dimension",
        "scores",
        "nonfinite",
        "dtype",
        "nested",
        "zero",
        "raw_budget",
        "tensor_budget",
        "image_budget",
        "nonunit",
    ],
)
def test_malformed_heads_and_work_limits_emit_nothing(problem):
    args, params = arguments(), {}
    if problem == "shape":
        args["tensors"]["keypoints"]["shape"] = [1, 2, 4]
    elif problem == "dimension":
        args["tensors"]["descriptors"]["shape"] = [1, 8, 128]
    elif problem == "scores":
        args["tensors"]["scores"]["values"][0] = 2
    elif problem == "nonfinite":
        args["tensors"]["descriptors"]["values"][0] = float("nan")
    elif problem == "dtype":
        args["tensors"]["descriptors"]["dtype"] = "uint8"
    elif problem == "nested":
        args["tensors"]["descriptors"]["values"][0] = [1]
    elif problem == "zero":
        args["tensors"]["descriptors"]["values"][:256] = [0] * 256
    elif problem == "raw_budget":
        params = {"maximum_features": 3, "minimum_count": 1}
    elif problem == "tensor_budget":
        params = {"maximum_tensor_elements": 10}
    elif problem == "image_budget":
        params = {"maximum_image_pixels": 1}
    else:
        params = {"normalize_descriptors": False}
    block = PackSuperPointFeatures(parameters=params)
    assert block.execute(args) is None and not block.current_emissions


def test_empty_heads_graphics_off_and_incomplete_count_are_normal_judgments():
    args = arguments()
    args["tensors"] = {
        k: tensor_record(np.empty(s), 2_000_000, allow_empty=True)
        for k, s in [("keypoints", (1, 0, 2)), ("descriptors", (1, 0, 256)), ("scores", (1, 0))]
    }
    block, result = pack(args)
    assert not result.passed and result.outputs["count"] == 0
    block.parameters["minimum_count"] = 0
    block.run_profile = profile("lean")
    result = block.execute(args)
    assert result.passed and not result.drawings and result.outputs["features"]["points"] == []
    matcher = SuperPointLightGlueMatch()
    result = matcher.execute(
        {
            "image": args["image"],
            "query_features": result.outputs["features"],
            "reference_features": result.outputs["features"],
        }
    )
    assert result is not None and not result.passed and result.outputs["pose"] is NO_VALUE
    assert matcher._matcher is None


def contextual():
    args = arguments()
    payload = {"ticket": TicketClock().issue("query"), "spatial": frame([120, 80], "camera")}
    return {k: ValuePacket(v, copy.deepcopy(payload)) for k, v in args.items()}, payload


@pytest.mark.parametrize("field", ["ticket", "spatial", "recipe_ref"])
def test_known_pack_context_mismatch_prevents_work(field):
    args, payload = contextual()
    bad = args["tensors"].payload
    if field == "ticket":
        bad[field] = TicketClock().issue("other")
    elif field == "spatial":
        bad[field]["root_id"] = "other"
    else:
        bad[field] = {"id": "p", "revision": 2, "digest": "1" * 64}
        args["image"].payload[field] = {"id": "p", "revision": 1, "digest": "1" * 64}
    block = PackSuperPointFeatures()
    assert block.execute(args) is None and not block.current_emissions


@pytest.mark.parametrize("field", ["ticket", "spatial", "recipe_ref"])
def test_matcher_checks_current_body_wire_context(field):
    wrapped, payload = contextual()
    _, packed = pack(wrapped)
    current = packed.outputs["features"]
    bad = copy.deepcopy(payload)
    if field == "ticket":
        bad[field] = TicketClock().issue("other")
    elif field == "spatial":
        bad[field]["root_id"] = "other"
    else:
        current["coordinate_context"][field] = {"id": "p", "revision": 1, "digest": "1" * 64}
        bad[field] = {"id": "p", "revision": 2, "digest": "1" * 64}
    block = SuperPointLightGlueMatch()
    assert (
        block.execute(
            {
                "image": wrapped["image"],
                "query_features": ValuePacket(current, bad),
                "reference_features": current,
            }
        )
        is None
    )
    assert not block.current_emissions


@pytest.mark.parametrize(
    "cls,settings",
    [
        (PackSuperPointFeatures, {"minimum_count": 8, "maximum_features": 4}),
        (PackSuperPointFeatures, {"normalize_descriptors": 1}),
        (PackSuperPointFeatures, {"graphics_maximum_features": 1.0}),
        (SuperPointLightGlueMatch, {"weights_sha256": "xx"}),
        (SuperPointLightGlueMatch, {"depth_confidence": -0.5}),
        (SuperPointLightGlueMatch, {"mixed_precision": True}),
    ],
)
def test_field_validation(cls, settings):
    with pytest.raises(ValidationError):
        cls(parameters=settings).validate_parameters()


@pytest.fixture
def native(monkeypatch):
    if not WEIGHTS.exists():
        pytest.skip("Optional local Apache-2.0 matcher not prepared")
    torch = pytest.importorskip("torch")
    pytest.importorskip("lightglue")
    old = torch.get_num_threads()
    torch.set_num_threads(2)

    def no_download(*a, **k):
        pytest.fail("Runtime must not download weights")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", no_download)
    try:
        yield
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("case", list(TRUTH["cases"]))
@pytest.mark.parametrize("run_profile", ["debug", "lean"])
def test_native_projects_contract_geometry_graphics_and_preserved_source(case, run_profile, native):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(run_profile)
    try:
        graph.run()
        assert not any(b.error for b in graph.blocks.values()), {
            b.id: b.error for b in graph.blocks.values() if b.error
        }
        for side in ("query", "reference"):
            result = graph.blocks[side + "_pack"].last_result
            assert result is not None and result.passed
            assert result.outputs["report"]["detector_executed"] is False
            np.testing.assert_array_equal(
                result.outputs["image"], graph.blocks[side].last_result.outputs["image"]
            )
            assert result.outputs["features"]["raw_indices"] == result.outputs["indices"]
        if not TRUTH["cases"][case]["matcher"]:
            return
        result = graph.blocks["match"].last_result
        assert result is not None and result.passed is TRUTH["cases"][case]["passed"]
        if result.passed:
            np.testing.assert_allclose(result.outputs["pose"]["origin"], [8, 6], atol=0.002)
            assert result.outputs["count"] >= 4 and result.outputs["inliers"] >= 4
        else:
            assert result.outputs["pose"] is NO_VALUE
        np.testing.assert_array_equal(
            result.outputs["image"], graph.blocks["query"].last_result.outputs["image"]
        )
        overlay = graph.blocks["overlay"].last_result.outputs["image"]
        np.testing.assert_array_equal(
            overlay, graph.blocks["filmstrip"].last_result.outputs["image"]
        )
        if run_profile == "lean":
            assert not result.drawings and not result.graphics
            np.testing.assert_array_equal(overlay, result.outputs["image"])
        elif result.outputs["count"]:
            assert result.drawings and np.any(overlay != result.outputs["image"])
        if case == "comparison":
            sift = graph.blocks["sift"].last_result
            assert sift is not None and sift.passed
            np.testing.assert_allclose(sift.outputs["pose"]["origin"], [8, 6], atol=0.2)
    finally:
        graph.close()


def test_native_cache_wrong_family_hash_exception_recovery_and_close(native, monkeypatch):
    graph = Graph.load(BASE / "translation.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["match"]
        args = {
            name: graph.blocks["query"].last_result.outputs["image"]
            if name == "image"
            else graph.blocks[
                "query_pack" if name == "query_features" else "reference_pack"
            ].last_result.outputs["features"]
            for name in ("image", "query_features", "reference_features")
        }
        assert block.last_result.passed
        old = block._matcher
        assert block.execute(args).passed and block._matcher is old
        block.parameters["weights_sha256"] = "0" * 64
        assert (
            block.execute(args) is None and "SHA-256" in block.error and not block.current_emissions
        )
        block.parameters["weights_sha256"] = TRUTH["matcher_sha256"]
        native_fit = cv2.findHomography

        def broken(*a, **k):
            raise cv2.error("Controlled fit error")

        monkeypatch.setattr(cv2, "findHomography", broken)
        assert block.execute(args) is None and "Controlled fit" in block.error
        monkeypatch.setattr(cv2, "findHomography", native_fit)
        assert block.execute(args).passed
        block.parameters["weights_path"] = str(Path(".local/models/sift_lightglue.pth").resolve())
        block.parameters["weights_sha256"] = ""
        assert block.execute(args) is None and not block.current_emissions
        block.close()
        assert block._matcher is None and block._digest_key is None
    finally:
        graph.close()


def test_json_xml_save_as_and_exact_regeneration(tmp_path, native):
    before = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    graph = Graph.load(BASE / "translation.json", builtin_registry())
    try:
        graph.save(tmp_path / "saved.json")
        for fmt in ("json", "xml"):
            if fmt == "json":
                restored = Graph.load(tmp_path / "saved.json", builtin_registry())
            else:
                export_project_xml(graph.to_dict(), tmp_path / "saved.xml")
                restored = Graph.from_dict(
                    read_project_xml(tmp_path / "saved.xml"),
                    builtin_registry(),
                    project_dir=tmp_path,
                )
            try:
                restored.run()
                assert restored.blocks["match"].last_result.passed
            finally:
                restored.close()
    finally:
        graph.close()
    spec = importlib.util.spec_from_file_location(
        "local_feature_examples", BASE / "build_example.py"
    )
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


def test_catalog_parameter_help_and_distinct_feature_family():
    for kind in ("ml.pack_superpoint_features", "ml.lightglue_superpoint"):
        block = builtin_registry().create(kind)
        block.validate_parameters()
        assert block.parameter_help.keys() == block.defaults.keys()
    _, result = pack()
    value = result.outputs["features"]
    value["model"] = "lightglue_sift_v1"
    with pytest.raises(ValidationError):
        feature_arrays(value)


def test_base_only_example_needs_no_optional_matcher_runtime(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_ml(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "lightglue"):
            raise ImportError("Optional ML deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_ml)
    graph = Graph.load(BASE / "pack_only.json", builtin_registry())
    try:
        graph.run()
        assert all(
            graph.blocks[side + "_pack"].last_result.passed for side in ("query", "reference")
        )
    finally:
        graph.close()


def test_optional_runtime_missing_has_install_guidance(tmp_path, monkeypatch):
    import builtins

    original = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("Controlled missing torch")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    path = tmp_path / "local.pth"
    path.write_bytes(b"inert fixture")
    _, result = pack()
    data = result.outputs["features"]
    block = SuperPointLightGlueMatch(parameters={"weights_path": str(path)})
    assert (
        block.execute(
            {"image": result.outputs["image"], "query_features": data, "reference_features": data}
        )
        is None
    )
    assert "Install PyTorch" in block.error


def test_preparation_preserves_files_and_rejects_changed_download(tmp_path, monkeypatch):
    import io

    path = Path(__file__).resolve().parents[1] / "scripts/prepare_ml_assets.py"
    spec = importlib.util.spec_from_file_location("prepare_local_weights", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"original"))
    dest = tmp_path / "weights.pth"
    module.prepare_lightglue(
        dest,
        url=module.SUPERPOINT_MATCHER_URL,
        expected_sha256=hashlib.sha256(b"original").hexdigest(),
    )
    receipt = json.loads(dest.with_suffix(".receipt.json").read_bytes())
    assert receipt["source"] == module.SUPERPOINT_MATCHER_URL
    with pytest.raises(FileExistsError):
        module.prepare_lightglue(dest)
    with pytest.raises(ValueError):
        module.prepare_lightglue(tmp_path / "changed.pth", expected_sha256="0" * 64)
    assert not (tmp_path / "changed.pth").exists()
    orphan = tmp_path / "orphan.pth"
    orphan.with_suffix(".receipt.json").write_bytes(b"preserve receipt")
    with pytest.raises(FileExistsError):
        module.prepare_lightglue(orphan)
    assert not orphan.exists()
