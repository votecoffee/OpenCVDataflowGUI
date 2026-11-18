"""Image descriptor identities, finite native PCA, residuals and portable model IO."""

import copy
import json

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.classifier_common import model_revision
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.principal_components import projected_contract
from opencv_dataflow.tickets import TicketClock


@pytest.fixture
def make():
    blocks = []
    registry = builtin_registry()

    def create(kind, **parameters):
        block = registry.create(kind, parameters=parameters)
        blocks.append(block)
        return block

    yield create
    for block in blocks:
        block.close()


def fitted(make, samples=None, **parameters):
    block = make("cv.fit_pca", **parameters)
    result = block.execute(
        {"samples": [[6, 20], [14, 20], [10, 19], [10, 21]] if samples is None else samples}
    )
    assert result is not None, block.error
    return result.outputs["model"]


@pytest.mark.parametrize("normalization", ["none", "l2"])
@pytest.mark.parametrize("selection", ["count", "variance"])
def test_high_dimensional_native_fit_agrees_with_independent_svd(make, normalization, selection):
    rng = np.random.default_rng(7401)
    samples = rng.normal(size=(9, 324))
    contract = {"model": "controlled_image_features_v1", "length": 324}
    group = FlowGroup(
        tuple(ValuePacket(row.tolist(), {"feature_contract": contract}) for row in samples),
        tuple(str(i) for i in range(9)),
    )
    model = fitted(make, group, normalization=normalization, selection=selection)
    if normalization == "l2":
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    _, singular, vectors = np.linalg.svd(samples - samples.mean(0), full_matrices=False)
    count = len(model["basis"])
    basis = np.asarray(model["basis"])
    np.testing.assert_allclose(model["mean"], samples.mean(0), atol=1e-15)
    np.testing.assert_allclose(basis.T @ basis, vectors[:count].T @ vectors[:count], atol=1e-12)
    np.testing.assert_allclose(model["eigenvalues"], singular[:count] ** 2 / 9, rtol=1e-12)
    assert model["feature_contract"] == contract and model["sample_count"] == 9
    assert model["revision"] == model_revision(model)


def test_fixed_mean_and_float64_precision_are_preserved(make):
    values = np.float64([[10_000_000, 1], [10_000_000.00001, 2], [10_000_000.00002, 4]])
    model = fitted(make, values.tolist(), mean=[10_000_000, 0], selection="count")
    mean, basis, eigenvalues = cv2.PCACompute2(values, np.float64([[10_000_000, 0]]))
    assert model["mean"] == mean.ravel().tolist() and model["centering"] == "provided"
    np.testing.assert_allclose(model["eigenvalues"], eigenvalues.ravel(), rtol=1e-9)
    actual = np.asarray(model["basis"])
    np.testing.assert_allclose(actual.T @ actual, basis.T @ basis, atol=1e-12)


def test_legacy_training_value_bounds_allow_the_corresponding_total_variance(make):
    model = fitted(make, [[1e12] * 128, [-1e12] * 128])
    assert len(model["basis"]) == 1
    assert model["eigenvalues"][0] == pytest.approx(128 * 1e24)


def test_projection_reconstruction_and_inclusive_error_judgment(make):
    model = fitted(make, selection="count", component_count=1)
    block = make("cv.transform_pca", check_reconstruction_error=True, error_limit=np.sqrt(2))
    result = block.execute({"model": model, "samples": [[13, 22]]})
    assert result is not None and result.passed, block.error
    assert result.outputs["samples"] == [[3]]
    assert result.outputs["reconstruction_errors"] == [np.sqrt(2)]
    assert result.score is None and result.final_judgment is None
    block.parameters["error_limit"] = np.nextafter(np.sqrt(2), 0)
    result = block.execute({"model": model, "samples": [[13, 22]]})
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "pca_reconstruction_error"
    restore = make("cv.transform_pca", direction="reconstruct")
    restored = restore.execute(
        {
            "model": model,
            "samples": ValuePacket(result.outputs["samples"], result.payloads["samples"]),
        }
    )
    assert restored is not None, restore.error
    assert restored.outputs["samples"] == [[13, 20]]
    assert restored.outputs["maximum_error"] is NO_VALUE
    assert restored.payloads["samples"]["feature_contract"]["model"].endswith("reconstruction_v1")


def test_descriptor_and_projection_revision_mismatches_prevent_native_calls(make, monkeypatch):
    contract = {"model": "image_histogram", "bins": 2}
    model = fitted(make, ValuePacket([[1, 0], [0, 1], [2, 0]], {"feature_contract": contract}))
    project = make("cv.transform_pca")
    original = cv2.PCAProject
    monkeypatch.setattr(
        cv2, "PCAProject", lambda *a: pytest.fail("Mismatched contract reached native PCA")
    )
    assert (
        project.execute(
            {
                "model": model,
                "samples": ValuePacket([1, 0], {"feature_contract": {**contract, "bins": 3}}),
            }
        )
        is None
    )
    assert "contract" in project.error and not project.current_emissions
    monkeypatch.setattr(cv2, "PCAProject", original)
    projected = project.execute(
        {"model": model, "samples": ValuePacket([1, 0], {"feature_contract": contract})}
    )
    assert projected is not None, project.error
    other = copy.deepcopy(model)
    other["mean"][0] += 0.1
    other["revision"] = model_revision(other)
    restore = make("cv.transform_pca", direction="reconstruct")
    assert (
        restore.execute(
            {
                "model": other,
                "samples": ValuePacket(projected.outputs["samples"], projected.payloads["samples"]),
            }
        )
        is None
    )
    assert "revision differs" in restore.error
    assert projected_contract(model) != projected_contract(other)
    project.parameters["contract_policy"] = "dimensions_only"
    bypassed = project.execute({"model": model, "samples": [1, 0]})
    assert bypassed is not None and not bypassed.outputs["report"]["contract_verified"]
    assert bypassed.payloads["samples"]["feature_contract"]["verification"] == "dimensions_only"


def test_l2_reconstruction_is_explicitly_normalized_and_zero_rows_are_rejected(make):
    model = fitted(make, [[2, 0], [0, 2], [3, 4]], selection="count", normalization="l2")
    project = make("cv.transform_pca")
    result = project.execute({"model": model, "samples": [[6, 8]]})
    assert result is not None, project.error
    restored = make("cv.transform_pca", direction="reconstruct").execute(
        {"model": model, "samples": result.outputs["samples"]}
    )
    np.testing.assert_allclose(restored.outputs["samples"], [[0.6, 0.8]], atol=1e-15)
    assert project.execute({"model": model, "samples": [0, 0]}) is None
    assert "zero descriptor" in project.error
    fit = make("cv.fit_pca", normalization="l2")
    assert fit.execute({"samples": [[0, 0], [1, 2]]}) is None


def test_current_ticket_grid_pyramid_and_review_pixels_survive(make):
    model = fitted(make)
    context = {
        "ticket": TicketClock("pca-session").issue("camera", "fixed", "part-17"),
        "spatial": frame([32, 32], "camera-1"),
        "pyramid": {
            "level": 1,
            "largest_size": [64, 64],
            "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        },
        "extra": {"encoder": 1234},
    }
    image = np.arange(1024, dtype=np.uint16).reshape(32, 32)
    block = make("cv.transform_pca")
    result = block.execute(
        {
            "model": model,
            "samples": ValuePacket([11, 20], context),
            "image": ValuePacket(image, context),
        }
    )
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], image)
    for key, value in context.items():
        assert result.payloads["samples"][key] == result.payloads["image"][key] == value
    assert result.drawings and result.graphics
    block.set_graphics_mode("off")
    lean = block.execute({"model": model, "samples": ValuePacket([11, 20], context)})
    assert lean is not None and not lean.drawings and not lean.graphics
    assert lean.outputs["samples"] == result.outputs["samples"]
    bad = {**context, "spatial": frame([32, 32], "other")}
    assert (
        block.execute(
            {
                "model": model,
                "samples": ValuePacket([11, 20], context),
                "image": ValuePacket(image, bad),
            }
        )
        is None
    )
    assert "frame differs" in block.error


@pytest.mark.parametrize("setting", [{"max_work": 1}, {"max_bytes_mb": 0.01}])
def test_fit_admission_prevents_native_work_and_can_recover(make, monkeypatch, setting):
    samples = np.random.default_rng(91).normal(size=(6, 324)).tolist()
    original = cv2.PCACompute2
    monkeypatch.setattr(cv2, "PCACompute2", lambda *a: pytest.fail("Budget reached native PCA"))
    block = make("cv.fit_pca", **setting)
    assert block.execute({"samples": samples}) is None and not block.current_emissions
    monkeypatch.setattr(cv2, "PCACompute2", original)
    block.parameters.update(max_work=64_000_000, max_bytes_mb=128)
    assert block.execute({"samples": samples}) is not None, block.error


def test_projection_admission_native_exception_and_recovery(make, monkeypatch):
    model = fitted(make)
    block = make("cv.transform_pca", max_work=1)
    original = cv2.PCAProject

    def fail(*args):
        raise RuntimeError("projection was called")

    monkeypatch.setattr(cv2, "PCAProject", fail)
    assert block.execute({"model": model, "samples": [11, 20]}) is None
    assert "Max work" in block.error and "projection was called" not in block.error
    block.parameters["max_work"] = 64_000_000
    assert block.execute({"model": model, "samples": [11, 20]}) is None
    assert "projection was called" in block.error and not block.current_emissions
    monkeypatch.setattr(cv2, "PCAProject", original)
    assert block.execute({"model": model, "samples": [11, 20]}) is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "bad"),
        ("normalization", "unknown"),
        ("sample_count", 1),
        ("total_variance", -1),
        ("retained_variance", 2),
        ("eigenvalues", [1]),
    ],
)
def test_tampered_or_rehashed_invalid_models_are_contained(make, field, value):
    model = fitted(make, selection="count")
    model[field] = value
    block = make("cv.transform_pca")
    assert block.execute({"model": model, "samples": [11, 20]}) is None
    if field != "revision":
        model["revision"] = model_revision(model)
        assert block.execute({"model": model, "samples": [11, 20]}) is None
    assert not block.current_emissions


def test_mixed_group_contracts_and_native_exceptions_do_not_emit(make, monkeypatch):
    group = FlowGroup(
        (ValuePacket([1, 2], {"feature_contract": {"bins": 2}}), ValuePacket([3, 4], {})),
        ("a", "b"),
    )
    fit = make("cv.fit_pca")
    assert fit.execute({"samples": group}) is None and "different preprocessing" in fit.error
    original = cv2.PCACompute2

    def fail(*args):
        raise RuntimeError("native PCA failed")

    monkeypatch.setattr(cv2, "PCACompute2", fail)
    assert fit.execute({"samples": [[1, 2], [3, 4]]}) is None
    assert "native PCA failed" in fit.error and not fit.current_emissions
    monkeypatch.setattr(cv2, "PCACompute2", original)
    assert fit.execute({"samples": [[1, 2], [3, 4]]}) is not None


def test_portable_save_load_legacy_and_existing_file_preservation(make, tmp_path):
    model = fitted(make)
    path = tmp_path / "model.json"
    save = make("cv.save_pca", path=str(path))
    assert save.execute({"model": model}) is not None, save.error
    before = path.read_bytes()
    assert b"\r" not in before and before.endswith(b"\n")
    assert save.execute({"model": model}) is None and path.read_bytes() == before
    load = make("cv.load_pca", path=str(path))
    result = load.execute({})
    assert result is not None and result.outputs["model"] == model, load.error
    legacy = {"model": "principal_components_v1", "mean": [1, 2], "basis": [[1, 0]]}
    path.write_bytes((json.dumps(legacy) + "\n").encode("utf-8"))
    result = load.execute({})
    assert result is not None, load.error
    project = make("cv.transform_pca")
    result = project.execute({"model": result.outputs["model"], "samples": [3, 4]})
    assert result.outputs["samples"] == [[2]] and result.outputs["report"]["legacy_model"]


@pytest.mark.parametrize("raw", [b'{"model":1,"model":2}', b'{"x":NaN}', b"x" * (8 * 1024**2 + 1)])
def test_model_loader_bounds_and_strict_json(make, tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    block = make("cv.load_pca", path=str(path))
    assert block.execute({}) is None and not block.current_emissions


@pytest.mark.parametrize(
    "kind,parameters",
    [
        ("cv.fit_pca", {"component_count": 0}),
        ("cv.fit_pca", {"normalization": "unknown"}),
        ("cv.fit_pca", {"minimum_variance": 1.1}),
        ("cv.fit_pca", {"mean": [True]}),
        ("cv.fit_pca", {"max_work": 0}),
        ("cv.transform_pca", {"max_bytes_mb": 0}),
        ("cv.transform_pca", {"check_reconstruction_error": 1}),
        ("cv.transform_pca", {"direction": "reconstruct", "check_reconstruction_error": True}),
        ("cv.transform_pca", {"contract_policy": "unknown"}),
    ],
)
def test_parameter_limits_and_enums_are_reported(make, kind, parameters):
    with pytest.raises(ValueError):
        make(kind, **parameters).validate_parameters()
