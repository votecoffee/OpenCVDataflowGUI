"""Descriptor covariance identity, numerical references and safe image outlier scoring."""

import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.classifier_common import model_revision
from opencv_dataflow.feature_covariance import inverse_covariance
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame
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
    block = make("cv.feature_covariance", **parameters)
    result = block.execute(
        {"samples": [[-2, -1], [-1, -2], [1, 2], [2, 1]] if samples is None else samples}
    )
    assert result is not None, block.error
    return result.outputs["model"]


@pytest.mark.parametrize("normalization,ddof", [("sample", 1), ("population", 0)])
@pytest.mark.parametrize("row_norm", ["none", "l2"])
def test_float64_group_fit_and_score_match_independent_covariance_and_quadratic_form(
    make, normalization, ddof, row_norm
):
    rows = np.random.default_rng(743).normal(size=(8, 5))
    contract = {"model": "image_features_test_v1", "dimensions": 5}
    grouped = FlowGroup(
        tuple(ValuePacket(row.tolist(), {"feature_contract": contract}) for row in rows),
        tuple(str(i) for i in range(8)),
    )
    model = fitted(make, grouped, normalization=normalization, feature_normalization=row_norm)
    reference = rows / np.linalg.norm(rows, axis=1, keepdims=True) if row_norm == "l2" else rows
    expected_mean, expected_cov = reference.mean(0), np.cov(reference, rowvar=False, ddof=ddof)
    np.testing.assert_allclose(model["mean"], expected_mean, atol=1e-15)
    np.testing.assert_allclose(model["covariance"], expected_cov, atol=1e-15)
    assert model["feature_contract"] == contract and model["revision"] == model_revision(model)
    query = np.array([[0.1, 0.3, -0.5, 0.2, 1.4], [0.2, 0.4, 0.9, 1.3, 0.1]])
    block = make("inspect.mahalanobis_distance", ridge=0.02)
    result = block.execute(
        {"model": model, "samples": ValuePacket(query.tolist(), {"feature_contract": contract})}
    )
    assert result is not None, block.error
    if row_norm == "l2":
        query /= np.linalg.norm(query, axis=1, keepdims=True)
    delta = query - expected_mean
    inverse = np.linalg.inv(expected_cov + np.eye(5) * 0.02)
    expected = np.sqrt(np.einsum("ni,ij,nj->n", delta, inverse, delta))
    np.testing.assert_allclose(result.outputs["distances"], expected, rtol=1e-12)
    assert result.outputs["report"]["contract_verified"] and result.score is None


def test_fixed_center_matches_independent_second_moment_and_large_offsets(make):
    rows = np.array([[1e7, 1], [1e7 + 0.0001, 2], [1e7 + 0.0002, 4]], np.float64)
    model = fitted(make, rows.tolist(), mean=[1e7, 0], normalization="population")
    centered = rows - [1e7, 0]
    np.testing.assert_allclose(model["covariance"], centered.T @ centered / 3, atol=1e-15)
    assert model["mean"] == [1e7, 0] and model["centering"] == "provided"


def test_full_image_descriptor_width_is_supported_without_pca(make):
    rows = np.random.default_rng(145).normal(size=(9, 144)).tolist()
    model = fitted(make, rows)
    block = make("inspect.mahalanobis_distance", ridge=0.01)
    result = block.execute({"model": model, "samples": rows[0]})
    assert result is not None and result.outputs["report"]["dimensions"] == 144, block.error
    assert np.isfinite(result.outputs["maximum_distance"])


def test_default_324_value_hog_descriptors_work_without_a_reduction_block(make):
    hog = make("inspect.hog")
    packets = []
    for seed in range(9):
        image = np.random.default_rng(seed).integers(0, 256, (32, 32), np.uint8)
        result = hog.execute({"image": image})
        assert result is not None, hog.error
        packets.append(ValuePacket(result.outputs["descriptor"], result.payloads["descriptor"]))
    model = fitted(make, FlowGroup(tuple(packets), tuple(str(i) for i in range(9))))
    assert len(model["mean"]) == 324
    block = make("inspect.mahalanobis_distance", ridge=0.001)
    result = block.execute({"model": model, "samples": packets[0]})
    assert result is not None and np.isfinite(result.outputs["maximum_distance"]), block.error


def test_regularization_inclusive_boundary_and_diagnostic_only_mode(make):
    model = fitted(make, normalization="population")
    block = make("inspect.mahalanobis_distance", check_distance=True, distance_limit=4)
    result = block.execute({"model": model, "samples": [2, -2]})
    assert result is not None and result.passed and result.outputs["maximum_distance"] == 4, (
        block.error
    )
    block.parameters["distance_limit"] = np.nextafter(4, 0)
    result = block.execute({"model": model, "samples": [2, -2]})
    assert (
        not result.passed
        and result.fail_reasons[0]["code"] == "mahalanobis_distance_out_of_tolerance"
    )
    block.parameters["check_distance"] = False
    assert block.execute({"model": model, "samples": [2, -2]}).passed
    singular = fitted(make, [[1, 1], [1, 1]])
    assert block.execute({"model": singular, "samples": [2, 2]}) is None
    assert "singular" in block.error
    block.parameters["ridge"] = 0.25
    result = block.execute({"model": singular, "samples": [2, 2]})
    assert result is not None, block.error
    assert result.outputs["maximum_distance"] == pytest.approx(np.sqrt(8))


def test_descriptor_contract_and_pca_revision_mismatch_prevent_native_scoring(make, monkeypatch):
    contract = {"model": "principal_components_projection_v1", "pca_revision": "training-space"}
    model = fitted(
        make, ValuePacket([[-2, -1], [-1, -2], [1, 2], [2, 1]], {"feature_contract": contract})
    )
    original = cv2.eigen
    monkeypatch.setattr(
        cv2, "eigen", lambda *a: pytest.fail("Mismatched features reached native work")
    )
    block = make("inspect.mahalanobis_distance")
    assert (
        block.execute(
            {
                "model": model,
                "samples": ValuePacket(
                    [1, 2], {"feature_contract": {**contract, "pca_revision": "other-space"}}
                ),
            }
        )
        is None
    )
    assert "preprocessing differs" in block.error and not block.current_emissions
    monkeypatch.setattr(cv2, "eigen", original)
    block.parameters["contract_policy"] = "dimensions_only"
    result = block.execute({"model": model, "samples": [1, 2]})
    assert result is not None and not result.outputs["report"]["contract_verified"], block.error
    assert result.payloads["distances"]["feature_contract"]["verification"] == "dimensions_only"


def test_current_image_ticket_pyramid_companion_metadata_and_lean_graphics(make):
    model = fitted(make)
    context = {
        "ticket": TicketClock("outliers").issue("camera", "fixed", "part-43"),
        "spatial": frame([32, 32], "fixture"),
        "extra": {"encoder": 21},
        "pyramid": {
            "level": 1,
            "largest_size": [64, 64],
            "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        },
    }
    image = np.arange(1024, dtype=np.uint16).reshape(32, 32)
    block = make("inspect.mahalanobis_distance", check_distance=True, distance_limit=1)
    result = block.execute(
        {
            "model": model,
            "samples": ValuePacket([2, -2], context),
            "image": ValuePacket(image, context),
        }
    )
    assert result is not None and not result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], image)
    for key, value in context.items():
        assert result.payloads["distances"][key] == result.payloads["image"][key] == value
    assert result.drawings and result.graphics and result.final_judgment is None
    block.set_graphics_mode("off")
    lean = block.execute(
        {
            "model": model,
            "samples": ValuePacket([2, -2], context),
            "image": ValuePacket(image, context),
        }
    )
    assert not lean.passed and not lean.drawings and not lean.graphics
    assert lean.outputs["distances"] == result.outputs["distances"]


@pytest.mark.parametrize("field", ["spatial", "ticket", "pyramid"])
def test_mismatched_review_image_is_rejected_before_native_inversion(make, monkeypatch, field):
    model = fitted(make)
    context = {
        "ticket": TicketClock("a").issue("camera"),
        "spatial": frame([8, 8], "a"),
        "pyramid": {
            "level": 0,
            "largest_size": [8, 8],
            "to_largest": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
    }
    other = copy.deepcopy(context)
    if field == "spatial":
        other[field] = frame([8, 8], "b")
    elif field == "ticket":
        other[field] = TicketClock("b").issue("camera")
    else:
        other[field]["level"] = 1
    monkeypatch.setattr(cv2, "invert", lambda *a, **k: pytest.fail("Wrong image reached inversion"))
    block = make("inspect.mahalanobis_distance")
    assert (
        block.execute(
            {
                "model": model,
                "samples": ValuePacket([1, 2], context),
                "image": ValuePacket(np.zeros((8, 8), np.uint8), other),
            }
        )
        is None
    )
    assert not block.current_emissions


@pytest.mark.parametrize(
    "kind,setting",
    [
        ("cv.feature_covariance", {"max_work": 1}),
        ("cv.feature_covariance", {"max_bytes_mb": 0.01}),
        ("inspect.mahalanobis_distance", {"max_work": 1}),
        ("inspect.mahalanobis_distance", {"max_bytes_mb": 0.01}),
    ],
)
def test_budget_checks_prevent_native_work_and_recover(make, monkeypatch, kind, setting):
    rows = np.random.default_rng(43).normal(size=(9, 144)).tolist()
    model = fitted(make, rows)
    original_cov, original_eigen = cv2.calcCovarMatrix, cv2.eigen
    monkeypatch.setattr(
        cv2, "calcCovarMatrix", lambda *a, **k: pytest.fail("Budget reached covariance")
    )
    monkeypatch.setattr(cv2, "eigen", lambda *a, **k: pytest.fail("Budget reached eigen"))
    block = make(kind, **setting)
    inputs = {"samples": rows, **({"model": model} if kind.startswith("inspect") else {})}
    assert block.execute(inputs) is None and not block.current_emissions
    monkeypatch.setattr(cv2, "calcCovarMatrix", original_cov)
    monkeypatch.setattr(cv2, "eigen", original_eigen)
    block.parameters.update(max_work=64_000_000, max_bytes_mb=128)
    if kind.startswith("inspect"):
        block.parameters["ridge"] = 0.01
    assert block.execute(inputs) is not None, block.error


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "changed"),
        ("sample_count", 0),
        ("feature_normalization", "bad"),
        ("normalization", "bad"),
        ("centering", "bad"),
        ("covariance", [[-1, 0], [0, 1]]),
    ],
)
def test_integrity_and_rehashed_invalid_models_are_contained(make, field, value):
    model = fitted(make)
    model[field] = value
    block = make("inspect.mahalanobis_distance")
    assert block.execute({"model": model, "samples": [1, 2]}) is None
    if field != "revision":
        model["revision"] = model_revision(model)
        assert block.execute({"model": model, "samples": [1, 2]}) is None
    assert not block.current_emissions


def test_json_model_io_legacy_numpy_rows_and_existing_file_preservation(make, tmp_path):
    model = fitted(make)
    path = tmp_path / "covariance.json"
    save = make("cv.save_feature_covariance", path=str(path))
    assert save.execute({"model": model}) is not None, save.error
    before = path.read_bytes()
    assert b"\r" not in before and before.endswith(b"\n")
    assert save.execute({"model": model}) is None and path.read_bytes() == before
    load = make("cv.load_feature_covariance", path=str(path))
    assert load.execute({}).outputs["model"] == model
    for bad in (b'{"model":1,"model":2}', b'{"x":NaN}', b"x" * (8 * 1024**2 + 1)):
        path.write_bytes(bad)
        assert load.execute({}) is None and not load.current_emissions
    legacy = {"model": "feature_covariance_v1", "mean": np.array([0, 0]), "covariance": np.eye(2)}
    mean, inverse = inverse_covariance(legacy, 2, 0)
    np.testing.assert_array_equal(mean, [0, 0])
    np.testing.assert_array_equal(inverse, np.eye(2))
    legacy = {**legacy, "mean": [0, 0], "covariance": [[1, 0], [0, 1]]}
    check = make("inspect.mahalanobis_distance")
    result = check.execute({"model": legacy, "samples": [3, 4]})
    assert result is not None and result.outputs["maximum_distance"] == 5, check.error
    assert result.outputs["report"]["legacy_model"]


def test_native_exception_and_zero_l2_features_do_not_emit_stale_results(make, monkeypatch):
    model = fitted(make, feature_normalization="l2")
    block = make("inspect.mahalanobis_distance", ridge=0.01)
    assert block.execute({"model": model, "samples": [1, 2]}) is not None
    assert (
        block.execute({"model": model, "samples": [0, 0]}) is None and not block.current_emissions
    )
    original = cv2.Mahalanobis

    def fail(*args):
        raise RuntimeError("native distance failed")

    monkeypatch.setattr(cv2, "Mahalanobis", fail)
    assert block.execute({"model": model, "samples": [1, 2]}) is None
    assert "native distance failed" in block.error and not block.current_emissions
    monkeypatch.setattr(cv2, "Mahalanobis", original)
    assert block.execute({"model": model, "samples": [1, 2]}) is not None


def test_static_numeric_review_and_pca_compatibility_use_unknown_context(make):
    model = fitted(make)
    block = make("inspect.mahalanobis_distance", samples=[[1, 2]])
    result = block.execute({"model": model, "image": np.zeros((8, 8), np.uint8)})
    assert result is not None and not result.outputs["report"]["contract_verified"], block.error
    pca = make("cv.fit_pca")
    pca_model = pca.execute({"samples": [[1, 2], [2, 4], [4, 5]]}).outputs["model"]
    project = make("cv.transform_pca", samples=[[2, 3]])
    assert project.execute({"model": pca_model, "image": np.zeros((8, 8), np.uint8)}) is not None


@pytest.mark.parametrize(
    "kind,p",
    [
        ("cv.feature_covariance", {"normalization": "invalid"}),
        ("cv.feature_covariance", {"feature_normalization": "invalid"}),
        ("cv.feature_covariance", {"mean": [True]}),
        ("inspect.mahalanobis_distance", {"ridge": -1}),
        ("inspect.mahalanobis_distance", {"distance_limit": -1}),
        ("inspect.mahalanobis_distance", {"check_distance": 1}),
        ("inspect.mahalanobis_distance", {"contract_policy": "invalid"}),
        ("inspect.mahalanobis_distance", {"max_work": 0}),
    ],
)
def test_parameter_ranges_and_choices_are_validated(make, kind, p):
    with pytest.raises(ValueError):
        make(kind, **p).validate_parameters()
