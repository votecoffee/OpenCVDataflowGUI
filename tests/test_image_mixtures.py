"""Native EM and independent Gaussian density, selections and portable image models."""

import json
import math

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.classifier_common import model_revision
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
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


def image():
    rng = np.random.default_rng(1042)
    values = np.empty((32, 48, 3), np.uint8)
    values[:, :24] = [40, 70, 180]
    values[:, 24:] = [180, 150, 40]
    return np.uint8(np.clip(values.astype(int) + rng.integers(-5, 6, values.shape), 0, 255))


def train(make, source=None, **parameters):
    block = make("ml.em_train_image", feature_space="bgr", **parameters)
    result = block.execute({"image": image() if source is None else source})
    assert result is not None, block.error
    return result.outputs["model"]


def direct_native(model):
    storage = cv2.FileStorage(
        json.dumps(model["native"]), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY
    )
    try:
        native = cv2.ml.EM_create()
        native.read(storage.root())
    finally:
        storage.release()
    return native


@pytest.mark.parametrize("kind", ["diagonal", "spherical", "generic"])
@pytest.mark.parametrize("initialization", ["auto", "manual"])
def test_native_training_modes_and_prediction_density_reference(make, kind, initialization):
    source = image()
    settings = {"initialization": initialization, "covariance_type": kind}
    if initialization == "manual":
        settings.update(
            initial_means=[[40 / 255, 70 / 255, 180 / 255], [180 / 255, 150 / 255, 40 / 255]],
            initial_weights=[0.4, 0.6],
        )
    model = train(make, source, **settings)
    block = make("ml.em_classify_image", chunk_size=137)
    block.exposed_outputs.append("posteriors")
    query = source.copy()
    query[9:14, 10:18] = [5, 240, 10]
    result = block.execute({"model": model, "image": query})
    assert result is not None, block.error
    native = direct_native(model)
    for y, x in ((0, 0), (10, 12), (20, 31), (31, 47)):
        # Match documented unit conversion, independently of the app's feature helper.
        vector = (query[y, x].astype(np.float32) / np.float32(255)).astype(np.float64)[None]
        (density, label), probabilities = native.predict2(vector)
        assert result.outputs["labels"][y, x] == label
        assert result.outputs["log_likelihood"][y, x] == pytest.approx(density, abs=1e-8)
        assert result.outputs["winning_posterior"][y, x] == pytest.approx(
            probabilities.max(), abs=1e-12
        )
        for index, packet in enumerate(result.outputs["posteriors"].items):
            assert packet.value[y, x] == pytest.approx(probabilities[0, index], abs=1e-7)
    np.testing.assert_array_equal(result.outputs["image"], query)
    np.testing.assert_array_equal(result.outputs["unknown_mask"][9:14, 10:18], 255)
    assert result.outputs["report"]["unknown_count"] == 40 and not result.passed
    assert result.fail_reasons[0]["code"] == "em_unknown_pixels"
    assert result.score == pytest.approx(1 - 40 / (32 * 48)) and result.final_judgment is None
    assert result.drawings and result.graphics
    assert model["revision"] == model_revision(model)
    assert model["training"]["initialization"] == initialization


def test_manual_initial_covariances_and_bounded_sampling_do_not_set_global_rng(make, monkeypatch):
    monkeypatch.setattr(cv2, "setRNGSeed", lambda *a: pytest.fail("Runtime changed global RNG"))
    model = train(
        make,
        initialization="manual",
        max_samples=80,
        initial_means=[[0.15, 0.27, 0.7], [0.7, 0.59, 0.15]],
        initial_covariances=[np.eye(3).tolist(), (np.eye(3) * 0.5).tolist()],
    )
    assert model["training"]["sample_count"] == 80
    assert model["training"]["sampling"] == "even_supported_row_major"
    assert model["native"]["means"]["data"][:3] < model["native"]["means"]["data"][3:]


@pytest.mark.parametrize("kind", ["diagonal", "spherical", "generic"])
def test_manual_gaussian_initialization_matches_independent_native_maximization(make, kind):
    source = image()
    values = (source.astype(np.float32) / np.float32(255)).reshape(-1, 3).astype(np.float64)
    means = np.float64([[0.15, 0.27, 0.7], [0.7, 0.59, 0.15]])
    cov = np.eye(3) * 0.01
    if kind == "generic":
        cov[0, 1] = cov[1, 0] = 0.001
    covs, weights = np.array([cov, cov]), np.float64([0.4, 0.6])
    terms = []
    for weight, mean, covariance in zip(weights, means, covs, strict=True):
        delta = values - mean
        quadratic = np.einsum("ni,ij,nj->n", delta, np.linalg.inv(covariance), delta)
        terms.append(np.log(weight) - 0.5 * (quadratic + np.linalg.slogdet(covariance)[1]))
    terms = np.column_stack(terms)
    probabilities = np.exp(terms - terms.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    native = cv2.ml.EM_create()
    native.setClustersNumber(2)
    native.setCovarianceMatrixType({"diagonal": 1, "spherical": 0, "generic": 2}[kind])
    native.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-6))
    assert native.trainM(values, probabilities)[0]
    model = train(
        make,
        source,
        covariance_type=kind,
        initialization="manual",
        initial_means=means.tolist(),
        initial_covariances=covs.tolist(),
        initial_weights=weights.tolist(),
    )
    order = sorted(range(2), key=lambda i: tuple(native.getMeans()[i]))
    np.testing.assert_allclose(
        np.array(model["native"]["means"]["data"]).reshape(2, 3),
        native.getMeans()[order],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        model["native"]["weights"]["data"], native.getWeights().ravel()[order], atol=1e-12
    )
    assert model["training"]["covariance_type"] == kind
    assert model["native"]["training_params"]["cov_mat_type"] == (
        "diagonal" if kind == "spherical" else kind
    )


def test_spherical_prediction_representation_matches_three_dimensional_gaussian(make):
    trainer = make(
        "ml.em_train_image",
        feature_space="bgr",
        covariance_type="spherical",
        components=1,
        covariance_floor=0.01,
    )
    source = np.full((12, 12, 3), 100, np.uint8)
    result = trainer.execute({"image": source})
    assert result is not None, trainer.error
    model = result.outputs["model"]
    block = make("ml.em_classify_image")
    query = np.full_like(source, 200)
    result = block.execute({"image": query, "model": model})
    assert result is not None, block.error
    mean = np.array(model["native"]["means"]["data"])
    value = (query[0, 0].astype(np.float32) / np.float32(255)).astype(float)
    expected = -0.5 * (((value - mean) ** 2 / 0.01).sum() + 3 * np.log(2 * np.pi * 0.01))
    np.testing.assert_allclose(result.outputs["log_likelihood"], expected, atol=1e-10)
    assert result.outputs["report"]["fitted_covariance_type"] == "spherical"
    assert result.outputs["report"]["prediction_covariance_type"] == "diagonal"


def test_one_component_closed_form_density_high_posterior_is_not_novelty_acceptance(make):
    source = np.full((12, 12), 100, np.uint8)
    trainer = make("ml.em_train_image", feature_space="gray", components=1, covariance_floor=0.01)
    fitted = trainer.execute({"image": source})
    assert fitted is not None, trainer.error
    model = fitted.outputs["model"]
    block = make("ml.em_classify_image", min_log_likelihood=-2)
    query = np.full_like(source, 250)
    result = block.execute({"model": model, "image": query})
    assert result is not None and not result.passed, block.error
    mean = model["native"]["means"]["data"][0]
    covariance = model["native"]["covs"][0]["data"][0]
    value = float(np.float32(250) / np.float32(255))
    expected = -0.5 * ((value - mean) ** 2 / covariance + math.log(2 * math.pi * covariance))
    np.testing.assert_allclose(result.outputs["log_likelihood"], expected, atol=1e-10)
    np.testing.assert_array_equal(result.outputs["winning_posterior"], 1)
    block.parameters["check_log_likelihood"] = False
    assert block.execute({"model": model, "image": query}).passed


@pytest.mark.parametrize("encoding", ["bgr", "rgb", "bgra", "rgba"])
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32", "float64"])
def test_fixed_color_scaling_encodings_alpha_and_current_metadata(make, encoding, dtype):
    source = image()
    model = train(make, source)
    encoded = source[..., ::-1] if encoding in ("rgb", "rgba") else source.copy()
    if "a" in encoding:
        encoded = np.dstack((encoded, np.full(source.shape[:2], 255, np.uint8)))
        encoded[:3, :4, 3] = 0
    if dtype == "uint16":
        encoded = encoded.astype(np.uint16) * 257
    elif dtype.startswith("float"):
        encoded = encoded.astype(dtype) / 255
    context = {
        "ticket": TicketClock("em").issue("camera", "fixed", "part-42"),
        "spatial": frame([48, 32], "camera-grid"),
        "pyramid": {
            "level": 1,
            "largest_size": [96, 64],
            "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        },
        "extra": {"encoder": 456},
    }
    tagged = ImageArray(encoded, encoding)
    block = make("ml.em_classify_image")
    result = block.execute({"model": model, "image": ValuePacket(tagged, context)})
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], encoded)
    assert result.outputs["image"].color_format == encoding
    for key, value in context.items():
        assert result.payloads["labels"][key] == result.payloads["image"][key] == value
    assert result.outputs["report"]["supported_pixels"] == (
        32 * 48 - 12 if "a" in encoding else 32 * 48
    )
    assert result.outputs["posteriors"] is NO_VALUE
    block.set_graphics_mode("off")
    lean = block.execute({"model": model, "image": ValuePacket(tagged, context)})
    assert lean.passed and not lean.drawings and not lean.graphics
    np.testing.assert_array_equal(lean.outputs["unknown_mask"], result.outputs["unknown_mask"])


def test_texture_mean_std_have_independent_window_reference_and_conservative_selection(make):
    y, x = np.indices((25, 29))
    source = np.uint8(100 + ((x + y) % 2) * 40)
    mask = np.full(source.shape, 255, np.uint8)
    mask[10:13, 12:15] = 0
    trainer = make(
        "ml.em_train_image",
        feature_space="gray_texture",
        texture_window=5,
        components=1,
        covariance_type="generic",
    )
    fitted = trainer.execute({"image": source, "mask": mask})
    assert fitted is not None, trainer.error
    model = fitted.outputs["model"]
    block = make("ml.em_classify_image", check_log_likelihood=False)
    result = block.execute({"model": model, "image": source, "mask": mask})
    assert result is not None, block.error
    native = direct_native(model)
    expected_support = np.zeros(source.shape, bool)
    for yy in range(2, 23):
        for xx in range(2, 27):
            window = mask[yy - 2 : yy + 3, xx - 2 : xx + 3]
            expected_support[yy, xx] = bool(np.all(window))
    np.testing.assert_array_equal(result.outputs["valid_mask"] != 0, expected_support)
    values = (source[2:7, 2:7].astype(np.float32) / np.float32(255)).astype(float)
    (density, _), _ = native.predict2(np.float64([[values.mean(), values.std()]]))
    assert result.outputs["log_likelihood"][4, 4] == pytest.approx(density, abs=1e-8)
    block.parameters["minimum_support_fraction"] = 1.0
    result = block.execute({"model": model, "image": source, "mask": mask})
    assert not result.passed and result.fail_reasons[0]["code"] == "em_feature_support"


def test_roi_component_filter_and_inclusive_unknown_fraction(make):
    source = image()
    model = train(make, source)
    mask = np.zeros(source.shape[:2], np.uint8)
    mask[4:24, 5:37] = 255
    block = make("ml.em_classify_image", accepted_components=[0], max_unknown_fraction=13 / 32)
    result = block.execute({"model": model, "image": source, "mask": mask})
    assert result is not None and result.passed, block.error
    assert result.outputs["report"]["unknown_fraction"] == 13 / 32
    block.parameters["max_unknown_fraction"] = np.nextafter(13 / 32, 0)
    assert not block.execute({"model": model, "image": source, "mask": mask}).passed
    assert np.all(result.outputs["labels"][mask == 0] == -1)
    assert not result.outputs["unknown_mask"][mask == 0].any()


@pytest.mark.parametrize(
    "kind,settings",
    [
        ("ml.em_train_image", {"max_work": 1}),
        ("ml.em_train_image", {"max_bytes_mb": 0.01}),
        ("ml.em_classify_image", {"max_work": 1}),
        ("ml.em_classify_image", {"max_bytes_mb": 0.01}),
    ],
)
def test_resource_guards_prevent_native_em_and_allow_recovery(make, monkeypatch, kind, settings):
    model = train(make)
    original = cv2.ml.EM_create
    monkeypatch.setattr(cv2.ml, "EM_create", lambda: pytest.fail("Budget reached native EM"))
    block = make(kind, **settings)
    inputs = {"image": image(), **({"model": model} if "classify" in kind else {})}
    assert block.execute(inputs) is None and not block.current_emissions
    monkeypatch.setattr(cv2.ml, "EM_create", original)
    block.parameters.update(max_work=128_000_000, max_bytes_mb=256)
    assert block.execute(inputs) is not None, block.error


@pytest.mark.parametrize(
    "tamper", ["revision", "weights", "covariance", "means", "settings", "version"]
)
def test_tampered_and_rehashed_invalid_models_are_rejected_before_native_read(
    make, monkeypatch, tamper
):
    model = train(make)
    if tamper == "revision":
        model["revision"] = "invalid"
    elif tamper == "weights":
        model["native"]["weights"]["data"] = [-1, 2]
    elif tamper == "covariance":
        model["native"]["covs"][0]["data"][0] = -1
    elif tamper == "means":
        model["native"]["means"]["rows"] = 500000
    elif tamper == "settings":
        model["native"]["training_params"]["nclusters"] = 17
    else:
        model["features"]["opencv_version"] = "4.12.0"
    monkeypatch.setattr(
        cv2.ml, "EM_create", lambda: pytest.fail("Invalid model reached native read")
    )
    block = make("ml.em_classify_image")
    assert block.execute({"model": model, "image": image()}) is None
    if tamper != "revision":
        model["revision"] = model_revision(model)
        assert block.execute({"model": model, "image": image()}) is None
    assert not block.current_emissions


def test_strict_portable_model_io_preserves_existing_files(make, tmp_path):
    model = train(make)
    path = tmp_path / "mixture.json"
    save = make("ml.em_save", path=str(path))
    assert save.execute({"model": model}) is not None
    before = path.read_bytes()
    assert save.execute({"model": model}) is None and path.read_bytes() == before
    load = make("ml.em_load", path=str(path))
    assert load.execute({}).outputs["model"] == model
    for raw in (b'{"model":1,"model":2}', b'{"value":NaN}', b"x" * 65537):
        path.write_bytes(raw)
        assert load.execute({}) is None and not load.current_emissions
    path.write_bytes(before)
    assert load.execute({}) is not None


def test_native_exception_is_contained_and_next_input_recovers(make, monkeypatch):
    model = train(make)
    block = make("ml.em_classify_image")
    original = cv2.ml.EM_create

    def fail():
        raise RuntimeError("EM native failure")

    monkeypatch.setattr(cv2.ml, "EM_create", fail)
    assert block.execute({"image": image(), "model": model}) is None
    assert "EM native failure" in block.error and not block.current_emissions
    monkeypatch.setattr(cv2.ml, "EM_create", original)
    assert block.execute({"image": image(), "model": model}) is not None


@pytest.mark.parametrize(
    "kind,p",
    [
        ("ml.em_train_image", {"components": 0}),
        ("ml.em_train_image", {"max_samples": 1}),
        ("ml.em_train_image", {"feature_space": "hsv"}),
        ("ml.em_train_image", {"covariance_type": "unknown"}),
        ("ml.em_train_image", {"texture_window": 4}),
        ("ml.em_train_image", {"initialization": "manual"}),
        (
            "ml.em_train_image",
            {"initialization": "manual", "initial_means": [[False, 0, 0], [1, 1, 1]]},
        ),
        ("ml.em_train_image", {"covariance_floor": 0}),
        ("ml.em_classify_image", {"min_posterior": 1.1}),
        ("ml.em_classify_image", {"accepted_components": [True]}),
        ("ml.em_classify_image", {"accepted_components": [1, 1]}),
        ("ml.em_classify_image", {"check_log_likelihood": 1}),
    ],
)
def test_parameter_validation_and_enums(make, kind, p):
    with pytest.raises(ValueError):
        make(kind, **p).validate_parameters()
