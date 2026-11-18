"""Independent visual-word distances, portable identity and bounded failure contracts."""

import copy
import json

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.classifier_common import model_revision
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.visual_words import FEATURE_DEFAULTS, FEATURES, contract


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


def record(method="sift", wta="2", seed=3):
    dimension = 128 if method == "sift" else 32
    rows = np.random.default_rng(seed).integers(0, 256, (32, dimension), np.uint8)
    if method == "sift":
        rows = rows.astype(np.float32)
    return {
        "model": FEATURES,
        "extractor": contract({**FEATURE_DEFAULTS, "method": method, "orb_wta_k": wta}),
        "image_size": [64, 64],
        "points": [[float(i + 1), 20.0] for i in range(32)],
        "descriptors": rows.tolist(),
    }


def trained(make, data=None):
    data = record() if data is None else data
    block = make("ml.bow_train", words=4)
    result = block.execute({"features": data})
    assert result is not None, block.error
    return result.outputs["model"]


@pytest.mark.parametrize("method,wta", [("sift", "2"), ("orb", "2"), ("orb", "3"), ("orb", "4")])
@pytest.mark.parametrize("normalization", ["l1", "l2", "hellinger"])
def test_native_assignments_and_histograms_match_independent_float_or_bit_references(
    make, method, wta, normalization
):
    data = record(method, wta)
    model = trained(make, data)
    encode = make("ml.bow_encode", normalization=normalization)
    result = encode.execute({"model": model, "features": data})
    assert result is not None, encode.error
    rows, centers = np.array(data["descriptors"]), np.array(model["centers"])
    if method == "sift":
        distances = np.linalg.norm(rows[:, None] - centers[None], axis=2)
    else:
        xor = np.bitwise_xor(rows[:, None].astype(np.uint8), centers[None].astype(np.uint8))
        distances = (
            np.unpackbits(xor, axis=2).sum(2)
            if wta == "2"
            else ((xor[..., None] >> np.array([0, 2, 4, 6])) & 3 != 0).sum((2, 3))
        )
        assert all(any(np.array_equal(center, row) for row in rows) for center in centers)
    ids = distances.argmin(1)
    np.testing.assert_array_equal(result.outputs["word_ids"], ids)
    np.testing.assert_allclose(
        result.outputs["distances"], distances[np.arange(len(rows)), ids], rtol=1e-6
    )
    histogram = np.bincount(ids, minlength=4).astype(float)
    histogram /= np.linalg.norm(histogram) if normalization == "l2" else histogram.sum()
    if normalization == "hellinger":
        histogram = np.sqrt(histogram)
    np.testing.assert_allclose(result.outputs["histogram"], histogram, atol=1e-15)
    assert result.score is None and result.final_judgment is None
    assert (
        result.payloads["histogram"]["feature_contract"]["codebook_revision"] == model["revision"]
    )


def test_training_is_repeatable_and_native_centroids_match_assigned_samples(make):
    data = record()
    first, second = trained(make, data), trained(make, data)
    assert first == second
    rows, centers = np.array(data["descriptors"]), np.array(first["centers"])
    ids = np.linalg.norm(rows[:, None] - centers[None], axis=2).argmin(1)
    for index, center in enumerate(centers):
        np.testing.assert_allclose(center, rows[ids == index].mean(0), rtol=1e-6)


@pytest.mark.parametrize("metric", ["cosine", "l2", "chi_square", "bhattacharyya"])
def test_gallery_rank_stable_ties_metric_references_and_inclusive_rejection(make, metric):
    data, model = record(), trained(make)
    encoder = make("ml.bow_encode")
    result = encoder.execute({"model": model, "features": data})
    histogram, identity = result.outputs["histogram"], result.payloads["histogram"]
    group = FlowGroup(
        tuple(ValuePacket(h, identity) for h in (histogram, histogram, list(reversed(histogram)))),
        ("first", "tie", "other"),
    )
    builder = make("ml.bow_gallery")
    gallery = builder.execute({"histograms": group}).outputs["model"]
    rank = make("ml.bow_rank", metric=metric, max_distance=0, top_k=3)
    query = ValuePacket(histogram, identity)
    found = rank.execute({"query": query, "gallery": gallery})
    assert found is not None, rank.error
    assert [row["label"] for row in found.outputs["results"]] == ["first", "tie"]
    assert found.outputs["distance"] == 0 and found.passed and found.score is None
    rank.parameters["max_distance"] = -1
    found = rank.execute({"query": query, "gallery": gallery})
    query_array, reference = np.array(histogram), np.array(histogram[::-1])
    if metric == "cosine":
        expected = 1 - (query_array @ reference) / np.linalg.norm(query_array) / np.linalg.norm(
            reference
        )
    elif metric == "l2":
        expected = np.linalg.norm(query_array - reference)
    elif metric == "chi_square":
        expected = 2 * np.sum((query_array - reference) ** 2 / (query_array + reference))
    else:
        expected = np.sqrt(max(0, 1 - np.sqrt(query_array * reference).sum()))
    assert found.outputs["results"][-1]["distance"] == pytest.approx(expected, abs=1e-6)
    empty = rank.execute({"query": ValuePacket([0] * 4, identity), "gallery": gallery})
    assert empty is not None and not empty.passed and empty.outputs["count"] == 0
    assert empty.outputs["distance"] is NO_VALUE and not empty.outputs["results"]


@pytest.mark.parametrize("field", ["ticket", "spatial", "pyramid", "recipe_ref"])
def test_review_context_conflicts_are_rejected_before_native_match(make, monkeypatch, field):
    data, model = record(), trained(make)
    context = {
        "ticket": TicketClock("one").issue("camera"),
        "spatial": frame([64, 64], "one"),
        "pyramid": {"level": 0, "largest_size": [64, 64], "to_largest": np.eye(3).tolist()},
        "recipe_ref": {"id": "one", "revision": 1},
    }
    other = copy.deepcopy(context)
    if field == "ticket":
        other[field] = TicketClock("two").issue("camera")
    elif field == "spatial":
        other[field] = frame([64, 64], "two")
    elif field == "pyramid":
        other[field]["level"] = 1
    else:
        other[field]["id"] = "two"
    monkeypatch.setattr(
        cv2, "BFMatcher", lambda *a, **k: pytest.fail("Conflicting image reached matcher")
    )
    block = make("ml.bow_encode")
    assert (
        block.execute(
            {
                "model": model,
                "features": ValuePacket(data, context),
                "image": ValuePacket(np.zeros((64, 64), np.uint8), other),
            }
        )
        is None
    )
    assert not block.current_emissions


@pytest.mark.parametrize("method", ["sift", "orb"])
def test_image_conversion_selection_and_lean_preserve_image_and_provenance(make, method):
    gray = np.random.default_rng(21).integers(0, 256, (128, 128), np.uint8)
    rgb = ImageArray(
        np.repeat((gray.astype(np.uint16) * 257)[..., None], 3, axis=2),
        "rgb",
        metadata={"custom": "kept"},
    )
    mask = np.zeros(gray.shape, np.uint8)
    mask[:, :90] = 255
    extractor = make("ml.visual_features", method=method, max_features=80)
    result = extractor.execute({"image": rgb, "mask": mask})
    assert result is not None and result.outputs["count"] > 4, extractor.error
    assert all(round(p[0]) < 90 for p in result.outputs["features"]["points"])
    np.testing.assert_array_equal(result.outputs["image"], rgb)
    assert result.outputs["image"].color_format == "rgb" and result.drawings
    model = trained(make, result.outputs["features"])
    encode = make("ml.bow_encode")
    debug = encode.execute(
        {
            "model": model,
            "features": ValuePacket(result.outputs["features"], result.payloads["features"]),
            "image": result.outputs["image"],
        }
    )
    assert debug is not None, encode.error
    encode.run_profile = PROFILES["lean"]
    lean = encode.execute(
        {
            "model": model,
            "features": ValuePacket(result.outputs["features"], result.payloads["features"]),
            "image": result.outputs["image"],
        }
    )
    assert lean.outputs["histogram"] == debug.outputs["histogram"]
    assert not lean.graphics and not lean.drawings and debug.graphics and debug.drawings
    assert lean.outputs["image"].metadata["custom"] == "kept"


def test_blank_empty_and_all_rejected_features_have_explicit_valid_failures(make):
    extractor = make("ml.visual_features")
    empty = extractor.execute({"image": np.zeros((64, 64), np.uint8)})
    assert empty is not None and empty.outputs["count"] == 0
    trainer = make("ml.bow_train", words=4)
    assert trainer.execute({"features": empty.outputs["features"]}) is None
    encode = make("ml.bow_encode")
    model = trained(make)
    result = encode.execute({"model": model, "features": empty.outputs["features"]})
    assert result is not None and not result.passed and result.outputs["histogram"] == [0] * 4
    encode.parameters["max_word_distance"] = 0
    result = encode.execute({"model": model, "features": record(seed=44)})
    assert result is not None and not result.passed and set(result.outputs["word_ids"]) == {-1}


@pytest.mark.parametrize(
    "kind,setting",
    [
        ("ml.bow_train", {"max_work": 1}),
        ("ml.bow_train", {"max_bytes_mb": 0.01}),
        ("ml.bow_encode", {"max_work": 1}),
        ("ml.bow_encode", {"max_bytes_mb": 0.01}),
    ],
)
def test_budgets_prevent_native_work_and_recover(make, monkeypatch, kind, setting):
    data, model = record(), trained(make)
    original = cv2.BFMatcher
    monkeypatch.setattr(cv2, "BFMatcher", lambda *a, **k: pytest.fail("Budget reached matcher"))
    block = make(kind, **setting)
    inputs = {"features": data, **({"model": model} if kind.endswith("encode") else {})}
    assert block.execute(inputs) is None and not block.current_emissions
    monkeypatch.setattr(cv2, "BFMatcher", original)
    block.parameters.update(max_work=128_000_000, max_bytes_mb=128)
    assert block.execute(inputs) is not None, block.error


def test_changed_models_settings_packet_contracts_and_normalization_are_rejected(make):
    data, model = record(), trained(make)
    encode = make("ml.bow_encode")
    altered = copy.deepcopy(model)
    altered["centers"][0][0] += 1
    assert encode.execute({"model": altered, "features": data}) is None
    different = record()
    different["extractor"]["settings"]["sift_sigma"] = 2
    assert encode.execute({"model": model, "features": different}) is None
    assert (
        encode.execute(
            {"model": model, "features": ValuePacket(data, {"feature_contract": {"wrong": True}})}
        )
        is None
    )
    encoded = encode.execute({"model": model, "features": data})
    identity = encoded.payloads["histogram"]
    gallery = (
        make("ml.bow_gallery")
        .execute({"histograms": ValuePacket(encoded.outputs["histogram"], identity)})
        .outputs["model"]
    )
    query = make("ml.bow_encode", normalization="l2").execute({"model": model, "features": data})
    rank = make("ml.bow_rank")
    assert (
        rank.execute(
            {
                "query": ValuePacket(query.outputs["histogram"], query.payloads["histogram"]),
                "gallery": gallery,
            }
        )
        is None
    )
    broken = copy.deepcopy(gallery)
    broken["entries"][0]["histogram"] = [0] * 4
    broken["revision"] = model_revision(broken)
    assert (
        rank.execute(
            {"query": ValuePacket(encoded.outputs["histogram"], identity), "gallery": broken}
        )
        is None
    )


def test_strict_json_persistence_preserves_existing_destinations_and_exact_models(make, tmp_path):
    model = trained(make)
    path = tmp_path / "words.json"
    save, load = make("ml.bow_save", path=str(path)), make("ml.bow_load", path=str(path))
    assert save.execute({"model": model}) is not None
    raw = path.read_bytes()
    assert save.execute({"model": model}) is None and path.read_bytes() == raw
    assert load.execute({}).outputs["model"] == model
    for invalid in (
        '{"model":1,"model":2}',
        '{"model": NaN}',
        json.dumps({**model, "revision": "wrong"}),
    ):
        path.write_text(invalid, encoding="utf-8", newline="\n")
        assert load.execute({}) is None and not load.current_emissions
    path.write_bytes(raw)
    assert load.execute({}).outputs["model"] == model


@pytest.mark.parametrize(
    "kind,settings",
    [
        ("ml.visual_features", {"orb_wta_k": "5"}),
        ("ml.visual_features", {"orb_first_level": 8}),
        ("ml.visual_features", {"orb_patch_size": 10}),
        ("ml.bow_train", {"words": 16, "max_samples": 8}),
        ("ml.bow_encode", {"max_word_distance": -0.5}),
        ("ml.bow_gallery", {"labels": [None]}),
        ("ml.bow_rank", {"minimum_results": 2, "top_k": 1}),
    ],
)
def test_configuration_rules_are_exposed_without_running_native_work(make, kind, settings):
    block = make(kind, **settings)
    with pytest.raises(ValueError):
        block.validate_parameters()


def test_native_exception_is_contained_without_stale_emissions_and_next_input_recovers(
    make, monkeypatch
):
    data, model = record(), trained(make)
    block = make("ml.bow_encode")
    inputs = {"features": data, "model": model}
    assert block.execute(inputs) is not None
    original = cv2.BFMatcher

    def broken(*args, **kwargs):
        raise cv2.error("controlled native error")

    monkeypatch.setattr(cv2, "BFMatcher", broken)
    assert block.execute(inputs) is None and not block.current_emissions and block.state == "error"
    monkeypatch.setattr(cv2, "BFMatcher", original)
    assert block.execute(inputs) is not None and block.current_emissions


@pytest.mark.parametrize("wta", ["2", "3", "4"])
def test_native_orb_extraction_and_training_use_the_declared_binary_metric(make, wta):
    image = np.random.default_rng(90).integers(0, 256, (128, 128), np.uint8)
    extractor = make("ml.visual_features", method="orb", orb_wta_k=wta, max_features=80)
    extracted = extractor.execute({"image": image})
    assert extracted is not None and extracted.outputs["count"] > 4, extractor.error
    model = trained(make, extracted.outputs["features"])
    assert model["metric"] == ("hamming" if wta == "2" else "hamming2")
    encoded = make("ml.bow_encode").execute(
        {"model": model, "features": extracted.outputs["features"]}
    )
    assert encoded is not None and encoded.passed


def test_transparent_pixels_and_mask_intersect_selected_feature_centers(make):
    gray = np.random.default_rng(52).integers(0, 256, (128, 128), np.uint8)
    image = ImageArray(np.dstack((gray, gray, gray, np.full_like(gray, 255))), "bgra")
    image[:64, :, 3] = 0
    mask = np.zeros_like(gray)
    mask[:, :80] = 255
    block = make("ml.visual_features", sift_contrast_threshold=0.01)
    result = block.execute({"image": image, "mask": mask})
    assert result is not None and result.outputs["count"] > 0, block.error
    assert all(round(x) < 80 and round(y) >= 64 for x, y in result.outputs["features"]["points"])


def test_large_orb_upscale_is_rejected_before_native_detector_creation(make, monkeypatch):
    block = make(
        "ml.visual_features", method="orb", orb_scale_factor=2, orb_levels=16, orb_first_level=15
    )
    monkeypatch.setattr(
        cv2, "ORB_create", lambda *a, **k: pytest.fail("Oversized pyramid reached detector")
    )
    assert block.execute({"image": np.zeros((128, 128), np.uint8)}) is None
    assert "16 million" in block.error and not block.current_emissions


def test_training_groups_refuse_mixed_extractors_or_forged_packet_identity(make):
    first, second = record(), record("orb")
    trainer = make("ml.bow_train", words=4)
    group = FlowGroup((ValuePacket(first, {}), ValuePacket(second, {})), ("one", "two"))
    assert trainer.execute({"features": group}) is None and "settings differ" in trainer.error
    group = FlowGroup((ValuePacket(first, {"feature_contract": {"wrong": True}}),), ("one",))
    assert trainer.execute({"features": group}) is None and "packet identity" in trainer.error
