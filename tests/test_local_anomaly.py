"""Local feature geometry, reference identity, independent distances and guarded I/O."""

import copy
import hashlib
import json

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.classifier_common import model_revision
from opencv_dataflow.local_anomaly import hash_image
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


@pytest.fixture
def make():
    blocks = []

    def create(kind="ml.local_features", **parameters):
        block = builtin_registry().create(kind, parameters=parameters)
        blocks.append(block)
        return block

    yield create
    for block in blocks:
        block.close()


def inputs(values=None, *, layout="NCHW", source=None):
    values = np.float32([[[[1, 1], [1, 1]], [[2, 2], [2, 2]]]] if values is None else values)
    if layout == "NHWC":
        values = values.transpose(0, 2, 3, 1)
    source = np.zeros((4, 4), np.uint8) if source is None else source
    prep = {
        "input_size": [4, 4],
        "network_size": [4, 4],
        "input_to_network": np.eye(3).tolist(),
        "network_to_input": np.eye(3).tolist(),
        "preprocessing_contract": {"scale": 1 / 255, "color_format": "gray"},
    }
    descriptor = {
        "name": "local",
        "shape": list(values.shape),
        "dtype": "float32",
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }
    call = {
        "artifact": {"sha256": "a" * 64, "bytes": 123},
        "contract_revision": "b" * 64,
        "configuration_verified": True,
        "inputs": [{"name": "image", "shape": [1, 1, 4, 4], "dtype": "float32"}],
        "outputs": [descriptor],
        "preprocessing": {"image": copy.deepcopy(prep["preprocessing_contract"])},
    }
    return {
        "image": source,
        "tensors": {
            "local": {
                "model": "tensor_v1",
                "shape": list(values.shape),
                "dtype": "float32",
                "values": values.ravel().tolist(),
            }
        },
        "preprocessing": prep,
        "receipt": {"model": "onnx_inference_receipt_v1", "schema_version": 1, "calls": [call]},
    }


def grid(make, **kwargs):
    return make().execute(inputs(**kwargs)).outputs["features"]


def trained(make, **parameters):
    reference = grid(make)
    result = make("ml.train_local_anomaly", **parameters).execute({"features": reference})
    assert result is not None
    return result.outputs["model"]


@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_feature_layout_receipt_source_identity_and_normalization(make, layout):
    data = inputs(layout=layout)
    block = make(layout=layout, normalization="l2")
    result = block.execute(data)
    assert result is not None and result.passed
    features = result.outputs["features"]
    np.testing.assert_allclose(features["features"], [[1 / np.sqrt(5), 2 / np.sqrt(5)]] * 4)
    assert features["source_sha256"] == hash_image(data["image"])
    assert result.outputs["report"]["identity"]["tensor_hash_checked"]
    assert not result.outputs["report"]["identity"]["source_pixels_hash_checked"]
    np.testing.assert_array_equal(result.outputs["image"], data["image"])


@pytest.mark.parametrize(
    "mutation", ["batch", "pooled", "dtype", "hash", "preprocessing", "empty_roi", "zero_l2"]
)
def test_invalid_feature_heads_receipts_and_selections_emit_nothing(make, mutation):
    data = inputs()
    parameters = {}
    if mutation == "batch":
        data = inputs(np.ones((2, 2, 2, 2)))
    elif mutation == "pooled":
        data = inputs(np.ones((1, 2, 1, 1)))
    elif mutation == "dtype":
        data["tensors"]["local"]["dtype"] = "int8"
    elif mutation == "hash":
        data["tensors"]["local"]["values"][0] += 1
    elif mutation == "preprocessing":
        data["preprocessing"]["preprocessing_contract"]["scale"] = 0.5
    elif mutation == "empty_roi":
        data["mask"] = np.zeros((4, 4), np.uint8)
    else:
        data = inputs(np.zeros((1, 2, 2, 2)))
        parameters["normalization"] = "l2"
    block = make(**parameters)
    assert block.execute(data) is None and block.state == "error"
    assert not block.current_emissions


def test_letterbox_padding_cells_and_mask_center_selection(make):
    data = inputs()
    data["image"] = np.zeros((4, 2), np.uint8)
    data["preprocessing"].update(
        input_size=[2, 4],
        network_size=[8, 4],
        input_to_network=[[1, 0, 3], [0, 1, 0], [0, 0, 1]],
        network_to_input=[[1, 0, -3], [0, 1, 0], [0, 0, 1]],
    )
    raw = np.ones((1, 1, 2, 4), np.float32)
    data["tensors"]["local"].update(shape=list(raw.shape), values=raw.ravel().tolist())
    descriptor = data["receipt"]["calls"][0]["outputs"][0]
    descriptor.update(shape=list(raw.shape), sha256=hashlib.sha256(raw.tobytes()).hexdigest())
    # Head cell centers at x=1,3,5,7; only x=3 maps into this two-pixel source.
    result = make().execute(data)
    assert result is not None
    assert result.outputs["features"]["valid"] == [False, True, False, False] * 2


def test_known_ticket_and_frame_mismatches_are_contained(make):
    data = inputs()
    clock = TicketClock()
    a, b = clock.issue("a"), clock.issue("b")
    data["image"] = ValuePacket(data["image"], {"ticket": a, "spatial": frame((4, 4), root_id="a")})
    data["receipt"] = ValuePacket(data["receipt"], {"ticket": b})
    block = make()
    assert block.execute(data) is None and "ticket" in block.error.lower()
    data = inputs()
    data["image"] = ValuePacket(data["image"], {"spatial": frame((4, 4), root_id="a")})
    data["tensors"] = ValuePacket(data["tensors"], {"spatial": frame((4, 4), root_id="b")})
    assert block.execute(data) is None and "frame" in block.error.lower()


def test_bank_training_sampling_limits_duplicates_and_contracts(make):
    a = grid(make)
    b = grid(make, source=np.ones((4, 4), np.uint8))
    train = make("ml.train_local_anomaly", bank_limit=3)
    assert train.execute({"features": [a, b]}) is None
    train.parameters["sampling"] = "even_stride"
    result = train.execute(
        {"features": FlowGroup((ValuePacket(a, {}), ValuePacket(b, {})), ("a", "b"))}
    )
    assert result is not None and len(result.outputs["model"]["samples"]) == 3
    assert result.outputs["report"]["candidate_rows"] == 8
    assert not result.outputs["report"]["known_good_labels_verified"]
    assert train.execute({"features": [a, a]}) is None and "Duplicate" in train.error
    b["feature_contract"]["preprocessing"]["scale"] = 1
    assert train.execute({"features": [a, b]}) is None and "contract" in train.error


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_native_distance_nearest_source_mapping_threshold_and_graphics(make, metric):
    model = trained(make, metric=metric)
    values = np.array([[[[4, 1], [1, 1]], [[6, 2], [2, 2]]]], np.float32)
    query = grid(make, values=values, source=np.ones((4, 4), np.uint8))
    threshold = 5 if metric == "euclidean" else 7
    block = make("ml.local_anomaly", distance_threshold=threshold)
    data = {"image": np.ones((4, 4), np.uint8), "features": query, "model": model}
    result = block.execute(data)
    assert result is not None and result.passed and result.score is None
    assert result.outputs["maximum_distance"] == threshold
    expected = np.zeros((4, 4), np.float32)
    expected[:2, :2] = threshold
    np.testing.assert_array_equal(result.outputs["distance"], expected)
    block.parameters["distance_threshold"] = threshold - 0.1
    result = block.execute(data)
    assert result is not None and not result.passed and result.outputs["anomaly_pixels"] == 4
    assert result.fail_reasons[0]["code"] == "anomaly.area" and result.drawings and result.graphics
    assert result.outputs["report"]["native_distances"] == [threshold, 0, 0, 0]
    block.parameters["allowed_anomaly_pixels"] = 4
    block.set_graphics_mode("off")
    lean = block.execute(data)
    assert lean is not None and lean.passed and not lean.drawings and not lean.graphics
    np.testing.assert_array_equal(lean.outputs["distance"], expected)


@pytest.mark.parametrize(
    "mutation", ["source", "contract", "model_revision", "work", "reused", "support"]
)
def test_query_rejection_and_recovery(make, mutation):
    model = trained(make)
    query = grid(make)
    data = {"image": np.zeros((4, 4), np.uint8), "features": query, "model": model}
    parameters = {}
    if mutation == "source":
        data["image"][0, 0] = 1
    elif mutation == "contract":
        query["feature_contract"]["model_revision"] = "c" * 64
    elif mutation == "model_revision":
        model["samples"][0][0] += 1
    elif mutation == "work":
        parameters["maximum_distance_work"] = 1
    elif mutation == "reused":
        parameters["reject_training_image"] = True
    else:
        query["valid"] = [False, True, True, True]
    block = make("ml.local_anomaly", **parameters)
    assert block.execute(data) is None and not block.current_emissions
    block.parameters.update(maximum_distance_work=64_000_000, reject_training_image=False)
    valid = {"image": np.zeros((4, 4), np.uint8), "features": grid(make), "model": trained(make)}
    assert block.execute(valid).passed


def test_inspection_mask_excludes_known_defect_and_empty_selection_is_invalid(make):
    model = trained(make)
    query = grid(make, values=np.array([[[[8, 1], [1, 1]], [[8, 2], [2, 2]]]]))
    mask = np.full((4, 4), 255, np.uint8)
    mask[:2, :2] = 0
    data = {"image": np.zeros((4, 4), np.uint8), "features": query, "model": model, "mask": mask}
    block = make("ml.local_anomaly")
    result = block.execute(data)
    assert result.passed and result.outputs["report"]["scored_pixels"] == 12
    assert result.outputs["maximum_distance"] == 0
    data["mask"][:] = 0
    assert block.execute(data) is None


def test_model_io_exact_reload_corrupt_json_and_existing_file_preservation(make, tmp_path):
    model = trained(make)
    path = tmp_path / "reference.json"
    save = make("ml.save_local_anomaly", path=str(path))
    assert save.execute({"model": model})
    before = path.read_bytes()
    assert save.execute({"model": model}) is None and path.read_bytes() == before
    load = make("ml.load_local_anomaly", path=str(path))
    assert load.execute({}).outputs["model"] == model
    path.write_bytes(b'{"model":"x","model":"y"}')
    assert load.execute({}) is None and "Duplicate" in load.error
    path.write_bytes(b'{"model":NaN}')
    assert load.execute({}) is None and "Nonfinite" in load.error
    model["samples"][0][0] += 0.5
    path.write_bytes(json.dumps(model).encode())
    assert load.execute({}) is None and "revision" in load.error


def test_model_validation_checks_all_l2_rows_and_large_banks(make):
    reference = grid(make)
    model = trained(make)
    model["samples"] = [[1.0, 2.0]] * 5000
    model["candidate_rows"] = 5000
    model["revision"] = model_revision(model)
    block = make("ml.local_anomaly")
    assert block.execute(
        {"image": np.zeros((4, 4), np.uint8), "features": reference, "model": model}
    )
    model["feature_contract"]["normalization"] = "l2"
    model["revision"] = model_revision(model)
    assert (
        block.execute({"image": np.zeros((4, 4), np.uint8), "features": reference, "model": model})
        is None
    )


@pytest.mark.parametrize(
    "kind,parameters",
    [
        ("ml.local_features", {"layout": "unknown"}),
        ("ml.local_features", {"maximum_feature_elements": True}),
        ("ml.train_local_anomaly", {"bank_limit": 1.5}),
        ("ml.local_anomaly", {"distance_threshold": -1}),
        ("ml.local_anomaly", {"reject_training_image": "yes"}),
    ],
)
def test_parameter_validation_fields(make, kind, parameters):
    with pytest.raises(ValueError):
        make(kind, **parameters).validate_parameters()
