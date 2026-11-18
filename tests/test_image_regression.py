"""Continuous values, receipt association, residual metrics and valid failure behavior."""

import copy
import hashlib

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


@pytest.fixture
def make():
    blocks = []

    def create(kind="ml.decode_regression", **p):
        b = builtin_registry().create(kind, parameters=p)
        blocks.append(b)
        return b

    yield create
    for block in blocks:
        block.close()


def inputs(values=None, *, detailed=True, row=None):
    tensor = np.array([[1.25, 0.5]] if values is None else values, np.float32)
    geometry = {
        "input_size": [4, 3],
        "network_size": [4, 3],
        "input_to_network": np.eye(3).tolist(),
        "network_to_input": np.eye(3).tolist(),
        "preprocessing_contract": {"color_format": "gray", "scale": 1 / 255},
    }
    descriptor = {"name": "head", "shape": list(tensor.shape), "dtype": "float32"}
    if detailed:
        descriptor["sha256"] = hashlib.sha256(tensor.tobytes()).hexdigest()
    call = {
        "artifact": {"sha256": "a" * 64, "bytes": 123},
        "contract_revision": "b" * 64,
        "configuration_verified": True,
        "inputs": [{"name": "image", "shape": [1, 1, 3, 4], "dtype": "float32"}],
        "outputs": [descriptor],
        "preprocessing": {"image": geometry["preprocessing_contract"]},
    }
    receipt = {"model": "onnx_inference_receipt_v1", "schema_version": 1, "calls": [call]}
    if row is not None:
        descriptor["shape"][0] = 3
        call["inputs"][0]["shape"][0] = 3
        receipt = {
            "model": "onnx_image_execution_receipt_v1",
            "schema_version": 1,
            "call": call,
            "native_row_index": row,
        }
    return {
        "image": np.full((3, 4), 100, np.uint8),
        "tensors": {
            "head": {
                "model": "tensor_v1",
                "shape": list(tensor.shape),
                "dtype": "float32",
                "values": tensor.ravel().tolist(),
            }
        },
        "preprocessing": geometry,
        "receipt": receipt,
    }


def measurement(make, value, reference, **p):
    return (
        make(reference_enabled=True, reference_value=reference, **p)
        .execute(inputs([[value]]))
        .outputs["measurement"]
    )


def test_scaled_values_reference_errors_are_measurements_not_confidence(make):
    block = make(
        output_name="head",
        output_index=0,
        value_scale=2,
        value_offset=-0.25,
        reference_enabled=True,
        reference_value=2,
        check_range=True,
        minimum_value=2,
        maximum_value=3,
        check_absolute_error=True,
        maximum_absolute_error=0.25,
        units="mm",
    )
    result = block.execute(inputs())
    assert result is not None and result.passed, block.error
    assert (
        result.outputs["value"] == 2.25
        and result.outputs["residual"] == 0.25
        and result.score is None
    )
    assert result.outputs["measurement"]["model_revision"] == "b" * 64
    assert result.outputs["report"]["identity"]["tensor_hash_checked"]
    assert result.outputs["report"]["identity"]["source_pixels_hash_checked"] is False
    assert result.drawings
    block.parameters["maximum_absolute_error"] = 0.249
    failed = block.execute(inputs())
    assert failed is not None and not failed.passed
    assert failed.fail_reasons[0]["code"] == "regression.residual" and block.state == "fail"
    block.parameters.update(check_absolute_error=False, maximum_value=2.2)
    assert block.execute(inputs()).fail_reasons[0]["code"] == "regression.range"


def test_absent_reference_has_no_reference_or_residual_event_and_optional_receipt_is_explicit(make):
    block = make()
    result = block.execute(inputs())
    assert result is not None, block.error
    assert all(
        result.outputs[k] is NO_VALUE and not block.current_emissions[k]
        for k in ("reference", "residual", "absolute_error")
    )
    data = inputs()
    data.pop("receipt")
    assert block.execute(data) is None
    optional = make(receipt_policy="optional").execute(data)
    assert optional is not None and optional.outputs["measurement"]["model_revision"] is None
    assert make(check_absolute_error=True).execute(inputs()) is None


@pytest.mark.parametrize(
    "values,layout,expected", [([2.5], "vector", 2.5), ([[[[3.5]], [[0.2]]]], "nchw_scalar", 3.5)]
)
def test_explicit_head_shapes_accept_only_single_image_values(make, values, layout, expected):
    result = make(layout=layout).execute(inputs(values))
    assert result is not None and result.outputs["value"] == expected


@pytest.mark.parametrize("values", [[[1], [2]], [[[1, 2], [3, 4]]], [[[[1, 2]]]]])
def test_batch_and_spatial_heads_are_never_implicitly_flattened(make, values):
    block = make()
    assert block.execute(inputs(values)) is None and "single-image" in block.error


@pytest.mark.parametrize(
    "change", ["hash", "artifact", "revision", "shape", "preprocessing", "missing_output"]
)
def test_receipt_mismatches_do_not_publish_stale_measurements(make, change):
    block = make()
    assert block.execute(inputs()) is not None
    bad = inputs()
    call = bad["receipt"]["calls"][0]
    if change == "hash":
        bad["tensors"]["head"]["values"][0] += 1
    elif change == "artifact":
        call["artifact"]["sha256"] = "bad"
    elif change == "revision":
        call["contract_revision"] = "bad"
    elif change == "shape":
        call["outputs"][0]["shape"] = [1, 3]
    elif change == "preprocessing":
        call["preprocessing"] = {}
    else:
        call["outputs"][0]["name"] = "another"
    assert block.execute(bad) is None and not block.current_emissions
    assert block.execute(inputs()) is not None


def test_batch_row_receipts_report_no_full_batch_hash_proof(make):
    result = make().execute(inputs(row=1))
    assert result is not None
    identity = result.outputs["report"]["identity"]
    assert identity["native_row_index"] == 1 and not identity["tensor_hash_checked"]
    assert make().execute(inputs(row=3)) is None


def test_known_ticket_frame_and_reference_identity_are_preserved_and_checked(make):
    context = {
        "ticket": TicketClock("camera").issue("camera"),
        "spatial": frame([4, 3], "camera"),
        "custom": {"kept": True},
    }
    other = TicketClock("other").issue("other")
    data = {name: ValuePacket(value, context) for name, value in inputs().items()}
    result = make().execute(data)
    assert result is not None
    assert result.payloads["value"]["ticket"] == context["ticket"]
    assert result.payloads["measurement"]["custom"] == {"kept": True}
    bad = {**data, "tensors": ValuePacket(inputs()["tensors"], {**context, "ticket": other})}
    assert make().execute(bad) is None
    bad = {
        **data,
        "receipt": ValuePacket(inputs()["receipt"], {**context, "spatial": frame([4, 3], "other")}),
    }
    assert make().execute(bad) is None
    reference = {**data, "reference": ValuePacket(1.0, {"ticket": other})}
    assert make().execute(reference) is None
    assert make(reference_ticket_policy="external").execute(reference) is not None


def test_resolved_output_name_participates_in_decoder_identity(make):
    first = make().execute(inputs()).outputs["measurement"]
    data = inputs()
    data["tensors"]["other"] = data["tensors"].pop("head")
    data["receipt"]["calls"][0]["outputs"][0]["name"] = "other"
    second = make().execute(data).outputs["measurement"]
    assert first["decoder_revision"] != second["decoder_revision"]


@pytest.mark.parametrize(
    "p",
    [
        {"output_index": 1.0},
        {"value_scale": "2"},
        {"units": None},
        {"check_range": 1},
        {"minimum_value": 2, "maximum_value": 1},
    ],
)
def test_configuration_limits_are_actionable(make, p):
    with pytest.raises(ValueError):
        make(**p).validate_parameters()


def test_native_independent_metric_formulas_constant_targets_limits_and_unicode_plot(make):
    rows = [
        measurement(make, value, reference, units="Âµm")
        for value, reference in [(1.1, 1), (1.8, 2), (3.3, 3)]
    ]
    block = make("ml.regression_metrics", limit_metric="rmse", maximum_error=0.2)
    result = block.execute({"measurements": rows})
    assert result is not None, block.error
    errors = np.array([row["value"] - row["reference"] for row in rows])
    assert result.outputs["mae"] == pytest.approx(np.abs(errors).mean())
    assert result.outputs["rmse"] == pytest.approx(np.sqrt((errors**2).mean()))
    assert result.outputs["bias"] == pytest.approx(errors.mean())
    assert result.outputs["report"]["r2"] == pytest.approx(1 - np.sum(errors**2) / 2)
    assert (
        not result.passed and result.score is None and result.outputs["plot"].shape == (260, 640, 3)
    )
    block.set_graphics_mode("off")
    quiet = block.execute({"measurements": rows})
    assert quiet.outputs["mae"] == result.outputs["mae"] and quiet.outputs["plot"] is NO_VALUE
    constant = make("ml.regression_metrics").execute(
        {"measurements": [measurement(make, 1, 1), measurement(make, 1.1, 1)]}
    )
    assert constant.outputs["report"]["r2"] is None


@pytest.mark.parametrize(
    "change", ["duplicate", "units", "model", "decoder", "reference", "residual"]
)
def test_invalid_or_incomparable_metric_records_are_rejected(make, change):
    row = measurement(make, 1, 1)
    other = copy.deepcopy(row)
    other["id"] = "second"
    if change == "duplicate":
        other["id"] = row["id"]
    elif change == "units":
        other["units"] = "pixels"
    elif change == "model":
        other["model_revision"] = "c" * 64
    elif change == "decoder":
        other["decoder_revision"] = "d" * 64
    elif change == "reference":
        other["reference"] = None
    else:
        other["residual"] = 1
    block = make("ml.regression_metrics")
    assert block.execute({"measurements": [row, other]}) is None and not block.current_emissions


def test_metric_group_keeps_each_source_ticket_and_rejects_excess_rows(make):
    row = measurement(make, 1, 1)
    other = measurement(make, 2, 2)
    group = FlowGroup((ValuePacket(row, {}), ValuePacket(other, {})), ("a", "b"))
    result = make("ml.regression_metrics").execute({"measurements": group})
    assert result is not None and result.outputs["report"]["sample_count"] == 2
    assert make("ml.regression_metrics", maximum_samples=1).execute({"measurements": group}) is None


def test_unsealed_model_preprocessing_changes_are_distinct_measurement_pipelines(make):
    first = inputs()
    first["receipt"]["calls"][0]["contract_revision"] = None
    a = make().execute(first).outputs["measurement"]
    second = copy.deepcopy(first)
    second["preprocessing"]["preprocessing_contract"]["scale"] = 1 / 127.5
    b = make().execute(second).outputs["measurement"]
    assert (
        a["model_revision"] == b["model_revision"]
        and a["decoder_revision"] != b["decoder_revision"]
    )
    assert make("ml.regression_metrics").execute({"measurements": [a, b]}) is None
