"""Image-linked continuous ONNX measurements and bounded residual evaluation."""

import copy
import hashlib
import math
import re
import uuid

import cv2
import numpy as np

from .classifier_common import canonical_json
from .core import Block, Port, Result
from .detection_decode import check_context, preprocessing_geometry
from .graphics_layers import GraphicDrawing, graphic_object
from .ml_common import selected_tensor
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket, require_same_frame
from .processing import image_value
from .tickets import ticket_key
from .unicode_text import TextStyle, draw_unicode

MEASUREMENT = "image_regression_measurement_v1"


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e12:
        raise ValidationError(name + " must be finite numeric data within +/-1e12", name)
    return float(value)


def text(value, name, *, empty=False):
    if not isinstance(value, str) or len(value) > 128 or (not empty and not value.strip()):
        raise ValidationError("Use bounded nonempty text", name)
    return value


def numeric_settings(block, p):
    for key, rule in block.parameter_ranges.items():
        if type(p[key]) not in (int, float) or (rule.whole and type(p[key]) is not int):
            raise ValidationError("Use an integer" if rule.whole else "Use a numeric value", key)


def receipt_identity(receipt, name, tensor, preprocessing):
    """Validate reported artifact/configuration and output evidence, not source pixels."""
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise ValidationError("Connect a versioned ONNX execution receipt", "receipt")
    row = None
    if receipt.get("model") == "onnx_inference_receipt_v1":
        calls = receipt.get("calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValidationError(
                "Single-image regression needs exactly one inference call", "receipt"
            )
        call = calls[0]
    elif receipt.get("model") == "onnx_image_execution_receipt_v1":
        call = receipt.get("call")
        row = receipt.get("native_row_index")
        if type(row) is not int or row < 0:
            raise ValidationError("Invalid batch-row receipt", "receipt")
    else:
        raise ValidationError("Choose a single-image or unpacked image-row receipt", "receipt")
    if not isinstance(call, dict):
        raise ValidationError("Missing ONNX call identity", "receipt")
    artifact = call.get("artifact")
    if (
        not isinstance(artifact, dict)
        or not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256", "")))
        or type(artifact.get("bytes")) is not int
        or not 1 <= artifact["bytes"] <= 256 * 1024**2
    ):
        raise ValidationError("Invalid ONNX artifact identity", "receipt")
    revision = call.get("contract_revision")
    if revision is not None and (
        not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{64}", revision)
    ):
        raise ValidationError("Invalid model contract revision", "receipt")
    descriptors = call.get("outputs")
    if not isinstance(descriptors, list) or len(descriptors) > 64:
        raise ValidationError("Invalid receipt output descriptions", "receipt")
    matches = [
        entry for entry in descriptors if isinstance(entry, dict) and entry.get("name") == name
    ]
    if len(matches) != 1:
        raise ValidationError(
            "Selected output is absent or repeated in the receipt", "output_name", "receipt"
        )
    output = matches[0]
    shape = output.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(v) is not int or v < 1 for v in shape)
        or output.get("dtype") != "float32"
    ):
        raise ValidationError("Invalid native output shape/type", "receipt")
    if row is None:
        if shape != list(tensor.shape):
            raise ValidationError("Tensor shape differs from receipt", "receipt")
    elif row >= shape[0] or [1, *shape[1:]] != list(tensor.shape):
        raise ValidationError("Unpacked tensor shape/row differs from receipt", "receipt")
    inputs = call.get("inputs")
    contract = preprocessing.get("preprocessing_contract")
    if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], dict):
        raise ValidationError("Regression receipt must describe one image input", "receipt")
    if (
        not isinstance(contract, dict)
        or call.get("preprocessing", {}).get(inputs[0].get("name")) != contract
    ):
        raise ValidationError(
            "Preprocessing contract differs from the inference receipt", "preprocessing", "receipt"
        )
    checked = False
    if row is None and "sha256" in output:
        expected = hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest()
        if output["sha256"] != expected:
            raise ValidationError(
                "Tensor content differs from detailed receipt", "tensors", "receipt"
            )
        checked = True
    return {
        "model_revision": revision or "sha256:" + artifact["sha256"],
        "artifact": copy.deepcopy(artifact),
        "contract_revision": revision,
        "configuration_verified": call.get("configuration_verified") is True,
        "native_row_index": row,
        "tensor_hash_checked": checked,
        "source_pixels_hash_checked": False,
    }


class DecodeRegression(Block):
    type_id, title = "ml.decode_regression", "Decode image regression"
    category = block_type = "ML"
    subtype = "Regression"
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"
    inputs = (
        Port("image"),
        Port("tensors", "data"),
        Port("preprocessing", "data"),
        Port("receipt", "data", required=False, required_when_connected=True),
        Port("reference", "number", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("value", "number"),
        Port("reference", "number", required=False),
        Port("residual", "number", required=False),
        Port("absolute_error", "number", required=False),
        Port("measurement", "data"),
        Port("report", "data"),
    )
    defaults = {
        "output_name": "",
        "output_index": 0,
        "layout": "vector",
        "measurement_name": "measurement",
        "units": "",
        "value_scale": 1.0,
        "value_offset": 0.0,
        "receipt_policy": "required",
        "reference_enabled": False,
        "reference_value": 0.0,
        "reference_ticket_policy": "same_ticket",
        "check_range": False,
        "minimum_value": 0.0,
        "maximum_value": 1.0,
        "check_absolute_error": False,
        "maximum_absolute_error": 1.0,
        "maximum_values": 10000,
        "maximum_image_pixels": 16_777_216,
    }
    parameter_choices = {
        "layout": ("vector", "nchw_scalar"),
        "receipt_policy": ("required", "optional"),
        "reference_ticket_policy": ("same_ticket", "external"),
    }
    parameter_ranges = {
        "output_index": NumericRange(0, 9999, whole=True),
        "value_scale": NumericRange(-1e6, 1e6),
        "value_offset": NumericRange(-1e12, 1e12),
        "reference_value": NumericRange(-1e12, 1e12),
        "minimum_value": NumericRange(-1e12, 1e12),
        "maximum_value": NumericRange(-1e12, 1e12),
        "maximum_absolute_error": NumericRange(0, 1e12),
        "maximum_values": NumericRange(1, 10000, whole=True),
        "maximum_image_pixels": NumericRange(1, 64_000_000, whole=True),
    }
    parameter_help = {
        "layout": "Vector accepts [M] or [1,M]; NCHW scalar requires "
        "[1,M,1,1]. Never flatten spatial maps or multiple "
        "images.",
        "output_index": "Zero-based measurement/channel in the selected output.",
        "value_scale": "Decoded value = raw native value * scale + offset; no "
        "confidence interpretation.",
        "receipt_policy": "Required retains checked model identity. Optional "
        "explicitly permits unidentified tensors.",
        "reference_enabled": "Use configured Reference value when no reference connector is wired.",
        "reference_ticket_policy": "Known reference tickets must match the image, unless "
        "External is explicitly chosen.",
        "maximum_absolute_error": "Inclusive residual limit when its check is enabled; "
        "requires a reference.",
        "maximum_values": "Bound selected tensor elements before decoding.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        numeric_settings(self, p)
        for key in ("measurement_name", "units", "output_name"):
            text(p[key], key, empty=key != "measurement_name")
        for key in ("reference_enabled", "check_range", "check_absolute_error"):
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox boolean", key)
        if p["minimum_value"] > p["maximum_value"]:
            raise ValidationError("Minimum exceeds Maximum", "minimum_value", "maximum_value")

    def process(self, inputs):
        p = self.parameters
        source = inputs["image"]
        image = image_value(source)
        if (
            image.shape[0] * image.shape[1] > p["maximum_image_pixels"]
            or max(image.shape[:2]) > 16384
        ):
            raise ValidationError(
                "Source exceeds regression image allowance", "maximum_image_pixels"
            )
        association = check_context(self, kind="Regression")
        preprocessing_geometry(inputs["preprocessing"], source)
        records = inputs["tensors"]
        if not isinstance(records, dict) or not records:
            raise ValidationError("Connect current image tensors", "tensors")
        name = p["output_name"] or next(iter(records))
        if not isinstance(records.get(name), dict) or records[name].get("dtype") != "float32":
            raise ValidationError("Choose a float32 tensor output", "output_name")
        tensor = selected_tensor(records, name, max_elements=p["maximum_values"])
        valid = (
            (tensor.ndim == 1 or (tensor.ndim == 2 and tensor.shape[0] == 1))
            if p["layout"] == "vector"
            else tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[2:] == (1, 1)
        )
        if not valid:
            raise ValidationError("Tensor is not a single-image scalar/vector head", "layout")
        values = tensor.reshape(-1)
        if p["output_index"] >= len(values):
            raise ValidationError("Output index exceeds the measurement head", "output_index")
        identity = {
            "model_revision": None,
            "tensor_hash_checked": False,
            "source_pixels_hash_checked": False,
        }
        if "receipt" in inputs:
            context = self.input_payloads.get("receipt", {})
            require_same_frame(
                (source, ValuePacket(None, context)),
                message="Receipt belongs to another image frame",
            )
            known = [
                c["ticket"]
                for c in (self.input_payloads.get("image", {}), context)
                if "ticket" in c
            ]
            if len(known) == 2 and ticket_key(known[0]) != ticket_key(known[1]):
                raise ValidationError("Receipt belongs to another image ticket", "receipt")
            identity = receipt_identity(inputs["receipt"], name, tensor, inputs["preprocessing"])
        elif p["receipt_policy"] == "required":
            raise ValidationError(
                "Expose and connect the inference Receipt output", "receipt_policy"
            )
        raw = number(float(values[p["output_index"]]), "raw_value")
        value = number(raw * p["value_scale"] + p["value_offset"], "value")
        reference = inputs.get(
            "reference", p["reference_value"] if p["reference_enabled"] else None
        )
        if reference is not None:
            reference = number(reference, "reference")
            if "reference" in inputs and p["reference_ticket_policy"] == "same_ticket":
                tickets = [self.input_payloads[k].get("ticket") for k in ("image", "reference")]
                if all(tickets) and ticket_key(tickets[0]) != ticket_key(tickets[1]):
                    raise ValidationError("Reference belongs to another ticket", "reference")
        if p["check_absolute_error"] and reference is None:
            raise ValidationError("Residual check requires a reference", "reference_enabled")
        residual = None if reference is None else number(value - reference, "residual")
        failures = []
        if p["check_range"] and not p["minimum_value"] <= value <= p["maximum_value"]:
            failures.append(
                {
                    "code": "regression.range",
                    "message": "Measurement is outside the configured value range",
                }
            )
        if p["check_absolute_error"] and abs(residual) > p["maximum_absolute_error"]:
            failures.append(
                {
                    "code": "regression.residual",
                    "message": "Absolute residual exceeds the configured limit",
                }
            )
        settings = {
            k: p[k]
            for k in (
                "output_name",
                "output_index",
                "layout",
                "measurement_name",
                "units",
                "value_scale",
                "value_offset",
            )
        }
        settings["output_name"] = name
        context = copy.deepcopy(self.input_payloads.get("image", {}))
        measurement = {
            "model": MEASUREMENT,
            "id": uuid.uuid4().hex,
            "name": p["measurement_name"],
            "units": p["units"],
            "value": value,
            "raw_value": raw,
            "reference": reference,
            "residual": residual,
            "absolute_error": None if residual is None else abs(residual),
            "model_revision": identity["model_revision"],
            "decoder_revision": digest(
                {
                    "decode": settings,
                    "preprocessing": inputs["preprocessing"].get("preprocessing_contract"),
                }
            ),
            "context": context,
            "passed": not failures,
        }
        outputs = {
            "image": source,
            "value": value,
            "reference": NO_VALUE if reference is None else reference,
            "residual": NO_VALUE if residual is None else residual,
            "absolute_error": NO_VALUE if residual is None else abs(residual),
            "measurement": measurement,
            "report": {
                "identity": identity,
                "association": association,
                "decoder": settings,
                "value_is_confidence": False,
            },
        }
        label = f"{p['measurement_name']}: {value:.6g} {p['units']}"
        if residual is not None:
            label += f"; residual {residual:+.4g}"
        color = (60, 60, 235) if failures else (70, 220, 130)
        return Result(
            outputs,
            passed=not failures,
            fail_reasons=failures,
            payloads={k: copy.deepcopy(context) for k in outputs},
            message=label,
            drawings={
                "regression": GraphicDrawing(
                    (
                        graphic_object(
                            "label",
                            [[8, 18]],
                            color,
                            text=label if len(label) <= 160 else label[:157] + "...",
                        ),
                    )
                )
            }
            if self.generate_graphics
            else {},
        )


class RegressionMetrics(Block):
    type_id, title = "ml.regression_metrics", "Evaluate regression measurements"
    category = block_type = "ML"
    subtype = "Regression"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("measurements", "any"),)
    outputs = (
        Port("report", "data"),
        Port("mae", "number"),
        Port("rmse", "number"),
        Port("bias", "number"),
        Port("maximum_absolute_error", "number"),
        Port("plot", required=False),
    )
    defaults = {
        "limit_metric": "none",
        "maximum_error": 1.0,
        "identity_policy": "same_model_and_decoder",
        "require_model_identity": True,
        "duplicate_policy": "reject",
        "maximum_samples": 10000,
    }
    parameter_choices = {
        "limit_metric": ("none", "mae", "rmse", "maximum_absolute_error", "absolute_bias"),
        "identity_policy": ("same_model_and_decoder", "allow_mixed"),
        "duplicate_policy": ("reject", "allow"),
    }
    parameter_ranges = {
        "maximum_error": NumericRange(0, 1e12),
        "maximum_samples": NumericRange(1, 10000, whole=True),
    }
    parameter_help = {
        "limit_metric": "All metrics are always reported; choose one inclusive inspection limit.",
        "identity_policy": "Names/units must agree. By default model and decoder "
        "revisions must also agree.",
        "duplicate_policy": "Reject reused measurement IDs. This does not certify "
        "independent capture groups.",
        "require_model_identity": "Reject records decoded without a model execution receipt.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        numeric_settings(self, p)
        if type(p["require_model_identity"]) is not bool:
            raise ValidationError("Use a checkbox boolean", "require_model_identity")

    def process(self, inputs):
        p = self.parameters
        data = inputs["measurements"]
        rows = (
            [entry.value for entry in data.items]
            if isinstance(data, FlowGroup)
            else [data]
            if isinstance(data, dict)
            else data
        )
        if not isinstance(rows, list) or not 1 <= len(rows) <= p["maximum_samples"]:
            raise ValidationError(
                "Supply a bounded nonempty measurement collection", "maximum_samples"
            )
        records = []
        ids = set()
        identities = set()
        names = set()
        for row in rows:
            if not isinstance(row, dict) or row.get("model") != MEASUREMENT:
                raise ValidationError("Use image regression measurement records")
            identifier = text(row.get("id"), "id")
            if identifier in ids and p["duplicate_policy"] == "reject":
                raise ValidationError("Duplicate regression measurement ID", "duplicate_policy")
            ids.add(identifier)
            name, units = text(row.get("name"), "name"), text(row.get("units"), "units", empty=True)
            model = row.get("model_revision")
            decoder = text(row.get("decoder_revision"), "decoder_revision")
            if model is not None:
                text(model, "model_revision")
            if p["require_model_identity"] and not model:
                raise ValidationError("Measurement lacks model identity", "require_model_identity")
            value, reference = (
                number(row.get("value"), "value"),
                number(row.get("reference"), "reference"),
            )
            residual = number(value - reference, "residual")
            if row.get("residual") is None or not math.isclose(
                number(row["residual"], "residual"), residual, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValidationError("Stored residual differs from value minus reference")
            records.append(
                {
                    "id": identifier,
                    "value": value,
                    "reference": reference,
                    "residual": residual,
                    "model_revision": model,
                    "decoder_revision": decoder,
                    "source_ticket": copy.deepcopy(row.get("context", {}).get("ticket")),
                }
            )
            identities.add((model, decoder))
            names.add((name, units))
        if len(names) != 1:
            raise ValidationError("Regression measurement names or units differ")
        if p["identity_policy"] == "same_model_and_decoder" and len(identities) != 1:
            raise ValidationError("Regression model/decoder revisions differ", "identity_policy")
        errors = np.array([r["residual"] for r in records])
        targets = np.array([r["reference"] for r in records])
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        bias = float(np.mean(errors))
        maximum = float(np.max(np.abs(errors)))
        total = float(np.sum((targets - targets.mean()) ** 2))
        r2 = None if total == 0 else float(1 - np.sum(errors**2) / total)
        values = {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "absolute_bias": abs(bias),
            "maximum_absolute_error": maximum,
        }
        passed = p["limit_metric"] == "none" or values[p["limit_metric"]] <= p["maximum_error"]
        name, units = next(iter(names))
        report = {
            "model": "image_regression_report_v1",
            "name": name,
            "units": units,
            "sample_count": len(records),
            **values,
            "r2": r2,
            "records": records,
            "capture_independence_verified": False,
            "identity_count": len(identities),
        }
        plot = NO_VALUE
        if self.generate_graphics:
            plot = np.full((260, 640, 3), 245, np.uint8)
            cv2.line(plot, (30, 130), (610, 130), (90, 90, 90), 1)
            scale = max(maximum, p["maximum_error"] if p["limit_metric"] != "none" else 0, 1e-12)
            indices = np.linspace(0, len(errors) - 1, min(len(errors), 512)).astype(int)
            for position, index in enumerate(indices):
                x = 30 + round(position * 580 / max(1, len(indices) - 1))
                y = 130 - round(float(errors[index]) / scale * 95)
                cv2.circle(plot, (x, y), 3, (210, 110, 40), -1)
            title = f"Residuals ({units}); MAE {mae:.4g} RMSE {rmse:.4g}"
            draw_unicode(
                plot,
                title if len(title) <= 160 else title[:157] + "...",
                (12, 22),
                (35, 35, 35),
                TextStyle(font_size=14),
            )
        context = copy.deepcopy(self.input_payloads.get("measurements", {}))
        for key in ("spatial", "pyramid"):
            context.pop(key, None)
        outputs = {
            "report": report,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "maximum_absolute_error": maximum,
            "plot": plot,
        }
        return Result(
            outputs,
            passed=passed,
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "regression.metrics",
                    "message": "Aggregate error exceeds the configured limit",
                }
            ],
            payloads={key: copy.deepcopy(context) for key in outputs},
            graphics={"regression residuals": plot} if plot is not NO_VALUE else {},
            message=f"{len(records)} samples; MAE {mae:.5g}, RMSE {rmse:.5g} {units}",
        )


REGRESSION_BLOCKS = (DecodeRegression, RegressionMetrics)
