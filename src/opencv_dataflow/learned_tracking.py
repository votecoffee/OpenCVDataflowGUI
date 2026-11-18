"""Native ViT tracking over the common seeded, ticketed sequence lifecycle."""

import copy
import math
from dataclasses import replace

import cv2
import numpy as np

from .core import Port
from .dnn_blocks import BACKENDS, TARGETS
from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .model_contracts import (
    CONTRACT_CHOICES,
    CONTRACT_DEFAULTS,
    CONTRACT_HELP,
    canonical,
    environment,
    model_bytes,
    resolve_contract,
)
from .native_trackers import TRACKER_DEFAULTS, TRACKER_HELP, NativeBoxTracker, box_center
from .onnx_loading import load_onnx_snapshot
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value
from .selection import contains
from .sequence import SEQUENCE_CHOICES


def preprocessing(p):
    return {
        "model": "native_vit_preparation_v1",
        "image": "canonical_bgr_uint8",
        "mean": copy.deepcopy(p["mean"]),
        "std": copy.deepcopy(p["std"]),
        "template_size": [128, 128],
        "search_size": [256, 256],
        "native_template_factor": 2,
        "native_search_factor": 4,
    }


class ScoredVit:
    """Keep native model state and current update score out of serializable sequence state."""

    def __init__(self, tracker, threshold, maximum_patch_side):
        self.tracker, self.threshold, self.maximum_patch_side = (
            tracker,
            threshold,
            maximum_patch_side,
        )
        self.score = None
        self.bounds = None

    def guard(self, bounds):
        if math.ceil(4 * math.sqrt(bounds[2] * bounds[3])) > self.maximum_patch_side:
            raise ValidationError(
                "Native search crop exceeds Maximum patch side", "maximum_patch_side"
            )

    def init(self, image, bounds):
        self.guard(bounds)
        self.bounds = tuple(bounds)
        return self.tracker.init(image, bounds)

    def update(self, image):
        self.guard(self.bounds)
        self.score = None
        located, bounds = self.tracker.update(image)
        score = float(self.tracker.getTrackingScore())
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError("Native ViT returned a nonfinite/out-of-range tracking score")
        self.score = score
        if located:
            self.bounds = tuple(bounds)
        return bool(located and score >= self.threshold), bounds


class VitSequence(NativeBoxTracker):
    type_id, title = "sequence.track_vit", "Track object box (ViT)"
    preview_safe = False
    inputs = tuple(
        replace(port, required_when_connected=True) if port.name in ("mask", "region") else port
        for port in NativeBoxTracker.inputs
    )
    project_path_parameters = ("model_path",)
    project_asset_parameters = (*NativeBoxTracker.project_asset_parameters, "contract_path")
    status_outputs = (
        *NativeBoxTracker.status_outputs,
        Port("image"),
        Port("tracking_score", "number", required=False),
        Port("receipt", "data", required=False),
    )
    defaults = {
        **TRACKER_DEFAULTS,
        **CONTRACT_DEFAULTS,
        "model_path": "",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "tracking_score_threshold": 0.2,
        "engine": "classic",
        "backend": "opencv",
        "target": "cpu",
        "fusion": True,
        "selection_mode": "center",
        "maximum_source_pixels": 2_100_000,
        "maximum_patch_side": 2048,
        "max_bytes_mb": 384.0,
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        **CONTRACT_CHOICES,
        "engine": ("classic", "new"),
        "backend": tuple(BACKENDS),
        "target": tuple(TARGETS),
        "selection_mode": ("center", "seed_only"),
        "receipt_mode": ("summary", "off"),
    }
    parameter_ranges = {
        "warmup": NumericRange(1, 10000, whole=True),
        "max_streams": NumericRange(1, 64, whole=True),
        "max_bytes_mb": NumericRange(0.01, 4096),
        "max_gap_ms": NumericRange(0, 3_600_000),
        "max_misses": NumericRange(1, 1000, whole=True),
        "minimum_contrast": NumericRange(0, 255),
        "maximum_center_jump": NumericRange(0, 65536),
        "tracking_score_threshold": NumericRange(0, 1),
        "maximum_source_pixels": NumericRange(1, 2_100_000, whole=True),
        "maximum_patch_side": NumericRange(32, 8192, whole=True),
    }
    parameter_help = {
        **TRACKER_HELP,
        **CONTRACT_HELP,
        "model_path": "Local ViT ONNX, or use a sealed Contract path; no automatic download.",
        "mean": "Native normalized BGR means; native tracker owns template/search preparation.",
        "std": "Three positive native channel divisors; defaults mirror TrackerVit.Params.",
        "tracking_score_threshold": "Window-weighted response threshold; not a probability.",
        "selection_mode": "Selections seed the rectangle; Center also gates accepted updates.",
        "initial_bounds": "Initial [x,y,w,h], at least 8x8; wired Seed takes precedence.",
        "maximum_patch_side": "Bound 4*sqrt(box area) search crop before native calls.",
        "max_bytes_mb": "Conservative native workspace/model allowance; not measured RSS.",
        "engine": "Explicit import engine; New is CPU-only here. No silent retry.",
        "draw_overlay": "Current drawable status/box; raster Overlay only when exposed.",
        "receipt_mode": "Model/settings/import evidence or Off; native tensors are not exposed.",
    }
    input_help = {
        "image": "Original uint8 gray/RGB/BGR/alpha; native tracker receives canonical BGR.",
        "seed": "Current-grid [x,y,w,h] or {bounds:[...]}, used only when a new model is needed.",
        "mask": "Seed support and accepted-center gate; not a native template mask.",
        "region": "Current-grid rectangle/circle/polygon, intersected with Mask.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = [n for n in self.exposed_outputs if n not in ("overlay", "receipt")]
        self._snapshot = None

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name, r in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or r.whole and type(p[name]) is not int:
                raise ValidationError("Use typed numeric tracker settings", name)
        for name in ("fusion", "draw_overlay"):
            if type(p[name]) is not bool:
                raise ValidationError("Use a checkbox boolean", name)
        for name in ("mean", "std"):
            if (
                not isinstance(p[name], list)
                or len(p[name]) != 3
                or any(
                    type(v) not in (int, float)
                    or not math.isfinite(v)
                    or abs(v) > 1e6
                    or name == "std"
                    and v < 1e-12
                    for v in p[name]
                )
            ):
                raise ValidationError("Use three finite means/positive divisors", name)
        bounds = p["initial_bounds"]
        if not isinstance(bounds, list) or len(bounds) not in (0, 4):
            raise ValidationError(
                "Initial bounds must be empty or [x,y,width,height]", "initial_bounds"
            )
        for i, value in enumerate(bounds):
            NumericRange(8 if i > 1 else 0, 32766, whole=True).check(
                value, "Initial bound", "initial_bounds"
            )
        if not isinstance(p["model_path"], str) or not isinstance(p["contract_path"], str):
            raise ValidationError(
                "Model/contract paths must be text", "model_path", "contract_path"
            )
        if p["contract_path"]:
            from .model_contracts import relative_path

            relative_path(p["contract_path"])
        if (p["backend"] == "cuda") != p["target"].startswith("cuda"):
            raise ValidationError("Select CUDA backend and target together", "backend", "target")
        if (p["target"] != "cpu" or not p["fusion"]) and p["engine"] != "classic":
            raise ValidationError("Select Classic for non-CPU or disabled fusion", "engine")

    def project_assets(self):
        assets = super().project_assets()
        if self.parameters["contract_path"]:
            from pathlib import Path

            path, contract = resolve_contract(self)
            _, identity = model_bytes(path, bounded=True)
            if identity != {k: contract["artifact"][k] for k in ("sha256", "bytes")}:
                raise ValidationError("Bundled ViT checksum differs", "contract_path")
            assets.append((Path(self.parameters["contract_path"]).parent / "model.onnx").as_posix())
        return assets

    def extra_context(self, inputs):
        path, contract = resolve_contract(self)
        raw, identity = model_bytes(path, bounded=True)
        if contract:
            if identity != {k: contract["artifact"][k] for k in ("sha256", "bytes")}:
                raise ValidationError("Bundled ViT checksum differs", "contract_path")
            if contract["preprocessing"] and canonical(contract["preprocessing"]) != canonical(
                preprocessing(self.parameters)
            ):
                raise ValidationError(
                    "Native ViT preparation differs from contract", "contract_path"
                )
        self._snapshot = (raw, identity, contract)
        return {
            "artifact": identity,
            "contract_revision": contract["revision"] if contract else None,
        }

    def prepare_image(self, source):
        return np.ascontiguousarray(super().prepare_image(source))

    def location_allowed(self, bounds, mask):
        return self.parameters["selection_mode"] == "seed_only" or contains(
            mask, box_center(bounds)
        )

    def tracked_overlay(self, source, bounds):
        if not self.generate_graphics or "overlay" not in self.exposed_outputs:
            return None
        return super().tracked_overlay(source, bounds)

    def new_tracker(self, image, bounds, mask, state):
        p = self.parameters
        raw, identity, contract = self._snapshot
        if p["selection_mode"] == "center" and not contains(mask, box_center(bounds)):
            raise ValidationError("Seed center is outside selection")
        if math.ceil(4 * math.sqrt(bounds[2] * bounds[3])) > p["maximum_patch_side"]:
            raise ValidationError(
                "Native search crop exceeds Maximum patch side", "maximum_patch_side"
            )
        self.reserve_native_state(
            state,
            identity["bytes"] * 8
            + image.nbytes * 4
            + p["maximum_patch_side"] ** 2 * 12
            + 16 * 1024**2,
        )
        selected = {k: p[k] for k in ("engine", "backend", "target", "fusion")}
        requested = dict(selected)
        cpu_only = p["target"] == "cpu" and p["backend"] in ("opencv", "default")
        available = (
            [cv2.dnn.DNN_TARGET_CPU]
            if cpu_only
            else [int(t) for t in cv2.dnn.getAvailableTargets(BACKENDS[p["backend"]])]
        )
        fallback = None
        if TARGETS[p["target"]] not in available:
            if p["fallback_policy"] == "error":
                raise ValidationError("Requested tracker DNN target unavailable", "target")
            selected.update(engine="classic", backend="opencv", target="cpu")
            fallback = "requested_target_unavailable"
            self.logger.warning("Tracker DNN target unavailable; explicitly falling back to CPU")
        engine = cv2.dnn.ENGINE_CLASSIC if selected["engine"] == "classic" else cv2.dnn.ENGINE_NEW
        net, imported = load_onnx_snapshot(raw, engine)
        if selected["engine"] == "classic":
            net.setPreferableBackend(BACKENDS[selected["backend"]])
            net.setPreferableTarget(TARGETS[selected["target"]])
            net.enableFusion(p["fusion"])
        native = cv2.TrackerVit.create(
            net, tuple(p["mean"]), tuple(p["std"]), p["tracking_score_threshold"]
        )
        state["model_receipt"] = {
            "model": "native_vit_receipt_v1",
            "artifact": identity,
            "contract_revision": contract["revision"] if contract else None,
            "configuration_verified": contract is not None,
            "preprocessing_verified": bool(contract and contract["preprocessing"]),
            "preprocessing": preprocessing(p),
            "environment": environment(),
            "model_import": imported,
            "requested_execution": requested,
            "selected_execution": selected,
            "requested_availability_scope": "requested_cpu_only"
            if cpu_only
            else "backend_discovery",
            "fallback": fallback,
            "native_placement_verified": False,
        }
        return ScoredVit(native, p["tracking_score_threshold"], p["maximum_patch_side"])

    def step(self, inputs, state, mask):
        native = state.get("native")
        step = super().step(inputs, state, mask)
        score = native.score if native is not None else None
        if (
            step.details["phase"] == "lost"
            and step.details["reason"] == "target_missing"
            and score is not None
            and score < self.parameters["tracking_score_threshold"]
        ):
            step.details["reason"] = "tracking_score_below_threshold"
        state["tracking_score"] = score
        step.details.update(
            tracking_score=score,
            native_updates=int(native is not None),
            native_confidence_available=True,
        )
        return step

    def process(self, inputs):
        source = inputs["image"]
        image_value(source, (np.uint8,))
        if (
            source.shape[0] * source.shape[1] > self.parameters["maximum_source_pixels"]
            or max(source.shape[:2]) > 32766
        ):
            raise ValidationError("Tracker image exceeds source guard", "maximum_source_pixels")
        self._snapshot = None
        try:
            result = super().process(inputs)
            status = result.outputs["status"]
            with self._memory_lock:
                state = self.memory["streams"].get(status["context"], {})
                score = state.get("tracking_score")
                receipt = copy.deepcopy(state.get("model_receipt"))
                status["last_accepted_bounds"] = (
                    list(state["bounds"]) if "bounds" in state else None
                )
            result.outputs.update(
                image=source,
                tracking_score=score if score is not None else NO_VALUE,
                receipt=receipt
                if receipt is not None and self.parameters["receipt_mode"] != "off"
                else NO_VALUE,
            )
            failed = status["phase"] == "lost" or (
                status["phase"] == "seed_required"
                and status.get("misses", 0) >= self.parameters["max_misses"]
            )
            if receipt is not None:
                receipt.update(
                    model_active="native" in state,
                    native_updates=status["native_updates"],
                    activation=status["activations"],
                    tracking_score_current=score,
                    track_id=status.get("track_id"),
                )
                if self.parameters["receipt_mode"] != "off":
                    result.outputs["receipt"] = receipt
            result.passed = not failed
            result.score = score
            if failed:
                result.fail_reasons = [
                    {"code": "vit_" + status["reason"], "message": status["reason"]}
                ]
            context = copy.deepcopy(self.input_payloads["image"])
            context["sequence"] = copy.deepcopy(status)
            if result.outputs["ready"] and "spatial" in context:
                result.outputs["tracked_region"]["coordinate_context"] = copy.deepcopy(
                    context["spatial"]
                )
            for name in result.outputs:
                if result.outputs[name] is not NO_VALUE:
                    result.payloads[name] = copy.deepcopy(context)
            if self.generate_graphics and self.parameters["draw_overlay"]:
                color = (40, 70, 240) if failed else (60, 200, 100)
                text = f"ViT: {status['phase']}" + (
                    f"; score {score:.3f}" if score is not None else ""
                )
                objects = [
                    graphic_object("text", [(5, min(18, source.shape[0] - 1))], color, text=text)
                ]
                if result.outputs["ready"]:
                    objects.append(graphic_box(*result.outputs["bounds"], color))
                result.drawings = {"ViT tracking": GraphicDrawing(tuple(objects))}
            result.message += " · " + status["phase"]
            return result
        finally:
            self._snapshot = None
