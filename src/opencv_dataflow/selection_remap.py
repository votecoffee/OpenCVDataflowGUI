"""Remap images and whole selections together, with explicit support and exact raster regions."""

import copy
import hashlib
import json
import math

import cv2
import numpy as np

from .analysis_tools import Remap, RemapPacked
from .core import Port
from .flow_warp import sample_mask
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, frame, require_same_frame
from .processing import finite, image_value
from .raster_regions import MAX_REGION_RUNS, region_from_mask
from .selection import selection_mask


def boundaries(mask, maximum):
    if maximum < 2:
        return ()
    h, w = mask.shape
    scale = min(1.0, math.sqrt(262144 / mask.size))
    size = max(1, round(w * scale)), max(1, round(h * scale))
    preview = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) if scale < 1 else mask
    contours, _ = cv2.findContours(preview, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    objects, remaining = [], maximum
    for contour in contours:
        points = contour.reshape(-1, 2).astype(float)
        if len(points) < 2:
            continue
        if remaining < 2:
            break
        if len(points) > remaining:
            points = points[np.linspace(0, len(points) - 1, remaining, dtype=int)]
        points = (points + 0.5) * [w / size[0], h / size[1]] - 0.5
        objects.append(graphic_object("polyline", points, (50, 220, 100), closed=True))
        remaining -= len(points)
    return tuple(objects)


class SelectionRemap:
    """Native Remap owns image sampling; this mixin adds selection/data contracts."""

    defaults = {
        **Remap.defaults,
        "border_values": [],
        "selection_rule": "complete_footprint",
        "selection_sampling": "linear",
        "minimum_coverage": 0.5,
        "emit_region": True,
        "minimum_selected_pixels": 1,
        "maximum_region_runs": 100000,
        "maximum_graphics_points": 4096,
        "maximum_pixels": 16_000_000,
        "maximum_working_bytes": 512 * 1024**2,
    }
    parameter_choices = {
        **Remap.parameter_choices,
        "selection_rule": ("complete_footprint", "coverage_threshold"),
        "selection_sampling": ("nearest", "linear"),
    }
    parameter_ranges = {
        "minimum_coverage": NumericRange(0, 1),
        "minimum_selected_pixels": NumericRange(0, 64_000_000, whole=True),
        "maximum_region_runs": NumericRange(1, MAX_REGION_RUNS, whole=True),
        "maximum_graphics_points": NumericRange(0, 100000, whole=True),
        "maximum_pixels": NumericRange(1, 64_000_000, whole=True),
        "maximum_working_bytes": NumericRange(1, 2 * 1024**3, whole=True),
    }
    parameter_help = {
        **Remap.parameter_help,
        "border_values": "Optional finite per-channel JSON list; empty uses scalar Border value.",
        "border": "Image extrapolation only; filled borders never add selection validity.",
        "selection_rule": "Require the full image footprint inside the selection, or use coverage.",
        "selection_sampling": "Sampling for Coverage; also used by Coverage threshold selection.",
        "minimum_coverage": "Coverage threshold only. Zero accepts any positive contribution.",
        "emit_region": "Encode an exact raster region. Disable to skip noisy-mask encoding.",
        "minimum_selected_pixels": "Minimum supported selected count; zero permits empty masks.",
        "maximum_region_runs": "Bound exact region encoding; independent of image/mask sampling.",
        "maximum_graphics_points": "Bound decimated diagnostic contours; zero omits drawings.",
        "maximum_pixels": "Source/output pixel limit; axes are also limited to 16384 pixels.",
        "maximum_working_bytes": "Estimated arrays; excludes native/RSS and runtime copies.",
    }
    input_help = {
        "image": "Original image; selection shares its destination-to-source maps.",
        "mask": "Optional uint8 selection, intersected with Region in the source image frame.",
        "region": "Optional source rectangle/circle/polygon/composite/transformed/raster region.",
    }
    output_help = {
        "mask": "Supported target selection: uint8, exactly zero or 255.",
        "region": "Exact JSON raster region on the output image grid; optional emission.",
        "coverage": "Nearest/linear source-selection contribution in [0,1]; not physical area.",
        "valid_mask": "Complete real-source image support, independent of border extrapolation.",
        "report": "Sampling policies, support/selection counts, source/target grids and budgets.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = self.description or "Carry a whole selection through the image's remap."
        self.documentation = self.documentation or (
            "<p>Image and selection share the same remap. Border fills never add inspection "
            "support. Exact raster regions preserve holes. See docs/selection-remap.md for "
            "coverage, conservative negative-kernel support, frames and resource bounds.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["emit_region"]) is not bool:
            raise ValidationError("Emit region must be boolean", "emit_region")
        values = p["border_values"]
        if not isinstance(values, list) or len(values) > 4:
            raise ValidationError(
                "Border values must be a list of up to four values", "border_values"
            )
        for value in values:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValidationError("Border values must be finite numbers", "border_values")
        for name, bounds in self.parameter_ranges.items():
            if isinstance(p[name], bool) or (bounds.whole and type(p[name]) is not int):
                raise ValidationError(f"{name} needs a typed numeric value", name)

    def border_fill(self, image):
        values = self.parameters["border_values"]
        if not values:
            return super().border_fill(image)
        channels = image.shape[2] if image.ndim == 3 else 1
        if len(values) != channels:
            raise ValidationError("Border values must match stored channels", "border_values")
        return tuple(finite(value, "Border value") for value in values)

    def process(self, inputs):
        p = self.parameters
        image = image_value(inputs["image"])
        map1, map2, lens = self.coordinate_maps(inputs)
        source_size, target_size = list(image.shape[1::-1]), list(map1.shape[1::-1])
        for size in (source_size, target_size):
            if max(size) > 16384 or math.prod(size) > p["maximum_pixels"]:
                raise ValidationError(
                    "Remap source/output exceeds pixel or axis limit", "maximum_pixels"
                )
        pixel_bytes = image.dtype.itemsize * (image.shape[2] if image.ndim == 3 else 1)
        estimate = math.prod(source_size) * (pixel_bytes + 12) + math.prod(target_size) * (
            pixel_bytes + 48
        )
        if estimate > p["maximum_working_bytes"]:
            raise ValidationError(
                "Selection remap exceeds Maximum working bytes", "maximum_working_bytes"
            )
        if map1.dtype == np.int16:
            if map2 is None:
                x, y = map1[..., 0].astype(np.float32), map1[..., 1].astype(np.float32)
            else:
                x, y = cv2.convertMaps(map1, map2, cv2.CV_32FC1)
        else:
            x, y = map1, map2
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValidationError("Remap coordinates must be finite")
        for name in ("mask", "region"):
            if name in inputs:
                require_same_frame(
                    (inputs["image"], ValuePacket(inputs[name], self.input_payloads[name])),
                    message="Selection must use the source image frame",
                )
        selected = selection_mask(inputs, inputs["image"])
        selected = (
            np.ones(image.shape[:2], np.uint8)
            if selected is None
            else (selected != 0).astype(np.uint8)
        )
        coverage = cv2.remap(
            selected.astype(np.float32),
            x,
            y,
            cv2.INTER_NEAREST if p["selection_sampling"] == "nearest" else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        coverage = np.clip(coverage, 0, 1)
        method = "nearest" if p["interpolation"] == "nearest" else "linear"
        halo = {"cubic": 2, "lanczos4": 4}.get(p["interpolation"], 0)
        real = sample_mask(np.ones(image.shape[:2], np.uint8), x, y, method, halo)
        if p["selection_rule"] == "complete_footprint":
            accepted = sample_mask(selected, x, y, method, halo)
        else:
            accepted = (
                coverage > 0 if p["minimum_coverage"] == 0 else coverage >= p["minimum_coverage"]
            )
        mask = np.where(accepted & real, 255, 0).astype(np.uint8)
        result = super().process(inputs)
        context = copy.deepcopy(result.payloads.get("image", self.input_payloads["image"]))
        if not result.payloads.get("image"):
            digest = hashlib.sha256(
                json.dumps(
                    [self.input_payloads["image"].get("spatial"), target_size], sort_keys=True
                ).encode()
            )
            for plane in (x, y):
                digest.update(np.ascontiguousarray(plane).data)
            context["spatial"] = frame(target_size, f"selection-remap:{digest.hexdigest()}")
        context.pop("pyramid", None)
        context.pop("coordinate_remap", None)
        context.pop("lens_remap_map", None)
        region = region_from_mask(mask, p["maximum_region_runs"]) if p["emit_region"] else NO_VALUE
        count = int(np.count_nonzero(mask))
        report = {
            "model": "selection_remap_v1",
            "direction": "destination_to_source",
            "source_size": source_size,
            "output_size": target_size,
            "source_selected_pixels": int(np.count_nonzero(selected)),
            "selected_pixels": count,
            "valid_pixels": int(np.count_nonzero(real)),
            "output_pixels": int(mask.size),
            "selection_rule": p["selection_rule"],
            "selection_sampling": p["selection_sampling"],
            "image_interpolation": p["interpolation"],
            "conservative_halo": halo,
            "minimum_coverage": p["minimum_coverage"],
            "estimated_working_bytes": estimate,
            "region_runs": len(region["runs"]) if region is not NO_VALUE else None,
            "source_spatial": self.input_payloads["image"].get("spatial"),
            "output_spatial": context["spatial"],
            "lens_map_id": lens["id"] if lens else None,
            "area_units": "target pixels; not calibrated physical area",
        }
        context["selection_remap"] = {
            k: report[k] for k in ("model", "direction", "source_spatial", "lens_map_id")
        }
        result.outputs.update(
            mask=mask,
            region=region,
            coverage=coverage,
            valid_mask=np.where(real, 255, 0).astype(np.uint8),
            report=report,
        )
        result.payloads = {
            name: copy.deepcopy(context)
            for name, value in result.outputs.items()
            if value is not NO_VALUE
        }
        result.passed = count >= p["minimum_selected_pixels"]
        if not result.passed:
            result.fail_reasons = [
                {
                    "code": "selection_remap.insufficient_selection",
                    "message": "Too few supported selected pixels",
                }
            ]
        result.message = f"{count} selected / {np.count_nonzero(real)} supported output pixels"
        if self.generate_graphics:
            objects = boundaries(mask, p["maximum_graphics_points"])
            if objects:
                result.drawings["Remapped selection"] = GraphicDrawing(
                    objects, "image", from_output=True
                )
        return result


class RemapSelection(SelectionRemap, Remap):
    type_id, title = "cv.remap_selection", "Remap image and selection"
    process_inputs_readonly = True
    inputs = (
        *Remap.inputs,
        Port("mask", required=False, visual_type="region"),
        Port("region", "data", required=False, visual_type="region"),
    )
    outputs = (
        Port("image"),
        Port("mask", visual_type="region"),
        Port("region", "data", visual_type="region"),
        Port("coverage"),
        Port("valid_mask", visual_type="region"),
        Port("report", "data"),
    )


class RemapSelectionPacked(SelectionRemap, RemapPacked):
    type_id, title = "cv.remap_selection_packed", "Remap image and selection (packed)"
    process_inputs_readonly = True
    input_help = {**SelectionRemap.input_help, **RemapPacked.input_help}
    inputs = (
        *RemapPacked.inputs,
        Port("mask", required=False, visual_type="region"),
        Port("region", "data", required=False, visual_type="region"),
    )
    outputs = RemapSelection.outputs
