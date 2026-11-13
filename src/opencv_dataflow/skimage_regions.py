"""Standard 2D regionprops, with native units and separate image-pixel collections."""

import math
import warnings

import numpy as np

from .collection_tools import collection_result
from .core import Block, Port
from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .histogram import HISTOGRAM_CHANNELS
from .images import image_format
from .label_measurement import check_label_plane, raster_digest, scope_digest
from .objects import collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import require_same_frame
from .processing import image_value
from .region_proposals import box_item
from .selection import selection_mask
from .tickets import ticket_key

INSTALL = "Install scikit-image 0.26.x with pip install .[skimage]"
STANDARD = (
    "label",
    "area",
    "bbox",
    "centroid",
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "equivalent_diameter_area",
    "euler_number",
    "extent",
    "solidity",
    "perimeter",
    "perimeter_crofton",
    "feret_diameter_max",
    "intensity_min",
    "intensity_max",
    "intensity_mean",
    "intensity_median",
    "intensity_std",
)
NUMERIC = (
    "area",
    "area_bbox",
    "area_convex",
    "area_filled",
    "axis_major_length",
    "axis_minor_length",
    "bbox",
    "centroid",
    "centroid_local",
    "centroid_weighted",
    "centroid_weighted_local",
    "eccentricity",
    "equivalent_diameter_area",
    "euler_number",
    "extent",
    "feret_diameter_max",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "intensity_max",
    "intensity_mean",
    "intensity_median",
    "intensity_min",
    "intensity_std",
    "label",
    "moments",
    "moments_central",
    "moments_hu",
    "moments_normalized",
    "moments_weighted",
    "moments_weighted_central",
    "moments_weighted_hu",
    "moments_weighted_normalized",
    "num_pixels",
    "orientation",
    "perimeter",
    "perimeter_crofton",
    "solidity",
)
COORDINATES = ("coords", "coords_scaled")
PATCHES = ("image", "image_convex", "image_filled", "image_intensity")
PROPERTIES = (*NUMERIC, "slice", *COORDINATES, *PATCHES)
MAX_ROI_WORK = 16_777_216


def _clean(value):
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _flatten(value, prefix, result):
    if isinstance(value, list):
        for index, item in enumerate(value):
            _flatten(item, prefix + "." + str(index), result)
    elif value is None or type(value) in (int, float):
        result[prefix] = value


def _nonfinite(value):
    return value is None or isinstance(value, list) and any(_nonfinite(v) for v in value)


class RegionProperties(Block):
    type_id, title = "skimage.region_properties", "Region properties (scikit-image)"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (
        Port("labels"),
        Port("intensity_image", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("collection", "data"),
        Port("count", "integer"),
        Port("points", "data"),
        Port("labels"),
        Port("measurements", "data"),
    )
    payload_reference = "labels"
    process_inputs_readonly = parallel_safe = True
    defaults = {
        "input_mode": "labels",
        "connectivity": 8,
        "ignored_labels": [-1, 0],
        "property_set": "standard",
        "properties": [],
        "include_coordinates": False,
        "include_images": False,
        "spacing_yx": [1.0, 1.0],
        "offset_yx": [0, 0],
        "units": "pixels",
        "include_alpha": False,
        "intensity_alignment": "same_frame",
        "cache": True,
        "unavailable": "report",
        "maximum_regions": 1000,
        "maximum_values": 1_000_000,
        "preview_count": 25,
    }
    parameter_choices = {
        "input_mode": ("labels", "binary"),
        "intensity_alignment": ("same_frame", "pixel_grid"),
        "connectivity": (4, 8),
        "property_set": ("standard", "comprehensive", "custom"),
        "unavailable": ("report", "error"),
    }
    parameter_ranges = {
        "maximum_regions": NumericRange(1, 10000, whole=True),
        "maximum_values": NumericRange(100, 2_000_000, whole=True),
        "preview_count": NumericRange(0, 250, whole=True),
    }
    parameter_help = {
        "input_mode": "Measure each label ID, or label connected nonzero binary pixels first.",
        "connectivity": "4 or 8 neighbors for binary labeling; existing label IDs are not split.",
        "ignored_labels": "IDs excluded after selection. Remove 0 to measure a class-zero region.",
        "property_set": "Standard measurements, all numeric/moment/tensor properties, or custom.",
        "properties": "Custom names from the documented regionprops catalog. Extra Python "
        "callables belong in user blocks; this field does not execute code.",
        "include_coordinates": "Also return coords and coords_scaled; may be large.",
        "include_images": "Return cropped binary/convex/filled/intensity arrays as data.",
        "spacing_yx": "Positive [row, column] spacing for native properties. Collection geometry "
        "and ordinary metrics stay in image pixels. Units are descriptive, not a calibration fit.",
        "offset_yx": "Native regionprops [row, column] origin offset. Does not move GUI graphics.",
        "units": "Label for native property units; choose consistently with spacing.",
        "include_alpha": "Include alpha in intensity channels; default excludes it.",
        "intensity_alignment": "Same frame checks known grids and tickets; pixel grid explicitly "
        "accepts separately captured/exported aligned intensity data.",
        "cache": "Cache native intermediate properties during this invocation only.",
        "unavailable": "Report null/reason for missing intensity, nonfinite or unsupported "
        "properties (e.g. anisotropic perimeters), or fail the block.",
        "maximum_regions": "Maximum distinct measured label IDs, even for sparse large IDs.",
        "maximum_values": "Total property-value budget, including opted-in coordinates/images.",
        "preview_count": "Maximum boxes/centroids to draw. Zero disables these optional graphics.",
    }
    input_help = {
        "labels": "Integer label plane (up to 2.1M pixels); binary mode uses nonzero pixels.",
        "intensity_image": "Optional same-grid gray/RGB/BGR image, retained in raw sample units.",
        "mask": "Clip measured pixels. Reuse this scope when reconstructing masks from labels.",
        "region": "Optional rectangle, circle or polygon, intersected with a wired mask.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in ("include_coordinates", "include_images", "include_alpha", "cache"):
            if type(p[name]) is not bool:
                raise ValidationError(f"{name} must be true or false", name)
        for name, positive in (("spacing_yx", True), ("offset_yx", False)):
            pair = p[name]
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(
                    type(v) not in (int, float)
                    or not np.isfinite(v)
                    or (positive and not 1e-9 <= v <= 1e9)
                    or (not positive and abs(v) > 1e9)
                    for v in pair
                )
            ):
                raise ValidationError("Use a finite [row, column] pair; spacing in 1e-9..1e9", name)
        ignored = p["ignored_labels"]
        if (
            not isinstance(ignored, list)
            or len(ignored) > 256
            or any(type(v) is not int or not -(2**31) <= v < 2**31 for v in ignored)
        ):
            raise ValidationError("Ignored labels must be up to 256 int32 IDs", "ignored_labels")
        properties = p["properties"]
        if (
            not isinstance(properties, list)
            or any(name not in PROPERTIES for name in properties)
            or len(properties) > len(PROPERTIES)
            or (p["property_set"] == "custom" and not properties)
        ):
            raise ValidationError("Choose documented regionprops property names", "properties")
        if not isinstance(p["units"], str) or not 1 <= len(p["units"]) <= 64:
            raise ValidationError("Units must contain 1..64 characters", "units")

    def process(self, inputs):
        try:
            from skimage.measure import label, regionprops
        except ImportError as exc:
            raise RuntimeError(INSTALL) from exc
        p = self.parameters
        source = inputs["labels"]
        check_label_plane(
            source.astype(np.uint8)
            if source.dtype == np.bool_ and p["input_mode"] == "binary"
            else source
        )
        scope = selection_mask({**inputs, "image": source}, source)
        labels = np.asarray(source).copy()
        if p["input_mode"] == "binary":
            labels = label(
                (labels != 0) & (scope != 0 if scope is not None else True),
                connectivity=int(p["connectivity"]) // 4,
            ).astype(np.int32)
        selected = ~np.isin(labels, p["ignored_labels"])
        if scope is not None:
            selected &= scope != 0
        ids, inverse = np.unique(labels[selected], return_inverse=True)
        if len(ids) > p["maximum_regions"]:
            raise ValidationError("Too many measured regions; tighten selection or label criteria")
        dense = np.zeros(labels.shape, np.int32)
        dense[selected] = inverse + 1
        intensity, channels = None, []
        if "intensity_image" in inputs:
            image = inputs["intensity_image"]
            image_value(image, (np.uint8, np.uint16, np.int16, np.float32, np.float64))
            if image.shape[:2] != labels.shape or image_format(image) not in HISTOGRAM_CHANNELS:
                raise ValidationError("Intensity image must match label dimensions and encoding")
            if p["intensity_alignment"] == "same_frame":
                require_same_frame((source, image), message="Intensity image and labels must align")
                a, b = (
                    self.input_payloads[key].get("ticket") for key in ("labels", "intensity_image")
                )
                if a is not None and b is not None and ticket_key(a) != ticket_key(b):
                    raise ValidationError("Intensity image and labels have different tickets")
            intensity = np.asarray(image)
            channels = list(HISTOGRAM_CHANNELS[image_format(image)])
            if intensity.ndim == 3 and len(channels) == 4 and not p["include_alpha"]:
                intensity, channels = intensity[..., :3], channels[:3]
        names = list(
            STANDARD
            if p["property_set"] == "standard"
            else (*NUMERIC, "slice")
            if p["property_set"] == "comprehensive"
            else p["properties"]
        )
        if p["include_coordinates"]:
            names.extend(COORDINATES)
        if p["include_images"]:
            names.extend(PATCHES)
        names = list(dict.fromkeys(names))
        native = regionprops(
            dense, intensity, cache=p["cache"], spacing=p["spacing_yx"], offset=p["offset_yx"]
        )
        pixels = regionprops(dense, intensity, cache=p["cache"])
        used, work, items, rows = 0, 0, [], []
        for prop, pixel in zip(native, pixels, strict=True):
            ident = int(ids[prop.label - 1])
            y, x, bottom, right = pixel.bbox
            work += (bottom - y) * (right - x)
            if work > MAX_ROI_WORK:
                raise ValidationError("Region bounding boxes exceed the bounded work budget")
            item = box_item(ident, x, y, right, bottom)
            metrics = item["metrics"]
            eigen = pixel.inertia_tensor_eigvals
            metrics.update(
                area=int(pixel.num_pixels),
                center_x=float(pixel.centroid[1]),
                center_y=float(pixel.centroid[0]),
                major_size=float(pixel.axis_major_length),
                minor_size=float(pixel.axis_minor_length),
                extent=float(pixel.extent),
                solidity=float(pixel.solidity),
                perimeter=float(pixel.perimeter),
                angle=(90 - math.degrees(pixel.orientation)) % 180
                if eigen[0] - eigen[1] > max(eigen[0], 1) * 1e-12
                else None,
            )
            metrics["rotated_aspect_ratio"] = (
                metrics["major_size"] / metrics["minor_size"] if metrics["minor_size"] else None
            )
            metrics["circularity"] = (
                4 * math.pi * metrics["area"] / metrics["perimeter"] ** 2
                if metrics["perimeter"]
                else None
            )
            for index, hu in enumerate(pixel.moments_hu, 1):
                metrics[f"hu_{index}"] = float(hu)
                metrics[f"hu_log_{index}"] = (
                    -math.copysign(1, hu) * math.log10(abs(hu)) if hu else 0.0
                )
            if intensity is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean = np.atleast_1d(pixel.intensity_mean)
                if intensity.ndim == 2:
                    metrics["intensity"] = _clean(mean[0])
                elif all(c in channels for c in ("R", "G", "B")):
                    metrics["intensity"] = _clean(
                        sum(
                            weight * mean[channels.index(channel)]
                            for channel, weight in (("R", 0.299), ("G", 0.587), ("B", 0.114))
                        )
                    )
            raw, unavailable = {}, {}
            for name in names:
                if name in COORDINATES:
                    estimate = 2 * int(pixel.num_pixels)
                elif name in PATCHES:
                    estimate = (bottom - y) * (right - x)
                    if name == "image_intensity" and intensity is not None and intensity.ndim == 3:
                        estimate *= intensity.shape[-1]
                else:
                    estimate = 1
                if used + estimate > p["maximum_values"]:
                    raise ValidationError(
                        "Region properties exceed the configured output-value budget"
                    )
                requires_intensity = (
                    name.startswith("intensity_") or "weighted" in name or name == "image_intensity"
                )
                if requires_intensity and intensity is None:
                    value, reason = None, "requires_intensity_image"
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            value = _clean(getattr(prop, name)) if name != "label" else ident
                        reason = "nonfinite_property" if _nonfinite(value) else None
                    except (
                        AttributeError,
                        NotImplementedError,
                        ValueError,
                        FloatingPointError,
                    ) as exc:
                        value, reason = None, str(exc)
                raw[name] = value
                used += max(estimate, int(np.asarray(value, dtype=object).size))
                if used > p["maximum_values"]:
                    raise ValidationError("Region properties exceed the output-value budget")
                if reason:
                    unavailable[name] = reason
                    if p["unavailable"] == "error":
                        raise ValidationError(f"Region {ident}, {name}: {reason}")
                if name in NUMERIC:
                    _flatten(value, "regionprops." + name, metrics)
            item["properties"], item["unavailable"] = raw, unavailable
            items.append(item)
            rows.append({"label": ident, "properties": raw, "unavailable": unavailable})
        data = collection("boxes", items, source, self.input_payloads["labels"])
        data["label_source"] = {
            "model": "measured_label_source_v1",
            "labels_sha256": raster_digest(labels),
            "scope_sha256": scope_digest(scope),
        }
        report = {
            "model": "skimage_region_properties_v1",
            "property_names": names,
            "spacing_yx": p["spacing_yx"],
            "offset_yx": p["offset_yx"],
            "units": p["units"],
            "coordinate_order": "row,column",
            "collection_geometry": "image_pixels_xy",
            "intensity_channels": channels,
            "rows": rows,
            "property_values": used,
            "unavailable_count": sum(len(row["unavailable"]) for row in rows),
        }
        result = collection_result(
            data, self.input_payloads["labels"], labels=labels, measurements=report
        )
        if self.generate_graphics:
            objects = []
            for item in items[: int(p["preview_count"])]:
                objects.extend(
                    (
                        graphic_box(*item["geometry"]["bounds"], (70, 220, 130)),
                        graphic_object(
                            "marker",
                            [(item["metrics"]["center_x"], item["metrics"]["center_y"])],
                            (70, 220, 130),
                        ),
                    )
                )
            if objects:
                result.drawings["measured regions"] = GraphicDrawing(
                    tuple(objects), source="labels"
                )
        result.message = (
            f"{len(items)} regions · {len(names)} properties · "
            f"{report['unavailable_count']} unavailable"
        )
        return result
