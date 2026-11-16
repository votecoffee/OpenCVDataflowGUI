"""Register large inspection scans and render one bounded mosaic tile at a time."""

import math

import cv2
import numpy as np

from .core import Port, Result
from .image_stitching import (
    _project_edges,
    _transform,
    mosaic_frame,
    mosaic_geometry,
    warp_mosaic_view,
)
from .images import ImageArray, image_format
from .mosaic_layout import IDENTITIES, MODEL, bind_layout, canvas_frame, seal_layout, source_ids
from .mosaic_photometry import FeatherMosaic
from .multiband_mosaic import (
    MULTIBAND_CHOICES,
    MULTIBAND_DEFAULTS,
    MULTIBAND_HELP,
    MULTIBAND_RANGES,
    MultibandMosaic,
    tile_window,
)
from .parameter_validation import NumericRange, ValidationError
from .payloads import moved_payload
from .processing import ImageTool, integer
from .scan_stitching import StitchScanSequence, _unpack, register_scan

_LAYOUT_MODEL = "tiled_scan_mosaic_v1"
_MAX_VIEWS = 64
_MAX_SIDE = 1_000_000
_MAX_WORKSPACE = 256 * 1024**2
_REGISTRATION_KEYS = {
    "motion_model",
    "max_features",
    "match_ratio",
    "max_descriptor_distance",
    "inlier_pixels",
    "min_inliers",
    "min_inlier_ratio",
    "max_scale_ratio",
    "max_rotation_degrees",
    "loop_closure",
    "max_loop_drift_pixels",
}
_INTERPOLATION = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}


def _views(inputs, *, min_views=2):
    unpacked = _unpack(
        inputs["views"], inputs.get("masks"), max_views=_MAX_VIEWS, min_views=min_views
    )
    if any(max(image.shape[:2]) >= 32767 for image in unpacked[1]):
        raise ValidationError("Each source view must be smaller than 32767 pixels per side")
    return unpacked


def _bounded_size(size):
    if len(size) != 2 or any(
        type(value) is not int or not 1 <= value <= _MAX_SIDE for value in size
    ):
        raise ValidationError("Mosaic size requires two integers in 1..1,000,000")
    return tuple(size)


def _read_layout(layout, group, images, pixels):
    if not isinstance(layout, dict) or layout.get("model") != _LAYOUT_MODEL:
        raise ValidationError("Connect a layout from Register scan layout")
    size = layout.get("image_size")
    if not isinstance(size, (list, tuple)):
        raise ValidationError("Layout is missing its mosaic image size")
    size = _bounded_size(size)
    records = layout.get("views")
    if not isinstance(records, list) or len(records) != len(pixels):
        raise ValidationError("Layout must contain one record per source view")
    transforms = []
    for index, (record, label, image, pixel) in enumerate(
        zip(records, group.labels, images, pixels, strict=True)
    ):
        if not isinstance(record, dict) or (
            record.get("index") != index
            or record.get("label") != label
            or record.get("source_size") != list(pixel.shape[1::-1])
            or record.get("source_dtype") != pixel.dtype.name
            or record.get("source_format") != image_format(image)
        ):
            raise ValidationError("Layout source order, labels, grid, depth or encoding changed")
        transforms.append(_transform(record.get("source_to_mosaic")))
    actual, shift = mosaic_geometry(pixels, transforms)
    if actual != size or not np.allclose(shift, np.eye(3), atol=1e-6, rtol=0):
        raise ValidationError("Layout dimensions do not match the source transforms")
    return size, transforms


class RegisterScanLayout(StitchScanSequence):
    type_id, title = "cv.register_scan_layout", "Register scan layout"
    outputs = (Port("layout", "data"),)
    inputs = (
        *StitchScanSequence.inputs,
        Port("source_ids", "data", required=False, required_when_connected=True),
    )
    defaults = {
        key: value
        for key, value in StitchScanSequence.defaults.items()
        if key in _REGISTRATION_KEYS
    }
    defaults = {
        **defaults,
        "source_identity": "ordered",
        "source_id_field": "source_id",
        "source_ids": [],
    }
    parameter_choices = {
        key: value
        for key, value in StitchScanSequence.parameter_choices.items()
        if key in _REGISTRATION_KEYS
    }
    parameter_choices = {**parameter_choices, "source_identity": IDENTITIES}
    parameter_ranges = {
        key: value
        for key, value in StitchScanSequence.parameter_ranges.items()
        if key in _REGISTRATION_KEYS
    }
    parameter_help = {
        key: value
        for key, value in StitchScanSequence.parameter_help.items()
        if key in _REGISTRATION_KEYS
    }
    parameter_help = {
        **parameter_help,
        "inlier_pixels": "Fit threshold and maximum adjacent RMS after optional loop adjustment.",
        "source_identity": "Ordered: legacy. Other modes identify reusable views by stable IDs.",
        "source_id_field": "Payload key containing a stable source ID (top-level metadata).",
        "source_ids": "Explicit IDs in received view order; wired Source IDs overrides.",
    }
    input_help = {
        "views": "Two to 64 ordered overlapping images, at most 128 MiB total.",
        "masks": "Optional uint8 source masks for feature detection and tile validity.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        field = p["source_id_field"]
        if (
            not isinstance(field, str)
            or not field.strip()
            or len(field) > 128
            or any(ord(c) < 32 for c in field)
        ):
            raise ValidationError(
                "Choose a visible metadata key up to 128 characters", "source_id_field"
            )
        if not isinstance(p["source_ids"], list) or len(p["source_ids"]) > 64:
            raise ValidationError("Source IDs must be a list of at most 64 strings", "source_ids")

    def process(self, inputs):
        images, pixels, masks, contexts = _views(inputs)
        mode = self.parameters["source_identity"]
        ids = None
        if mode != "ordered":
            ids = source_ids(
                inputs["views"],
                contexts,
                mode,
                self.parameters["source_id_field"],
                inputs.get("source_ids", self.parameters["source_ids"]),
            )
        elif "source_ids" in inputs or self.parameters["source_ids"]:
            raise ValidationError(
                "Source IDs require a stable source identity mode", "source_identity"
            )
        layout, transforms, failure = register_scan(images, pixels, masks, self.parameters)
        if failure is not None:
            pair, reason = failure
            return Result(
                {"layout": layout},
                passed=False,
                fail_reasons=[reason],
                message=f"Views {pair['previous_index'] + 1} and {pair['view_index'] + 1}: "
                f"{pair['status'].replace('_', ' ')}",
            )
        size, to_mosaic = mosaic_geometry(pixels, transforms)
        _bounded_size(size)
        layout.update(model=_LAYOUT_MODEL, image_size=list(size), views=[])
        for index, (image, pixel, transform, label, context) in enumerate(
            zip(images, pixels, transforms, inputs["views"].labels, contexts, strict=True)
        ):
            layout["views"].append(
                {
                    "index": index,
                    "label": label,
                    "source_size": list(pixel.shape[1::-1]),
                    "source_dtype": pixel.dtype.name,
                    "source_format": image_format(image),
                    "source_to_mosaic": (to_mosaic @ transform).tolist(),
                    "source_spatial": context.get("spatial"),
                    "source_ticket": context.get("ticket"),
                }
            )
        if ids is not None:
            layout["model"] = MODEL
            layout["source_identity"] = {"mode": mode, "field": self.parameters["source_id_field"]}
            for record, source_id in zip(layout["views"], ids, strict=True):
                record["source_id"] = source_id
            layout = seal_layout(layout)
        return Result(
            {"layout": layout},
            message=f"Registered {len(images)} views for a {size[0]} × {size[1]} tiled mosaic",
        )


class RenderMosaicTile(ImageTool):
    type_id, title, subtype = "cv.render_mosaic_tile", "Render mosaic tile", "Geometry"
    process_inputs_readonly = True
    payload_reference = "views"
    inputs = (
        Port("views", "any"),
        Port("layout", "data"),
        Port("masks", "any", required=False, required_when_connected=True),
        Port("tile_index", "number", required=False, required_when_connected=True),
        Port("source_ids", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("coverage", visual_type="region"),
        Port("overlap", visual_type="region"),
        Port("tile", "data"),
    )
    defaults = {
        "tile_size": 512,
        "tile_index": 0,
        "interpolation": "linear",
        "blend": "prefer_first",
        "feather_width": 32,
        "source_ids": [],
        "missing_views": "error",
        "grid_policy": "strict",
        **MULTIBAND_DEFAULTS,
    }
    parameter_choices = {
        "interpolation": tuple(_INTERPOLATION),
        "blend": ("prefer_first", "prefer_last", "feather", "multiband"),
        "missing_views": ("error", "fail", "allow"),
        "grid_policy": ("strict", "pixel_indices"),
        **MULTIBAND_CHOICES,
    }
    parameter_ranges = {
        "tile_size": NumericRange(16, 2048, whole=True),
        "tile_index": NumericRange(0, _MAX_SIDE**2, whole=True),
        "feather_width": NumericRange(1, 512, whole=True),
        **MULTIBAND_RANGES,
    }
    parameter_help = {
        "tile_size": "Maximum width and height. Edge tiles are cropped to the mosaic bounds.",
        "tile_index": "Zero-based tile in row-major order. A connected input overrides this value.",
        "interpolation": "Linear output requires every contributing source pixel to be valid.",
        "blend": "Choose priority, feather or multiband. Filter halos preserve tile boundaries.",
        **MULTIBAND_HELP,
        "feather_width": "Distance from valid source edges, in mosaic pixels, to full weight.",
        "source_ids": "Optional IDs in received order, overriding the layout identity rule.",
        "missing_views": "Error: no output. Fail: failed partial. Allow: passing partial.",
        "grid_policy": "Verify recorded coordinates or explicitly assert calibrated pixel indices.",
    }
    input_help = {
        "views": "Legacy order, or stable-ID layouts with reordered/allowed missing views.",
        "layout": "Versioned layout from Register scan layout; no feature refitting is performed.",
        "masks": "Optional ordered uint8 validity masks in source coordinates.",
        "tile_index": "Optional row-major tile index for a loop or per-tile inspection.",
        "source_ids": "Optional explicit IDs for current group entries; not training order.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["source_ids"], list) or len(p["source_ids"]) > 64:
            raise ValidationError("Source IDs must be a list of at most 64 strings", "source_ids")

    def process(self, inputs):
        images, pixels, masks, contexts = _views(inputs, min_views=1)
        group = inputs["views"]
        stable = isinstance(inputs["layout"], dict) and inputs["layout"].get("model") == MODEL
        binding, selected = None, None
        if stable:
            size, transforms, selected, binding = bind_layout(
                inputs["layout"],
                group,
                images,
                pixels,
                contexts,
                self.parameters,
                inputs.get("source_ids"),
            )
            order = [current for _, current in selected]
            images, pixels, masks, contexts = (
                [values[i] for i in order] for values in (images, pixels, masks, contexts)
            )
        else:
            if "source_ids" in inputs or self.parameters["source_ids"]:
                raise ValidationError("Explicit source IDs require a v2 stable layout")
            size, transforms = _read_layout(inputs["layout"], group, images, pixels)
        width, height = size
        side = int(self.parameters["tile_size"])
        columns, rows = math.ceil(width / side), math.ceil(height / side)
        index = integer(
            inputs.get("tile_index", self.parameters["tile_index"]),
            "Tile index",
            0,
            columns * rows - 1,
        )
        x, y = index % columns * side, index // columns * side
        tile_size = (min(side, width - x), min(side, height - y))
        feather = self.parameters["blend"] == "feather"
        # DIST_MASK_3 advances about 0.955 pixels per horizontal step. A halo
        # this wide reaches the weight plateau before entering the output tile.
        halo = math.ceil(int(self.parameters["feather_width"]) / 0.95) + 2 if feather else 0
        left, top = max(0, x - halo), max(0, y - halo)
        right = min(width, x + tile_size[0] + halo)
        bottom = min(height, y + tile_size[1] + halo)
        if self.parameters["blend"] == "multiband":
            left, top, right, bottom, halo = tile_window(
                (x, y), tile_size, size, self.parameters["multiband_bands"]
            )
        render_size = (right - left, bottom - top)
        channels = 1 if pixels[0].ndim == 2 else pixels[0].shape[2]
        bytes_per_pixel = 5 * channels * pixels[0].itemsize + 32
        if feather:
            bytes_per_pixel += 32 + 24 * channels
        workspace = render_size[0] * render_size[1] * bytes_per_pixel
        if workspace > _MAX_WORKSPACE:
            raise ValidationError("Tile exceeds the 256 MiB workspace limit; reduce Tile size")
        shape = (render_size[1], render_size[0], *pixels[0].shape[2:])
        blender = FeatherMosaic(shape, int(self.parameters["feather_width"])) if feather else None
        if self.parameters["blend"] == "multiband":
            blender = MultibandMosaic(shape, pixels[0].dtype, self.parameters, base_bytes=workspace)
        mosaic = np.zeros(shape, dtype=pixels[0].dtype) if blender is None else None
        counts = np.zeros(shape[:2], dtype=np.uint8)
        to_window = np.eye(3)
        to_window[:2, 2] = [-left, -top]
        interpolation = _INTERPOLATION[self.parameters["interpolation"]]
        contributors = []
        area = np.s_[y - top : y - top + tile_size[1], x - left : x - left + tile_size[0]]
        for i, (image, mask, transform, context) in enumerate(
            zip(pixels, masks, transforms, contexts, strict=True)
        ):
            edges = _project_edges(image, transform)
            if (edges.max(axis=0) < [left - 1, top - 1]).any() or (
                edges.min(axis=0) > [right, bottom]
            ).any():
                continue
            local = to_window @ transform
            support = (
                np.ones(image.shape[:2], np.float32)
                if mask is None
                else (mask != 0).astype(np.float32)
            )
            valid = warp_mosaic_view(support, local, render_size, interpolation) == 1
            if not valid.any():
                continue
            warped = warp_mosaic_view(image, local, render_size, interpolation)
            if blender is not None:
                blender.add(warped, valid)
            else:
                use = valid if self.parameters["blend"] == "prefer_last" else valid & (counts == 0)
                np.copyto(mosaic, warped, where=use if channels == 1 else use[..., None])
            np.add(counts, valid, out=counts)
            np.minimum(counts, 2, out=counts)
            if not valid[area].any():
                continue
            contributors.append(
                {
                    "index": i,
                    "label": group.labels[i],
                    "source_to_mosaic": transform.tolist(),
                    "source_to_tile": (
                        np.array([[1.0, 0.0, -x], [0.0, 1.0, -y], [0.0, 0.0, 1.0]]) @ transform
                    ).tolist(),
                    "source_spatial": context.get("spatial"),
                    "source_ticket": context.get("ticket"),
                }
            )
            if stable:
                calibrated, current = selected[i]
                record = inputs["layout"]["views"][calibrated]
                contributors[-1].update(
                    index=calibrated,
                    input_index=current,
                    label=group.labels[current],
                    source_id=record["source_id"],
                    registered_label=record["label"],
                )
        if blender is not None:
            mosaic = blender.finish(pixels[0].dtype)
        if mosaic.dtype.kind == "f" and not np.isfinite(mosaic).all():
            raise ValidationError("Tile rendering produced nonfinite samples")
        # Copy the requested tile so a result never retains the larger halo arrays.
        image = ImageArray(mosaic[area].copy(), image_format(images[0]))
        counts = counts[area].copy()
        tile = {
            "model": _LAYOUT_MODEL,
            "index": index,
            "count": columns * rows,
            "columns": columns,
            "rows": rows,
            "origin": [x, y],
            "image_size": list(tile_size),
            "mosaic_size": list(size),
            "covered_pixels": int(np.count_nonzero(counts)),
            "overlap_pixels": int(np.count_nonzero(counts >= 2)),
            "views": contributors,
        }
        if stable:
            tile.update(
                model=MODEL, layout_id=inputs["layout"]["layout_id"], source_matching=binding
            )
        if self.parameters["blend"] == "multiband":
            tile["multiband"] = {
                **blender.info,
                "halo": halo,
                "window_origin": [left, top],
                "window_size": list(render_size),
            }
        context = {**self.input_payloads["views"]}
        context.pop("pyramid", None)
        context["spatial"] = (
            canvas_frame(inputs["layout"])
            if stable
            else mosaic_frame(contexts[0].get("spatial", {}), transforms[0], size, self.id)
        )
        tile_to_mosaic = np.eye(3)
        tile_to_mosaic[:2, 2] = [x, y]
        payload = moved_payload(context, tile_to_mosaic, tile_size)
        payload["mosaic"] = {
            key: tile[key] for key in ("origin", "image_size", "mosaic_size", "index")
        }
        if stable:
            payload["mosaic"].update(layout_id=tile["layout_id"], source_matching=binding)
        passed = not (
            stable and binding["missing_source_ids"] and self.parameters["missing_views"] == "fail"
        )
        return Result(
            {
                "image": image,
                "coverage": (counts != 0).astype(np.uint8) * 255,
                "overlap": (counts >= 2).astype(np.uint8) * 255,
                "tile": tile,
            },
            payloads=dict.fromkeys(("image", "coverage", "overlap", "tile"), payload),
            passed=passed,
            fail_reasons=[] if passed else ["mosaic_sources_missing"],
            message=f"Rendered tile {index + 1}/{columns * rows} at ({x}, {y})"
            + (f"; missing {len(binding['missing_source_ids'])} sources" if stable else ""),
        )
