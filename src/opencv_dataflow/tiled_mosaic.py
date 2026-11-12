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
from .mosaic_photometry import FeatherMosaic
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


def _views(inputs):
    unpacked = _unpack(inputs["views"], inputs.get("masks"), max_views=_MAX_VIEWS)
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
    defaults = {
        key: value
        for key, value in StitchScanSequence.defaults.items()
        if key in _REGISTRATION_KEYS
    }
    parameter_choices = {
        key: value
        for key, value in StitchScanSequence.parameter_choices.items()
        if key in _REGISTRATION_KEYS
    }
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
    }
    input_help = {
        "views": "Two to 64 ordered overlapping images, at most 128 MiB total.",
        "masks": "Optional uint8 source masks for feature detection and tile validity.",
    }

    def process(self, inputs):
        images, pixels, masks, contexts = _views(inputs)
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
    }
    parameter_choices = {
        "interpolation": tuple(_INTERPOLATION),
        "blend": ("prefer_first", "prefer_last", "feather"),
    }
    parameter_ranges = {
        "tile_size": NumericRange(16, 2048, whole=True),
        "tile_index": NumericRange(0, _MAX_SIDE**2, whole=True),
        "feather_width": NumericRange(1, 512, whole=True),
    }
    parameter_help = {
        "tile_size": "Maximum width and height. Edge tiles are cropped to the mosaic bounds.",
        "tile_index": "Zero-based tile in row-major order. A connected input overrides this value.",
        "interpolation": "Linear output requires every contributing source pixel to be valid.",
        "blend": "Choose source priority or feather edges. Feather halos preserve tile boundaries.",
        "feather_width": "Distance from valid source edges, in mosaic pixels, to full weight.",
    }
    input_help = {
        "views": "The same ordered source grids and labels used for registration.",
        "layout": "Versioned layout from Register scan layout; no feature refitting is performed.",
        "masks": "Optional ordered uint8 validity masks in source coordinates.",
        "tile_index": "Optional row-major tile index for a loop or per-tile inspection.",
    }

    def process(self, inputs):
        images, pixels, masks, contexts = _views(inputs)
        group = inputs["views"]
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
        context = {**self.input_payloads["views"]}
        context.pop("pyramid", None)
        context["spatial"] = mosaic_frame(
            contexts[0].get("spatial", {}), transforms[0], size, self.id
        )
        tile_to_mosaic = np.eye(3)
        tile_to_mosaic[:2, 2] = [x, y]
        payload = moved_payload(context, tile_to_mosaic, tile_size)
        payload["mosaic"] = {
            key: tile[key] for key in ("origin", "image_size", "mosaic_size", "index")
        }
        return Result(
            {
                "image": image,
                "coverage": (counts != 0).astype(np.uint8) * 255,
                "overlap": (counts >= 2).astype(np.uint8) * 255,
                "tile": tile,
            },
            payloads=dict.fromkeys(("image", "coverage", "overlap", "tile"), payload),
            message=f"Rendered tile {index + 1}/{columns * rows} at ({x}, {y})",
        )
