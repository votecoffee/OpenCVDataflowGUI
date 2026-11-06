"""Stitch two registered inspection views with explicit coverage and coordinates."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame, metadata, moved_payload
from .processing import ImageTool, image_value
from .registration import matrix_value
from .selection import selection_mask
from .tickets import ticket_key

_INTERPOLATION = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}
_MAX_SIDE = 16_384
_MAX_WORKSPACE = 256 * 1024**2


def _transform(value, source_size=None, target_size=None):
    if isinstance(value, dict):
        for side, actual in (("source", source_size), ("target", target_size)):
            recorded = value.get(f"{side}_image_size", value.get(f"{side}_size"))
            if recorded is not None and actual is not None and recorded != list(actual):
                raise ValidationError("Transform must map Image B to Image A with matching sizes")
        value = value.get("source_to_target")
    try:
        matrix = np.asarray(value, np.float64)
        if matrix.shape == (2, 3):
            matrix = np.vstack((matrix, [0, 0, 1]))
        return matrix_value(matrix, (3, 3))
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValidationError(f"Invalid Image B to A transform: {exc}", "matrix") from exc


def _project_edges(image, to_anchor):
    h, w = image.shape[:2]
    # Work from the outer pixel edges so a magnified view keeps its full footprint.
    corners = np.array(
        [[-0.5, -0.5, 1], [w - 0.5, -0.5, 1], [-0.5, h - 0.5, 1], [w - 0.5, h - 0.5, 1]]
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        projected = corners @ to_anchor.T
        denominator = projected[:, 2]
        if not (np.all(denominator > 0) or np.all(denominator < 0)):
            raise ValidationError("Transform crosses infinity inside a mosaic view")
        return projected[:, :2] / denominator[:, None]


def mosaic_canvas(images, transforms, *, extra_bytes_per_pixel=0):
    edges = np.concatenate(
        [_project_edges(image, matrix) for image, matrix in zip(images, transforms, strict=True)]
    )
    # Round out at output pixel edges. Tiny numerical drift at integral bounds
    # does not need an additional uncovered row or column.
    lower_edge, upper_edge = edges.min(axis=0) + 0.5, edges.max(axis=0) - 0.5
    with np.errstate(over="ignore", invalid="ignore"):
        lower = np.floor(
            np.where(
                np.abs(lower_edge - np.rint(lower_edge)) < 1e-9, np.rint(lower_edge), lower_edge
            )
        )
        upper = np.ceil(
            np.where(
                np.abs(upper_edge - np.rint(upper_edge)) < 1e-9, np.rint(upper_edge), upper_edge
            )
        )
        size = upper - lower + 1
    if not np.isfinite(size).all() or np.any(size > _MAX_SIDE):
        raise ValidationError(f"Mosaic exceeds the {_MAX_SIDE:,}-pixel dimension limit")
    width, height = map(int, size)
    first = images[0]
    channels = 1 if first.ndim == 2 else first.shape[2]
    bytes_per_pixel = 5 * channels * first.itemsize + 10 + extra_bytes_per_pixel
    if width * height * bytes_per_pixel > _MAX_WORKSPACE:
        raise ValidationError("Mosaic exceeds the 256 MiB workspace limit")
    a_to_mosaic = np.eye(3)
    a_to_mosaic[:2, 2] = -lower
    return (width, height), a_to_mosaic


def _selected(inputs, side):
    source = inputs[f"image_{side}"]
    mask = selection_mask(
        {
            "image": source,
            "mask": inputs.get(f"mask_{side}"),
            "region": inputs.get(f"region_{side}"),
        },
        source,
    )
    if mask is not None and not mask.any():
        raise ValidationError(f"Image {side.upper()} selection must include at least one pixel")
    return np.ones(source.shape[:2], dtype=bool) if mask is None else mask != 0


def warp_mosaic_view(image, matrix, size, interpolation):
    return cv2.warpPerspective(
        image, matrix, size, flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def mosaic_frame(anchor, anchor_to_mosaic, size, block_id):
    if anchor.get("root_id") is None:
        return frame(size, f"{block_id}:image")
    return moved_payload({"spatial": anchor}, np.linalg.inv(anchor_to_mosaic), size)["spatial"]


class StitchImagePair(ImageTool):
    type_id, title, subtype = "cv.stitch_image_pair", "Stitch image pair", "Geometry"
    process_inputs_readonly = True
    payload_reference = "image_a"
    inputs = (
        Port("image_a"),
        Port("image_b"),
        Port("transform", "data", required=False, required_when_connected=True),
        Port("mask_a", required=False, required_when_connected=True),
        Port("region_a", "data", required=False, required_when_connected=True),
        Port("mask_b", required=False, required_when_connected=True),
        Port("region_b", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("coverage", visual_type="region"),
        Port("overlap", visual_type="region"),
        Port("layout", "data"),
    )
    defaults = {
        "transform_source": "shared_frame",
        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "interpolation": "linear",
        "blend": "prefer_a",
        "min_overlap_pixels": 1,
        "ticket_policy": "same_ticket",
    }
    parameter_choices = {
        "transform_source": ("shared_frame", "matrix"),
        "interpolation": tuple(_INTERPOLATION),
        "blend": ("prefer_a", "prefer_b", "average"),
        "ticket_policy": ("same_ticket", "anchor_a"),
    }
    parameter_ranges = {"min_overlap_pixels": NumericRange(0, _MAX_SIDE**2, whole=True)}
    parameter_help = {
        "transform_source": "Use a common recorded image frame or the editable matrix.",
        "matrix": "Affine or projective matrix mapping Image B pixels into Image A coordinates.",
        "interpolation": "Resample Image B. Linear requires all contributing pixels to be valid.",
        "blend": "Choose which image to use in the overlap or average their samples.",
        "min_overlap_pixels": "Withhold the mosaic below this count. Zero permits disjoint tiles.",
        "ticket_policy": "Anchor A allows distinct tickets for offline stitching and reports both.",
    }
    input_help = {
        "image_a": "Anchor view. Its pixel grid defines the mosaic scale and orientation.",
        "image_b": "Second view with matching pixel depth and channel encoding.",
        "transform": "Optional Image B to Image A transform. Overrides Transform source.",
        "mask_a": "Select valid pixels in Image A coordinates.",
        "mask_b": "Select valid pixels in Image B coordinates.",
        "region_a": "Optional region in Image A coordinates.",
        "region_b": "Optional region in Image B coordinates.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["transform_source"] == "matrix":
            _transform(p["matrix"])

    def registration(self, inputs, a_frame, b_frame, first, second):
        if "transform" in inputs:
            return _transform(inputs["transform"], second.shape[1::-1], first.shape[1::-1])
        if self.parameters["transform_source"] == "matrix":
            return _transform(self.parameters["matrix"])
        if (
            a_frame.get("root_id") is None
            or a_frame["root_id"] != b_frame.get("root_id")
            or a_frame["root_size"] != b_frame["root_size"]
        ):
            raise ValidationError("Images need a shared coordinate frame or a B to A transform")
        return _transform(np.linalg.solve(a_frame["to_root"], b_frame["to_root"]))

    def process(self, inputs):
        a, b = inputs["image_a"], inputs["image_b"]
        first, second = image_value(a), image_value(b)
        encoding = image_format(a)
        if (
            first.dtype != second.dtype
            or first.shape[2:] != second.shape[2:]
            or encoding != image_format(b)
            or encoding not in ("gray", "bgr", "rgb", "bgra", "rgba")
        ):
            raise ValidationError("Use matching grayscale or RGB/BGR views with the same depth")
        context_a, context_b = metadata(a), metadata(b)
        ticket_a, ticket_b = context_a.get("ticket"), context_b.get("ticket")
        p = self.parameters
        if p["ticket_policy"] == "same_ticket" and ticket_a and ticket_b:
            if ticket_key(ticket_a) != ticket_key(ticket_b):
                raise ValidationError("Stitch images from the same ticket or choose Anchor A")
        a_frame, b_frame = context_a.get("spatial", {}), context_b.get("spatial", {})
        b_to_a = self.registration(inputs, a_frame, b_frame, first, second)
        size, a_to_mosaic = mosaic_canvas((first, second), (np.eye(3), b_to_a))
        b_to_mosaic = a_to_mosaic @ b_to_a
        width, height = size
        x, y = map(int, a_to_mosaic[:2, 2])
        ah, aw = first.shape[:2]
        a_area = np.s_[y : y + ah, x : x + aw]
        a_selected, b_selected = _selected(inputs, "a"), _selected(inputs, "b")
        a_valid = np.zeros((height, width), dtype=bool)
        a_valid[a_area] = a_selected
        interpolation = _INTERPOLATION[p["interpolation"]]
        b_valid = (
            warp_mosaic_view(b_selected.astype(np.float32), b_to_mosaic, size, interpolation) == 1
        )
        overlap = a_valid & b_valid
        coverage = a_valid | b_valid
        overlap_count = int(np.count_nonzero(overlap))
        layout = {
            "model": "image_mosaic_v1",
            "image_size": list(size),
            "covered_pixels": int(np.count_nonzero(coverage)),
            "overlap_pixels": overlap_count,
            "views": [
                {
                    "input": side,
                    "source_to_mosaic": transform.tolist(),
                    "source_spatial": context.get("spatial"),
                    "source_ticket": context.get("ticket"),
                }
                for side, transform, context in (
                    ("image_a", a_to_mosaic, context_a),
                    ("image_b", b_to_mosaic, context_b),
                )
            ],
        }
        primary = context_a if ticket_a else context_b
        payload = {key: primary[key] for key in ("ticket", "processing_time") if key in primary}
        payload["spatial"] = mosaic_frame(a_frame, a_to_mosaic, size, self.id)

        passed = bool(b_valid.any()) and overlap_count >= int(p["min_overlap_pixels"])
        output = NO_VALUE
        if passed:
            mosaic = np.zeros((height, width, *first.shape[2:]), dtype=first.dtype)
            condition = a_selected if first.ndim == 2 else a_selected[..., None]
            np.copyto(mosaic[a_area], first, where=condition)
            warped_b = warp_mosaic_view(second, b_to_mosaic, size, interpolation)
            blend = p["blend"]
            if blend == "average" and overlap_count:
                values_a, values_b = mosaic[overlap], warped_b[overlap]
                mixed = cv2.addWeighted(values_a, 0.5, values_b, 0.5, 0)
                mosaic[overlap] = mixed.reshape(values_a.shape)
            use_b = b_valid if blend == "prefer_b" else b_valid & ~a_valid
            condition = use_b if second.ndim == 2 else use_b[..., None]
            np.copyto(mosaic, warped_b, where=condition)
            if mosaic.dtype.kind == "f" and not np.isfinite(mosaic).all():
                raise ValidationError("Stitching produced nonfinite samples")
            output = ImageArray(mosaic, encoding)
        return Result(
            {
                "image": output,
                "coverage": coverage.astype(np.uint8) * 255,
                "overlap": overlap.astype(np.uint8) * 255,
                "layout": layout,
            },
            payloads=dict.fromkeys(("image", "coverage", "overlap", "layout"), payload),
            passed=passed,
            fail_reasons=[] if passed else ["stitch_insufficient_overlap"],
            message=f"{width} × {height} mosaic canvas · {overlap_count} overlapping pixels",
        )
