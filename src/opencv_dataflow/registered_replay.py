"""Independent registered frames on one grid, with exact per-frame validity."""

import copy

import cv2
import numpy as np

from .core import Port, Result
from .image_stitching import mosaic_frame, warp_mosaic_view
from .images import ImageArray, image_format
from .mosaic_layout import MODEL, bind_layout, canvas_frame
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket, moved_payload
from .processing import ImageTool
from .tiled_mosaic import _read_layout, _views

_INTERPOLATION = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}
_MAX_SIDE = 16384


def support_bounds(mask):
    """Bound occupied rows/columns without allocating an Nx2 point list."""
    ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not len(xs):
        return None
    return [int(xs[0]), int(ys[0]), int(xs[-1] - xs[0] + 1), int(ys[-1] - ys[0] + 1)]


def area(bounds):
    x, y, w, h = bounds
    return np.s_[y : y + h, x : x + w]


def native_timelapser(bounds, size, crop):
    # The native API clips rectangles, not arbitrary masks. Exact support is
    # applied before placement and after its rectangular intersection crop.
    boxes = [box for box in bounds if box is not None]
    if crop == "canvas":
        boxes = [*boxes, [0, 0, *size]]
    corners = [(b[0], b[1]) for b in boxes]
    sizes = [(b[2], b[3]) for b in boxes]
    if crop == "intersection":
        # The wheel's direct TimelapserCrop constructor faults; its factory
        # correctly initializes the derived native object (isolated audit).
        native = cv2.detail.Timelapser.createDefault(cv2.detail.Timelapser_CROP)
        x, y = max(b[0] for b in boxes), max(b[1] for b in boxes)
        right = min(b[0] + b[2] for b in boxes)
        bottom = min(b[1] + b[3] for b in boxes)
    else:
        native = cv2.detail.Timelapser.createDefault(cv2.detail.Timelapser_AS_IS)
        x, y = min(b[0] for b in boxes), min(b[1] for b in boxes)
        right = max(b[0] + b[2] for b in boxes)
        bottom = max(b[1] + b[3] for b in boxes)
    native.initialize(corners, sizes)
    return native, [x, y, right - x, bottom - y]


class ReplayRegisteredViews(ImageTool):
    type_id, title, subtype = "cv.replay_registered_views", "Replay registered views", "Geometry"
    process_inputs_readonly = True
    parallel_safe = True
    payload_reference = "views"
    flow_primary = ("frames",)
    inputs = (
        Port("views", "any"),
        Port("layout", "data"),
        Port("masks", "any", required=False, required_when_connected=True),
        Port("source_ids", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("frames", "any"),
        Port("coverage", "any", visual_type="region"),
        Port("image"),
        Port("common", visual_type="region"),
        Port("info", "data"),
    )
    defaults = {
        "crop": "canvas",
        "backend": "preserve",
        "interpolation": "linear",
        "frame_order": "received",
        "preview_frame": 0,
        "source_ids": [],
        "missing_views": "error",
        "grid_policy": "strict",
        "maximum_workspace_mib": 256,
    }
    parameter_choices = {
        "crop": ("canvas", "union", "intersection"),
        "backend": ("preserve", "native_detail"),
        "interpolation": tuple(_INTERPOLATION),
        "frame_order": ("received", "calibrated"),
        "missing_views": ("error", "fail", "allow"),
        "grid_policy": ("strict", "pixel_indices"),
    }
    parameter_ranges = {
        "preview_frame": NumericRange(0, 63, whole=True),
        "maximum_workspace_mib": NumericRange(1, 256, whole=True),
    }
    parameter_help = {
        "crop": "Full canvas, tight union, or common-valid bounding crop with holes zeroed.",
        "backend": "Preserve supports layout types. Native detail requires uint8 gray/RGB/BGR.",
        "interpolation": "Linear requires every contributing source sample to be valid.",
        "frame_order": "Received preserves acquisition order; calibrated uses saved order.",
        "preview_frame": "Zero-based output frame shown as Image; does not discard other frames.",
        "source_ids": "Stable IDs in received order; wired IDs override the saved identity rule.",
        "missing_views": "Error: no output. Fail: failed partial. Allow: incomplete replay.",
        "grid_policy": "Verify saved coordinates, or explicitly assert calibrated pixel indices.",
        "maximum_workspace_mib": "Replay-buffer estimate; snapshots/runtime copies are additional.",
    }
    input_help = {
        "views": "One to 64 current images; member tickets/timestamps remain attached.",
        "layout": "Successful v1/v2 planar layout. No feature fitting or blending is performed.",
        "masks": "Optional uint8 validity masks, in received image order and source coordinates.",
        "source_ids": "Optional explicit stable IDs for received entries, requiring a v2 layout.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["source_ids"], list) or len(p["source_ids"]) > 64:
            raise ValidationError("Source IDs must be a list of at most 64 strings", "source_ids")

    def process(self, inputs):
        images, pixels, masks, contexts = _views(inputs, min_views=1)
        group, layout, p = inputs["views"], inputs["layout"], self.parameters
        stable = isinstance(layout, dict) and layout.get("model") == MODEL
        binding = None
        if stable:
            size, transforms, selected, binding = bind_layout(
                layout, group, images, pixels, contexts, p, inputs.get("source_ids")
            )
        else:
            if "source_ids" in inputs or p["source_ids"]:
                raise ValidationError("Explicit source IDs require a v2 stable layout")
            size, transforms = _read_layout(layout, group, images, pixels)
            selected = [(i, i) for i in range(len(images))]
        if max(size) > _MAX_SIDE:
            raise ValidationError(
                "Replay canvas exceeds 16,384 pixels per side; use tiled rendering"
            )
        if p["preview_frame"] >= len(selected):
            raise ValidationError("Preview frame is outside the received group", "preview_frame")
        native_backend = p["backend"] == "native_detail"
        if native_backend and (
            pixels[0].dtype != np.uint8 or pixels[0].shape[2:] not in ((), (3,))
        ):
            raise ValidationError(
                "Native detail replay requires uint8 gray/RGB/BGR images", "backend"
            )
        channels = 1 if pixels[0].ndim == 2 else pixels[0].shape[2]
        sample_bytes = channels * pixels[0].itemsize
        estimate = (
            size[0]
            * size[1]
            * (
                2 * len(selected) * (sample_bytes + 1)
                + 6 * sample_bytes
                + 32
                + (36 if native_backend else 0)
            )
        )
        if estimate > p["maximum_workspace_mib"] * 1024**2:
            raise ValidationError(
                "Replay exceeds workspace budget; use tiled rendering or fewer views"
            )
        entries = list(zip(selected, transforms, strict=True))
        if p["frame_order"] == "received":
            entries.sort(key=lambda item: item[0][1])
        interpolation = _INTERPOLATION[p["interpolation"]]
        supports, boxes = [], []
        union, common = np.zeros(size[::-1], bool), np.ones(size[::-1], bool)
        for (_, current), transform in entries:
            support = np.ones(pixels[current].shape[:2], np.float32)
            if masks[current] is not None:
                support[:] = masks[current] != 0
            valid = warp_mosaic_view(support, transform, size, interpolation) == 1
            supports.append(valid)
            boxes.append(support_bounds(valid))
            union |= valid
            common &= valid
        union_box, common_box = support_bounds(union), support_bounds(common)
        crop = (
            [0, 0, *size]
            if p["crop"] == "canvas"
            else union_box
            if p["crop"] == "union"
            else common_box
        )
        info = {
            "model": "registered_view_replay_v1",
            "layout_model": layout["model"],
            "layout_id": layout.get("layout_id"),
            "backend": p["backend"],
            "crop": p["crop"],
            "frame_order": p["frame_order"],
            "mosaic_size": list(size),
            "union_bounds": union_box,
            "common_bounds": common_box,
            "union_pixels": int(np.count_nonzero(union)),
            "common_pixels": int(np.count_nonzero(common)),
            "workspace_estimate_bytes": estimate,
            "source_matching": binding,
            "frames": [],
        }
        base = (
            canvas_frame(layout)
            if stable
            else mosaic_frame(contexts[0].get("spatial", {}), transforms[0], size, self.id)
        )
        if union_box is None or crop is None:
            info.update(
                status="no_common_support" if p["crop"] == "intersection" else "no_valid_support"
            )
            return Result(
                {
                    "frames": NO_VALUE,
                    "coverage": NO_VALUE,
                    "image": NO_VALUE,
                    "common": common.astype(np.uint8) * 255,
                    "info": info,
                },
                payloads={"common": {"spatial": base}},
                passed=False,
                fail_reasons=["replay_" + info["status"]],
                message=info["status"].replace("_", " "),
            )
        x, y, w, h = crop
        output_to_canvas = np.array([[1.0, 0, x], [0, 1.0, y], [0, 0, 1.0]])
        spatial = moved_payload({"spatial": base}, output_to_canvas, [w, h])["spatial"]
        common = common[area(crop)].copy()
        native = None
        try:
            if native_backend:
                native, native_box = native_timelapser(boxes, size, p["crop"])
                info["native_bounds"] = native_box
            frames, coverage, labels = [], [], []
            for output_index, (((calibrated, current), transform), valid, box) in enumerate(
                zip(entries, supports, boxes, strict=True)
            ):
                warped = warp_mosaic_view(pixels[current], transform, size, interpolation)
                warped[~valid] = 0
                if native is not None and box is not None:
                    source = warped[area(box)].astype(np.int16)
                    if source.ndim == 2:
                        source = np.repeat(source[..., None], 3, axis=2)
                    native.process(source, np.full(source.shape[:2], 255, np.uint8), tuple(box[:2]))
                    placed = native.getDst().get()
                    local_crop = [x - native_box[0], y - native_box[1], w, h]
                    output = placed[area(local_crop)].astype(np.uint8)
                    if pixels[current].ndim == 2:
                        output = output[..., 0].copy()
                else:
                    output = warped[area(crop)].copy()
                valid = valid[area(crop)].copy()
                if p["crop"] == "intersection":
                    valid &= common
                output[~valid] = 0
                context = copy.deepcopy(contexts[current])
                source_pyramid = context.pop("pyramid", None)
                record = {
                    "index": output_index,
                    "calibrated_index": calibrated,
                    "input_index": current,
                    "label": group.labels[current],
                    "source_id": layout["views"][calibrated].get("source_id"),
                    "source_ticket": copy.deepcopy(context.get("ticket")),
                    "source_spatial": copy.deepcopy(context.get("spatial")),
                    "source_pyramid": source_pyramid,
                    "source_to_frame": (np.linalg.inv(output_to_canvas) @ transform).tolist(),
                    "source_to_mosaic": transform.tolist(),
                    "valid_pixels": int(np.count_nonzero(valid)),
                }
                info["frames"].append(record)
                context.update(
                    spatial=spatial,
                    registered_replay={
                        "layout_id": layout.get("layout_id"),
                        "origin": [x, y],
                        "mosaic_size": list(size),
                        "crop": p["crop"],
                        **record,
                    },
                )
                frames.append(
                    ValuePacket(ImageArray(output, image_format(images[current])), context)
                )
                coverage.append(ValuePacket(valid.astype(np.uint8) * 255, copy.deepcopy(context)))
                labels.append(group.labels[current])
        finally:
            native = None
        info.update(status="ready", origin=[x, y], image_size=[w, h])
        failed = (
            binding is not None and binding["missing_source_ids"] and p["missing_views"] == "fail"
        )
        preview = frames[p["preview_frame"]]
        return Result(
            {
                "frames": FlowGroup(tuple(frames), tuple(labels)),
                "coverage": FlowGroup(tuple(coverage), tuple(labels)),
                "image": preview.value,
                "common": common.astype(np.uint8) * 255,
                "info": info,
            },
            payloads={"image": preview.payload, "common": {"spatial": spatial}},
            passed=not bool(failed),
            fail_reasons=["mosaic_sources_missing"] if failed else [],
            message=f"Replayed {len(frames)} independent views on a {w} × {h} common grid",
        )
