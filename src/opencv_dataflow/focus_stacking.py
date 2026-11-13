"""Focus fusion of registered image batches with explicit sampling and ambiguity maps."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS
from .payloads import NO_VALUE, FlowGroup, check_spatial, frame, metadata, require_same_frame
from .processing import image_value
from .selection import selection_mask


def stack_values(group):
    if not isinstance(group, FlowGroup) or not 2 <= len(group.items) <= 16:
        raise ValidationError("Focus stacking needs a batch of 2–16 registered images")
    images = [entry.value for entry in group.items]
    if any(not isinstance(image, np.ndarray) or image.ndim not in (2, 3) for image in images):
        raise ValidationError("Every focus entry must be an image")
    shape = images[0].shape
    if min(shape[:2]) < 2 or max(shape[:2]) > 16384 or shape[0] * shape[1] > 4_194_304:
        raise ValidationError(
            "Images need 2 pixels per axis, at most 4,194,304 pixels and 16384 per axis"
        )
    if (
        sum(image.shape[0] * image.shape[1] for image in images) > 16_777_216
        or sum(image.size for image in images) > 64_000_000
    ):
        raise ValidationError("Stack exceeds 16,777,216 image pixels or 64 million channel samples")
    encoding, depth = image_format(images[0]), images[0].dtype
    channels = () if encoding == "gray" else (4,) if encoding in ("rgba", "bgra") else (3,)
    if encoding not in ("gray", "rgb", "bgr", "rgba", "bgra") or shape[2:] != channels:
        raise ValidationError(
            "Use grayscale or RGB(A)/BGR(A) images with matching channel metadata"
        )
    for entry in group.items:
        image = entry.value
        if image.shape != shape or image.dtype != depth or image_format(image) != encoding:
            raise ValidationError("Focus images must share size, depth and channel order")
        image_value(image, (np.uint8, np.uint16, np.float32, np.float64))
        try:
            check_spatial(metadata(entry), list(shape[1::-1]))
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        if len(channels) and channels[0] == 4:
            if metadata(entry).get("alpha_representation", "straight") != "straight":
                raise ValidationError("Unassociate premultiplied focus images first")
            maximum = np.iinfo(depth).max if depth.kind == "u" else 1
            if np.any(image[..., 3] < 0) or np.any(image[..., 3] > maximum):
                raise ValidationError("Alpha must lie within integer full scale or floating [0,1]")
    return images, encoding


def luminance(image, encoding, scale):
    signal = image[..., :3] if image.ndim == 3 else image
    values = np.asarray(signal, np.float64) / scale
    if values.ndim == 2:
        return values
    weights = (0.299, 0.587, 0.114) if encoding.startswith("rgb") else (0.114, 0.587, 0.299)
    return np.sum(values[..., :3] * weights, axis=2)


def focus_energy(gray, p):
    border = BORDERS[p["border"]]
    if p["prefilter"] != "off":
        k = int(p["prefilter"])
        gray = cv2.GaussianBlur(gray, (k, k), p["prefilter_sigma"], borderType=border)
    options = {"ksize": int(p["aperture"]), "borderType": border}
    if p["method"] == "tenengrad":
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, **options)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, **options)
        energy = dx * dx + dy * dy
    else:
        derivative = cv2.Laplacian(gray, cv2.CV_64F, **options)
        energy = derivative * derivative
    size = int(p["window"])
    energy = cv2.boxFilter(energy, -1, (size, size), normalize=True, borderType=border)
    if not np.isfinite(energy).all():
        raise ValidationError("Focus response is not finite")
    return np.maximum(energy, 0)


class FocusStack(Block):
    type_id, title = "cv.focus_stack", "Focus stack images"
    category = block_type = "Image processing"
    subtype = "Image fusion"
    parallel_safe = process_inputs_readonly = True
    payload_reference = "images"
    inputs = (
        Port("images", "any"),
        Port("coverage", "any", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("focus_index"),
        Port("valid_mask", visual_type="region"),
        Port("confidence"),
        Port("contributor_count"),
        Port("response"),
        Port("info", "data"),
        Port("source"),
        Port("preview", required=False),
    )
    defaults = {
        "anchor_index": 0,
        "alignment": "same_frame",
        "method": "tenengrad",
        "aperture": "3",
        "window": 9,
        "prefilter": "off",
        "prefilter_sigma": 0.0,
        "border": "reflect_101",
        "support": "strict",
        "minimum_contributors": 2,
        "minimum_energy": 1e-6,
        "minimum_confidence": 0.05,
        "minimum_valid_fraction": 0.0,
        "blend": "hard",
        "blend_power": 2.0,
        "weight_radius": 0,
    }
    parameter_choices = {
        "alignment": ("same_frame", "pixel_grid"),
        "method": ("tenengrad", "laplacian_energy"),
        "aperture": ("1", "3", "5", "7"),
        "prefilter": ("off", "3", "5", "7", "9", "15", "31"),
        "border": ("constant", "replicate", "reflect_101", "reflect_edge"),
        "support": ("strict", "extrapolate"),
        "blend": ("hard", "weighted"),
    }
    parameter_ranges = {
        "anchor_index": NumericRange(0, 15, whole=True),
        "window": NumericRange(1, 101, whole=True, odd=True),
        "prefilter_sigma": NumericRange(0, 50),
        "minimum_contributors": NumericRange(1, 16, whole=True),
        "minimum_energy": NumericRange(0),
        "minimum_confidence": NumericRange(0, 1),
        "minimum_valid_fraction": NumericRange(0, 1),
        "blend_power": NumericRange(0.1, 16),
        "weight_radius": NumericRange(0, 31, whole=True),
    }
    parameter_help = {
        "anchor_index": "Defines the output grid, alpha and fallback pixels.",
        "alignment": "Check known grids, or declare equal pixel indices registered upstream.",
        "method": "Local mean squared Sobel gradients or squared Laplacian of luminance.",
        "aperture": "Derivative kernel size. Keep it fixed when comparing energy thresholds.",
        "window": "Odd local energy averaging window; neighborhoods retain full-image context.",
        "prefilter": "Optional Gaussian kernel before derivatives; Off performs no prefiltering.",
        "prefilter_sigma": "Gaussian sigma; zero derives it from the selected kernel size.",
        "border": "Native filter extrapolation at image boundaries.",
        "support": "Require captured neighborhoods, or permit image-edge filter padding.",
        "minimum_contributors": "Required number of captures with supported scoring neighborhoods.",
        "minimum_energy": "Inclusive energy floor after shared scaling; positive energy required.",
        "minimum_confidence": "Inclusive (best-second)/best contrast; not a probability or depth.",
        "minimum_valid_fraction": "Required reliable selected fraction; at least one pixel.",
        "blend": "Copy the winning capture exactly, or blend with relative focus-energy weights.",
        "blend_power": "Weighted mode only: exponent applied to each score relative to the best.",
        "weight_radius": "Weighted mode only: box-smooth weights with radius 0–31 pixels.",
    }
    input_help = {
        "images": "Batch of 2–16 registered uint8/16 or float32/64 intensity images.",
        "coverage": "Optional same-order batch of uint8 support masks, before derivative erosion.",
        "mask": "Output-center selection on the anchor grid; outside pixels retain the anchor.",
        "region": "Anchor-grid selection intersected with the mask. Does not cut neighborhoods.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "focus_index", "valid_mask", "info"]
        self.documentation = (
            "<p>Fuse registered captures using local luminance focus energy. Align images upstream "
            "with registration/warp blocks, supplying coverage masks for resampling gaps. "
            "Focus indices are original zero-based batch indices, with -1 where no reliable fusion "
            "occurs. Exact ties prefer the anchor, then batch order. Flat/ambiguous pixels retain "
            "the anchor; they do not silently acquire an invented focus level.</p>"
            "<p>Unsigned images share integer full scale; floating images share one batch peak. "
            "Scores use this scale, not independent per-frame normalization. Strict support erodes "
            "coverage by the prefilter/derivative/window halo. Alpha is excluded from scoring and "
            "retained from the anchor; transparent samples cannot supply neighborhoods. Weighted "
            "fusion may soften detail, while hard fusion retains original pixel values. Focus "
            "level/confidence are diagnostics, not surface depth or visibility proof.</p>"
            '<p><a href="https://docs.opencv.org/5.0/main_modules/imgproc_filter.html">'
            "OpenCV filter reference</a>.</p>"
        )

    def process(self, inputs):
        p, group = self.parameters, inputs["images"]
        images, encoding = stack_values(group)
        n, index = len(images), int(p["anchor_index"])
        if index >= n or p["minimum_contributors"] > n:
            raise ValidationError("Anchor or contributor count is outside the image batch")
        if p["alignment"] == "same_frame":
            try:
                require_same_frame(group.items)
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
        anchor, context = images[index], metadata(group.items[index])
        h, w = anchor.shape[:2]
        selected = selection_mask(
            {**inputs, "image": ImageArray(anchor, encoding, context)}, anchor
        )
        selected = np.ones((h, w), bool) if selected is None else selected != 0
        if not selected.any():
            raise ValidationError("Select at least one output pixel")
        coverage = inputs.get("coverage")
        if coverage is not None and (
            not isinstance(coverage, FlowGroup) or len(coverage.items) != n
        ):
            raise ValidationError("Coverage must have one ordered mask for every focus image")
        alpha = encoding in ("rgba", "bgra")
        planes = [np.asarray(image)[..., :3] if alpha else np.asarray(image) for image in images]
        scale = (
            float(np.iinfo(anchor.dtype).max)
            if anchor.dtype.kind == "u"
            else max(float(np.max(np.abs(value))) for value in planes)
        )
        scale = scale or 1.0
        halo = (
            (0 if p["prefilter"] == "off" else int(p["prefilter"]) // 2)
            + max(1, int(p["aperture"]) // 2)
            + p["window"] // 2
        )
        kernel = np.ones((2 * halo + 1, 2 * halo + 1), np.uint8)
        best, second = np.full((h, w), -1.0), np.full((h, w), -1.0)
        winner, count = np.full((h, w), -1, np.int32), np.zeros((h, w), np.uint8)
        scores, supports = {}, {}
        for i in [index, *[j for j in range(n) if j != index]]:
            support = np.ones((h, w), np.uint8)
            if coverage is not None:
                entry = coverage.items[i]
                value = entry.value
                if (
                    not isinstance(value, np.ndarray)
                    or value.shape != (h, w)
                    or value.dtype != np.uint8
                ):
                    raise ValidationError("Each coverage mask must be uint8 on its capture grid")
                try:
                    check_spatial(metadata(entry), [w, h])
                    require_same_frame([group.items[i], entry])
                except ValueError as exc:
                    raise ValidationError(str(exc)) from exc
                support &= value != 0
            if alpha:
                support &= images[i][..., 3] > 0
            support = cv2.erode(
                support,
                kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0 if p["support"] == "strict" else 1,
            ).astype(bool)
            energy = focus_energy(luminance(images[i], encoding, scale), p)
            score = np.where(support, energy, -1)
            better = score > best
            second = np.where(better, best, np.maximum(second, score))
            best = np.where(better, score, best)
            winner[better] = i
            count += support
            if p["blend"] == "weighted":
                scores[i], supports[i] = energy, support
        confidence = np.zeros((h, w), np.float64)
        np.divide(best - second, best, out=confidence, where=(best > 0) & (count > 1))
        confidence[(best > 0) & (count == 1)] = 1
        confidence = np.clip(confidence, 0, 1)
        valid = (
            selected
            & (count >= p["minimum_contributors"])
            & (best > 0)
            & (best >= p["minimum_energy"])
            & (confidence >= p["minimum_confidence"])
        )
        if alpha:
            valid &= anchor[..., 3] > 0
        output = np.asarray(anchor).copy()
        destination = output[..., :3] if alpha else output
        if p["blend"] == "hard":
            for i, pixels in enumerate(planes):
                chosen = valid & (winner == i)
                destination[chosen] = pixels[chosen]
        else:
            numerator = np.zeros(planes[index].shape, np.float64)
            total = np.zeros((h, w), np.float64)
            for i, pixels in enumerate(planes):
                ratio = np.zeros((h, w), np.float64)
                np.divide(scores[i], best, out=ratio, where=valid & supports[i])
                weights = np.where(valid & supports[i], ratio.clip(0, 1) ** p["blend_power"], 0)
                radius = int(p["weight_radius"])
                if radius:
                    weights = cv2.boxFilter(
                        weights,
                        -1,
                        (2 * radius + 1, 2 * radius + 1),
                        borderType=BORDERS[p["border"]],
                    )
                weights *= valid & supports[i]
                numerator += (
                    pixels.astype(np.float64)
                    / scale
                    * (weights[..., None] if numerator.ndim == 3 else weights)
                )
                total += weights
            normalizer = np.maximum(total, np.finfo(float).tiny)
            fused = numerator / (normalizer[..., None] if numerator.ndim == 3 else normalizer)
            fused = fused.clip(-1, 1) * scale
            if anchor.dtype.kind == "u":
                fused = np.rint(fused).clip(0, np.iinfo(anchor.dtype).max)
            destination[valid] = fused[valid].astype(anchor.dtype)
        fraction = float(valid.sum() / selected.sum())
        passed = bool(valid.any() and fraction >= p["minimum_valid_fraction"])
        payload = copy.deepcopy(context)
        for key in ("outcome", "processing_time"):
            payload.pop(key, None)
        payload.update(
            copy.deepcopy(
                {
                    key: value
                    for key, value in self.input_payloads["images"].items()
                    if key in ("ticket", "recipe_ref")
                }
            )
        )
        payload.setdefault("spatial", frame([w, h]))
        payload["focus_stack"] = {
            "frames": n,
            "anchor_index": index,
            "blend": p["blend"],
            "units": "current_pixel",
        }
        map_payload = copy.deepcopy(payload)
        map_payload.pop("alpha_representation", None)
        info = {
            "model": "focus_stack_v1",
            "frames": n,
            "anchor_index": index,
            "method": p["method"],
            "blend": p["blend"],
            "score_scale": scale,
            "score_units": "squared_scaled_luminance",
            "support_halo": int(halo),
            "selected_pixels": int(selected.sum()),
            "valid_pixels": int(valid.sum()),
            "valid_fraction": fraction,
            "rejection_counts": {
                "insufficient_support": int(
                    np.count_nonzero(selected & (count < p["minimum_contributors"]))
                ),
                "low_energy": int(
                    np.count_nonzero(selected & ((best <= 0) | (best < p["minimum_energy"])))
                ),
                "ambiguous": int(
                    np.count_nonzero(selected & (confidence < p["minimum_confidence"]))
                ),
                "transparent_anchor": int(np.count_nonzero(selected & (anchor[..., 3] <= 0)))
                if alpha
                else 0,
            },
            "mean_confidence": float(confidence[valid].mean()) if valid.any() else None,
            "chosen_pixels": [int(np.count_nonzero(valid & (winner == i))) for i in range(n)],
            "sources": [
                {
                    "index": i,
                    "label": group.labels[i],
                    "ticket": metadata(entry).get("ticket"),
                    "spatial": metadata(entry).get("spatial"),
                }
                for i, entry in enumerate(group.items)
            ],
        }
        outputs = {
            "image": ImageArray(output, encoding),
            "focus_index": np.where(valid, winner, -1).astype(np.int32),
            "valid_mask": np.uint8(valid) * 255,
            "confidence": np.where(selected, confidence, 0),
            "contributor_count": np.where(selected, count, 0).astype(np.uint8),
            "response": np.where(selected & (count > 0), best, 0),
            "info": info,
            "source": anchor,
            "preview": NO_VALUE,
        }
        graphics, drawings = {}, {}
        if self.generate_graphics:
            labels = np.uint8(np.maximum(outputs["focus_index"], 0) * (255 / max(1, n - 1)))
            preview = cv2.applyColorMap(labels, cv2.COLORMAP_TURBO)
            preview[~valid] = 0
            outputs["preview"] = ImageArray(preview, "bgr")
            graphics["focus levels"] = outputs["preview"]
            label = graphic_object(
                "text",
                [[5, 16]],
                (60, 230, 120) if passed else (60, 90, 240),
                text=f"Focus coverage {fraction:.1%}",
            )
            drawings["focus coverage"] = GraphicDrawing((label,), "image", True)
        contexts = {name: map_payload for name in outputs}
        contexts.update(image=payload, source=context)
        return Result(
            outputs,
            payloads=contexts,
            graphics=graphics,
            drawings=drawings,
            passed=passed,
            score=fraction,
            fail_reasons=[] if passed else ["focus_stack_insufficient_coverage"],
            message=f"Fused {int(valid.sum())}/{int(selected.sum())} selected pixels",
        )
