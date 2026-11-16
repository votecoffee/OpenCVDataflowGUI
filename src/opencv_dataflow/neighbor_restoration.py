"""Conservative blur-model checked reuse of detail from registered neighbors."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .focus_stacking import focus_energy, luminance, stack_values
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket, check_spatial, frame, metadata
from .selection import selection_mask

_MAX_WORKSPACE = 256 * 1024**2


def _model(image, p):
    size = p["blur_size"]
    if p["blur_model"] == "gaussian":
        return cv2.GaussianBlur(
            image, (size, size), p["blur_sigma"], borderType=cv2.BORDER_REFLECT_101
        )
    shape = (size, 1) if p["blur_model"] == "horizontal_box" else (1, size)
    return cv2.blur(image, shape, borderType=cv2.BORDER_REFLECT_101)


def _erode(mask, radius):
    return cv2.erode(
        mask.astype(np.uint8),
        np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


class RestoreFromNeighbors(Block):
    type_id, title = "sequence.restore_neighbors", "Restore detail from neighbors"
    category = block_type = "Image processing"
    subtype = "Video restoration"
    parallel_safe = process_inputs_readonly = True
    payload_reference = "frames"
    inputs = (
        Port("frames", "any"),
        Port("coverage", "any", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("source"),
        Port("valid_mask", visual_type="region"),
        Port("restored_mask", visual_type="region"),
        Port("donor_index"),
        Port("contributor_count"),
        Port("contributions", "any"),
        Port("info", "data"),
        Port("preview", required=False),
    )
    defaults = {
        "target_index": 0,
        "temporal_radius": 2,
        "alignment": "same_frame",
        "blur_model": "horizontal_box",
        "blur_size": 7,
        "blur_sigma": 1.5,
        "maximum_model_error": 0.02,
        "change_radius": 3,
        "energy_window": 5,
        "minimum_energy": 1e-6,
        "minimum_sharpness_ratio": 1.1,
        "blend": "best",
        "mix": 1.0,
        "minimum_restored_fraction": 0.0,
        "no_neighbor": "pass_through",
    }
    parameter_choices = {
        "alignment": ("same_frame", "pixel_grid"),
        "blur_model": ("horizontal_box", "vertical_box", "gaussian"),
        "blend": ("best", "weighted"),
        "no_neighbor": ("pass_through", "fail"),
    }
    parameter_ranges = {
        "target_index": NumericRange(0, 15, whole=True),
        "temporal_radius": NumericRange(1, 15, whole=True),
        "blur_size": NumericRange(3, 63, whole=True, odd=True),
        "blur_sigma": NumericRange(0, 30),
        "maximum_model_error": NumericRange(0, 2),
        "change_radius": NumericRange(0, 63, whole=True),
        "energy_window": NumericRange(1, 63, whole=True, odd=True),
        "minimum_energy": NumericRange(0),
        "minimum_sharpness_ratio": NumericRange(1, 100),
        "mix": NumericRange(0.001, 1),
        "minimum_restored_fraction": NumericRange(0, 1),
    }
    parameter_help = {
        "target_index": (
            "Zero-based capture whose pixels, support, ticket and coordinate grid are retained."
        ),
        "temporal_radius": "Use only neighbors within this many batch positions of the target.",
        "alignment": (
            "Same frame requires every known matching root/grid; pixel grid explicitly asserts "
            "upstream registration."
        ),
        "blur_model": (
            "Expected target blur applied to a sharp neighbor for the scene-consistency check."
        ),
        "blur_size": "Odd box length or square Gaussian size in current pixels.",
        "blur_sigma": "Gaussian only; zero derives sigma from the kernel size.",
        "maximum_model_error": (
            "Maximum absolute channel difference between blurred donor and target, divided by "
            "shared scale."
        ),
        "change_radius": (
            "Exclude this halo around model mismatches, including changed or moving foreground."
        ),
        "energy_window": "Odd local averaging size for native 3x3 Sobel luminance energy.",
        "minimum_energy": (
            "Minimum donor energy in squared shared-scale luminance units; positive energy "
            "required."
        ),
        "minimum_sharpness_ratio": (
            "Donor energy must strictly exceed target energy times this factor."
        ),
        "blend": (
            "Best copies the highest-energy donor; weighted mixes all accepted donors by energy."
        ),
        "mix": (
            "Convex mix of restored detail and original target; one uses the accepted donor result."
        ),
        "minimum_restored_fraction": (
            "Required restored fraction of selected, originally supported target pixels."
        ),
        "no_neighbor": (
            "Pass through unchanged when no pixels qualify, or report an inspection failure."
        ),
    }
    input_help = {
        "frames": ("Batch of 2-16 registered gray/RGB(A)/BGR(A) uint8/16 or float32/64 images."),
        "coverage": (
            "Optional same-order uint8 support masks on each capture grid. Target support is "
            "never expanded."
        ),
        "mask": (
            "Target-grid output selection. Outside pixels are copied exactly; neighborhoods use "
            "full images."
        ),
        "region": "Optional target-grid region intersected with mask.",
    }
    output_help = {
        "image": "Restored target; original type, alpha, ticket and pixel grid.",
        "source": "Unmodified target capture.",
        "valid_mask": "Original target coverage intersected with nonzero target alpha.",
        "restored_mask": "Pixels actually replaced or mixed; all other target samples are exact.",
        "donor_index": "Strongest accepted batch index, or -1 where no donor was used.",
        "contributor_count": "Number of eligible donors; best mode uses only the strongest.",
        "contributions": "Target-grid masks of actual contributions, in original batch order.",
        "info": "Scale, support, eligible/used counts and contributing capture tickets.",
        "preview": "Optional colored donor-index map; omitted with result graphics off.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "restored_mask", "info"]
        self.documentation = (
            "<p>Restore a selected blurred target from sharper registered neighbors. The expected "
            "target blur is applied to each donor; channel mismatches and their configurable halo "
            "exclude detected changes. Entire scoring and blur neighborhoods must "
            "be supported. Best ties prefer batch order. Alpha stays the target's.</p>"
            "<p>This is an application equivalent, not the unavailable native WeightingDeblurer. "
            "It requires upstream registration and a suitable blur model. Small, low-contrast or "
            "model-indistinguishable changes can escape the gate; accepted pixels are not proof "
            "of unchanged parts. Use raw images for critical defect decisions; qualify tolerances "
            "on representative captures. Registration and blind deconvolution are separate.</p>"
        )

    def process(self, inputs):
        p, group = dict(self.parameters), inputs["frames"]
        # NumericRange accepts integral floats and numeric text for real-valued
        # fields. Use its typed results locally, without mutating saved values.
        for name, limits in self.parameter_ranges.items():
            p[name] = limits.check(p[name], name, name)
        # Bound workspace before float conversion, filters or full finite-value scans.
        if isinstance(group, FlowGroup) and group.items:
            raw = [entry.value for entry in group.items]
            if all(isinstance(a, np.ndarray) and a.ndim in (2, 3) for a in raw):
                h, w = raw[0].shape[:2]
                channels = 1 if raw[0].ndim == 2 else raw[0].shape[2]
                estimate = sum(a.nbytes for a in raw) + h * w * (256 + channels * 64 + len(raw))
                if estimate > _MAX_WORKSPACE:
                    raise ValidationError(
                        "Neighbor restoration exceeds the 256 MiB workspace budget"
                    )
        images, encoding = stack_values(group)
        n, target = len(images), p["target_index"]
        if target >= n:
            raise ValidationError("Target index is outside the frame batch", "target_index")
        contexts = [metadata(entry) for entry in group.items]
        if p["alignment"] == "same_frame":
            grids = [ctx.get("spatial") for ctx in contexts]
            if any(not grid or grid.get("root_id") is None for grid in grids):
                raise ValidationError(
                    "Same-frame mode requires every capture's known registered grid"
                )
            reference = grids[target]
            if any(
                grid["root_id"] != reference["root_id"]
                or grid["root_size"] != reference["root_size"]
                or not np.allclose(grid["to_root"], reference["to_root"], rtol=0, atol=1e-9)
                for grid in grids
            ):
                raise ValidationError("Register every capture onto the same coordinate frame first")
        source, context = images[target], contexts[target]
        h, w = source.shape[:2]
        coverage = inputs.get("coverage")
        if coverage is not None and (
            not isinstance(coverage, FlowGroup) or len(coverage.items) != n
        ):
            raise ValidationError("Coverage needs one ordered mask per frame")
        supports = []
        alpha = encoding in ("rgba", "bgra")
        for i, image in enumerate(images):
            support = np.ones((h, w), bool)
            if coverage is not None:
                entry, mask = coverage.items[i], coverage.items[i].value
                if (
                    not isinstance(mask, np.ndarray)
                    or mask.shape != (h, w)
                    or mask.dtype != np.uint8
                ):
                    raise ValidationError(
                        "Coverage must be uint8 on its corresponding capture grid"
                    )
                try:
                    check_spatial(metadata(entry), [w, h])
                except ValueError as exc:
                    raise ValidationError(str(exc)) from exc
                a, b = contexts[i].get("spatial"), metadata(entry).get("spatial")
                if b is not None and (a is None or b != a):
                    raise ValidationError("Coverage grid differs from its capture")
                support &= mask != 0
            if alpha:
                support &= image[..., 3] > 0
            supports.append(support)
        selected = selection_mask(
            {**inputs, "image": ImageArray(source, encoding, context)}, source
        )
        selected = np.ones((h, w), bool) if selected is None else selected != 0
        if not selected.any():
            raise ValidationError("Select at least one output pixel")
        usable = selected & supports[target]
        planes = [image[..., :3] if alpha else image for image in images]
        scale = (
            float(np.iinfo(source.dtype).max)
            if source.dtype.kind == "u"
            else (max(float(np.max(np.abs(a))) for a in planes) or 1.0)
        )
        target_signal = np.asarray(planes[target], np.float64) / scale
        focus_parameters = {
            "method": "tenengrad",
            "aperture": "3",
            "window": p["energy_window"],
            "prefilter": "off",
            "border": "reflect_101",
        }
        target_energy = focus_energy(luminance(source, encoding, scale), focus_parameters)
        halo = max(p["blur_size"] // 2, 1 + p["energy_window"] // 2)
        target_support = _erode(supports[target], halo)
        numerator = np.zeros_like(target_signal)
        total, best = np.zeros((h, w)), np.zeros((h, w))
        winner, count = np.full((h, w), -1, np.int32), np.zeros((h, w), np.uint8)
        records, masks = [], []
        for i, image in enumerate(images):
            accepted = np.zeros((h, w), bool)
            record = {"index": i, "label": group.labels[i], "ticket": contexts[i].get("ticket")}
            if i == target or abs(i - target) > p["temporal_radius"]:
                record["state"] = "target" if i == target else "outside_radius"
            else:
                signal = np.asarray(planes[i], np.float64) / scale
                predicted = _model(signal, p)
                error = np.abs(predicted - target_signal)
                if error.ndim == 3:
                    error = np.max(error, axis=2)
                mismatch = error > p["maximum_model_error"]
                if alpha:
                    mismatch |= image[..., 3] != source[..., 3]
                radius = p["change_radius"]
                changed = (
                    cv2.dilate(mismatch.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8))
                    != 0
                )
                energy = focus_energy(luminance(image, encoding, scale), focus_parameters)
                supported = usable & target_support & _erode(supports[i], halo)
                sharper = (
                    (energy > 0)
                    & (energy >= p["minimum_energy"])
                    & (energy > target_energy * p["minimum_sharpness_ratio"])
                )
                accepted = supported & ~changed & sharper
                better = accepted & (energy > best)
                if p["blend"] == "weighted":
                    # Track relative weights without multiplying tiny energy by
                    # tiny signal or overflowing a sum of unscaled responses.
                    rescale = np.ones((h, w))
                    np.divide(best, energy, out=rescale, where=better)
                    numerator *= rescale[..., None] if signal.ndim == 3 else rescale
                    total *= rescale
                winner[better], best[better] = i, energy[better]
                if p["blend"] == "best":
                    numerator[better] = signal[better]
                else:
                    weights = np.zeros((h, w))
                    np.divide(energy, best, out=weights, where=accepted)
                    numerator += signal * (weights[..., None] if signal.ndim == 3 else weights)
                    total += weights
                count += accepted
                record.update(
                    state="considered",
                    supported_pixels=int(supported.sum()),
                    changed_pixels=int(np.count_nonzero(supported & changed)),
                    not_sharper_pixels=int(np.count_nonzero(supported & ~sharper)),
                    maximum_model_error=float(error[supported].max()) if supported.any() else None,
                )
            masks.append(accepted)
            records.append(record)
        restored = count > 0
        if p["blend"] == "weighted":
            np.divide(
                numerator,
                total[..., None] if numerator.ndim == 3 else total,
                out=numerator,
                where=(total > 0)[..., None] if numerator.ndim == 3 else total > 0,
            )
        mixed = ((1 - p["mix"]) * target_signal + p["mix"] * numerator).clip(-1, 1) * scale
        if source.dtype.kind == "u":
            mixed = np.rint(mixed).clip(0, np.iinfo(source.dtype).max)
        output = np.asarray(source).copy()
        destination = output[..., :3] if alpha else output
        destination[restored] = mixed[restored].astype(source.dtype)
        if p["blend"] == "best" and p["mix"] == 1:
            # Full best-neighbor replacement preserves even Float64 source bits.
            for i, pixels in enumerate(planes):
                chosen = winner == i
                destination[chosen] = pixels[chosen]
        fraction = float(restored.sum() / usable.sum()) if usable.any() else 0.0
        passed = bool(
            usable.any()
            and fraction >= p["minimum_restored_fraction"]
            and (restored.any() or p["no_neighbor"] == "pass_through")
        )
        payload = copy.deepcopy(context)
        for key in ("outcome", "processing_time"):
            payload.pop(key, None)
        payload.setdefault("spatial", frame([w, h]))
        payload["neighbor_restoration"] = {
            "target_index": target,
            "frames": n,
            "blur_model": p["blur_model"],
        }
        map_payload = copy.deepcopy(payload)
        map_payload.pop("alpha_representation", None)
        contributions = []
        for i, accepted in enumerate(masks):
            contributing = accepted if p["blend"] == "weighted" else winner == i
            records[i].update(
                eligible_pixels=int(accepted.sum()), contributed_pixels=int(contributing.sum())
            )
            contributions.append(
                ValuePacket(ImageArray(np.uint8(contributing) * 255, "gray"), map_payload)
            )
        info = {
            "model": "neighbor_restoration_v1",
            "target_index": target,
            "frames": n,
            "blur_model": p["blur_model"],
            "scale": scale,
            "support_halo": halo,
            "selected_supported_pixels": int(usable.sum()),
            "restored_pixels": int(restored.sum()),
            "restored_fraction": fraction,
            "neighbors": records,
            "state": "restored" if restored.any() else "unchanged",
        }
        outputs = {
            "image": ImageArray(output, encoding),
            "source": source,
            "valid_mask": np.uint8(supports[target]) * 255,
            "restored_mask": np.uint8(restored) * 255,
            "donor_index": winner,
            "contributor_count": count,
            "contributions": FlowGroup(tuple(contributions), group.labels),
            "info": info,
            "preview": NO_VALUE,
        }
        graphics, drawings = {}, {}
        if self.generate_graphics:
            preview = cv2.applyColorMap(
                np.uint8(np.maximum(winner, 0) * 255 / max(1, n - 1)), cv2.COLORMAP_TURBO
            )
            preview[~restored] = 0
            outputs["preview"] = ImageArray(preview, "bgr")
            graphics["contributing neighbors"] = outputs["preview"]
            label = graphic_object(
                "text",
                [[5, 16]],
                (60, 230, 120) if passed else (60, 90, 240),
                text=f"Restored {fraction:.1%}",
            )
            drawings["neighbor restoration"] = GraphicDrawing((label,), "image", True)
        return Result(
            outputs,
            payloads={
                **{name: map_payload for name in outputs},
                "image": payload,
                "source": context,
            },
            graphics=graphics,
            drawings=drawings,
            passed=passed,
            score=fraction,
            fail_reasons=[] if passed else ["neighbor_restoration_insufficient_detail"],
            message=f"Restored {int(restored.sum())}/{int(usable.sum())} supported selected pixels",
        )
