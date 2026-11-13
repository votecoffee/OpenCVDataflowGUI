"""CPU source-over composition with explicit alpha association and image context."""

import copy

import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, draw_object, graphic_object
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, metadata, require_same_frame
from .processing import ImageTool, image_value
from .selection import selection_mask
from .tickets import ticket_key

FORMATS = ("gray", "bgr", "rgb", "bgra", "rgba")
DEPTHS = ("uint8", "uint16", "float32", "float64")


def unit_samples(image):
    values = np.asarray(image, np.float64)
    return values / np.iinfo(image.dtype).max if image.dtype.kind == "u" else values


def split_layer(image, requested, name):
    raw = image_value(image, (np.uint8, np.uint16, np.float32, np.float64))
    encoding = image_format(image)
    if encoding not in FORMATS:
        raise ValidationError(f"{name} needs grayscale or RGB(A)/BGR(A) pixels")
    expected_channels = 1 if encoding == "gray" else 4 if encoding in ("bgra", "rgba") else 3
    actual_channels = 1 if raw.ndim == 2 else raw.shape[2]
    if actual_channels != expected_channels:
        raise ValidationError(f"{name} encoding does not match its channel layout")
    values = unit_samples(raw)
    has_alpha = raw.ndim == 3 and raw.shape[2] == 4
    alpha = values[..., 3] if has_alpha else np.ones(raw.shape[:2], np.float64)
    if np.any((alpha < 0) | (alpha > 1)):
        raise ValidationError(f"{name} floating alpha must be in [0,1]")
    representation = requested
    if representation == "auto":
        representation = (
            metadata(image).get("alpha_representation", "straight") if has_alpha else "straight"
        )
    if representation not in ("straight", "premultiplied"):
        raise ValidationError(f"{name} has an unknown alpha representation; choose it explicitly")
    if representation == "premultiplied" and not has_alpha:
        raise ValidationError(f"{name} premultiplied input needs an embedded alpha channel")
    if raw.ndim == 2:
        rgb = np.repeat(values[..., None], 3, axis=2)
    else:
        rgb = values[..., :3]
        if encoding in ("bgr", "bgra"):
            rgb = rgb[..., ::-1]
    if representation == "premultiplied" and np.any(rgb[alpha == 0] != 0):
        raise ValidationError(f"{name} premultiplied colors must be zero where alpha is zero")
    return rgb, alpha, representation


def unassociate(colors, alpha):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        output = np.divide(
            colors, alpha[..., None], out=np.zeros_like(colors), where=alpha[..., None] > 0
        )
    if not np.isfinite(output).all():
        raise ValidationError(
            "Unassociated colors exceed float64; rescale or retain premultiplied data"
        )
    return output


def srgb_to_linear(colors):
    if np.any((colors < 0) | (colors > 1)):
        raise ValidationError("Linear sRGB blending requires straight colors in [0,1]")
    return np.where(colors <= 0.04045, colors / 12.92, ((colors + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(colors):
    return np.where(
        colors <= 0.0031308, colors * 12.92, 1.055 * np.maximum(colors, 0) ** (1 / 2.4) - 0.055
    )


def associated_working(colors, alpha, representation, space):
    if space == "stored":
        return colors if representation == "premultiplied" else colors * alpha[..., None]
    straight = unassociate(colors, alpha) if representation == "premultiplied" else colors
    return srgb_to_linear(straight) * alpha[..., None]


def encoded_colors(associated, alpha, representation, space):
    if space == "stored" and representation == "premultiplied":
        return associated
    straight = unassociate(associated, alpha)
    if space == "linear_srgb":
        straight = linear_to_srgb(straight)
    return straight if representation == "straight" else straight * alpha[..., None]


def pack_image(rgb, alpha, encoding, dtype):
    colors = (
        rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        if encoding == "gray"
        else rgb[..., ::-1]
        if encoding in ("bgr", "bgra")
        else rgb
    )
    values = np.dstack((colors, alpha)) if encoding in ("bgra", "rgba") else colors
    if not np.isfinite(values).all():
        raise ValidationError("Composited colors are nonfinite; rescale floating-point images")
    clipped = int(np.count_nonzero((values < 0) | (values > 1))) if dtype.kind == "u" else 0
    if dtype.kind == "u":
        values = np.rint(np.clip(values, 0, 1) * np.iinfo(dtype).max)
    elif np.any(np.abs(values) > np.finfo(dtype).max):
        raise ValidationError(
            "Composited colors exceed the output depth; choose float64 or rescale"
        )
    return ImageArray(values.astype(dtype), encoding), clipped


class AlphaOver(ImageTool):
    type_id, title = "cv.alpha_over", "Source-over alpha composition"
    subtype = "Image math"
    payload_reference = "background"
    inputs = (
        Port("foreground"),
        Port("background"),
        Port("alpha", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("alpha"),
        Port("info", "data"),
        Port("foreground_alpha"),
        Port("preview", required=False),
    )
    defaults = {
        "foreground_representation": "auto",
        "background_representation": "auto",
        "output_representation": "straight",
        "output_format": "background",
        "output_depth": "background",
        "blend_space": "stored",
        "opacity": 1.0,
        "alignment": "same_frame",
        "ticket_policy": "same_ticket",
    }
    parameter_choices = {
        "foreground_representation": ("auto", "straight", "premultiplied"),
        "background_representation": ("auto", "straight", "premultiplied"),
        "output_representation": ("straight", "premultiplied"),
        "output_format": ("background", *FORMATS),
        "output_depth": ("background", *DEPTHS),
        "blend_space": ("stored", "linear_srgb"),
        "alignment": ("same_frame", "pixel_grid"),
        "ticket_policy": ("same_ticket", "background"),
    }
    parameter_ranges = {"opacity": NumericRange(0, 1)}
    parameter_help = {
        "foreground_representation": (
            "Auto reads alpha_representation metadata; untagged RGBA is straight."
        ),
        "background_representation": "Association of background colors with its embedded alpha.",
        "output_representation": (
            "Premultiplied output requires BGRA/RGBA and carries association metadata."
        ),
        "output_format": "Preserve the background encoding, or explicitly convert the result.",
        "output_depth": (
            "Preserve background depth or choose a depth; integer colors use full-scale units."
        ),
        "blend_space": (
            "Blend stored component values or decode/encode unit-range sRGB around blending."
        ),
        "opacity": "Multiply foreground embedded alpha, optional alpha map, and associated colors.",
        "alignment": (
            "Same frame validates identity; pixel grid explicitly pairs equal-size image grids."
        ),
        "ticket_policy": (
            "Require matching known tickets, or explicitly use background ownership across sources."
        ),
    }
    input_help = {
        "foreground": (
            "Top layer; gray/RGB(A)/BGR(A) uint8/16 or float32/64, registered with background."
        ),
        "background": (
            "Canvas layer and output coordinate/ticket reference. Mixed "
            "depths/orders are converted."
        ),
        "alpha": "Optional matching uint8/16 or floating [0,1] map multiplying foreground alpha.",
        "mask": (
            "Optional uint8 selection of output centers; other pixels represent the background."
        ),
        "region": "Optional circle, rectangle or polygon intersected with the selection mask.",
    }
    process_inputs_readonly = True
    parallel_safe = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "alpha", "info"]
        self.documentation = (
            "<p>Source-over composition uses Ao = Af + Ab(1-Af) and associated color "
            "Co = Cf Af + Cb Ab(1-Af). Straight output divides Co by Ao where Ao is positive. "
            "Fully transparent selected pixels have zero color.</p>"
            '<p><a href="https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators">'
            "W3C source-over reference</a>. This CPU implementation handles alpha association, "
            "pixel depths, color order and optional linear sRGB blending explicitly.</p>"
        )

    def process(self, inputs):
        p, foreground, background = self.parameters, inputs["foreground"], inputs["background"]
        h, w = background.shape[:2]
        if h * w > 2_100_000 or max(h, w) > 16384:
            raise ValidationError("Alpha composition is limited to 2.1 MP and 16,384 per axis")
        if foreground.shape[:2] != (h, w):
            raise ValidationError("Foreground and background must use matching pixel grids")
        sources = [foreground, background]
        if "alpha" in inputs:
            sources.append(inputs["alpha"])
        if p["alignment"] == "same_frame":
            require_same_frame(sources, message="Alpha layers/map use different coordinate frames")
        contexts = [metadata(value) for value in sources]
        known = [c for c in contexts if "ticket" in c]
        if (
            p["ticket_policy"] == "same_ticket"
            and len({ticket_key(c["ticket"]) for c in known}) > 1
        ):
            raise ValidationError(
                "Alpha layers/map use different tickets; explicitly choose background policy"
            )
        fg, fa, fg_repr = split_layer(foreground, p["foreground_representation"], "Foreground")
        bg, ba, bg_repr = split_layer(background, p["background_representation"], "Background")
        encoding = (
            image_format(background) if p["output_format"] == "background" else p["output_format"]
        )
        dtype = (
            background.dtype if p["output_depth"] == "background" else np.dtype(p["output_depth"])
        )
        association = p["output_representation"]
        if association == "premultiplied" and encoding not in ("bgra", "rgba"):
            raise ValidationError("Premultiplied output requires BGRA/RGBA to retain alpha")
        selection = selection_mask(
            {"image": background, **{k: inputs[k] for k in ("mask", "region") if k in inputs}},
            background,
        )
        selected = np.ones((h, w), bool) if selection is None else selection != 0
        factor = np.full((h, w), float(p["opacity"]), np.float64)
        if "alpha" in inputs:
            matte = image_value(inputs["alpha"], (np.uint8, np.uint16, np.float32, np.float64))
            if matte.shape != (h, w):
                raise ValidationError("Alpha map must be monochrome with matching dimensions")
            matte = unit_samples(matte)
            if np.any((matte < 0) | (matte > 1)):
                raise ValidationError("Floating alpha map must be in [0,1]")
            factor *= matte
        factor *= selected
        effective = fa * factor
        alpha = np.clip(effective + ba * (1 - effective), 0, 1)
        fg_work = associated_working(fg, fa, fg_repr, p["blend_space"])
        bg_work = associated_working(bg, ba, bg_repr, p["blend_space"])
        with np.errstate(over="ignore", invalid="ignore"):
            associated = fg_work * factor[..., None] + bg_work * (1 - effective[..., None])
        colors = encoded_colors(associated, alpha, association, p["blend_space"])
        image, clipped = pack_image(colors, alpha, encoding, dtype)
        # Preserve exact endpoint/background samples when representation and storage agree.
        if (
            encoding == image_format(background)
            and dtype == background.dtype
            and association == bg_repr
        ):
            take = (~selected) | ((effective == 0) & (ba > 0))
            image[take] = np.asarray(background)[take]
        if (
            encoding == image_format(foreground)
            and dtype == foreground.dtype
            and association == fg_repr
        ):
            image[effective == 1] = np.asarray(foreground)[effective == 1]
        quantized_zero = (
            selected & (image[..., 3] == 0)
            if encoding in ("rgba", "bgra")
            else np.zeros((h, w), bool)
        )
        if quantized_zero.any():
            image[quantized_zero, :3] = 0
        report = {
            "model": "alpha_composition_v1",
            "operator": "source_over",
            "foreground_representation": fg_repr,
            "background_representation": bg_repr,
            "output_representation": association,
            "output_format": encoding,
            "output_depth": str(dtype),
            "blend_space": p["blend_space"],
            "opacity": float(p["opacity"]),
            "selected_pixels": int(selected.sum()),
            "active_foreground_pixels": int((effective > 0).sum()),
            "transparent_pixels": int((alpha == 0).sum()),
            "opaque_pixels": int((alpha == 1).sum()),
            "clipped_samples": clipped,
            "quantized_zero_alpha_pixels": int(quantized_zero.sum()),
            "sources": [
                {k: copy.deepcopy(c[k]) for k in ("ticket", "spatial") if k in c}
                for c in contexts[:2]
            ],
        }
        context = copy.deepcopy(contexts[1])
        if "ticket" not in context and known:
            for key in ("ticket", "processing_time"):
                if key in known[0]:
                    context[key] = copy.deepcopy(known[0][key])
        context.pop("alpha_representation", None)
        context["alpha_composition"] = copy.deepcopy(report)
        image_context = {**context, "alpha_representation": association}
        values = {
            "image": image,
            "alpha": ImageArray(alpha, "gray"),
            "info": report,
            "foreground_alpha": ImageArray(effective, "gray"),
            "preview": NO_VALUE,
        }
        payloads = {
            "image": image_context,
            "alpha": context,
            "info": context,
            "foreground_alpha": context,
        }
        if self.generate_graphics:
            # Render actual transparency for diagnostics, independent of image widget alpha support.
            stored = unit_samples(image)
            display_alpha = (
                stored[..., 3] if encoding in ("rgba", "bgra") else np.ones((h, w), np.float64)
            )
            rgb = np.repeat(stored[..., None], 3, axis=2) if encoding == "gray" else stored[..., :3]
            if encoding in ("bgr", "bgra"):
                rgb = rgb[..., ::-1]
            display_associated = (
                rgb if association == "premultiplied" else rgb * display_alpha[..., None]
            )
            y, x = np.mgrid[:h, :w]
            checker = np.where(((x // 12 + y // 12) % 2) != 0, 0.72, 0.48)
            view = np.clip(
                display_associated + checker[..., None] * (1 - display_alpha[..., None]), 0, 1
            )
            preview = np.rint(view[..., ::-1] * 255).astype(np.uint8)
            label = graphic_object("label", [[8, 18]], (70, 220, 140), text="Source over")
            draw_object(preview, label, np.eye(3))
            result = Result(
                values, payloads=payloads, message="Composed foreground over background"
            )
            result.drawings["Alpha composition"] = GraphicDrawing((label,), from_output=True)
            values["preview"] = preview
            payloads["preview"] = context
            result.graphics["Alpha composition / checkerboard"] = preview
            return result
        return Result(values, payloads=payloads, message="Composed foreground over background")
