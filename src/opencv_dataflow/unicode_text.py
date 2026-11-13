"""FreeType glyph coverage composited directly into the original sample depth."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

DEFAULT_FONT = Path(__file__).with_name("fonts") / "NotoSans-Regular.ttf"
MAX_TEXT_PIXELS = 1_048_576


@dataclass(frozen=True)
class TextStyle:
    font_file: str = ""
    font_size: int = 18
    font_index: int = 0
    direction: str = "auto"
    outline: int = 1


def text_masks(text, style):
    """Return foreground/outline alpha masks and their baseline-relative bounds.

    Fonts are local to this invocation, so branch workers share no mutable font
    handles. Only glyph coverage is rasterized; no source image goes through PIL.
    """
    text = "".join(" " if ord(c) < 32 or 127 <= ord(c) < 160 else c for c in text[:160])
    if not text.strip():
        return None
    if not 8 <= style.font_size <= 128 or not 0 <= style.outline <= 8:
        raise ValueError("Text size must be 8..128 and outline 0..8 pixels")
    if style.direction not in ("auto", "ltr", "rtl"):
        raise ValueError("Text direction must be auto, ltr or rtl")
    raqm = features.check_feature("raqm")
    if style.direction != "auto" and not raqm:
        raise ValueError("Explicit text direction requires a Pillow build with Raqm")
    options = {"direction": style.direction} if style.direction != "auto" else {}
    layout = ImageFont.Layout.RAQM if raqm else ImageFont.Layout.BASIC
    # Loading through bytes avoids keeping a font file open on Windows.
    path = Path(style.font_file) if style.font_file else DEFAULT_FONT
    try:
        font = ImageFont.truetype(
            BytesIO(path.read_bytes()),
            style.font_size,
            index=style.font_index,
            layout_engine=layout,
        )
    except OSError as exc:
        raise ValueError(f"Cannot load text font {path.name}: {exc}") from exc
    bounds = font.getbbox(text, anchor="ls", stroke_width=style.outline, **options)
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        return None
    if width * height > MAX_TEXT_PIXELS:
        raise ValueError("Text exceeds the one-megapixel glyph coverage budget")
    masks = []
    for stroke in (0, style.outline):
        mask = Image.new("L", (width, height))
        ImageDraw.Draw(mask).text(
            (-left, -top),
            text,
            font=font,
            fill=255,
            anchor="ls",
            stroke_width=stroke,
            **options,
        )
        masks.append(np.asarray(mask))
    return (*masks, bounds)


def _blend(image, coverage, color):
    active = coverage != 0
    if not active.any():
        return
    target = image[..., :3] if image.ndim == 3 else image
    color = np.asarray(color, np.float64)
    if target.ndim == 2:
        color = float(color @ [0.114, 0.587, 0.299])
    alpha = coverage[active].astype(np.float64) / 255
    if target.ndim == 3:
        alpha = alpha[:, None]
    with np.errstate(invalid="ignore"):
        values = target[active].astype(np.float64) * (1 - alpha) + color * alpha
    # Exact opaque glyph pixels must not inherit a NaN/Inf from the background.
    values = np.where(alpha == 1, color, values)
    if np.issubdtype(target.dtype, np.integer):
        limits = np.iinfo(target.dtype)
        values = np.clip(np.rint(values), limits.min, limits.max)
    target[active] = values.astype(target.dtype)


def draw_unicode(image, text, anchor, color, style=None, *, label=True):
    style = style or TextStyle()
    masks = text_masks(text, style)
    if masks is None:
        return
    foreground, outline, (left, top, _, _) = masks
    height, width = image.shape[:2]
    rows, cols = foreground.shape
    x, y = map(int, anchor)
    if label:
        x = max(0, min(x, width - cols - 2))
        y = y - rows - 4 if y >= rows + 4 else y + 4
        y = max(0, min(y, height - rows))
    else:
        x, y = x + left, y + top
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + cols), min(height, y + rows)
    if x0 >= x1 or y0 >= y1:
        return
    crop = np.s_[y0 - y : y1 - y, x0 - x : x1 - x]
    target = image[y0:y1, x0:x1]
    if style.outline:
        _blend(target, outline[crop], (0, 0, 0))
    _blend(target, foreground[crop], color)
