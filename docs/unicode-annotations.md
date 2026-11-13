# Unicode image annotations

**Add annotation layer** creates frame-aware drawing objects. Select **Label**
and enter text or connect the **Text** input, then connect **Result graphics**
to **Overlay result graphics**. Layer selection and ticket/frame checks apply
as with other result graphics. Text remains an object until composition; it is
not a separate result image.

The FreeType renderer uses [Pillow](https://pillow.readthedocs.io/en/stable/reference/ImageFont.html)
and works without a Qt application, display server or GUI thread. It rasterizes
glyph coverage only, and blends onto the original samples. Unsigned 8/16-bit and
floating color canvases retain their sample depth. RGB/BGR order and alpha are
preserved by the overlay block; grayscale composition uses a three-channel
canvas. Float colors use the existing 0..1 drawing convention. Source pixels
outside glyph/outline coverage are untouched. The input image is never modified.

| Parameter | Meaning |
| --- | --- |
| Font file | Empty uses bundled Noto Sans. Otherwise a project-relative or absolute TTF/OTF/TTC path. Include your font and its license with a shared project. |
| Font size | 8..128 rendered pixels. Coordinate transforms move the anchor; they do not rotate or scale the glyphs. |
| Font index | 0..255; face index in a collection, usually zero. |
| Text direction | Auto uses Raqm when available, basic layout otherwise. Explicit ltr/rtl requires Raqm. |
| Text outline | 0..8 rendered pixels of black outline. Zero disables it. |

The anchor follows the source image frame through crops, resize/pyramid and
other supported transforms. Labels prefer space above the anchor, otherwise
below it, and are clamped to the image. Oversize labels clip without changing
their font size or stored text. Labels allow up to 160 characters; control
characters become spaces for display. Rendering has a one-megapixel coverage
budget. Font handles are local to each invocation for parallel branch safety.
Invalid/unreadable fonts and unsupported explicit direction fail the annotation
block through the ordinary exception/validation diagnostics, without emissions.

The bundled [Noto Sans font](../src/opencv_dataflow/fonts/README.md) covers
Latin, Greek and Cyrillic. It is not a universal font. Unsupported characters
use the chosen font's missing-glyph symbol; there is no automatic font fallback.
Arabic/Indic and other complex scripts require both an appropriate font and a
Pillow build with Raqm. Auto layout on a basic-only build does not establish
correct complex-script rendering. The current Windows installation is basic-only;
the public fixture qualifies Latin, Greek and Cyrillic, not complex scripts.

Existing ASCII result labels from other tools retain OpenCV Hershey rendering.
Non-ASCII result labels/text use the bundled FreeType renderer automatically,
including OCR/barcode result graphics and filmstrip overlays. Plugin authors can
pass `text_style=TextStyle(...)` to `graphic_object` to select FreeType explicitly
for any text. Use an absolute runtime font path in a drawing object; resolve
saved project-relative parameters against `block.project_dir`. No Qt objects or
mutable font handles travel with the graphics. Lean/inherit and Graphics Off
can omit optional drawing work; Graphics On explicitly requests it.

Try the [original public example](<../examples/opencv tools/unicode_annotations/README.md>).
