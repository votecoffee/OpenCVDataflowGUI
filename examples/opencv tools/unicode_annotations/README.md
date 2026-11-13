# Unicode inspection labels

Open `project.json` and run once. Three **Add annotation layer** blocks place
Latin, Greek and Cyrillic inspection messages over an original 16-bit image.
**Overlay result graphics** combines their selectable drawing layers, and
**Set filmstrip image** selects the composed result for the ticket.

Try changing a label, font size, outline or RGB color. A wired **Text** input
can supply OCR or inspection text. Glyphs are alpha-composited in the original
sample depth; the source remains unchanged. The saved Graphics **On** setting
also renders labels in Lean. Graphics **Off** returns the unannotated image.

The graph needs no downloads or plugin. The bundled Noto Sans supports these
three scripts. Choose a project-relative TTF/OTF/TTC and face index for another
font. Complex scripts and explicit left/right direction require a Pillow build
with Raqm; automatic basic layout is not a substitute for shaping. Font size is
in rendered pixels, and transforms move the anchor. Long labels clip at the
image boundary. See [annotation details](../../../docs/unicode-annotations.md).

The fixture, generator and project are original work under the repository MIT
license. The unmodified Noto Sans font is Copyright 2018 The Noto Project
Authors, licensed under SIL OFL 1.1; [font provenance and license](../../../src/opencv_dataflow/fonts/README.md).
The project contains the same attribution in a layout comment.

Regenerate with `python "examples/opencv tools/unicode_annotations/generate.py"`.
