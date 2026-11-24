# Run-length binary morphology

Open [compare_opening.json](compare_opening.json) and select **Run Once**.
The graph opens three synthetic plate silhouettes with a 9 × 9 rectangle.
One branch processes horizontal foreground runs and decodes the result for
display. The other uses dense OpenCV morphology. **Require exact pixel
agreement** passes with maximum absolute error **0**. The opening removes
five isolated pixels and the narrow tabs.

Use **Encode binary mask runs** after segmentation of a monochrome image.
Choose bright or dark polarity, and optionally wire a same-frame mask or
region to select source pixels. Wire the resulting `runs` data through as many
**Run-length binary morphology** blocks as needed. Decode when a downstream
tool or preview needs a raster. The data wires retain the source ticket and
pixel grid. The saved image is an original synthetic fixture covered by the
repository license.

## Kernel and storage

Choose erosion, dilation, opening, closing, gradient, top hat or black hat.
The kernel editor supports standard shapes and custom binary grids with one
contiguous span per active row. The operation anchor and the cross construction
anchor are separate. The default exterior is foreground for erosion and
background for dilation, matching OpenCV's neutral constant border. Set
**Erosion border** to **background** for an all-zero exterior.

The Python implementation follows the [OpenCV run-length morphology
interface](https://docs.opencv.org/5.0/extra_modules/ximgproc_run_length_morphology.html)
for binary operations. The run record is JSON with model
`binary_mask_runs_v1`, `image_size` in `[width, height]` order and
`runs` as sorted `[row, start, exclusive_end]` triples. Adjoining runs
on the same row must be merged.

The result has **801 runs**, or **9,612 bytes** as packed int32 triples,
compared with **3,145,728 bytes** for the uint8 raster. The report's
`packed_run_bytes` excludes Python and JSON overhead. In a separate large-tile
check, opening three 320 × 256 rectangles on a 16,384 × 16,384 virtual grid
with the same kernel peaked at **1.58 MiB traced allocations** for one block
execution. A single dense uint8 raster for that grid would occupy **256 MiB**.

Caps limit run counts and morphology work. Encoding and decoding allow at
most 64 million dense pixels, with a lower configurable decode limit.
Highly fragmented masks can exceed the run limit. Use dense morphology for
those masks or when the kernel needs gaps within a row.
