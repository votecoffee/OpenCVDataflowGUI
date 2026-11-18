# Multi-scale learned super-resolution

**Multi-scale learned super-resolution** (`ml.super_resolution_multi`) is under
**OpenCV > Deep Neural Networks (dnn) > Image restoration**. The pinned LapSRN x8
frozen model contains learned 2×, 4× and 8× heads. One native named DNN forward
returns every requested output; the block does not run three independent models.
The existing single-image super-resolution block retains its settings and behavior.

## Select outputs

**Output scales** selects two or three distinct values from `[2,4,8]`.
**Output nodes** supplies equally many distinct native names in the same order.
The pinned export uses `NCHW_output_2x`, `NCHW_output_4x`, `NCHW_output_8x`.
Order is preserved in the **Images** FlowGroup, labelled `x2`, `x4`, `x8`.
Other matching local LapSRN exports need their own checksum, names and qualification.
No missing or wrong-size head is silently replaced by interpolation.

**Image** previews the largest requested scale by default. **Preview scale** can
select another requested scale; Baseline, Processed mask and Geometry describe
that same preview. Optional **Image x2/x4/x8** connectors expose each requested
scale directly. Unrequested scales emit no event. Expose these connectors in the
sidebar before wiring them. Use group/parallel tools for the grouped images.
This is an ordered group of reconstructions, not a newly measured Gaussian pyramid.

Info retains the ordered head/scale mappings, all output sizes/transforms, chosen
preview, exact model SHA-256/bytes, cache reuse, native-forward count, dtype/color,
selection counts and requested backend/target. An empty selection uses zero
native calls and reports no inferred model artifact. Model files are never
downloaded by graph opening or execution.

## Native Python delivery

The supported Python 3.14.3 / OpenCV 5.0.0 contrib 5.0.0.93 Windows wheel exposes
`upsampleMultioutput` with an input list, but it returns `None` without delivering
its computed images to that Python list. That binding remains unqualified.
The implemented equivalent loads the exact hashed TensorFlow snapshot with the
Classic DNN engine and requests the named heads in one `Net.forward`.
TensorFlow is not a Python dependency. There is no diagnostic second forward
or native-binding retry during execution.

Luminance preparation, native float-conversion rounding, interpolated chroma
and 8-bit reconstruction match the native LapSRN path. Separate named native
heads verify all requested outputs; every 8× output also matches native single
`upsample` pixels exactly across qualified source encodings/depths. Native
construction shares the registration lock with the existing single-image tool.
Only CPU is qualified; target availability and requested settings do not prove
actual per-layer device placement. No performance improvement is claimed.

## Pixels, selections and coordinates

GRAY/BGR/RGB/BGRA/RGBA inputs retain their encoding and output depth. The learned
working path is 8-bit. Non-8-bit images require explicit **Convert full scale**:
uint16 or float32 `[0,1]` is quantized to uint8 for inference, then restored to
source depth. HDR clipping and metric-depth inference are not implicit. Alpha is
interpolated, never inferred. Baseline interpolation controls unselected pixels,
alpha and the preview baseline. Float baselines clip interpolation overshoot to
`[0,1]`, with clipped samples counted per output.

Optional masks and circle/rectangle/polygon regions intersect in the source grid.
The full source remains model context; selection repeats with nearest-neighbor
mapping at each output. Unselected pixels use the chosen baseline. An empty
selection produces all requested baselines without reading or running a model.

Each image and group member keeps its own nominal half-pixel output-to-input map:
`x_input = (x_output+0.5)/scale-0.5`, likewise for y. Ticket, recipe, auxiliary
payload and the existing pyramid mapping are preserved/composed independently
for each scale. A `super_resolution` payload records the scale, node and source
size, explicitly identifying reconstruction. Group/Info metadata stays in the
input frame; image ports and preview metadata use their respective output frames.

Learned feature positions are not calibrated by this nominal map. The known disk
example measures center errors about **0.59/0.16/0.23 original-reference pixels**
at 2×/4×/8×. At 8× the same learned output erases **230 dark detail pixels** in the
specified crack ROI. Lower average pixel error does not establish preservation of
every feature or improvement in every measurement. Reconstructed detail is not
additional measured evidence or qualified physical metrology.

## Bounds and lifecycle

Output budgets count the aggregate `input_pixels * sum(scale²)` before visible
output allocation or inference; axes also stay within 16,384 pixels. Snapshot
bytes and input pixels have separate bounds. These budgets do not bound native
workspace or every application-side result copy. Two-scale requests remain
bounded by their requested output grids.

Each activation hashes a bounded model snapshot. Cache misses import an owned
temporary snapshot that is removed before forward or on import error. Changed
bytes, wrong heads/shapes, exceptions and nonfinite outputs clear native state
and emit no partial scale group. Failed-input admission retains processed history
and emits nothing. Reset/close release the block's private native network.
Independent instances use ordinary parallel branch scheduling.

[Thirteen runnable projects](<../examples/opencv tools/multiscale_super_resolution/README.md>)
cover geometry, color/alpha/depth, selections, ordering/subsets, explicit errors,
noise and a CC0 photo. The separate direct-native benchmark retains native-size
truth, pixel errors, localization and erased/false detail. Save As/XML retain
model bytes, attribution and the model license.
