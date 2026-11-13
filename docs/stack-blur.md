# Stack blur

`cv.stack_blur` wraps OpenCV `stackBlur` for large-radius inspection smoothing.
Gaussian, box, median and edge-preserving filters remain available. Stack blur
has a separable triangular response and replicated borders; it approximates
Gaussian smoothing and also broadens inspection edges.

The [original example suite](<../examples/opencv tools/stack_blur/README.md>)
compares matched-variance Gaussian and stack responses, noise, edge position and
edge width. A separate native-call benchmark records local timing across three
odd kernel sizes. Performance is measured separately from graph overhead.

## Parameters and data

- **Kernel** is the positive odd X size, 1..2047. **Kernel Y** uses X when zero,
  or specifies its own positive odd size.
- **Oversize policy** follows native clamping to the largest fitting odd size,
  or rejects an oversized kernel. Info reports requested and effective sizes.
- **Arithmetic** defaults to float32 working samples, with integer output
  restored by round-to-nearest and clipping. It preserves sample units and
  original depth; it does not normalize intensity. Float32 accumulation has
  finite precision, especially for wide kernels or large values.
- **Native** arithmetic permits uint8 with effective kernels up to 255 and
  float32. The current build produced integer artifacts for high-bit-depth data
  and some larger integer kernels in controlled probes. Those paths are rejected
  before the native call. Use float32 arithmetic for uint16/int16.
- **Alpha mode** preserves alpha exactly by default. Filter smooths all channels
  independently. Preserving alpha on premultiplied inputs is rejected; filter all
  channels or first unassociate alpha. The block does not change alpha association.
- **Maximum pixels** bounds whole-image work to at most 16,777,216 pixels.
  A conservative native scratch estimate is capped at 128 MiB. Nonfinite samples,
  unsafe floating accumulations and unsupported encodings fail without output.

Inputs support uint8, uint16, int16 and float32 gray/BGR/RGB/BGRA/RGBA. Float64 and
HSV/Lab are rejected with conversion guidance. The host image port uses one,
three or four channels. Physical DepthArray planes retain their dedicated depth
tool contract. A 1×1 effective kernel returns an independent identity image.

## Selection, results and provenance

Mask and region intersect the write selection. Neighborhoods are computed from
the full source image; only selected output pixels are replaced. Image-edge
replication is separate from the selection boundary. Empty selections return an
unchanged copy without native filter work.

Output depth, channel order, ticket, coordinate frame, pyramid context and
companion payloads are retained. Info records effective controls, working/output
depths, selected pixels, work estimate and no-op status. An optional drawable
selection outline is separate from image pixels and honors graphics-off settings.
It is capped at 128 contours/8192 vertices, with truncation reported in Info.
The underlying selection always retains full precision.

The shared block runtime supplies validation icons, diagnostic exceptions,
timings, ticket events and last-input previews. Parameters and enums persist in
JSON/XML. Run block and live preview use the standard captured-input path.

## References and interpretation

For radius r, the normalized one-dimensional stack weights are proportional to
r+1−|x| and their variance is r(r+2)/6. The example solves the finite Gaussian
kernel's sigma to match that variance on identical support. This controls one
source of comparison bias; the two filter responses still differ.

Native API and supported data types: [OpenCV image filtering](https://docs.opencv.org/5.0/main_modules/imgproc_filter.html).
The wrapper calls the installed library; it does not contain a copied stack-filter implementation.
