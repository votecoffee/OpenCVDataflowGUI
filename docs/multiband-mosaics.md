# Multiband inspection mosaics

**Stitch image pair**, **Stitch scan sequence** and **Render mosaic tile** offer
**Blend → Multiband**. They remain under **OpenCV → Image Stitching → Geometry**.
Existing projects keep their original priority/feather/graph-cut settings.
[Seven runnable projects](<../examples/opencv tools/multiband_mosaic/README.md>)
demonstrate known pair geometry, automatic scan registration, gain correction,
tiled output and a deliberate native-depth rejection using a credited PCB photo.

## Choose a backend

| Control | Behavior |
| --- | --- |
| Multiband backend: Float64 | Default multiband implementation. Mask-aware floating Gaussian/Laplacian pyramids preserve uint16 and floating signal precision until the final output cast. |
| Multiband backend: Native detail | CPU `cv2.detail.MultiBandBlender`, restricted to uint8 gray or three-channel RGB/BGR. Gray is replicated for the native three-channel contract and reduced afterward. |
| Multiband bands | Zero to six reductions. Zero averages overlap without pyramid reductions. Small images cap the effective count; reports retain requested and effective values. |
| Multiband native weights | Float32 or int16 weight maps for the native backend; ignored by Float64. |

The native backend uses signed integer image bands even when weights are float.
Its rounded result differs from Float64, and even identical overlapping uint8
images can change by several sample units. It is an explicit compatibility option,
not a more accurate or qualified GPU path. CPU calls temporarily disable OpenCL
in the calling thread and restore the caller's setting on success or exception.
The installed OpenCV 5 build's int16 OpenCL weight kernel produced incorrect
output during qualification; it is not used here. Native GPU/OpenCL blending
remains unqualified. See the [OpenCV 5 API](https://docs.opencv.org/5.0/extra_modules/classcv_1_1detail_1_1MultiBandBlender.html)
and [native implementation](https://github.com/opencv/opencv/blob/5.x/modules/stitching/src/blenders.cpp).

Native detail rejects uint16, floating and alpha-channel inputs rather than
silently reducing them to uint8. With scan exposure correction, native input is
rounded/clamped to uint8 before feed; the report counts clipped samples.
Float64 retains corrected values through composition and casts only at output.
Changing backend can therefore change photometric results.

## Validity and pixel values

All views need matching depth, channel count and encoding. The existing pair
mask/region inputs and ordered scan/tile mask groups select valid source pixels.
Warping still honors nearest/linear sampling and conservative source validity.
No-coverage pixels remain zero, and Coverage/Overlap describe original warped
support rather than the wider pyramid support. A legitimate black pixel can
have coverage, so always consume the mask when excluding unsupported samples.

The Float64 implementation normalizes Gaussian signal samples by valid support,
then accumulates weighted signed Laplacian bands. Invalid zeros do not become
photometric observations. Dynamic amplitude normalization avoids intermediate
overflow for finite high-range floating images. Reconstructed nonfinite/out-of-range
float output is rejected by normal block validation. Integer output rounds and
clips to its source type range.

Pixels supported by exactly one view are restored from that warped view after
configured exposure correction and output casting. Blending affects overlap;
it does not fill unsupported holes. RGB/BGR ordering is retained. In the Float64
backend, alpha is another scalar channel with the same weights; this is not
premultiplied-alpha or opacity compositing. Scan exposure fitting continues to
exclude alpha from its gain/offset estimate.

Multiband composition can overshoot, soften or ghost features when views disagree.
It does not solve registration, moving-part rejection or physical illumination
calibration. Use registration diagnostics, support masks and original source
samples for measurements that require unmodified intensities. On a controlled
100/140-DN constant-step fixture, Float64 reduced the largest interior boundary
jump from 40 DN for priority composition and 20 DN for overlap averaging to 13 DN.
That fixture is not a general image-quality guarantee.

## Dense and tiled execution

Pair and scan tools render the complete bounded mosaic. Scan registration,
optional loop adjustment, overlap checks and gain/offset fitting remain the same;
multiband changes the composition stage. Graph-cut selection and multiband are
separate blend choices in this version. The fitted source transforms and tickets
remain in Layout. Output depth, channel encoding and coordinate provenance follow
the existing mosaic contracts.

For larger mosaics, **Register scan layout** estimates geometry without allocating
the full canvas. **Render mosaic tile** crops a requested window and retains a
filter halo of `8 × 2^effective_bands` pixels. Window origins align to that power
of two in global mosaic coordinates, preserving the decimation phase. The output
is cropped back to the requested tile and does not retain halo arrays. Tile
reports the halo, render-window origin/size and local-to-mosaic transforms.
Adjacent tiles were checked against dense composition over odd sizes, uint16,
floating RGB/RGBA and native uint8. Integer reconstruction/casting can differ by
one sample unit at rounding boundaries; floating results agree within the tested
numerical tolerances.

Source batches remain bounded by the existing limits (128 MiB; eight views for
dense scan registration, 64 for tiled registration/rendering). A conservative
preflight estimate includes destination/source pyramids, weights, padded native
grids, single-view restoration, reconstruction/cast arrays and the renderer's
existing work buffers. An estimate above 256 MiB is rejected before allocating
pyramids or native preparation. Reduce tile size or bands if necessary. The
reported `workspace_upper_bound_bytes` is an algorithm allocation estimate,
not an operating-system memory limit; source batches, graph snapshots, previews,
allocator overhead and unrelated application work also consume memory.

The blender is local to one execution. It does not retain native pyramids or
accumulators as trained state. Lean/graphics-off still produces the same primary
image and judgments. Errors use ordinary exception containment with no output
event; valid registration/overlap failures retain their existing failure reasons.

For repeated captures, [stable source identity and layout persistence](mosaic-layout-reuse.md)
allow reordered or missing views without changing calibrated priority or canvas
coordinates. Missing-view policies explicitly distinguish an invalid request,
a failed partial image and an allowed partial image.
