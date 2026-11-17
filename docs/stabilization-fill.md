# Fill stabilization borders for review

`sequence.fill_stabilization_borders` transfers supported samples from neighboring
corrected frames into the selected frame's uncovered display pixels. **Image and
Coverage remain current measured data. Only Display contains review fills.**
Try the [seven projects](<../examples/opencv tools/stabilization_fill/README.md>).

## Input geometry and source identity

Connect matching Corrected and Coverage groups from recorded stabilization or
motion-trajectory replay. Supply 2–64 same-size/depth/encoding gray, RGB, BGR,
RGBA or BGRA images using uint8, uint16, float32 or float64. Coverage is required:
one uint8 mask per frame, with identical labels/order and dimensions. Known
mask grids must match their corresponding image. Known per-image/mask tickets
must match unless Coverage tickets is explicitly Ignore.

Target index chooses the current capture. Every image, map and report keeps
that member's ticket, spatial/pyramid coordinates and custom metadata; an outer
batch ticket does not replace it. Donor tickets and original indices remain in
Report. Positive straight alpha contributes support; premultiplied inputs require
unassociation first. Intentional transparency inside geometric coverage is not
permission to fill the image from another capture.

Spatial alignment requires known roots. Within one root, the tool pulls a target
pixel through `inverse(neighbor.to_root) × current.to_root` to find the neighbor's
corrected-image pixel. Matrices must be bounded, invertible affine mappings.
Recorded stabilization already publishes the inverse smoothed pose as To root,
so differently shifted corrected grids can share a coherent scene reference.
Different known roots normally skip as separate segments; the Error policy
rejects them. Inconsistent root extents or unknown roots reject.

Pixel grid alignment is an explicit assertion that corresponding pixel indices
are registered. It permits missing or independent roots and applies no geometric
mapping. Equal image dimensions alone do not establish alignment. The tool does
not estimate motion or optical flow and cannot repair an inaccurate upstream fit.

## Filling and consistency

Temporal radius bounds candidates by ordered batch position. Nearest mode visits
closest captures first, with earlier index breaking a tie; the first eligible
sample wins at each pixel. Mean uses a convex average of eligible samples.
Original depth and channel order remain intact. Nearest can copy original sample
bits when no resampling is needed. Linear resampling requires full measured
kernel support; out-of-bounds pixels, donor coverage holes and transparent donor
pixels cannot contribute. Floating resampling/averaging uses shared bounded
units, with separate alpha scaling, to handle tiny and HDR samples.

By default, only geometric holes connected to the image border can be filled.
All permits internal geometric holes too. Optional target-grid Mask and Region
intersect to select fill centers. Maximum fill distance limits Euclidean distance
from current measured support; zero disables that limit. The requested-hole
denominator is computed before distance filtering, so a stricter distance limit
does not hide unfilled requests. No measured target pixel is changed.

Overlap consistency first requires Minimum overlap pixels shared measured
samples. Maximum error compares the largest channel difference in common units:
integer full scale, or one maximum absolute color value over the float batch
(one for all-zero data). Alpha uses its own full scale. At least Minimum
consistent fraction of shared pixels must satisfy this limit. A rejected donor
contributes nothing. Detected mismatches are dilated by Change radius to exclude
a local halo from filling. Selection limits output centers, not overlap evidence.
Consistency Off explicitly disables this photometric check.

These are scene/lighting consistency proxies. Changes wholly outside current
coverage, low-contrast changes, occlusion and temporal motion can remain
unobservable. A neighbor is another acquisition, even if its image matches
perfectly in the overlap. Fills are for review, not current defect evidence.

## Outputs and judgments

| Output | Contract |
| --- | --- |
| Image | Current samples, with zeros outside original measured support. No fills. |
| Coverage | Original current geometric support intersected with positive alpha. Never expanded. |
| Display | Image plus accepted neighboring samples in originally uncovered pixels. |
| Filled mask | Actual review-filled pixels, disjoint from Coverage. |
| Remaining mask | All unavailable pixels after filling, including unselected/interior holes. |
| Display support | Union of Coverage and Filled mask; review availability only. |
| Donor index | First contributing original batch index, or −1; the sole donor in Nearest mode. |
| Contributors | Eligible donor count per pixel; Nearest uses only the first. |
| Report | Target identity, requested/measured/filled/remaining counts, fraction, completion, work estimate and per-donor mapping/overlap/provenance. |
| Preview | Optional measured/filled/missing color map on the current grid. |

Fill fraction divides Filled pixels by selected eligible geometric holes before
distance filtering. With no requested holes it is one. Minimum fill fraction
sets the required fraction; Unfilled Fail additionally requires every requested
hole to be filled. At least one current measured pixel is always required for
a passing result. Complete refers to requested holes; Remaining mask can still
include intentional transparent, unselected or excluded internal holes.

A valid partial/unavailable fill is an inspection failure when those thresholds
require it. Its current image, review image and diagnostic masks remain available.
Invalid settings, metadata or unexpected native exceptions use the common
contained block-error path without current output events. A later activation
can recover normally. These judgments describe review availability, not part
quality or recovered current measurement.

Display is tagged `stabilization_border_fill.review_only: true`; Image is false
and marked as requiring Coverage. This tool rejects its own filled Display as
a new measured input. The tag is provenance, not a universal runtime prohibition:
use Image/Coverage explicitly for downstream inspection. Drawable fill/missing
contours and text refer to Display and can be aggregated for the final filmstrip.
Graphics disabled omits drawings and Preview, retaining all numerical outputs.

## Bounds and qualification

Frames are bounded to 16,777,216 pixels and 16,384 pixels per axis. Shared input
guards cap the frame/mask batch at 128 MiB. The configurable array-work estimate
defaults to 512 MiB and is checked before conversion, support preparation or
filtering. The implementation visits one donor map at a time. Estimates exclude
graph snapshots, retained history and native workspaces; they are not a process
memory or runtime guarantee. The tool is stateless, read-only and parallel-safe.

Analytic ramps verify projection, subpixel samples, depth, alpha, HDR/tiny means,
coverage, masks/regions, changing donors and strict incomplete results. Saved
examples provide sealed known jitter and a separate native fitted sequence.
Some corrected crops retain no donor for a few corners: those pixels stay
missing and explicitly allowed partial examples declare that condition.

This is an application equivalent built from native remapping, component,
distance and morphology operations. The supported wheel has a `videostab`
namespace, but its `ConsistentMosaicInpainter`, `MotionInpainter`,
`ColorAverageInpainter`, `ColorInpainter`, `calcFlowMask` and
`completeFrameAccordingToFlow` bindings are absent. It does not claim numerical
parity with those [native C++ inpainting interfaces](https://github.com/opencv/opencv_contrib/blob/5.x/modules/videostab/include/opencv2/videostab/inpainting.hpp).
