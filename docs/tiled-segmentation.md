# Segmentation across image tiles

Use existing Crop/Resize blocks and ordinary `ml.dnn` inference on parallel
branches. Enable **Export segmentation** on each `ml.decode_segmentation`, join
the exported records with `flow.join`, and connect the group to
`ml.fuse_tile_segmentation`. Connect the original acquisition image to the fusion
block as its reference grid. No new inference engine or hidden tile scheduler is
introduced. The [nine examples](<../examples/opencv tools/segmentation_tile_fusion/README.md>)
include a separate full-frame inference branch for comparison.

## Transport and decisions

Export is off by default to avoid retaining full class planes. The optional
`segmentation` connector carries an array-backed `segmentation_tile_v1` record on
an `any` port: signed int32 labels, float32 C,H,W exclusive class probabilities
when available, ordered class names/count, and the tile's coordinate context.
This is an in-memory runtime value, not a JSON tensor or a saved trained model.
Class-ID heads export labels without inventing probabilities. Binary heads
export background and foreground probabilities. Existing activation, resampling,
confidence, ignored-class and tile ROI controls still apply. Labels of `-1`
identify pixels that may not contribute; other class planes remain intact.

Choose the fusion mode explicitly:

| Mode | Sampling and decision | Confidence output |
| --- | --- | --- |
| `probability_mean` | Bilinear per-class probabilities, normalized over valid neighbors, then weighted mean and argmax | Winning fused probability |
| `label_vote` | Nearest local label, then weighted class votes and argmax | Winning vote fraction; this is not model confidence |

Ties choose the lowest class ID. Probability fusion applies argmax after
averaging; it does not preserve a tile decoder's non-0.5 binary decision
threshold. Use the final confidence and class selections deliberately. Label
voting preserves each tile's already-decided label. Classes must have the same
count, names and order on every tile. Multi-label/sigmoid class sets, instance
identity merging and logits from incomparable models are outside this contract.

Uniform weights give every supported tile an equal contribution. Feather
weights rise from the tile's rectangular pixel boundary to one over **Feather
width**, in local tile pixels. They are independent of part boundaries and
invalid holes. Valid neighboring probabilities are renormalized before feather
weighting, so ignored pixels do not bleed background/part scores into the result.
Feathering reduces a tile edge's influence; it cannot recover details a model
did not observe or guarantee seam-free predictions for an arbitrary network.

## Coordinates, selection and missing coverage

Fusion validates acquisition tickets, known image roots, root dimensions,
record-versus-wire grid and pyramid metadata, and known recipe revisions.
Supported relative maps are crop, scale, flip and right-angle axis swaps. Shear,
arbitrary rotation, perspective and nonlinear maps are rejected. Output pixels
sample each tile by its inverse pixel-center mapping. Numeric score sampling
uses OpenCV's float-map linear interpolation and replicated edge neighbors,
with a separate strict tile footprint test. See the official
[OpenCV geometric transformation reference](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html).
Nearest label sampling is explicit `floor(coordinate + 0.5)` with border clamping
inside that footprint.

The original image, format, depth, ticket, reference grid and auxiliary payload
are preserved. Fused int32 labels, selected uint8 mask and float32 confidence
use the reference grid. Optional mask/region inputs select the final reference
ROI. Per-class counts, fractions and mean confidence are reported. Ignored or
low-confidence final winners become `-1`; no runner-up class is substituted.
Selected fraction uses all ROI pixels as its denominator.

`coverage` is a uint16 count of tiles with valid support at each pixel before
the final class/confidence gate; `weight` is their summed float32 weight.
These diagnostic maps cover the entire reference grid, even outside the final
ROI. Unknown/uncovered output pixels stay `-1` with zero confidence. The minimum
coverage judgment uses supported pixels divided by ROI pixels; an empty ROI has
zero coverage. Expected tile count checks group completeness, not spatial
coverage. Source tile boundaries are half-open pixel-edge footprints.

Any supplied failed tile adds `segmentation.tile_failed` to the aggregate's
ordinary fail result while retaining current fused outputs. Area and coverage
failures likewise emit inspectable current results and reasons. If a branch
does not emit, use Join's existing timeout/completion policy; fusion never treats
a missing event as a completed tile. Malformed data, mismatched contexts or
resource violations are errors handled by the normal block executor, with no
new output event. A valid next invocation can recover normally.

## Resource and qualification limits

Groups contain 1..64 distinct, bounded labels. Class count is 1..256. Inputs,
reference pixels, class accumulation, and conservative tiles x classes x pixels
work have explicit budgets checked before accumulation. Optional class export
also uses the decoder's resample budget. These limits bound declared numerical
arrays/work, not process RSS, scheduler copies or a native model's workspace.
The full acquisition still has to fit the reference pixel budget. This is not
disk-backed gigapixel processing. The two modes do not require a GPU.

Debug runs produce drawable selected-region contours and a semantic color map;
normal graphics optimization settings suppress these without changing numbers.
Fused graphics use aggregate pass/fail state. Connect `result_graphics` through
Overlay result graphics and Filmstrip image to review the final source image.

Qualification uses original MIT pixel-dependent ONNX equations and a controlled
image with parts crossing both overlap boundaries. All nine projects compare
against independently generated labels and a separate native whole-image
execution. CPU Classic/New engines, downscaled-tile coordinates, ROI, failed
tile, coverage, empty and area outcomes are checked. These are deterministic
plumbing and edge-continuity references, not a learned model accuracy benchmark
or physical inspection acceptance. A real model may need context halos, different
overlap/feather widths and separate source-labeled qualification.
