# Visible support for pick candidates

**Filter candidates by visible support** (`inspect.visible_support`) compares each
candidate's expected template appearance with the current image. Connect its
filtered **Matches** and unchanged **Image** to **Select pick candidate** for
separate detector-score, angle, clearance and downstream ranking. The original
detector score is retained. This block supplies additional appearance evidence;
it does not redefine that score.

## Build the inspection

1. Supply the current Image, expected Template and candidate Matches. Matches can
   be an ordinary template-instance list or a `pattern_matches_v1` envelope.
2. Optionally supply Template mask or Template region to select the object's
   expected pixels and exclude background or holes. Both selections intersect.
3. Set Absolute tolerance in raw image units, Minimum visible fraction and
   Minimum observed fraction. Run once and inspect Candidates and Report.
4. Connect accepted Matches to pick selection. Keep obstacle occupancy, pick-area
   and clearance checks there; touching objects can have perfect appearance support.

Scene Mask/Region optionally restrict observed support. They do not shrink the
expected template denominator. Template and scene masks are uint8, single channel,
and match their respective grids; nonzero selects. Regions support the common
shape/composition/transformed/raster contracts, including holes. Known selection
frames must agree with their image.

## Transform conventions

**Transform mode** is a dropdown. Auto prefers the first available representation:

| Mode | Template-to-scene mapping |
| --- | --- |
| Transform | A pattern record's `image_pattern_pose_v1` transform, using its full source template grid. |
| Bounds | Fixed-scale template bounds, with `subpixel_top_left` when present; dimensions must equal the template. |
| Pose | A supplied `image_pose_v1`, with template coordinates centered at Template origin. |

Auto never falls back when a present transform/bounds record is malformed. Pattern
transforms and bounds take precedence over an offset pick pose, so a shifted datum
does not shift the appearance comparison. Pose-only records must use the template's
axes; Template origin is empty for `[(width-1)/2,(height-1)/2]`, or an explicit
`[x,y]` in template pixels. Extra pose-axis rotations must be handled upstream.

Known target frames map into the current image through their shared coordinate
root, including compatible crops/resizes. Different roots, unresolvable dimensions,
stale declared tickets, singular mappings and perspective-horizon crossings are
rejected. Missing metadata cannot prove coordinate or acquisition identity.

For record transforms, **Template grid policy** normally checks known training
source frames against the supplied template. **Assert pixel alignment** permits
a separately loaded template with different frame identity after the application
author verifies that its pixels align with the transform's original source grid.
Dimensions still must match; the assertion is reported and does not authenticate
template content or calibration. Training-template tickets may differ from current
acquisition tickets. Candidate/image tickets must agree when both are declared.

## What the fractions mean

Each selected template pixel is projected into the scene. Nearest or linear
sampling obtains appearance and real-source support. Linear comparison uses
explicit **1/32-pixel phases** with float64 native remap; support uses the same
sample coordinates. This avoids narrowing raw image intensities and makes the
qualified native sampling convention explicit. Image borders are constant only
for computation; filled values cannot become observed support. Linear samples
require all contributing scene/mask pixels, not just an in-bounds center.

| Candidate support field | Definition |
| --- | --- |
| `expected_samples` | Selected template pixels, including those missing in the scene. |
| `in_frame_samples` | Expected samples with complete real-image sampling support. |
| `observed_samples` | In-frame samples also inside the scene Mask/Region. |
| `matching_samples` | Observed samples whose channel difference is within tolerance. |
| `visible_fraction` | Matching / expected samples. Used for appearance qualification. |
| `observed_fraction` | Observed / expected samples. Used for coverage qualification. |
| `observed_agreement_fraction` | Matching / observed samples; diagnostic only, null when unobserved. |
| `mean_observed_error` | Mean absolute error, or mean worst-channel error in All channels mode. |

The fractions count template-grid samples, not projected scene area or calibrated
physical area. A clipped object cannot improve its score by losing its missing
samples from the denominator. Zero observed samples always rejects the candidate,
even with both configured fraction thresholds at zero. Every candidate retains
its original fields plus a `support` object with the resolved transform, counts,
fractions, acceptance and reasons. Filtered Matches preserve the original list
or pattern envelope, metadata, candidate order and scores.
`support.index` identifies the original candidate; pick selection's own `index`
refers to the filtered list it receives.

Grayscale comparison uses floating BGR luminance weights. All channels requires
every compared BGR channel to meet tolerance. RGB/BGR order is interpreted from
the image encoding and alpha is ignored. Supported raw depths are uint8, uint16,
int16, float32 and float64; values are not normalized between depths or exposures.
For example, tolerance 20 means twenty stored counts for uint16 as well as uint8.
Convert HSV/Lab images explicitly first. Physical depth planes require depth tools.

## Judgments, graphics and limits

**Minimum matches** defaults to one accepted candidate. A valid shortage produces
a failed inspection, with all candidate diagnostics and filtered outputs retained.
Zero permits an empty filtered list. Pick consumers can reject failed inputs;
the `required_three` example does so while the filmstrip retains current graphics.
Malformed data/settings or native errors are contained block exceptions without
current output events. The tool is stateless, preview-safe, parallel-safe and
readonly; Lean skips diagnostic outlines/text while retaining numerical results.

Drawable outlines and percentage labels show accepted/rejected candidates on the
original image without altering numerical pixels. Maximum graphics candidates
limits this optional work. There is no per-candidate full-scene warp or retained
image cache. Defaults cap 64 candidates, 8 million source pixels, 262,144 full
template pixels, 16,777,216 total template-grid samples and an estimated 256 MiB
of arrays; limits are checked before comparison/sampling allocations. Axes are
capped at 16,384 pixels and projected coordinates at one million. The working-byte
estimate excludes runtime input copies, Python metadata and native workspaces;
it is not a process RSS guarantee. Oversized lists are rejected, never truncated.

Appearance disagreement can come from occlusion, damage, lighting, focus or pose
error. Lookalike material can agree despite being a different or occluding object.
One comparison does not recover true hidden area, establish collision clearance,
validate a grasp or qualify physical robot motion. Validate tolerances on the
intended parts, lighting and pose accuracy. See
[four controlled examples](<../examples/opencv tools/visible_support/README.md>) and
[patterns and poses](patterns-edges.md).

Native references: [OpenCV remapping](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html)
and [template matching](https://docs.opencv.org/4.13.0/d4/dc6/tutorial_py_template_matching.html).
