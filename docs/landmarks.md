# Ordered image landmarks

**OpenCV > Deep Neural Networks (dnn)** contains **Decode keypoints and
heatmaps** (`ml.decode_landmarks`) and **Pose from ordered landmarks**
(`ml.landmark_pose`). Connect the original image, tensors and preprocessing from
one inference activation. The decoder preserves the image's pixels, depth and
color encoding and emits drawable result objects separately.

This is an explicit single-object landmark adapter. It does not discover multiple
people/parts, associate instances, infer visibility, or guess an export's axes or
coordinate conventions. A model contract pins inference settings/bytes; the
decoder settings and names are saved with the project.

## Heads and coordinate conventions

| Format | Shape | Meaning |
| --- | --- | --- |
| Heatmaps | NCHW, NHWC, CHW or HWC | One peak per selected channel; N must be 1. |
| X/Y/score | K by 3 or 3 by K, optionally preceded by batch 1 | One explicit ordered coordinate and confidence per row. |

Use **Output name** to select a float32 `tensor_v1` head; empty selects the first
output. Choose probability scores or sigmoid logits. A probability must be in
0..1. Confidence is tested inclusively against **Minimum confidence**. For
heatmaps it is the original integer-peak score, even after fitting a position.

**Channels** selects distinct raw channels/rows in an explicit order; empty uses
all, including any background map unless you exclude it. Optional **Names** must
match that selected order. Every requested landmark keeps its index and raw
channel, including rejected landmarks. An empty, correctly shaped head is valid;
the count criteria determine its judgment.

Output coordinates use continuous, integer-origin **source pixel centers**: the
upper-left pixel center is (0,0). Heatmap cell index `p`, head extent `H` and
network extent `N` map to network pixel-edge coordinates as follows, independently
on each axis:

| Head alignment | Network edge coordinate |
| --- | --- |
| Half pixel (default) | `(p + 0.5) * N / H` |
| Align corners | `0.5 + p * (N - 1) / (H - 1)`; a one-cell axis uses `N / 2` |
| Asymmetric | `p * N / H + 0.5` |

The matching inference resize/rounded-letterbox inverse maps network edges to
source edges; subtracting 0.5 produces source centers. Direct X/Y rows declare
network pixels or normalized fractions multiplied by the full network width and
height, plus whether their origin denotes pixel centers or edges. These settings
must match the export. Source-outside coordinates are rejected rather than
clamped into the image.

## Selection, fitting and partial results

Mask and Region intersect in the original source grid. A connected selection
that has no event prevents execution. **Selection** peak search (default) uses
only heatmap cell centers that map into the source and selection, excluding
letterbox padding. **Entire head** finds the global peak and filters the resulting
position afterward. Neither policy alters the model's inference input.

Equal maxima reject by default. **First** explicitly chooses the first maximum
in row-major order while retaining the ambiguity flag. Refinement can be off,
quadratic, or log-quadratic (default). A bounded local least-squares fit must have
a sufficiently negative Hessian, an acceptable residual and a bounded offset.
The whole fit window must lie in the head and eligible selection; logarithmic
fitting requires positive values. If qualification fails, keep the integer peak
(default) or reject it. Each record reports its refinement status and fit RMSE.
This local model is suitable for an isolated concave peak; it is not a universal
subpixel accuracy guarantee.

`landmarks` is `image_landmarks_v1`, with `image_size`, the coordinate convention,
ordered `items` and source context. Each item contains `index`, `channel`, `name`,
`position`, `confidence`, `accepted`, `reason`, `refinement`, `fit_rmse` and
`ambiguous`. Reasons distinguish absent eligible peaks, ambiguity, rejected fits,
outside-source/selection positions and low confidence. Rejected positions remain
diagnostic; a position can be null if there was no eligible cell.

`indices` lists accepted logical indices and `count` counts them. By default,
`points` emits only when every requested landmark is accepted, preserving ordered
correspondence for downstream geometry. **Accepted only** opts into a compact
point list; use `indices` to retain its identities. **Minimum count** and
**Require all** govern the normal `landmark.count` pass/fail judgment. A failed
inspection still carries current diagnostic records and unchanged source pixels.

## Poses and result graphics

Connect ordered records and an image to **Pose from ordered landmarks**. Choose
an accepted origin index and either a second accepted direction index or a fixed
counterclockwise angle. The pair's minimum separation is checked in source
pixels. Local X/Y/angle offsets apply in that source frame before shared-root
mapping to the target image. The output is the ordinary `image_pose_v1`, usable
by **Pose-relative region**, masks and other inspections. Missing/rejected
landmarks or a short baseline produce a normal fail with `found=false` and no
pose event, rather than reusing a previous pose.

The decoder's optional result graphics show accepted/rejected points in
green/red, with names/confidences. **Connections** joins selected logical pairs
only when both are accepted. **Graphics maximum landmarks** caps diagnostic
markers/links without truncating numeric output; `graphics_truncated` reports it.
Use the normal graphics overlay/filmstrip controls to render these objects.
Lean or graphics-off execution skips their construction.

Tickets, processing start, recipe, spatial and pyramid metadata follow the
source image. Known ticket/recipe/frame conflicts reject before work. Pose
conversion also checks record-body/wire context. Tensor counts, raw/selected
landmarks, head/source axes and pixel counts are bounded. Malformed data and
native exceptions are contained by the base executor and emit no current result;
the next valid activation can recover. Blocks are stateless and opt into the
existing preview/parallel contracts.

## Reproducible controls

The [twelve landmark projects](<../examples/opencv tools/landmarks/README.md>)
use original uint16 Gaussian images and an image-dependent, untrained ONNX
network. Analytic Gaussian centers, independently calculated intensity centroids,
letterbox, ordered partial/empty records, source-depth retention, ROI/pose,
filmstrip overlays and a sealed model bundle are checked in Debug and Lean.
These controls qualify decoding/geometry and persistence, not learned model
accuracy, physical localization, arbitrary exports or accelerators.
