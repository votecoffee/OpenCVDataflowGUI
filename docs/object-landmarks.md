# Model-specific landmark skeletons

Use the existing **Decode keypoints and heatmaps** (`ml.decode_landmarks`) to
bind a model's ordered landmarks to names and a skeleton. The decoder now supports
direct heads with auxiliary columns, multiple confidence components and an
independent object-confidence head. The [nine examples](<../examples/opencv tools/object_landmarks/README.md>)
combine existing image preparation, sealed ONNX inference and decoding blocks;
no separate person-specific block or optional runtime is needed.

## Configure the head

For direct coordinates, choose Rows `[K,C]`, Channels first `[C,K]`, or Flat rows
`[K*C]`/`[1,K*C]`. An optional leading image batch must be one. Coordinate columns
defines C (3–16), X/Y columns selects two distinct column IDs, and Score columns
selects distinct remaining columns. Extra columns remain in the original tensor;
the 2D decoder does not assign physical meaning to Z or world coordinates.
The original three-column XY/score configuration remains the default.

Each selected score is independently interpreted as a probability or passed
through sigmoid. Minimum, Product or Mean combines those activated components;
the result is tested inclusively against Minimum confidence. Components are
retained on direct landmark records. For example, columns 3/4 of MediaPipe Pose
are visibility/presence logits; Minimum requires both activated scores to meet
the threshold. Presence alone makes a different claim from visibility and is a
separate saved configuration. These scores do not establish correctness.

Object score name optionally selects one scalar float32 head from the same
activation. It has its own probability/sigmoid choice and inclusive threshold.
A low object score rejects every requested point and fails the block even if
Minimum count is zero. Diagnostic records remain available, with no compact
Points event. The optional Object confidence output is hidden by default.
Malformed/missing scalar heads produce errors, rather than a passing fallback.
Main and object heads share the configured serialized-element budget.

## Geometry and skeleton identity

Channels selects model rows in a fixed order; Names and Connections use that
logical order. Rejected records keep their original identity and raw channel.
Links are drawn only when both endpoints are accepted. Rejected graphics can
show diagnostic red points (the existing default) or hide them entirely.
The report lists drawn point/edge indices for review, without changing numerical
landmarks or compact Points policies. Graphics limits and Lean still apply.

Direct coordinates declare network-pixel or normalized units. Pixel centers and
Pixel edges retain the existing resize mapping. Scaled indices explicitly uses
`source = network_coordinate * inverse_scale + inverse_offset`, without a
half-pixel shift. This matches exports using zero-origin scaled indices; it is
not automatically inferred from a model name. Rounded letterbox offsets remain
part of the declared inverse map. See [the general landmark contract](landmarks.md)
for source-center coordinates, heatmap alignment, selection and pose conversion.

Original image pixels, depth and encoding remain unchanged. Ticket, spatial,
pyramid and recipe context follow the current image. Known conflicts and missing
wired selections gate work through the shared runtime; failed-input rejection
preserves prior input history. Native exceptions are contained and suppress
current emissions. A missing/rejected endpoint cannot create a pose from a
previous result. Saved projects retain all decoder settings and skeleton fields;
JSON/XML and sealed model Save As are verified.

## Qualified pretrained example

The included OpenCV Zoo MediaPipe Pose model is unchanged, **5,557,238 bytes**,
SHA-256 `9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f`.
It is pinned to Zoo revision `48b10fa505aeb51fbd127fc60a5c660040e823b7` with
its Apache 2.0 license. A NASA public-domain portrait of Eileen Collins is supplied
through scikit-image. Source/license/hash/change records and canvas notes accompany
every project. No downloads or scikit-image installation are needed to run them.

Existing blocks prepare float32 RGB NHWC `[1,256,256,3]`, normalize by 1/255,
and request Classic/OpenCV/CPU execution. The sealed bundle verifies model,
network settings and preprocessing. The decoder selects rows 0–32 from the
39×5 `Identity` head, excluding six auxiliary rows; `Identity_1` is the scalar
object score. The supplied image is used directly. This baseline does not add
the upstream person detector or its rotation/crop alignment. Float32 interpolation
precedes normalization; comparisons use that declared preprocessing.

On the fixed portrait, the configured Minimum of visibility/presence at 0.8
accepts **28/33 points and 28/35 links**. Logical/raw indices **16,18,19,20,22**
are suppressed. Presence alone accepts all 33; Require all intentionally fails
the 28-point case. The upper-body subset accepts 16/17. A rejected right wrist
produces no pose. Blank input has object score about **7.3e-12**, zero accepted
points and no links. Setting the portrait's object threshold to 1 rejects it too.

Independent direct native inference and NumPy coordinate/sigmoid/skeleton rules
match the full saved pipelines within **0.0002 source pixels** and **1e-6 score**
on these controls, with exact accepted identities and links. Separate numerical
fixtures cover layout/column permutations, all score aggregations, inclusive
thresholds, resize/letterbox origins, malformed inputs and recovery.

The portrait's object score is about **0.99995**, yet the model confidently
predicts body points unsupported by this cropped view. Neither this score nor
its visibility estimates is annotated visibility truth or an accuracy result.
These examples qualify a configured single-object adapter and expose its limits;
multi-person association, other trained exports, physical localization and
accelerators need their own evidence.

Primary references: [OpenCV Zoo model and license](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/pose_estimation_mediapipe),
[native model conventions](https://github.com/opencv/opencv_zoo/blob/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/pose_estimation_mediapipe/mp_pose.py),
and [scikit-image's NASA portrait record](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut).
