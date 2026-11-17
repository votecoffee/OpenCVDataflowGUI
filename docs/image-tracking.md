# Image-detection tracking

`sequence.track_image` associates detections across acquisitions and predicts
center, log width/height and angle through bounded gaps. OpenCV's Float64 Kalman
filter supplies prediction/correction; a shared maximum-cardinality, minimum-cost
assignment matches eligible observations globally. This is an image-plane
constant-velocity baseline with optional appearance association. It does not
classify occlusion or prove physical identity.

## Connect and run

Connect the current image and its current detections. `inspect.blobs`, measured
contour collections and shape/texture pattern matches can supply detections.
`ml.decode_detections` and tile-fusion box collections also connect directly.
An explicit empty list means a frame with no detections and advances prediction.
A connected input with no event holds execution and does not advance memory.
The ordinary graph scheduler clears that activation's displayed result when a
required event is absent. The tracker serializes its own memory updates while
independent branches can run in parallel.

A custom detector can return a list, or
`{"model":"image_detections_v1","count":1,"items":[...]}`. Each item has:

- `center`: finite `[x,y]` in current-image pixels.
- Optional `size`: positive `[width,height]` along the object's local axes.
- Optional `angle_degrees`: counterclockwise image-pose degrees.
- Optional `score` in `[0,1]` (default one), and integer/text `label`.
- Optional nonnegative integer `class_id`, bounded detection `id`, and an
  `appearance` descriptor under the contract below.

Collections need the current dimensions, coordinate frame and a zero origin.
Their clockwise principal-axis angles are converted to pose convention; measured
major/minor sizes are preferred, otherwise width/height are used. Axis-aligned
width/height with a separate angle are an approximation; use explicit oriented
records or pattern poses for precise local dimensions. Pattern poses require a
positive uniform affine scale and matching current grid. Their original model
scale, local origin and offset outline are retained. Rectify projective,
nonuniform or sheared poses before tracking. Bound tickets must agree, including
embedded pattern-pose tickets; numerical ticket IDs are not elapsed times.

## Clocks, identity and uncertainty

Clock choices are explicit `timestamp_ms`, video media time, deterministic
nominal intervals, or automatic selection in that order. Supply synchronized
acquisition timestamps for cameras and irregular sampling. Duplicate/decreasing,
nonfinite or too-small intervals are invalid. Switching clock type requires
reset/reacquisition. Gaps above Maximum interval expire tracks before native
prediction, then current detections can establish new tracks.

Each state holds `[x,y,log_width,log_height,angle]` and the five corresponding
velocities. Time is in seconds internally; input timestamps are milliseconds.
Position noise is pixels, angle noise degrees, size noise log size; process noise
is acceleration standard deviation in the corresponding unit per second squared.
Joseph covariance updates retain positive semidefiniteness. Unmeasured dimensions
stay explicitly unavailable; a fallback size never becomes a measurement.

Use a 180-degree period for undirected principal axes and 360 for directed poses.
Innovation wraps around the chosen period. Association gates include center
travel, normalized squared Mahalanobis innovation, known label, size ratio and
wrapped angle. Gated matching minimizes total normalized cost after maximizing
accepted pairs. Near-tied candidates are flagged as ambiguous. Perfectly
indistinguishable crossings cannot guarantee identity; appearance/label evidence
and application checks are still needed.

## Optional appearance association

**Appearance policy** defaults to **off**, preserving motion-only behavior and
avoiding feature parsing/storage. **If both** uses descriptors when both track
and observation supply them; missing descriptors fall back to geometry and
retain the established template. **Required** rejects eligible detector records
without descriptors. Low-score detections excluded by the existing score filter
need no descriptor. Explicit empty detections remain a valid missed observation.

Each record can carry `appearance = {"space": "model/preprocessing-revision",
"vector": [0.2, 0.8, ...]}`. Use a stable signature naming model, weights and
preprocessing. Vectors have 1–4,096 finite nonzero numeric components, with a
configurable lower ceiling. Normalization is stable for large finite values.
Different known spaces/dimensions reject and retire sequence history; reset or
reacquire when changing the descriptor model. Recovery uses a fresh generation.

Cosine distance is in `[0,2]`. Its configured maximum is a hard association gate;
**Appearance weight** blends normalized appearance and motion costs after all
hard gates pass. Zero weight still retains the appearance gate. Known class IDs
join label gating: equal text cannot merge different known classes unless label
policy is explicitly ignored. Global assignment still maximizes eligible matches
before minimizing total cost.

**Appearance update rate** applies a normalized exponential update only on matched
descriptor evidence. Zero freezes an established template; missing/local-flow
observations and predictions do not manufacture fresh features. Geometry gates
and expiry remain active: appearance cannot resurrect an expired identity.

`ml.box_appearance`, under **OpenCV → Video Analysis (video) → Object tracking**,
accepts current-grid box collections with XYWH geometry or blobs/contours with
measured axis-aligned bounds. It computes a normalized HSV hue/saturation joint
histogram (up to 512 bins) or grayscale histogram (up to 256). Input is explicitly
uint8 gray/BGR/RGB/alpha; convert other depths deliberately. Alpha-zero pixels
are excluded. ROI inset, mask and region restrict support without changing image
pixels or detections. Box count and summed ROI pixels, including overlap, are
bounded before native work. Insufficient support either omits appearance while
retaining the record or rejects, according to policy. Existing descriptors are
replaced so stale features cannot survive missing support. Sampling ROI outlines
are optional drawable graphics.

External DNN/user blocks can supply the same field directly. No learned
re-identification model is qualified by histograms. Similar colors, background,
illumination changes and occlusions can defeat this appearance baseline.

IDs include a generation and are not reused after expiry, reset, error, stream
replacement or reopen. Recipe/grid/encoding changes follow common SequenceBlock
context policy. State is bounded, copyable to execution workers and not serialized
as trained/runtime history in saved projects. Validation or native errors use the
common fail-safe path: no output event, diagnostic exception where appropriate,
and retired sequence memory before recovery.

## Results and inspection use

`tracks` is a measured collection, `poses` has corresponding current-image poses,
`count` counts emitted tracks, and `report` includes all active tracks, association
costs/ambiguity, expiry reasons, elapsed time and the ten-state covariance order.
Primary outputs obey normal sequence readiness/warmup. Tentative tracks are
hidden by default until enough observations confirm them. Every emitted item
includes observed/confirmed flags, misses, hits, confidence, velocity, standard
deviations and covariance. Center-only tracks explicitly have no size/orientation;
use their centers/poses or result graphics, and turn off bounds when sending them
to Draw collection. Sized track widths/heights refer to local measured axes.

Confidence is last score times confirmation fraction and exponential time decay,
halved for ambiguous assignment. It is a heuristic, not a calibrated probability,
pass judgment or collision/clearance measurement. Minimum confidence filters
outputs; expiry separately bounds missed-frame count, unobserved duration,
position/size/angle uncertainty, finite state and offscreen distance. The report
retains tentative/filtered state and gives explicit expiry/capacity-drop reasons.

Poses map the first local model grid to the current image, retain the current
ticket and expose observed/orientation-available flags. Gate downstream inspection
on those flags when fresh observations are required. Mask/region selection tests
detection and predicted centers, rather than clipping tracked shapes.

Result graphics are drawable outlines/markers and identity/state text, green for
observed and amber for predicted. They can be selected, aggregated and overlaid
on the final source image. Global/per-block graphics-off suppresses their
construction without changing estimates. The source image remains unchanged.

## Reproducible checks

[Two annotated examples](<../examples/opencv tools/image_tracking/README.md>) use
actual native blob measurements and original MIT images with independent motion
annotations. They cover 180-degree wraparound, short and long gaps, same-ID
reacquisition, expiry/new-ID reacquisition, covariance, current tickets and final
filmstrip overlays. Unit checks independently evaluate native transition and
Joseph covariance math, irregular intervals, 360-degree wraparound, center-only
crossing/reordered detections, ambiguous ties, scale/angle/label gates, partial
measurements, worker memory isolation, XML, invalid input/native errors, graphics
off and editor controls without executing native work. Synthetic accuracy bounds
qualify this baseline; they do not establish production tracking performance.

The appearance extension retains `class_id` and current `source_detection_id`;
predictions have no current source detection ID. `appearance_available` and
`appearance_space` describe the retained template, not visibility. Association
reports include cosine distance and whether appearance was used.

[Six detection/appearance projects](<../examples/opencv tools/detection_tracking/README.md>)
add crossing, hidden swaps, visible detector misses, expiry/re-entry and pinned
YOLOX on translated CC0 photographs. Independent annotations stay outside the
graph. The appearance swap has zero ID switches versus two motion-only. Dim-part
controls retain three misses and one fragment; shorter expiry adds an ID switch.
Learned cup detections retain identity through a hidden frame. These are
controlled sequences, not a real-video benchmark or learned re-identification.

For offline reports, import `evaluate_tracking` from
`opencv_dataflow.tracking_evaluation`. Supply ordered frames, each with explicit
`truth` and `predictions` lists of `{id, bbox: [x,y,width,height], label?}`. Every
frame must be annotated; empty truth means no labeled objects, not missing
annotation. Class-compatible global IoU assignment determines TP/FP/FN. Identity
switches compare the last match across absence; fragments require labeled misses
between matches. Callers explicitly choose observed or all tracks: the function
does not hide predictions or errors. The example evaluator requires current
events, verifies file hashes and scores observed tracks while reporting
predictions separately. Precision, recall, MOTA, mean matched IoU and per-frame
associations are supplied; no HOTA/IDF1 is claimed. Bounds are 10,000 frames, 128
objects per side/frame, 50,000 records and two million comparisons before matching.
