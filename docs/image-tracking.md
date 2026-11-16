# Image-detection tracking

`sequence.track_image` associates detections across acquisitions and predicts
center, log width/height and angle through bounded gaps. OpenCV's Float64 Kalman
filter supplies prediction/correction; a shared maximum-cardinality, minimum-cost
assignment matches eligible observations globally. This is an image-plane
constant-velocity baseline, not an occlusion classifier or appearance tracker.

## Connect and run

Connect the current image and its current detections. `inspect.blobs`, measured
contour collections and shape/texture pattern matches can supply detections.
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
