# Periodic detector refresh and local tracking

Use `flow.detection_cadence` with `sequence.track_refreshed` to run an existing
full-frame detector less often while observing local image motion between its
refreshes. No graph feedback cycle, custom detector implementation or compiled
backend is required. The tracker extends [image-detection tracking](image-tracking.md)
with bounded local evidence and a separate detector-freshness limit.

## Connect an arbitrary detector

| From | To |
| --- | --- |
| Source Image | Cadence Image and Track with periodic detections Image |
| Cadence Image | Your detector's Image |
| Cadence Refresh | Track with periodic detections Refresh |
| Your detector's measured collection or pattern matches | Track with periodic detections Detections |

The cadence emits its Image only on frame zero and every Period acquisitions
after that. Refresh and Report emit on every eligible input. Optional Force is
a current boolean level: true requests an extra refresh and restarts the interval.
Reset, recipe/grid changes and stream contexts follow SequenceBlock behavior.
Warmup must remain one so the initial detector event is delivered. The tool is
under **Custom > Control flow > Scheduling**; the tracker is under **OpenCV >
Video Analysis (video)**.

An inactive detector branch emits no event between refreshes. The tracker still
receives the original image and false Refresh. On true Refresh, current detector
results are mandatory; an explicit empty collection/list is a valid no-detection
observation. Detector results on a false Refresh, missing refresh results or
mismatched tickets/grids are invalid. Upstream detector exceptions use ordinary
graph error handling, rather than silently switching that refresh to local work.
A detector ToolBlock can use the same wiring when its ports follow these contracts.

## Local image evidence

Local processing requires a **uint8** image of at most 4,194,304 pixels. Convert
other depths deliberately before this block; the source image is never changed.
At a full detector match, corners are selected inside that track's image ROI,
intersected with the current mask/region. Corner work uses cropped ROIs, not a
second full-frame object detector. Center-only detections use the configured
fallback ROI while size/orientation stay explicitly unavailable.

Between refreshes, native pyramidal Lucas-Kanade follows those corners, checks
native patch error or minimum eigenvalue, and optionally verifies backward
round-trip displacement. A native robust similarity fit estimates translation,
uniform scale and rotation. Minimum point count, two-axis spread, retained-inlier
fraction, residual, center-shift, scale and angle gates reject weak fits. All
meaningful corner/LK/fit controls have ranges, tooltips and dropdowns for enum
choices. RANSAC and LMedS are available; RANSAC confidence and native LK errors
are algorithm controls, not object probabilities.

Accepted local measurements can update only their original identity. They cannot
create tracks or switch to another identity through global assignment. Only a
full detector match discovers a new object. Rejected local evidence advances the
ordinary predictor, clears local points and waits for the next detector match to
reseed. It does not seed features from a predicted box on replacement background.
Local points are not replenished between detector refreshes.

Optional appearance gates/updates apply to full detector evidence. A required
appearance policy accepts identity-bound local-flow geometry without fabricating
a fresh descriptor or updating the retained template. Reports mark appearance
unused locally; local source IDs are `local_flow:<track-id>`. Missing descriptors
on a required full refresh reject.

Global detector assignment corrects the filtered state and reseeds local features.
This corrects bounded drift; a detector outside the configured association gates
can establish a new identity instead. There is no guarantee of identity through
indistinguishable parts, unobservable motion or arbitrary occlusion.

## Freshness, results and bounds

Maximum detector age frames counts eligible tracker activations since the last
full detector match. Maximum detector age ms separately uses the configured
acquisition/media/nominal clock. Either limit expires a track even when local
motion continues to succeed. Ordinary missed-frame, elapsed-time, covariance,
interval-gap and offscreen limits still apply. Allow enough freshness for the
chosen cadence; an intentionally shorter limit expires tracks between refreshes.

Tracks and corresponding poses include `measurement_source` (`detector`,
`local_flow` or `prediction`), `detector_observed`, `last_detector_frame`,
`detector_age_frames` and `detector_age_ms`. Ordinary observed/confirmed flags,
current tickets/grids, velocities and covariance remain available. Local image
evidence counts toward the normal minimum observations; it is not a fresh
full-frame detection. Filter those flags before operations that require one.

Local confidence uses the last detector score and retained local-inlier fraction,
then the ordinary confirmation/time/ambiguity factors. It is a heuristic. The
Kalman covariance follows the declared measurement/process noise; correlated
local estimates are not independent metrology or a calibrated probability.

Report gives refresh/frame index, local acceptance/rejection reasons and expiry
reasons. Cadence/tracker `detector_calls` count emitted/consumed refresh events;
for the provided graph they match independently counted detector executions.
These counters are not a profiler of arbitrary nested detector internals.
Graphics are selectable drawable objects: green detector evidence, blue local
evidence, amber prediction. Compose them onto the source image and select that
image for the filmstrip. Lean/graphics-off skips annotation construction while
retaining the estimates and source pixels.

The common sequence memory budget bounds copyable arrays and track metadata.
Limits include at most 4096 corners across configured tracks and a pre-native
sum of points times window area times pyramid levels. Backward checking can add
one LK call per track. These are bounded work controls, not exact native RSS,
latency or runtime speed guarantees. Runtime reset/close/errors retire history;
worker copies are independent and saved projects store configuration, not live
optical-flow history.

## Evidence and examples

[Three original annotated projects](<../examples/opencv tools/detector_refresh/README.md>)
show entry, local tracking, loss/expiry, detector-call budgets and deliberate
freshness expiry. Actual native blob detections drive them; independent annotations
record intended object motion. Period four executes the detector four times over
sixteen images; the every-frame baseline executes it sixteen times. Tests verify
actual processing calls, not just reported counters. This establishes a call
budget, without claiming a measured elapsed-time speedup.

Native translations and a rotated/scaled image are compared with independent
transforms. A controlled injected local-fit bias verifies correction at the next
detector refresh. Mask/ticket guards, invalid depth and work ceilings, native
failure recovery, no local births, forced cadence/reset, worker/XML persistence,
dropdowns without native execution and Debug/Lean replay cover the surrounding
contracts. These are synthetic baseline checks, not production tracking acceptance.
