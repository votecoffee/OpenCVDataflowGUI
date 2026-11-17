# SimpleDeflectometry E0 contracts — draft v1

The initial API/CLI implements registered temporal X/Y wrapped phase comparison
and a separate known-plane reflection API. Native retained teaching and cached
execution are delivered in [E2](RECIPE_CONTRACT.md). Absolute screen decoding
and shape reconstruction remain implementation targets. Start with
registered monochrome, stationary-surface sequences and imported known geometry.
Keep specular reflection separate from diffuse projector-camera ray intersection.

## Capture and reference

Retain original reference frames, exact observed order/IDs, X/Y axis, screen
pattern identity, frequency/period, phase origin and shift angles. Include dark/
white frames when the sequence supplies them. Missing steps are errors, not
interpolated data. A reference and test must match the setup, source grid,
radiometry and observed sequence. Caller capture IDs do not prove hardware
synchronization, stationary parts or actual frame order.

The implemented manifest is `simple_deflectometry_capture/v1`, with `sequence`
containing `setup_id`, `pixel_grid`, `exposure_s`, optional `response` and `axes`
X/Y records. Each axis contains observed `frame_ids` and `phase_angles` in radians,
optional known `period_px`/`origin_rad`, and direction +1/-1. Unknown periods/origins
remain null; the initial comparison never converts phase to absolute screen pixels.
`frames` lists X then Y `{id, image}` records in observed order, relative to the
manifest. `source_encoding`, `input_scale`, optional mask and JSON provenance are
explicit. See [a retained analytic capture](studies/reference/capture.json).

Normal processing declares `response: linear` and a common positive exposure.
An explicit `uncalibrated_codes` mode permits null exposure and fits normalized
code values without claiming calibrated radiometry. Its manifest encoding must
also be `uncalibrated_codes`; linear and code-value sequences cannot silently
match. The result records this scope. This diagnostic mode does not enable metric
geometry or physical accuracy claims.

Declare source encoding/scale, common exposure, original camera dimensions,
crop/resampling transform and screen pixel coordinates. Linearize exactly once;
do not normalize each frame independently. Saturated, nonfinite, weak-modulation
and poor-fit pixels retain separate reasons. Invalid values never become zeros
in numeric output; source masks never imply a measured surface outside coverage.

Use array-preserving runtime bundles and bounded numeric, checksummed retained
recipes, following the existing Simple Plugins conventions. No pickle, implicit
source replacement, automatic downloads or projector/display control on run.
Multi-frequency/Gray-coded observations may resolve screen coordinates later;
single-frequency phase alone does not resolve integer periods.

## Result levels

| Level | Required evidence | Output and explicit limit |
| --- | --- | --- |
| Wrapped reference difference | Matching X/Y phase sequences and registered reference | Wrapped phase change in radians, modulation/residual/valid reasons. Whole-period changes remain ambiguous. |
| Absolute screen correspondence | Verified coded or multi-frequency sequence, screen origin/axes | Observed screen [u,v] in pixels and reference displacement. No surface height follows from correspondence alone. |
| Reference-plane normal/slope estimate | Known camera, metric screen embedding and explicit reference-plane anchor | Apparent normal/slope on the assumed plane. Actual displacement from the plane creates model error. |
| Absolute reconstructed shape | Sufficient geometry, anchors/boundaries and independently checked reconstruction | Future capability; never enabled by the first two levels. |

For wrapped comparison, use the circular difference
`delta = atan2(sin(test - reference), cos(test - reference))`, with documented
range [-pi, pi]. Show the signed X/Y differences and magnitude using fixed shared
scales. Label the period ambiguity for every such result; proximity to pi can
signal a wrap boundary but cannot detect all whole-period ambiguity.

Do not average wrapped values across seams, unwrap across holes, fabricate a
continuous mask or promote a small fit residual to confidence in a correct code.
Retain reference/test invalid masks independently and compare only their valid
intersection. Report active/reference-valid/test-valid/comparable counts and
fractions with explicit denominators. Reference baseline repeatability and
application thresholds must be supplied or measured, not inferred from one image.

## Physical frames and reflection

Use a right-handed camera frame, x right/y down/z forward, camera origin [0,0,0].
Keep original pinhole intrinsics, OpenCV distortion and source-pixel transforms.
Represent a planar screen by an origin `O_s_m` and orthonormal in-plane unit axes
`e_u`, `e_v` in the camera frame, positive pitch in meters per screen pixel,
screen dimensions and explicit zero-based pixel-center origin. Then:

`S_camera(u,v) = O_s_m + u*pitch_u_m*e_u + v*pitch_v_m*e_v`.

Retain display surface/glass-layer conventions and uncertainty; a nominal planar
screen is an assumption. Do not silently swap projector intrinsics for this
physical screen embedding. Off-screen correspondence remains invalid.

For a normalized camera ray `d` pointing from camera toward the object and a
known assumed surface point `P = t*d`, the directions **away from the surface**
are `v_camera = -d` and `v_screen = normalize(S_camera - P)`. The camera-facing
reflection normal is their normalized bisector:

`n = normalize(v_camera + v_screen)`.

Require a forward ray/plane intersection, finite distances and a nondegenerate
bisector. Orient n toward the camera; retain grazing/near-degenerate evidence.
For graph-form surface z(x,y), only when n_z is safely nonzero, local slope
estimates are `dz/dx = -n_x/n_z`, `dz/dy = -n_y/n_z`. Compare reference/test normals
or slopes under the same anchor. This reference-plane estimate is not an
independent measurement of P or absolute height. A moved part or changed screen
pose can mimic a surface change; alignment/pose validity must remain explicit.

## Training and assessment

Show retained reference frames immediately when reopening, with mask/ROI and
sequence/screen/camera/anchor review. Provide zoom/pan and editable regions,
independent test previews, changed-draft invalidation, cancellation and stale-job
guards. Freeze preview inputs; Stage only a fresh reference result, update the
inspector draft and preserve applied state until Apply. Cancel retains prior
training. Do not ask users to paste arrays to teach a normal application.

Provide source-aligned reference/test, cyclic X/Y phase, modulation, fit residual,
signed change, vector displacement and unavailable-reason views. Compare captures
on fixed scales; show source coverage and rejected evidence. Output unpainted
numeric arrays separately from review images. An availability judgment is distinct
from a learned/specified surface-deviation judgment.

Use native fly-around graphics only for supplied/calibrated geometric entities.
An assumed plane/screen/ray review must be labeled **setup geometry**, not measured
shape. No point-cloud height is synthesized from wrapped phase. Graphics-off
must preserve numeric maps, validity, reports and judgments exactly.

## Reuse and execution

Reuse the validated N-step least-squares phase basis, source normalization and
reason semantics where justified. Keep reflection geometry in this tool. Decide
the smallest explicit shared module/package boundary during E1; avoid private
imports into another tool's GUI or a second diffuse-triangulation implementation.

Cache the sequence fit basis and fixed source rays/screen constants by retained
setup. Each capture recomputes signal fits, correspondence and validity. Bound
stack/model allocations before loading; use at most the existing 16-million
capture-sample initial scope, with larger tiled/streaming work explicitly deferred.
Independent instances may execute in parallel. Benchmark generic throughput
separately from graphics and avoid application-specific parameter tuning.
