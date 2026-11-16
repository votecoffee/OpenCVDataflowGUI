# Reference-based rolling-shutter correction

`inspect.scanline_motion` (**OpenCV > Video Analysis (video)**) estimates horizontal
and vertical displacement as a smooth function of reference-image row. Connect
its **Flow** and **Support** to `cv.warp_flow` (**Warp image with dense flow**),
with the current capture as **Image** and the stable reference as **Target**.
Keep the warper's **target_to_source** direction. Expose its Target and Flow
support inputs in the editor when wiring manually. The
[six saved examples](<../examples/opencv tools/rolling_shutter/README.md>) are
already connected.

This is a reference-based row-translation baseline. It does not implement the
native C++ [MoreAccurateMotionWobbleSuppressor](https://github.com/opencv/opencv_contrib/blob/5.x/modules/videostab/src/wobble_suppression.cpp),
which this Python wheel does not expose. It is not a physical exposure/readout
calibration, an automatic arbitrary-video stabilizer or a general projective
rolling-shutter solver. Global camera motion can be included in its fit, but
x-dependent rotation/perspective, parallax, changing scenes and high-frequency
vibration need different models or upstream handling.

## Fitting and validation

Use matching gray/RGB/BGR/RGBA/BGRA images at uint8, uint16, float32 or float64
depth. The two inputs must share shape, depth and encoding; their acquisition
roots and tickets can differ. Images need at least 24 pixels per axis, fewer than
32767 per axis, at most 2.1 million pixels and the block's 256 MiB conservative
workspace estimate. A work proxy bounds features, LK window area and pyramid
levels before filtering. The proxy is per direction; forward and backward LK
both run. No global thread settings change.

Color is converted to luminance with its declared channel order. Both captures
share one scale and one luminance range when creating the native 8-bit tracking
images; neither is normalized independently. Extreme finite Float64 magnitudes
remain bounded during conversion. The report records the scale/range. Original
pixel data is never replaced by this tracking representation. Straight alpha is
required; transparent pixel centers cannot supply tracks. Empty/flat selections
produce insufficient evidence, rather than a passing identity field.

Native Shi-Tomasi reference corners are tracked into the current image with
pyramidal Lucas-Kanade, then checked by backward tracking and patch error. Use
**Reference mask/region** to retain static texture across rows and exclude parts
expected to move. **Mask/region** restrict accepted current-image track centers.
These selections intersect and check their own coordinate frames. They select
centers, not all pyramid/filter context: leave adequate clearance around moving
parts. If a current mask also describes invalid image support, wire it to the
warper's **Mask** input as well; a fitting selection alone does not cut output
coverage.

Accepted pairs are sorted by reference row then x. Every fourth pair is withheld
from fitting. A deterministic Huber iteration and one training-inlier refit
estimate two polynomials, with selectable degree 1-5:

```
u = 2 * y / (height - 1) - 1
source_x = reference_x + sum(coefficient[k][0] * u**k)
source_y = reference_y + sum(coefficient[k][1] * u**k)
```

Coefficients are reported in ascending power order. Require sufficient training
inliers, withheld validation inliers, row span and no excessive gaps between
inlier rows. Only the observed inlier-row interval plus **Extrapolation rows** is
supported. Set extrapolation to zero to withhold all unobserved edge rows. Bounds
also reject excessive displacement, excessive changes between adjacent rows and
unsafe vertical row ordering (`1 + vertical flow step`). These checks do not
establish global physical correctness or general invertibility.

## Outputs, failures and coordinates

| Output | Contract |
| --- | --- |
| `flow` | Float32 reference-grid **target-to-source** pull displacement, or no event when fitting fails. |
| `support` | UInt8 validated row support on the reference grid. Unsupported rows have zero vectors and zero support; always connect both outputs. |
| `report` | Model/units, tracking and holdout counts, residuals, coefficients, observed/supported row extent, geometric guards and rejection reason. |

The report includes all surviving track pairs and their fitting/validation roles.
Optional drawable arrows show a bounded sample in reference coordinates; green
denotes training inliers, yellow denotes held-out tracks, red denotes rejected
training tracks. Graphics-off and Lean mode skip these drawings. Held-out arrows
are identified by role; inspect residual counts for their acceptance.

The field carries the current ticket/custom payload on the reference's spatial
and pyramid grid. A content-bound image-pair manifest seals both source and
reference pixels/type/encoding/geometry, the field hash and its direction. The
warper checks the source and any connected target anchor before sampling.
Changing a paired image, vectors or direction rejects stale use.

The existing warper resamples the **current** capture, retaining its depth,
encoding, ticket and acquisition timestamp. It does not replace moving objects
with the reference. Its coverage intersects fitted support, donor support,
image bounds and interpolation footprints. Unsupported borders remain explicit;
they do not gain valid pixels through extrapolation. Result images use reference
coordinates, and `map_x`/`map_y` retain the actual output-to-source mapping for
nonlinear provenance. See [dense-flow examples and contracts](<../examples/opencv tools/dense_flow_warp/README.md>).

Insufficient features/tracks, degenerate rows, failed consensus/validation,
inadequate row coverage, excessive motion or unsafe geometry produce an ordinary
failed inspection, a reason, zero support and **no flow emission**. No stale
previous field is re-emitted. Dependent correction blocks therefore do not run.
Malformed images/settings and unexpected native errors use the common contained
validation/exception path. No trained state, cursor or camera resources persist.

## Qualification and limits

Independent analytic row motion and inverse bisection/array sampling generate the
fixtures. Native feature/LK fitting is checked against known horizontal and
vertical cubic motion, near-edge field predictions, image-error reduction,
lossless FFV1 replay and acquisition timing. A moving-part project fits only a
static strip, then verifies that current part pixels remain at the corrected
position. Tests also cover robust outliers, all model degrees, withheld-track
rejection, insufficient row span/gaps, strong out-of-model motion, row folding,
resource guards, image-pair corruption, reset/concurrency, XML/reopen and
offscreen controls.

Track/holdout agreement is not proof of scene rigidity, a visibility score or a
defect judgment. Moving objects can dominate tracks; exclude them and validate
the reference/model on representative captures. Repeated patterns, blur,
lighting changes and an already distorted reference can invalidate the fit.
Retain raw captures for inspection qualification. The synthetic fixtures establish
the stated software behavior, not camera or production acceptance.
