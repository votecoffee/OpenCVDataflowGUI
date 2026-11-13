# Circle and ordered arc fitting

Open [project.json](project.json) and select **Run once**. The original 16-bit
illustrative part image accompanies polar-generated, image-coordinate points.
**Vector data** holds these fixture observations; it does not run an ML model.
Replace it with an ordered contour/blob collection or points from an edge tool
when building an inspection. **Fit circle or arc** fits supplied geometry; it
does not detect edges from image pixels.

The independent generating circle has center **(128, 96)** and radius **60 px**.
The partial arc spans **210°** across the angular branch cut and contains
Gaussian radial noise plus an outlier every ninth sample. On the pinned build:

| Branch | Center | Radius | Radial RMSE | Used points |
| --- | --- | ---: | ---: | ---: |
| Robust partial arc | (128.001, 96.020) | 59.981 px | 0.168 px | 161/181 |
| Ordinary fit, outliers retained | (127.454, 96.076) | 60.240 px | 4.576 px | 181/181 |
| Noisy full circle | (128.000, 96.010) | 60.002 px | 0.098 px | 180/180 |

The ordinary comparison deliberately permits up to 10 px RMSE and relaxed point
ordering so its biased model remains visible. The robust and full-circle
branches use the default checks. The full-circle fit also drives **Region from
fitted shape**, showing how to make a reusable inspection region from the fit.
All seven processing blocks should pass. Generating truth and bounds are in
[truth.json](assets/truth.json); automated checks require robust center and
radius errors below 0.1 px.

Open [short_arc.json](short_arc.json) to try the negative case. Its five-degree
arc must fail the default minimum-span check. The primary **Fit**, **Center**,
**Radius**, and **Arc** outputs are withheld. Candidate values and failure
reasons remain in **Measurements**, with diagnostic graphics available. This
project intentionally returns a failed run; its other two blocks should pass.

## Controls and results

- **RANSAC** rejects points beyond the inclusive radial tolerance, then refits
  until the consensus is stable. Trial limit, adaptive confidence target, local
  random seed, geometric iteration limit and pixel convergence tolerance are
  configurable. **Least squares** fits every eligible unique point. Its inlier
  fraction is 1 because it does not reject outliers. Neither score nor RANSAC's
  confidence setting is an inspection-pass probability.
- **Points** accepts an ordered list or a contour/blob collection. Select the
  current collection item and, for a blob, its outer or inner boundary. Exact
  duplicate coordinates vote once and do not artificially reduce uncertainty.
- **Auto** treats detected boundaries and repeated-endpoint lists as closed.
  If selection or consensus removes part of a closed boundary, Auto describes
  an open arc starting after the largest gap. **Open** and **Closed** override
  that convention. Positive angles run clockwise from +X in image coordinates.
  End angles agree with signed sweep; the last observed bearing is also
  reported. These bearings follow point order and are not an object pose.
- Angular support is the smallest enclosing angle interval of the used points,
  excluding their largest circular gap. It is an envelope, not proof of
  continuous observation, visibility or occlusion. Gaps, backtracking and
  multiple turns have separate order checks. Closed-path sweep can be 360°
  while observed support is less than 360°.
- Minimum distinct support, inlier fraction, angular span, radius limits and
  Jacobian condition qualify the circle. Too-short or unstable support withholds
  its primary geometry. Arc order and optional RMSE judgments are separate:
  a qualified circle can remain available when the arc order or RMSE check
  fails. Wire failure policies decide whether downstream tools accept it.
- **Measurements** includes candidate geometry, counts, convergence and
  consensus information, radial RMSE, conditioning, arc details and uncertainty.
  **Residuals** preserves source indices and signed radial errors in a bounded
  table; zero sample limit disables that table. **Points** outputs the complete
  fitting support in source order.

Uncertainty uses the radial-residual Jacobian for parameters `[center_x,
center_y, radius]`, with covariance `SSE/(n-3) * inverse(J.T @ J)` evaluated
through SVD. It assumes independent, identically distributed radial errors and
is conditional on the selected support. Three-point fits have no residual
degrees of freedom and report unavailable uncertainty. It does not include
calibration error, correlated edge samples, robust-selection bias or systematic
measurement error. [Circle-fitting problem reference](https://arxiv.org/abs/cs/0301001).

## Frames, graphics and bounds

Fitting uses centered, scaled float64 coordinates. Collection origin offsets
are removed. An optional image maps compatible named root frames into its own
pixel grid before fitting; unrelated roots, horizon crossings and different
known tickets are rejected. With an anonymous same-size reference image, the
known point frame is retained. Output payloads preserve ticket, spatial,
pyramid and companion context. Supplied image pixels, depth and encoding are
unchanged. Fitted dimensions are pixels of the fitting grid; rectify to a
calibrated workplane before interpreting them as physical dimensions.

An optional mask and region select points at their nearest pixel centers before
fitting. Without an image, a declared point or selection grid supplies the
selection dimensions. Selection does not manufacture closure evidence.
Diagnostics show used points in green, excluded eligible points in red,
unselected points in gray, a gray complete candidate circle and a colored
ordered arc. Graphics can be disabled independently of the raw fit. Diagnostic
point markers are bounded to 1,000; the solver uses all eligible points.

Limits are 100,000 input rows, finite coordinates within ±10M pixels, 64M image
or selection pixels, 16,384 per image axis, 32M RANSAC point trials and 32M
geometric point iterations. Refinement permits at most five consensus rounds
and sixteen line-search attempts per geometric iteration. These fixtures
qualify the declared CPU fitting behavior, not a particular manufactured part.

```powershell
opencv-dataflow-run "examples/opencv tools/circle_arc_fitting/project.json"
# Expected failure: insufficient angular support
opencv-dataflow-run "examples/opencv tools/circle_arc_fitting/short_arc.json"
```

Regenerate with `python "examples/opencv tools/circle_arc_fitting/generate.py"`.
All images, points, code and projects are original material under the repository
MIT license. Both projects contain an attribution/comment block.

[All examples](../../README.md)
