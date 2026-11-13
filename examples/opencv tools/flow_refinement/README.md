# Variational optical-flow refinement

Open `flow_refinement.json`, then select **Run**. A deliberately biased seed field
is refined between two original texture images. Select **Refine dense optical
flow** to inspect support, before/after residuals and the optional color flow
view. Result graphics are arrows and a status label on the source image. The
downstream **Warp image with dense flow** uses the refined field and its support
to align the source toward the target.

- `flow_refinement.json`: known (+2,+1)-pixel motion, initialized at (+1.25,+0.5).
- `partial_support.json`: an editable source selection and an editable target
  exclusion region. The region's mask keeps the target's coordinate frame;
  `target_support.png` is an independent reference for the default mask.
- `residual_gate_failure.json`: intentionally rejects a mean residual above one
  gray level. The flow event is withheld, diagnostic outputs remain available,
  and downstream warping does not execute. The CLI intentionally exits with failure.

## Block contract

The block is under **OpenCV → Video Analysis (video) → Video processing**.
Connect source A, target B and an A-indexed `DenseFlow` or float32/64 H×W×2 seed.
Vectors are **source-to-target displacement in local source pixels**:
`B(x+dx,y+dy)` is compared with `A(x,y)`. The seed is copied; inputs are not mutated.
The output keeps A's ticket, grid, pyramid and acquisition data, with both source
identities recorded under the `flow_refinement` payload. It does not reticket the
result as B. Connect `support` alongside `flow` when consuming the result.

Inputs must be uint8 gray or color images on equal-size grids, at least 8 pixels
per axis and at most 2.1 million pixels. Color inputs are converted to canonical
grayscale for the solver; stored alpha is not a solver weight. Use explicit
upstream conversions for other depths and support masks to exclude transparency.
Known frames must agree by default. This example deliberately uses `pixel_grid`
alignment for two separately loaded images whose pixel correspondence is known.
Seed and support fields must still belong to their declared image grids.

All seven native controls are exposed: outer fixed-point iterations, inner SOR
iterations, smoothness alpha, brightness delta, gradient gamma, relaxation omega,
and robust-penalty epsilon. Epsilon is not a convergence threshold. Defaults match
the installed OpenCV solver; this fixture explicitly uses 10 outer/20 inner
iterations, alpha 1 and delta 10. See the
[OpenCV API](https://docs.opencv.org/5.0/main_modules/classcv_1_1VariationalRefinement.html).

The solver uses full-frame context and has no native cost mask. `mask` and `region`
intersect to limit output writes and statistics; vectors outside this selection
retain the initial field. Source/target support masks and `border_margin` exclude
untrusted areas, neighboring support holes and displaced donors from diagnostics.
They do not remove pixels from the optimization energy. Unsupported residuals are
NaN and unsupported flow-view pixels are black. Border extrapolation never gains
valid support. `border_margin` is an explicit diagnostic exclusion band, not a
guarantee that native boundary effects cannot propagate farther.

Optional gates reject inadequate supported fraction, excessive mean residual or
increased residual on common before/after support. No valid support always fails.
The score is supported/selected fraction, not a probability of correct motion.
Photometric agreement and support do not prove visibility, identity or accuracy;
occlusions, brightness changes and low texture can violate this local model.
This is single-scale refinement of a suitable seed, not a replacement for a
coarse-to-fine estimator. Work and displacement bounds fail explicitly.

## Fixture and verification

`build_example.py` generates all images, the Middlebury `initial.flo` seed, graphs
and `truth.json`. These are original MIT-licensed synthetic tool fixtures, with
no customer or third-party assets; each project includes an attribution comment.
The seed file itself has no frame metadata. Its association is supplied by this
project's known pair; arbitrary files need their own verified association.

The reference checks interior endpoint error against (+2,+1), independent of the
solver. It requires mean error below 0.6 pixels and below 65% of the seed's error,
plus a comparable image residual below 60% of its initial value. Targeted tests
also cover native controls/parity, color encodings, donor support, selections,
ticket/grid ownership, failure containment, XML presets and Debug/Lean runs.
These controlled checks do not qualify captured-scene motion accuracy.
