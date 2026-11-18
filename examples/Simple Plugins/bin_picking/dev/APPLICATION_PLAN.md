# SimpleBinPicking example application plan

## Goal and evidence

Propose and assess suction contacts from a calibrated first-return image.
Output ranked alternatives, TCP/pregrasp transforms, contact/clearance states,
rejection reasons and a native RGB cloud. These are offline file-only graphs:
no device acquisition, outputs or robot motion. Passing means a geometrically
eligible alternative under this finite evidence model, not a physical pick.

## Inputs, frames and calibration

- Generated controls: integral uint16 axial-Z samples in mm, fixed pinhole
  camera, bin +Z up, supplied bin-to-camera transform and visible raised patch.
- T-LESS: real uint16 PrimeSense axial-Z, 0.1 mm per stored sample. Retain raw
  source hashes, camera dimensions/intrinsics and published world extrinsics.
  Canonicalize only the documented integer principal-point offsets; padding
  remains unknown. RGB is appearance, not geometry. No object GT/grasp labels.
- Explicit camera and bin-pose graph nodes are ordinary upstream calibration.
  They do not imply detected parts or synchronized sensors. Hardware accuracy,
  exposure pairing and independent bin calibration remain unqualified.

## Block choice and graph

Live catalog/schema discovery identified `inspect.select_pick` as an existing
image-plane candidate selector. It does not check physical suction contact,
first-return visibility, bin walls or a swept 3D tool. The optional
`3d.simple_bin_picking` block supplies that reusable geometric computation.
Use existing `input.file`, `3d.camera` and `3d.pose` blocks for file/calibration
inputs, then the single cached picker. Seal the recipe in the project.
The builder uses `ProjectSession`, static validation before execution and
the normal graph scheduler. Validation does not run processing.

## Frozen acceptance cases

Generated good/empty/missing/holes cases share one trained recipe. The narrow
bin control changes only the explicit workspace to exercise configured
collision. Good yields 10 alternatives; each negative yields none and absent
TCP/pregrasp. Real reference 3 and held-out 13/24 share identical recipe bytes
and settings; all yield zero eligible picks. Keep all failure reports.

Verify numerical equality with graphics disabled, original-member SHA-256,
normalized pixel identity, recipe identity and Save As portability. Exercise
native retained teaching, revision invalidation, Stage/Apply/Cancel and cached
selection/layers offscreen. These checks establish software behavior only.

## Review and remaining qualification

Native flyaround graphics show cups, TCP axes, approach lines, bin and tool
envelopes, plus observed collision and unknown-space witnesses. Every candidate
remains in raw output even when its graphic is omitted from the overview.
Keep original scene measurements separate from derived result geometry.

Next: independent licensed real bins, suction-label comparison, support/noise
models that respect calibrated tolerances, known-part and jaw grasps, retained
acquisition metadata, timing/memory measurements, full robot geometry/IK and
observed seal/pick outcomes. Do not interpret pose labels as grasp labels or
use the proposed list as a batch motion plan; reobserve after each pick.
