# Native CAD-jaw training

`3d.simple_bin_picking_jaw` opens **Teach CAD jaw grasps…** in its Configuration
inspector. Version `0.1.0.dev5` restores the canonical CAD surface, named grasp
frames, tool dimensions, mounting transform and retained depth/pose reference.
The popup edits a private draft. **Stage recipe** proposes a new checksummed
asset; **Apply configuration** in the inspector activates it. Cancel preserves
the applied parameters and original training file.

## Begin or reopen

Existing `.sbpj` recipes restore their actual CAD rows and taught jaw graphics
immediately. Select a named grasp to inspect its contacts, fingers, fixed palm,
sweeps, retract direction and TCP axes. The separate measured-reference tab
shows the original camera-visible depth, with missing samples unfilled.

For a new part, choose **New from bin + CAD…**:

1. Import a calibrated metric `.sbp` bin/reference recipe. The existing suction
   trainer can establish its camera, floor and bin limits; its suction settings
   do not control jaw planning.
2. Import a PLY/PCD CAD point surface with outward normals and at most 50,000
   points. Declare m/mm for unmarked coordinates. Embedded unit/frame metadata
   is respected. Coordinates are converted to meters without moving the original
   CAD origin. Triangle meshing/normal inference is not performed here.
3. The editor seeds an envelope grasp for adjustment. This is an authored
   proposal, not a supported contact or force-closure result. Select actual
   opposing CAD contacts and a retract direction before evaluating it.

The first workflow uses a supplied calibrated bin and sampled CAD surface.
Integrated triangle-mesh sampling and bin calibration inside this popup remain
follow-ups; no package installation, mesh executable or robot call is hidden in
opening or using the editor.

## Teach contacts and mounting

In **Contacts**, choose **Pick first contact** or **Pick second contact**, then
pick an actual visible CAD point. Orbit to expose the opposite face. Picking
uses the complete retained source and returns its original row; generated jaw
overlays never become pickable CAD rows. A changed source invalidates old picks.
The precise retained float64 row supplies the contact, independently of the
viewer’s float32 display coordinates.

Adjust contact XYZ numerically in **mm** for fine control. **Pick retract tip**
uses a CAD point relative to the current grasp center; alternatively enter a
direction vector. Closing +X follows first-to-second contact. Retract +Z is the
entered direction projected perpendicular to +X. Coincident contacts and parallel
closing/retract directions are rejected. Apply the contact edit to the draft
before switching grasps. Duplicate, remove and bounded undo support alternatives.

Use wheel zoom, middle/right-drag pan, Orbit and XY/XZ/YZ views. **Fit CAD + jaws**
frames the derived geometry without inserting it into the CAD source. Sweeps
can be hidden for precise contact work. Editing contact overlays does not run
the matcher or rebuild the source projection.

**Gripper** controls jaw stroke, opening margin, finger/pad/palm geometry,
approach distance and clearance, displayed in mm. **Evidence / workers** controls
pose/contact tolerances, sample spacing, support/contradiction percentages,
overlap suppression, candidate count and 1–8 workers. **TCP mount** edits the
grasp-from-TCP translation in mm and axis-angle rotation vector in degrees.
Grasp +Z still owns approach/retract when the TCP mount is rotated. Untouched
controls retain their stored values instead of being read back from rounded
display fields.

## Freeze a reference and preview

**Capture connected reference** copies the wired depth, explicit current
`part_instance_v1` pose records and capture ID, plus any connected camera,
bin-pose or instance labels. It also works before the target has run successfully.
The retained setup must already identify the calibrated bin and CAD frame.
Camera calibration, units, part/bin frames, labels and capture IDs are checked.
Snapshot/publication IDs are provenance, not proof of synchronized exposure.

**Import reference…** accepts a bounded JSON record containing `capture_id`,
`part_instances`, optional `camera`/`bin_pose`/integer `instances`, and a numeric
NPY depth file. Untagged depth uses the retained scale. Unit-bearing connected
depth must be meters with scale 1. New captures replace stale reference RGB
rather than associating an old image with new depth.

The captured depth, optional labels and supplied part poses are saved inside the
new `.sbpj` asset. Reopening can therefore **Preview reference** without graph
inputs. Legacy recipes retain their depth/model/grasps, but lack pose-reference
records: capture or import those records once. Poses are never inferred from the
CAD origin, a cached result or dataset metadata without an explicit input.

**Preview connected** instead freezes one current publication for diagnostic
evaluation without replacing the retained reference. A bounded background
worker evaluates the draft. Edits, cancellation, changed CAD/capture, closing the
popup or a newer connected publication prevent a stale result from being staged.
Native operations finish their current bounded call before cancellation is
observed. The worker never executes the graph or robot.

## Assess, stage and export

The candidate table separates CAD contact support, current pose fit, geometric
score, clearance state and reasons. Select a candidate and toggle layers to
review cached native colored clouds without re-planning. Green is eligible,
red rejected, orange unknown and magenta collision; cyan CAD and blue sweeps are
derived graphics. Raw rejected candidates remain visible. Missing depth does
not become free space through CAD assistance.

The **Pose fit** layer colors the tested CAD samples as supported, contradictory,
occluded, unknown, ambiguous or label-mismatched. Counts below the table cover
every tested ray; display points are a bounded sample per state. Hide Model and
the gripper layers to see this evidence clearly. [Color and residual guide](JAW_POSE_FIT.md).

Only a completed preview of the current draft enables **Stage recipe**. Expected
rejection/unknown is a valid preview outcome and stays rejected after Apply.
Staging writes a unique asset and updates only the inspector draft. The original
recipe is never replaced. Cancel before staging makes no new asset; discarding
an already staged inspector draft can leave an unused unique asset for later
review rather than deleting potentially useful training.

**Export cached review…** writes a fresh folder with the selected colored PLY,
complete frozen preview recipe and report/provenance. The export uses the
previewed geometry/capture even if the current draft has since changed.

This workflow evaluates finite geometric contact and first-return gripper
clearance. Native teaching does not qualify force closure, friction, transported
part volume, continuous/full-robot collision, IK or physical pick success.
