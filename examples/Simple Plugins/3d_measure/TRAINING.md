# Teach a new 3D measurement

Requires Simple3DMeasure `0.1.0.dev1` (planes), `0.1.0.dev2` (lines), or
`0.1.0.dev3` with optional SciPy (minimum-zone flatness), and an OpenCVDataflowGUI build containing
the native cloud-teaching API. Older hosts explicitly fall back to recipe
review/import. Keep `mm` or `m` and coordinate frames explicit; units are never
inferred from object size.

1. Add Simple3DMeasure, connect a calibrated point cloud and save the project.
   Open **Train 3D measurements…**. Import a PLY/PCD/NPZ reference, or run the
   source once and use **Capture reference… → Connected scene**. With dev4,
   RGB, source pixel rows/size and bounded JSON context are retained when supplied.
   **Show source RGB** changes point appearance; **Reference context…** shows
   read-only source metadata. That capture choice
   uses the scene's declared frame. For existing tools, **Completed scene +
   fixture** captures consumed inputs together and preserves the recipe's
   measurement frame. No fixture is estimated by this editor.
2. Choose **New plane region** or **New line region**, then **Draw region**
   around a planar patch or narrow line samples. The cyan
   points show the samples inside the proposed XYZ volume. Orbit, zoom with
   the wheel, and pan with middle/right drag to inspect it. Drag colored XYZ
   face handles to extend or trim the volume. XY/XZ/YZ views help separate
   axes. Draw a datum and any target surface regions; name them clearly.
3. **Add measurement…**. Select the datum/target and a measurement type.
   A surface range needs one region; step height and angle need two.
   **Minimum-zone flatness (all selected)** also uses one plane region. Its
   geometry budget bounds hull work; exceeding it leaves the value unavailable.
   This measurement includes samples excluded from the datum fit. In review,
   orange/violet zone planes and support markers show which points limit it.
   For a point-to-plane/line measurement, choose Pick source point in the viewer,
   click the desired visible point and use it in the measurement dialog.
   Optional lower/upper limits use the displayed physical units or degrees.
4. **Train / preview reference**. Training runs on one frozen draft in a
   background worker. The selected region shows green fit inliers and red
   fitting exclusions, its counts and learned physical band, and RGB datum
   axes. Blue plane datum Z follows the signed normal; a line's +X follows
   its direction. **Flip axis** changes the normal/direction
   sign, then requires retraining. The table shows training-reference values;
   they are not independent accuracy or acceptance evidence.
5. **Review trained / completed run…** opens the native flyaround evidence
   viewer. Capture an independent completed run to inspect its fit and values.
   Scene/fixture inputs are copied from the same completed block execution.
   Change layers/regions without refitting; export the visible PLY, portable
   recipe, raw report, hashes and capture identity.
6. **Stage recipe**, then inspector **Apply**. Stage only edits the inspector
   draft; Cancel leaves active settings intact. Reopening restores the complete
   reference XYZ, regions, measurements, normal orientation and fit bands.

Region, name, band, normal, measurement and limit edits share bounded Undo/Redo
(32 definition snapshots). Deleting a region removes its dependent
measurements; Undo restores them. Reference replacement clears history and
invalidates old jobs; a change of units/frame also clears incompatible
definitions. A same-frame replacement keeps the defined regions for explicit
retraining. The prior trained recipe cannot be staged after any edit, even
after Undo, until the current draft is trained again. Late results from an
older draft/source are discarded.

A drawn patch seeds a 3D axis-aligned volume, so it can include points hidden
in the current view. The wireframe and cyan selection show the actual volume.
Picking returns an original sensor row, independently of wireframes, axes and
display colors. Replacing a source clears the pick. The editor does not infer
missing surfaces, physical accuracy, object identity, or dimension uncertainty.

With dev4, the recipe also retains supplied source RGB, pixel rows and bounded
calibration/acquisition metadata. Completed captures retain source/measurement
frames and fixture/snapshot identity. These descriptive fields do not apply
calibration or establish synchronized acquisition. Older XYZ-only recipes
explicitly lack this context. See [reference context and exports](REFERENCE_CONTEXT.md).
Closing a trainer detaches its bounded worker and discards its result;
interrupting an already-running NumPy fit remains future work. Additional
primitives, oriented regions and sections remain in the roadmap.
See [line-specific definitions and observed-support limits](LINES.md).
See [minimum-zone definitions, dependencies and numerical limits](FLATNESS.md).
