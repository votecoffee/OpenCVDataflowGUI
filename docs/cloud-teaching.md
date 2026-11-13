# Native point-cloud teaching

`CloudTeachingWidget` is a reusable frozen-source editor for plugin dialogs.
It is separate from graph execution, model fitting and recipe activation.
The normal `CloudViewer` continues to provide live result flyaround previews.
No GPU or additional dependency is required.

```python
from opencv_dataflow.cloud_teaching import CloudTeachingWidget
from opencv_dataflow.cloud_teaching_geometry import CloudBox

editor = CloudTeachingWidget(dialog)
editor.set_cloud(private_cloud_snapshot)
editor.set_boxes([CloudBox("datum", (-20, -10, -1), (20, 10, 1))], "datum")
editor.box_created.connect(add_region_to_draft)
editor.box_changed.connect(change_region_in_draft)
editor.point_picked.connect(use_source_point)
```

The widget privately copies source XYZ, colors, normals, pixel mapping and
metadata. Its point arrays are immutable. Picking uses **every source row**,
with one frontmost row per rounded pixel in the current orthographic view.
Coincident depth ties choose the smallest original row index. A pick contains
`row`, `xyz`, `units`, `frame` and `source_identity`. The identity hashes XYZ,
units and frame; it is not an acquisition ticket or timestamp. A plugin with
higher-precision training XYZ can use the returned original row index against
its own retained array after checking the frozen editor identity.

Pick mode selects sensor rows independently of generated overlays. Hidden
backside points never replace a foreground point at the same screen pixel.
Orbiting changes what is visible. This is view-space point occlusion, not a
mesh surface reconstruction or sensor uncertainty model.

Use the wheel to zoom, middle/right drag to pan, or Orbit mode and left drag
to orbit. **Focus selected** frames the selected region. XY/XZ/YZ buttons make
planar adjustments easy. Draw mode seeds an
axis-aligned XYZ box from the visible source patch, with a small physical
padding. The **volume** can include other/hidden samples inside its bounds;
it is not a stored 2D mask. Drag the colored X-/X+, Y-/Y+, Z-/Z+ face handles
to resize. Rotate the view when a face's movement axis points into the screen.
Crossing the opposite face is rejected. `CloudBox` values use source units
and coordinates. Consumers own names, uniqueness, undo/redo and validation.

`set_point_colors(rgb_bytes)` changes display colors only, reusing the cached
full-source projection. It does not change source rows, identity or picking.
`set_boxes`, `set_segments` and `set_datum` update derived overlays without
projecting the source again. Datum blue Z follows the supplied signed normal;
the tangent X/Y axes are constructed for display. Segment records are
`(start_xyz, end_xyz, rgb, text)` and orbit with the cloud. Overlay geometry is
never inserted into the source or returned as a picked sensor point.

Replacing the source clears picks, regions, overlays and an active drag.
Plugins must also invalidate drafts/jobs tied to the previous capture.
Use `PluginDialogContext.snapshot_processed_inputs` when a reference requires
scene and fixture from the same completed block execution. Copy latest
`snapshot_input_cloud` only when a single declared scene is sufficient.
Acquisition pairing remains the upstream graph's responsibility.

The initial API provides axis-aligned volumes and source point picking.
Oriented boxes, section/brush selection, sensor-depth uncertainty, accelerated
large-cloud interaction and interrupting in-flight NumPy operations remain
future work. Rendering/picking is a training activity; this API adds no work
to a trained tool's numerical execution.
