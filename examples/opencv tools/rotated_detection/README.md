# Rotated detection and pose inspection

Open a project and run once. The original arithmetic ONNX model and synthetic
images are included; no downloads, plugins or optional packages are needed.

| Project | Expected result |
| --- | --- |
| [stretch.json](stretch.json) | Six retained detections; non-uniform scaling preserves source parallelograms. |
| [letterbox.json](letterbox.json) | Six detections through rounded letterbox mapping and polygon clipping. |
| [channels.json](channels.json) | Same results from a channels-first tensor. |
| [radians.json](radians.json) | Same geometry from radian angle columns. |
| [objectness.json](objectness.json) | Explicit angle/objectness/class-score rows. |
| [explicit_ids.json](explicit_ids.json) | Explicit angle/score/class-ID rows. |
| [native_nms.json](native_nms.json) | Five results on this build: native containment handling suppresses the small inner part. |
| [agnostic.json](agnostic.json) | Five results: the identical other-class proposal is suppressed too. |
| [selected.json](selected.json) | One detection whose original center lies inside the selected region. |
| [pose_region.json](pose_region.json) | Sort by score, retain the supplied affine pose, and inspect a local region. |
| [count_failure.json](count_failure.json) | Intentional failed count judgment: six results when seven are required; graphics remain available. |
| [empty.json](empty.json) | Intentional failed count judgment with zero scores; valid empty collection. |

The two thin crossing proposals survive rotated suppression; their axis-aligned
bounding boxes overlap almost completely. The small nested proposal survives the
default corrected polygon IoU; native OpenCV behavior is available for comparison.
Overlay result graphics feeds Set filmstrip image in every project. Debug shows
outlines/labels/width axes; Lean retains numerical results and omits drawings.

Images, models, graphs and analytic references are original MIT work by
OpenCVDataflowGUI contributors, with attribution on each canvas. No customer or
third-party assets or trained weights are included. Fixed proposal positions and
angles use RGB image means for scores. This tests wiring, geometry and processing;
it does not recognize arbitrary parts. `truth.json` records independent source
corners, retained IDs and exact asset hashes.

`build_example.py` regenerates the collection in a fresh folder while preserving
different existing assets. See [rotated detection contracts](../../../docs/rotated-detection.md)
for row conventions, pose units, suppression, selection and numerical limits.
