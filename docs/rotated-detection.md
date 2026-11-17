# Rotated object detection

**Decode rotated detections** (`ml.decode_rotated_detections`) is under **OpenCV →
Deep Neural Networks (dnn) → Object detection**. Connect the original Image and
matching inference Tensors/Preprocessing. For image batches, first select or
unbatch and unpack a single-image inference result. Loading a graph downloads no
models. This decoder owns no native model state and permits independent branches.

## Explicit head and angle formats

| Format | Row fields |
| --- | --- |
| `xywh_scores_angle` | center X/Y, width/height, class scores, angle |
| `xywha_scores` | center X/Y, width/height, angle, class scores |
| `xywha_objectness` | center X/Y, width/height, angle, objectness, class scores |
| `xywha_score_class` | center X/Y, width/height, angle, score, class ID |

Rows `[N,C]`, channels-first `[C,N]` and one leading batch dimension are explicit
choices. Outputs must be finite float32 `tensor_v1` records. Raw anchor/grid heads,
angle distributions, quadrilateral heads and quantized tensors need separate
adapters; no export format is guessed. Score/objectness probability or sigmoid,
labels, explicit-ID offset, class filters and strict score gates follow the
[axis-aligned decoder](object-detection.md). Each score-head row selects its best
class. Native NMS can be disabled for already suppressed exports.

Select degrees/radians and positive image-clockwise/counterclockwise angles.
The angle always describes the **width axis**; sides are never silently swapped
to make width the major dimension. Default period 180 describes an unoriented
axis. Period 360 is an explicit assertion that the model supplies direction, not
a guarantee that box geometry identifies a part's front. Normalized XYWH values
are expanded by network width/height **before** rotation in network pixels.

## Exact source footprints and poses

Preprocessing must give matching sizes and inverse positive scale/offset maps.
Four network rectangle corners map through the exact rounded stretch/letterbox
inverse. Non-uniform scaling generally produces a **parallelogram**; the decoder
preserves it instead of fitting a replacement rotated rectangle. Border policy
clips the source polygon, discards crossings or keeps the full outline. Completely
invisible detections drop. Clipping changes the visible footprint and bounding box,
without moving the predicted center or pose. Masks/regions filter the original
center's pixel cell; they do not alter inference or cut object footprints.

Image pixels/encoding/depth remain unchanged. The box collection carries ticket,
spatial/pyramid, recipe and timing context. Metrics include original predicted
center/width-axis angle, final polygon area/bounds, unclipped area, visible
fraction, score/class and full source-axis edge lengths. These are model box
metrics, not measured foreground area. Major/minor sizes are null when the
mapped axes are not orthogonal. Geometry retains full/clipped corners, canonical
network XYWHA, local corners and border/orthogonality flags.

Each item supplies an `image_pose_v1` affine pose. Local units are **network input
pixels**, with origin at the predicted center and model width/height axes retained.
`angle_degrees` in the pose uses the application's counterclockwise convention;
the collection metric uses image-clockwise degrees. Results mark orientation as
model-predicted, without asserting calibrated measurement.

Sort/filter the collection, then use **Pose from feature → Angle mode: Feature**.
This tool now prefers a supplied item pose and preserves its scale/shear; ordinary
blob/line items still use their axis metric. Fixed angle mode keeps its existing
behavior. **Region relative to pose** and offsets work in that model frame.
The additional Matches output uses the existing pose-match container for parallel
inspection/pick consumers; it contains detector proposals, not trained-template
recognition. A source pyramid level remains explicit, and shared-root pose mapping
can move the frame onto another image level.

## Suppression and limits

Suppression uses **unclipped network rectangles**. A shared invertible affine map
preserves polygon IoU, including when source footprints are parallelograms. Classes
are suppressed separately by default, avoiding cross-class coordinate offsets.
Results sort by score then original row ID. Adaptive eta and pre-NMS top-K are
applied per class, or globally in class-agnostic mode.

Default **Polygon IoU** uses native rotated intersections with correct
intersection/union area for containment. The installed OpenCV 5 native
`NMSBoxesRotated` treats a full-containment relation as overlap 1, so a small inner
box can suppress even when true IoU is low. **OpenCV** mode deliberately preserves
native behavior. Both modes are qualified by nested and crossing controls; the
native implementation is documented in
[OpenCV 5 NMS](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/src/nms.cpp).

Conservative bounds skip disjoint intersections. Candidate/result/pixel/raw-row
limits and a worst-case pair-comparison budget bound work; excess work rejects
instead of silently truncating. Coordinates stay within one million pixels and
rectangles that lose area at native float32 precision reject. Native overlap and
threshold decisions remain subject to floating-point precision. Allocation by
inference, runtime snapshots and arbitrary plugins is outside these bounds.

Count limits produce normal failed inspections with `detection.count`; empty
collections remain valid outputs. Bad types, geometry, known ticket/frame/recipe
mismatches, exhausted budgets and native exceptions emit no current outputs and
use the normal logger/exception framework. Valid later input can recover. Graphics
are drawable outlines, width-axis markers and labels; Off/Lean omits them. Use
Overlay result graphics and Set filmstrip image to retain rendered results.

[Twelve original MIT projects](<../examples/opencv tools/rotated_detection/README.md>)
qualify native image-dependent ONNX score heads, independent analytic corners,
suppression choices, source selection, affine pose regions and filmstrip overlays.
The proposals/angles are fixed numerical controls, without trained recognition,
industrial detection accuracy, physical pose calibration or throughput claims.
