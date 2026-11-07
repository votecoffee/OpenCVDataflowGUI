# Pose-relative inspection

Named projects in this guide refer to optional local examples. See the
[public example gallery](../examples/public/README.md) for bundled projects.

An image pose defines a local 2D coordinate frame in an image. Use it to locate an
inspection on a part after pattern matching or blob/line/contour detection. These
are image-plane transforms, not 3D robot poses or calibrated physical units.

## Blocks

| Block | Inputs | Behavior |
| --- | --- | --- |
| Define image pose (`pose.define`) | image | Fixed origin, angle and positive uniform scale |
| Pose from feature (`pose.from_feature`) | image, features | Frame from a ranked measured object or pattern match |
| Offset pose (`pose.offset`) | pose | Compose a local x/y/angular offset |
| Region relative to pose (`region.pose`) | image, pose | Position a rectangle, circle or polygon in the target image |

Pose blocks are under **Inspection > Poses**; relative regions are under **Image
processing > Regions**. Define/feature blocks draw local axes on their overlay.
Local x is red, local y is green. Pose outputs include the `pose`, one origin in
`points`, and the local x-axis `angle_degrees`.

## Pattern origins and offsets

Both trainable matchers expose `pose_offset_x`, `pose_offset_y` and
`pose_offset_angle`. Their default pose origin is the center of the training
selection's bounding box. The default axes follow the fitted pattern transform.
Offsets are measured in training/model pixels, not target-image pixels; scale and
rotation follow the pattern. Changing these runtime offsets needs no retraining.

Positive pose angles rotate counterclockwise, following OpenCV's convention. At
zero angle local x points right and local y points down. Compose offsets as
`new_pose = old_pose @ translation(dx,dy) @ rotation(angle)`.
For example, a +90-degree pose at `(80,70)` with scale 2 and local offset `(10,5)`
has new origin `(90,50)`. The offset is expressed in the old frame, before the
additional angular offset rotates the new axes.

Matcher `points`, match `center`, corners and the original model-to-image
`transform` continue to describe the detected pattern. Each match additionally
contains its offset `pose`; the pose port supplies the best match's pose. Use
**Pose from feature** with `features` connected to `matches` to select another
match by zero-based `index`. Existing match poses, including matcher offsets,
are retained; the feature block can add another local offset.

A negative pattern inspection returns an empty pose `{}`. Pose/relative-region
blocks reject that input and clear prior results, preventing an old position from
being inspected as though it came from the current run.

## Blob, contour and line poses

Connect a detector's `collection` output to **Pose from feature > features**,
along with the image to inspect. Filter/sort the collection first when selecting
the largest, leftmost or another ranked object. Index 0 means the first current
item; it is not a persistent tracked identity. An empty collection or unavailable
index is an execution error.

The origin is the measured centroid/line center. Choose `angle_mode=fixed` to use
`angle_degrees` (the default), or `feature` to use the measured principal axis or
line direction. Collection angles are image-clockwise and are converted to pose
counterclockwise angles. Principal axes have a 180-degree ambiguity. Symmetric
objects without a stable orientation reject feature-angle mode; select a fixed
angle for those objects. Feature poses start at unit scale in their working image.

Contour coordinate offsets are accounted for. Spatial payloads let these tools
map a pose from a crop, resized image or pyramid level to another image with the
same known root. Both position and local axes are transformed. Different known
roots are rejected even when dimensions match. Without a shared known root,
dimensions must match and coordinates are assumed to use that same image grid.

## Define regions in the local frame

Connect a pose to **Region relative to pose**, and connect the target image to
its image port. Define rectangle `[x,y,width,height]`, circle center/radius or
polygon vertices in local pose pixels. Negative positions are allowed: the default
rectangle `[-20,-20,40,40]` surrounds the pose origin. Click **Draw / edit region…**
to draw or change it over the captured image. Pointer coordinates, drag handles
and shape movement are transformed into local pose pixels; circles remain circles
in the local frame even when they appear scaled or distorted on the image. The
editor shows local axes and identifies its coordinate mode. Numeric fields remain
available for exact local geometry. Apply saves local parameters, so future poses
move the region; Cancel leaves the block unchanged.

**Shape mask** (`mask.shape`, under Image processing > Masks) has its own **Draw /
edit mask…** entry. It outputs a uint8 binary mask, a reusable region definition
and an overlay. Leave its optional `pose` port disconnected to draw in image pixels,
or connect a pose/coordinate frame to draw local geometry that follows it. Both
mask and relative-region editors support rectangles, circles, polygons, inversion,
resize/move handles and transactional acceptance. The existing **Shape region**
editor continues to define image-space regions.

Editors capture the image and pose together. Run once before editing a newly
connected frame. A cached pair can be reopened after applying geometry while the
same node remains selected. A fresh missing/failed pose is rejected; it cannot
silently reuse an earlier frame. Stop continuous execution before editing.

The region follows the full pose matrix. A scaled circle can become an ellipse;
a homography retains projective distortion. The local shape is rasterized in a
tight bounding box, then mapped with nearest-neighbor sampling and clipped to the
target image. This preserves geometry on either side of the local origin.
`invert` selects the target-image area outside the positioned shape. Degenerate
matrices, projective horizon crossings and oversized local rasters are rejected.
The existing region limits are 16384 pixels per axis and 64 million pixels.

Outputs are an ordinary region, binary mask and preview overlay. Connect `region`
or `mask` to downstream histogram, edge, threshold, blob or other selection-aware
blocks. Regions can also participate in existing union/intersection/difference
operations. Pose and region definitions remain JSON serializable for save/load,
background execution and reuse in nested ToolBlocks.

The `image_pose_v1` data contains a full `source_to_target` 3-by-3 matrix, target
dimensions, target coordinate payload, origin, angle and local x/y scales. The
angle/scales summarize the Jacobian at the origin for projective poses. Use the
matrix for actual mapping. Connect `pose` directly to **Transform points >
transform** to map local points into the target frame.

## Example

Open pose-relative inspection, then press F5:

1. Train the synthetic geometric reference and find it rotated +10 degrees.
2. Set the matcher's pose origin on its circle with offsets `(20.5,-17.5)` from
   the trained rectangle center. The expected image origin is near `(142.65,106.71)`.
3. Pass through an **Offset pose** block, initially zero. Change its local offsets
   to move the inspection independently of matching.
4. Position a radius-18 circle and calculate a histogram using that moving region.
5. Draw or edit the **Draw a local mask** block and inspect its separate masked
   histogram. The region and mask branches follow the same pose.

No camera, asset download or output writer is needed. Headless verification:

```powershell
.\.venv\Scripts\opencv-dataflow-run examples/applications/pose_inspection/pose_inspection.json --repeat 3
```
