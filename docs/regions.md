# Masks and image-space regions

**Shape region** creates a rectangle, circle or polygon using the inspector's shape
dropdown and numeric/JSON fields. Connect an image to size the mask and preview;
the source image is not modified. It outputs a uint8 `mask` (selected pixels 255),
a JSON-compatible `region` definition, and a tinted `overlay`.

- Rectangle: `[x,y,width,height]`; width/height are positive pixel counts. The
  rectangle covers x through x+width−1 and y through y+height−1.
- Circle: `[x,y]` center and positive integer radius; the drawn boundary is included.
- Polygon: at least three integer `[x,y]` vertices with nonzero area; edges are
  closed automatically. Use simple, non-self-intersecting polygons.
- Invert selects everything outside the shape. Shapes extending beyond the image
  are clipped to the image boundary. A wholly excluded shape produces an empty mask.

Fields are in full input-image pixel coordinates. Configure them numerically in
the inspector or use the interactive editor below.

## Regions from fitted shapes

Wire the `fit` output of **Fit line, ellipse or enclosing circle** to
**Region from fitted shape** and connect the image in the same coordinate frame.
The block accepts ellipse and enclosing-circle fits. It retains the fitted
center and orientation and adds **Margin** pixels to each semi-axis. Negative
margins shrink the shape. The resulting semi-axes must stay positive.

The mask includes the rasterized boundary. **Invert** selects the outside.
The reusable `region` uses the `ellipse` shape with `center`, full
`axis_lengths`, and clockwise image-space `angle_degrees`. An enclosing
circle uses equal axis lengths. The outputs carry the image's frame.
[Fitted ellipse inspection band example](<../examples/custom tools/fitted_shape_region/README.md>)

## Draw and edit on an image

**Shape mask** has a **Draw / edit mask…** editor and an optional `pose` input.
**Region relative to pose** has a **Draw / edit region…** editor for local geometry.
These use the same drawing/resize/move gestures described below, while mapping
through the supplied coordinate frame. See [pose-relative editing](poses.md) for
local coordinates, captured pose behavior and an example project.

1. Connect an image to Shape region and **Run once** to capture a frame. A new
   region can also use its upstream block's latest output before the region has run.
2. Select the region node and click **Draw / edit region…** in its inspector.
3. Choose a shape from the dropdown, or click **Draw new** to replace the current
   shape of the same type. Drag between rectangle corners; drag from a circle's
   center to its radius. Click polygon vertices, then Enter, double-click or
   right-click to finish. Backspace/Delete removes the last unfinished vertex.
4. Drag yellow handles to resize a rectangle, change a circle's center/radius,
   or move individual polygon vertices. Drag inside a shape away from its handles
   to move the entire shape. Shape movement tests the shape interior even when
   inversion selects the outside.
5. **Apply region** applies the inspector configuration, clears stale results and
   marks the project modified. Run again to update outputs, then save normally.
   **Cancel** leaves the block and any existing inspector drafts untouched.

Escape cancels an active drawing/drag gesture and restores its prior geometry;
otherwise it closes the dialog. An unfinished or zero-area polygon cannot be
applied. Pointer coordinates map through the fitted image rectangle, including
letterboxing and resizing; clicks in the margins do not start shapes. Drag
endpoints clamp to the image bounds. Shapes moved partly outside an image retain
their geometry and use the normal clipped mask behavior.

The dialog edits a **captured frame**, not a live stream. While the same node stays
selected, its last captured source remains available for repeated edits after
applying configuration. Run again after changing upstream images or geometry.
Changing selection or the source connection invalidates that editing cache. It is
not saved in the project. Drawing is disabled during graph execution; stop a
continuous run before editing. Numeric fields remain available for exact pixel values.

## Connecting selections

Supported blocks accept optional `mask` image and/or `region` data inputs. A mask
must be uint8, single channel, with the same width/height as the working image;
every nonzero byte selects a pixel. When both inputs are present they intersect.
No selection preserves previous whole-image behavior.

Connect a Shape region's `mask` to **Bitwise images** inputs to combine masks using
AND, OR, XOR or NOT; connect the resulting image to another block's `mask` input.
**Filter points by region** filters a detected point list while retaining global
coordinates. Point membership uses the nearest raster pixel (`np.rint`).

The region data contract is `image_region_v1`, with `image_size: [width,height]`,
`shape`, the corresponding shape fields and `invert`. A size mismatch is an error;
regions are never resized automatically. Use **Transform region** to explicitly
carry a selection into a resized/cropped/rotated/warped image's coordinate system.
Equal dimensions alone do not prove that two images share the same coordinate frame.

## Combine and measure regions

**Combine regions** takes `image`, `region_a` and optional `region_b`. Its
operation dropdown offers:

| Operation | Selected pixels |
| --- | --- |
| union | Inside A or B |
| intersection | Inside both A and B |
| difference | Inside A and outside B (A minus B) |
| xor | Inside exactly one of A and B |
| invert | Outside A; B is unused |

All operations except invert require B. Both regions and every nested child must
match the supplied image dimensions. Outputs are the usual mask, region definition
and tinted overlay. The output `region` can connect to another Combine regions or
any existing region-enabled tool. Holes and disjoint shapes are preserved; edit
the original Shape region blocks to change the geometry. Expressions are bounded
to 16 nested levels and 128 total shapes/operators per rasterization.

**Measure selected region** accepts the normal optional mask and region inputs;
without either it measures the full image. It reports:

- `area`: count of selected pixels, rather than contour-integrated area.
- `coverage`: selected pixel count divided by total image pixel count, in [0,1].
- `bounds`: `[x,y,width,height]` enclosing all selected pixels, or `[]` when empty.
- `centroid`: one `[[x,y]]` point, or `[]` when empty, compatible with point tools.
- `data`: these measurements with explicit pixel units and image dimensions.

The centroid can lie in a hole or between disconnected islands. Minimum/maximum
area and coverage control pass/fail; maximum area zero means unlimited. Failed
inspections retain their measurements for downstream reporting. This block does
not convert area or coordinates into calibrated physical units.

Open [combined regions](<../examples/custom tools/combined_regions/combined_regions.json>)
for a ring selection unioned with a separate rectangle, then measured and used by
histogram, threshold and component tools. The component tool uses `clip` mode:
center filtering would reject a ring whose centroid lies in its excluded hole.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/custom tools/combined_regions/combined_regions.json" --repeat 3
```

## Transform regions between image coordinate systems

**Transform region** accepts a source `region`, a target `image` (used for dimensions
and overlay), and an optional `transform`. Outputs are the transformed region, its
binary mask, a tinted overlay and the actual source-to-target transform.

To follow an image operation, connect its **image** output to Transform region's
**image**, and its **geometry** output to **transform**. Connect the region defined
on the operation's input image to **region**. This works with crop, resize, flip,
90-degree/arbitrary rotation, image borders, pyramids and affine/perspective warps.
For a chain, transform the region after each image operation.

The **Direction** dropdown controls the mapping:

- `forward`: follow geometry from its input image to its output image. Geometry's
  `output_to_input` matrix is inverted automatically. For registration payloads with
  `source_to_target`, the matrix is used directly.
- `inverse`: go from the operation's output coordinates back to its input, or reverse
  a registration matrix. Supply the region from that output frame and the original
  input image as the target.

Geometry must declare `input_size` and `output_size`; both are checked against the
source region and target image. Registration payloads may declare `source_size` and
`target_size`, which are checked when present. A raw 2×3 affine or 3×3 projective
matrix is also accepted; when the port is disconnected, the inspector's **Matrix**
field supplies it. Raw matrices are source-to-target before applying Direction.
Ambiguous mapping keys, nonfinite/singular matrices and inconsistent dimensions
produce configuration/execution errors. Equal dimensions cannot establish frame
identity; connect data from the corresponding image operation.

**Interpolation** offers `nearest` (default, discrete mask samples) and `linear`
(resample coverage then apply **Threshold**, in (0,1]). Outputs always contain
only 0 and 255. Samples beyond the source image are unselected, even when the
source region is inverted. Image border replication/reflection does not create
new selected pixels. Mask resampling follows continuous pixel-center geometry;
its boundaries can differ from the image's interpolation or legacy nearest-resize
index rounding. Reverse transforms cannot recover pixels lost to clipping or sampling.

Composite regions and holes are retained in the expression. Each transform
rasterizes its source selection before sampling, so repeated transformations can
accumulate discretization. They share the expression nesting/node limits above.
Transformed source and target frames are limited to 16384 pixels per axis and
64 million pixels total. The output transform includes source/target dimensions
and connects directly to **Transform points** for associated annotations.

Open [transformed regions](<../examples/custom tools/transformed_regions/transformed_regions.json>) for a 12-block project carrying a circular
inspection region through crop, resize and arbitrary rotation. It measures the
selection, computes its color histogram and inspects the same part in the final
image. The project needs no external images or camera.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/custom tools/transformed_regions/transformed_regions.json" --repeat 3
```

This is image-space affine/projective mapping. Lens-distortion/nonlinear remapping
and calibrated physical-area conversion are not implemented by this block.

## Processing and result policies

| Tools | Selection behavior |
| --- | --- |
| Gaussian (ordinary and G-API), box/median/bilateral filters, Filter2D, unsharp, separable/box filter, gamma, CLAHE, blend | Write processed pixels only inside selection; retain original samples outside when channels match |
| Threshold, automatic/adaptive threshold, InRange, morphology, Canny, gradient and corner response | Write selected outputs and zero outside |
| Automatic threshold | Learn Otsu/Triangle threshold from selected samples only; empty selection is an error |
| Equalize histogram | Learn and apply equalization from selected samples; preserve outside with matching channels |
| Normalize | Pass intersected selection to native normalization; outside is zero in the allocated output |
| Histogram | Count selected samples only, independently per channel; automatic floating range uses those samples |
| Bitwise, absolute difference | Outside output is zero |
| Good features, FAST, ORB | Native detector mask; ORB also filters pyramid keypoint centers against the original mask and keeps descriptors aligned |
| Contours, connected components | `center` keeps full objects whose centroid is selected; `clip` intersects the binary image before extraction, which can split objects and change area |
| Hough circles | Keep circles whose centers are selected |
| Hough line segments | Keep segments when both endpoints are selected; this does not imply the entire segment lies within a concave region |
| Template matching | `region` or `search_mask` selects legal template centers in source coordinates; `mask` remains the separate template-sized native mask |

Image dimensions and coordinates are preserved. Whole-object filtering reports
the complete original object's area and geometry, including pixels outside the
region. Contour offsets affect reported coordinates; region membership is tested
in the original local image before applying that offset.

For filters, derivatives, CLAHE, adaptive threshold and morphology, neighboring
pixels outside the region still contribute to native computations. Selection
controls where results are written; it does not create a cropped border or invent
values for masked neighbors. Changing output depth casts preserved outside samples
to that depth without intensity rescaling. When grayscale output has different
channels from its source, outside output is zero instead of preserving incompatible
color samples. Raw results and their associated graphics use the same selection.

Empty selections give unchanged images for preservation policies, zero images for
zero policies, empty histograms/detections, and normal configured inspection
pass/fail evaluation. Automatic threshold and template search report an execution
error when no samples/candidate centers are available. Template response maps remain
full-sized; only choosing the reported best match is restricted by the search region.

Regions, connections and node layouts persist in saved projects and nested
ToolBlocks. Encapsulation exposes unbound optional mask/region inputs as optional
public terminals; hide unused ports with the inspector's exposure controls.

## Example

Use [regions](<../examples/custom tools/regions/regions.json>). It contains
circle, rectangle and polygon overlays, a one-part component measurement, a
one-part contour inspection, a selected histogram and a circular blur. It runs
without a camera or external assets:

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/custom tools/regions/regions.json" --repeat 3
```

For Lena, open an existing Lena example, add Shape region, connect the source image
to it and use e.g. circle center `[300,300]`, radius `110`. Connect `region` to a
histogram, ORB or filter block to inspect only that part of the source image.

Native references: [drawing functions](https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html)
[feature detection masks](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html), and
[geometric image transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).
