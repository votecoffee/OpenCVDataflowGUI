# Blob, line, contour and circle collections

The toolbox includes **Find blobs** (connected components), **Find contours**,
**Hough line segments**, **Find line segments (LSD)** and **Hough circles**. Each exposes a structured
`collection` output usable by the same filtering, sorting and drawing tools.
Existing detector outputs remain available. Older saved nodes retain their port
exposure choices; expose the new collection output in the inspector if needed.

Find blobs and contour extraction treat nonzero uint8 pixels as foreground; connect
threshold, InRange or other segmentation tools first. Connected components retains
its existing expected-count inspection. Find blobs offers **Check count**, off by
default, so finding any number (including zero) is a valid detection operation.
Both support the existing mask/region and whole-object/clip controls.

Hough segments consume an edge image. LSD consumes a grayscale image directly,
and exposes all native constructor controls: refinement dropdown, detection scale,
sigma scale, quantization, angle tolerance, log epsilon, density and bin count.
LSD coordinates refer to its input image, not its internal scaled working image.
Optional selections keep lines whose endpoints both lie in the selection, matching
the existing Hough selection policy. This does not guarantee an entire segment
lies within a concave region.

**Approximate convex contour** accepts an ordered outline or a contour/blob
collection item. It reduces a convex boundary toward a requested side count
and reports the resulting area. An optional budget limits the area added
over the convex hull. [Convex contour example](<../examples/opencv tools/convex_contour_approximation/README.md>)

## Measurements and conventions

| Criteria | Meaning |
| --- | --- |
| area | Foreground pixel count for blobs, geometric area for contours and fitted disks |
| x, y, width, height | Axis-aligned pixel bounds. Circle bounds use the rounded center and radius. |
| center_x, center_y | Mass centroid for blobs/contours, endpoint midpoint for lines, fitted center for circles |
| aspect_ratio | Axis-aligned width / height |
| angle | Principal mass-axis orientation for blobs/contours, segment orientation for lines; degrees in [0,180), clockwise in image coordinates |
| length | Euclidean endpoint distance for lines; undefined for blobs/contours |
| perimeter | Native closed boundary length including blob holes, or fitted circle circumference |
| major_size, minor_size | Long/short sides of the minimum-area fitted rectangle in continuous coordinates |
| rotated_aspect_ratio | major_size / minor_size when the short side is nonzero |
| extent | area / axis-aligned bounding-box area |
| solidity | area / convex-hull area; blob hull area is rasterized pixel count, contour hull area is geometric |
| circularity | 4π × area / perimeter², without artificial clamping |
| radius | Fitted circle radius in pixels. Center-only detections have no radius. |
| strength | For Hough circles, native support in the selected gradient method. Center-only detections have no strength. |
| distance | Centroid/midpoint distance to the block's configured reference point |
| line_width, precision, nfa | Native LSD width, dimensionless precision parameter, and −log10(NFA) significance when advanced refinement supplies it |
| hu_1 … hu_7 | Native Hu invariants from binary-label moments for blobs or contour moments for contours |
| hu_log_1 … hu_log_7 | −sign(h) × log10(abs(h)); exact zero maps to zero |

An ideal filled 40×20 pixel rectangle has blob area 800, while its border contour
has geometric area 39×19 = 741. Fitted rectangle sides likewise use continuous
boundary distances and may differ from pixel bounding dimensions. These are
intentional measurement definitions. Pixel-area circularity can exceed one for
small raster objects; it is not silently clipped.

Symmetric shapes can have no defined principal-axis angle. Zero-area contours have
undefined Hu moments; lines do not have closed-region area/Hu/perimeter metrics.
Degenerate fitted rectangles may have undefined rotated aspect ratios. Such values
are `null`, never fabricated zeros or NaNs. Raw Hu invariants are approximately
translation/scale/rotation invariant under sampling, not an exact shape identity.
The seventh changes sign under reflection; signed logs can magnify near-zero noise.

Blob measurements and Hu moments include holes in the binary label. Their geometry
retains all boundary loops for exact mask reconstruction. Contour objects follow
the chosen retrieval mode: a hole may be a separate contour object, rather than
being subtracted from the parent contour's native geometric area.

## Filter object collection

Choose the **Metric** dropdown and enable minimum and/or maximum bounds. Both
boundaries are inclusive by default; the inclusion checkboxes provide exclusive
bounds. A disabled bound is unbounded. Equal enabled bounds select an exact scalar
value. **Invert range** selects the complement of the numeric interval; e.g. an
inverted [10,170] angle interval selects near-horizontal undirected lines.

The **Missing** dropdown chooses reject (default), keep, or error for undefined
metrics. It operates independently of numeric inversion. Optional `mask` and
`region` inputs then intersect the numeric result using the object's centroid or
line midpoint; the collection's coordinate origin is handled automatically.
Objects retain their full geometry and measurements; this is not contour clipping.

Outputs are the accepted `collection`, its `count` and centroid/midpoint `points`,
plus a `rejected` collection and `rejected_count`. Relative order and object IDs are
preserved. Chain filters for AND conditions, such as width → height → area → aspect
→ Hu bounds. Connect either accepted or rejected collections to drawing/sorting
blocks to inspect each branch. Empty output is valid and does not itself assert
an expected inspection count.

## Clip line segments

Connect a line collection from a detector. An optional rectangular `region`
sets the clipping window in the source image frame. The block intersects that
window with the image bounds. Without a region, it uses the full image.
The last pixel center on each edge is included.

**Minimum length** applies after clipping. Zero includes point contacts.
The block keeps the object ID and detector strength for clipped spans and
recomputes geometric measurements. Other detector measurements are cleared
for changed spans. Rejected items carry their original geometry and metrics.
The `clipped_count` counts retained spans whose endpoints changed.
The output retains collection coordinates and can feed **Draw object collection**.
[Line clipping example](<../examples/custom tools/clip_line_segments/README.md>)

## Sort object collection

Choose a primary metric and ascending/descending order, with an optional secondary
metric and independent direction. Sorting is stable: ties in both keys keep their
input order. Object IDs are not renumbered. Distance uses the finite `[x,y]`
reference point in the object's reported coordinates.

Undefined sort values go last by default, or first/error via the dropdown. Placement
is independent of ascending/descending direction. Outputs include the reordered
collection, count and points in the same order. The error policy validates all
selected sort criteria, including the optional secondary criterion.

## Draw object collection

Connect the original working image and a filtered/sorted collection. The block
produces an overlay, a selected-object mask and a count. Choose BGR color,
thickness, bounding rectangles, centroid markers, and rank/ID/no labels. Rank labels
start at 1 and show current list order; IDs identify the original detector records
within that run and do not track objects between camera frames.

Blob masks preserve holes. Contour masks fill the individual selected contour
interiors; they do not infer unselected hole/parent relationships. Line masks use
the chosen display stroke thickness, not an inferred physical line width.
Circle masks fill the fitted disk or mark a single pixel in center-only mode.
Reported contour offsets are subtracted for drawing on the local image and retained
in coordinates/measurements. Image dimensions and known coordinate frames must
match; use the matching pyramid level/frame rather than silently rescaling data.

## Collection and coordinate contract

`vision_collection_v1` contains `kind` (blobs/contours/lines/circles), `count`, `items`,
`image_size`, `origin`, and spatial/pyramid `context`. Each item has a stable run-local
`id`, `metrics` and shape geometry. Circle geometry stores a center and radius.
Contour hierarchy references identify
native detector contour IDs, which may not remain present after filtering; they
are not indices into a sorted collection. Blob hierarchy is local to its boundary
loops. Existing legacy outputs remain unchanged in shape.

Collections carry the same optional wire payload as their source image and embed
spatial/pyramid context for standalone JSON use. Filters/sorts preserve it, including
the image's originating pyramid level. All measurements remain in working-image
pixel units. No automatic physical-area, radius or Hu rescaling is applied. Contour
offsets are reflected in the coordinate mapping; `points` can feed the existing
Read payload / Transform points path to the largest/root image.

Filtering/sorting/rendering accept up to 100000 items and images up to 16384 pixels
per axis / 64 million pixels. Geometry is checked before drawing. Collections and
level selections pass through nested ToolBlocks and save/load with normal projects.

## Example

Use [object collections](<../examples/custom tools/object_collections/object_collections.json>). Its 16 blocks threshold synthetic parts, filter
and sort two blobs left-to-right, isolate a round contour by circularity, and select
two long vertical LSD segments by length and angle. Each branch has a ranked
overlay. No camera or optional image download is required.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/custom tools/object_collections/object_collections.json" --repeat 3
```

Native references: [shape descriptors and Hu moments](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html)
and [line segment detector](https://docs.opencv.org/4.x/db/d73/classcv_1_1LineSegmentDetector.html).
The precision/significance outputs are retained from the
[native LSD implementation](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/lsd.cpp).
