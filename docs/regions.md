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

Fields are in full input-image pixel coordinates. Shapes are configured in the
inspector; direct mouse drawing/editing is future work.

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
regions are never resized automatically. Regenerate regions for resized/cropped
images and transform their coordinates explicitly for rotated/warped images.
Equal dimensions alone do not prove that two images share the same coordinate frame.

## Processing and result policies

| Tools | Selection behavior |
| --- | --- |
| Gaussian G-API, box/median/bilateral filters, Filter2D, unsharp, separable/box filter, gamma, CLAHE, blend | Write processed pixels only inside selection; retain original samples outside when channels match |
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

Use **File > Load regions example** or open `examples/regions.json`. It contains
circle, rectangle and polygon overlays, a one-part component measurement, a
one-part contour inspection, a selected histogram and a circular blur. It runs
without a camera or external assets:

```powershell
.\.venv\Scripts\opencv-dataflow-run examples/regions.json --repeat 3
```

For Lena, open an existing Lena example, add Shape region, connect the source image
to it and use e.g. circle center `[300,300]`, radius `110`. Connect `region` to a
histogram, ORB or filter block to inspect only that part of the source image.

Native references: [drawing functions](https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html)
and [feature detection masks](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html).
