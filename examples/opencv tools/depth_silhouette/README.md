# Calibrated depth silhouette

Open [depth_silhouette.json](depth_silhouette.json) and run once. The graph
fits a datum to a synthetic tilted work surface and measures two raised,
rotated rectangular parts. Each silhouette includes edge pixels where the
depth sample was deliberately removed.

| Part | Known length × width × thickness | Measured length × width × thickness | Missing boundary pixels |
| --- | --- | --- | ---: |
| A | 70 × 35 × 12 mm | 70.86 × 35.91 × 12.00 mm | 36 |
| B | 46 × 24 × 20 mm | 46.90 × 24.72 × 20.00 mm | 21 |

The block reports boundary and face coverage, and supplies a mask of the
unusable boundary pixels. The result drawing shows the fitted rectangle and
flags invalid boundary cells. [Fixture dimensions and missing-pixel coordinates](assets/ground_truth.json)

## Connect your own measurement

Supply aligned camera-Z depth, matching undistorted pinhole intrinsics, a
metric support plane, and a mask for one complete visible top face. Use
**Separate support plane and foreground** to fit the datum from the support
surface. Keep missing depth cells in the silhouette mask so the block can
count them. A mask from an independent optical silhouette can provide that
coverage if it has been registered to the depth grid.

Length and width are the long and short sides of the minimum-area rectangle
on the datum. The block back-projects each usable boundary pixel's
footprint assuming its local surface is parallel to the datum. Thickness
is the median perpendicular height above the datum. These estimates
describe the visible top and its height above the support surface. Parts
with hidden overhangs or a gap under them need additional views or constraints.

The example's dimensions are exact before rasterization. The result retains
up to 1 mm of length or width error from pixelized edges. Use the boundary
and face coverage limits to withhold dimensions on incomplete selections.
By default, the block withholds dimensions when the silhouette touches the
image border.

The original synthetic raster is by the OpenCVDataflowGUI contributors
(MIT). Rebuild it with `python build_example.py --output <new-directory>`.
