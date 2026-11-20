# Depth-backed region reprojection

Open [depth_region_projection.json](depth_region_projection.json) and
run once. A rectangle selects a patch on the source fisheye camera's
depth grid. The source depth, target depth and rigid translation use
millimeters. The block returns axial target camera Z in meters.

The target camera is omnidirectional. A nearer target object obscures
part of the patch. A separate hole has no target measurement. The
**Require Agreement** policy keeps only projections within 12 mm of the
target measurement.

| Result | Count |
| --- | ---: |
| Selected source pixels | 3,723 |
| Valid source depth samples | 3,681 |
| Colliding projections | 1,138 |
| Unique candidate target pixels | 2,543 |
| Occluded target pixels | 132 |
| Unverified target pixels | 11 |
| Retained target pixels | 2,400 |

In Debug the coverage display shows retained pixels in green,
occlusions in red and unverified pixels in blue. Both profiles expose
the component masks. **Measure transferred support** reads the exact
output region on the target grid and reports 2,400 pixels. The
projection assigns the nearest Z to each
sampled target pixel and leaves other pixels empty.

The calibrated camera poses and depth rasters are original synthetic
assets by the OpenCVDataflowGUI contributors (MIT). The automated check
compares the mask and target depths against independent lens equations.
Rebuild into a fresh directory with
`python build_example.py --output <directory>`.
[OpenCV omnidirectional model](https://docs.opencv.org/5.0/main_modules/omnidir_8hpp.html)
