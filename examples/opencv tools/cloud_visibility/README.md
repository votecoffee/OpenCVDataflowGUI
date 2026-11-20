# Calibrated point-cloud visibility

Open [cloud_visibility.json](cloud_visibility.json) and click **Run once**.
The project renders overlapping samples onto an 80 × 64 pinhole image and
checks the predicted surface against observed camera Z. The assets are
original and synthetic. The expected depth and visibility PNGs come from
the fixture geometry independently of the projection block.

The expected nearest depth, coverage, and masks match across the full grid.
The cloud covers **2,508 pixels**. Of its samples, **2,370** agree with the
reference, **560** are hidden behind a nearer modeled sample, and **48**
are hidden behind a closer measured surface. A sensor hole leaves **45**
unverified, and an observed surface farther than predicted leaves another
**45** unverified.

## Use your own cloud

Connect a point cloud in meters or millimeters and the target pinhole
intrinsics. Connect a rigid cloud-to-camera transform when frame names
differ. Select a camera Z range and optional target mask or region. The
block rounds projections to the nearest pixel center and keeps the
nearest sample there. Equal-depth samples use the earliest input index
for raster color. Uncovered depth pixels are NaN, and the coverage mask
identifies all pixels with projected depth.

The projected depth and RGB are predictions from the cloud. Connect
`reference_depth` to compare with observed camera Z on the target grid.
Points near the nearest modeled surface are visible when no reference
is connected. With a reference connected, visible samples must also
agree within its tolerance. Samples behind the reference are occluded.
Missing measurements and predictions in front of the measured surface
are unverified. More than one status can occur at the same pixel.
In the result image, green means visible, red means occluded and blue
means unverified. Overlapping statuses combine those colors.

Each output cloud retains the original input frame, units and attributes.
Raster outputs use the target grid. The score is the visible fraction of
in-frame samples inside the selection and Z range. A pass indicates that
the projection has coverage, including cases where visibility is zero.
Each raster pixel represents a point sample. Sparse clouds can leave gaps.

The [fixture builder](build_example.py) regenerates the cloud and PNGs.
Visibility labels use 0 for uncovered, 1 for measured agreement, 2 for a
closer measured occluder, 3 for missing depth, and 4 for a farther
measurement. Bit 4 (value 16) marks self-occlusion at that pixel.

Original assets by OpenCVDataflowGUI contributors under [MIT](../../../LICENSE).
[OpenCV camera projection](https://docs.opencv.org/doc/doxygen/html/d4/d93/group__calib.html).
