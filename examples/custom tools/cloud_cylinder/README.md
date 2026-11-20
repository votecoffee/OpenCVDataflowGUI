# Measure point-cloud cylinder and bore diameters

Open `cloud_cylinder.json` and run once. The two branches fit partial
cylindrical walls from independent PLY scans with noisy wall points and
interleaved clutter. In each block's **Results**, select **Cylinder fit**
to see green inliers, red-orange rejected rows, and the fitted axis.

| Surface | Known diameter | Measured diameter | Observed arc | Inliers |
| --- | ---: | ---: | ---: | ---: |
| Shaft exterior | 42.000 mm | 41.965 mm | 140.1° | 590 of 745 |
| Bore interior | 25.000 mm | 25.029 mm | 103.9° | 584 of 739 |

The shaft scan is in millimeters and has no normals. The block estimates
normals on a bounded subset to seed the fit. The bore scan is in meters and
includes inward-facing normals. An optional fixture pose can seed the
fit with its Z axis and origin. Both branches report geometry in meters
and keep the original units, attributes, and order in the split clouds.

Set the radial inlier tolerance for the scan's noise. The fit requires a
radius greater than twice that tolerance and sufficient angular and axial
support. The reported angle excludes the largest gap in supported sectors.
The axial support is the 2nd to 98th percentile range on the fitted axis.
These ranges describe the observed wall. To inspect a specific cylinder
when several surfaces compete, crop the cloud to the target first.

**Bore** mode signs points toward the bore center as positive residuals,
so a positive error means reduced clearance. **Outer** mode signs points
away from the axis as positive. Optional diameter limits can fail the
inspection while the measured diameter and fitted axis remain available.
For invalid or insufficiently supported fits, the block emits a report
with the failure reason and withholds the geometry.

The example uses original synthetic geometry by OpenCVDataflowGUI
contributors under the project's MIT license. Run
`build_example.py --output <empty directory>` to recreate the point clouds,
the saved graph, and the numerical ground truth.
