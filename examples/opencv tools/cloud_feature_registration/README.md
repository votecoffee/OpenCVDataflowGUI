# Local cloud feature registration

Open [cloud_feature_registration.json](cloud_feature_registration.json)
and run once. The two synthetic scans crop different portions of one
asymmetric surface, with independent noise, dropped samples and clutter.
The graph estimates the source-to-target pose from local geometry.

| Result | Value |
| --- | ---: |
| Reciprocal descriptor matches | 805 |
| Geometrically consistent pairs | 441 |
| Inlier RMS | 0.372 mm |
| Source overlap within 4 mm | 55.6% |
| Target overlap within 4 mm | 57.0% |
| Translation error against known pose | 0.074 mm |
| Rotation error against known pose | 0.0051° |

In Debug, **Registered views** shows the transformed source in orange
and the fixed target in cyan. Both profiles expose the pose, aligned cloud,
inlier indices, residuals and overlap figures. When the input clouds carry
pixel mappings, the corresponding source and target pixels appear in each
inlier record. Increase **Minimum Overlap Fraction** to 0.8 to see the
failure report with the pose withheld.

Use matching m/mm units and comparable point spacing in both views.
Set the neighborhood radius to contain enough points for the selected
neighbor count. The radius, thresholds and report error use meters.
The block supports up to 250,000 points per input. Voxel sampling can
bring denser scans within that limit.

The synthetic PLY assets are by the OpenCVDataflowGUI contributors
(MIT). Rebuild in a fresh directory with
`python build_example.py --output <directory>`.
[SciPy nearest-neighbor query reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html)
