# Trace point-cloud inspection attributes

Open [cloud_inspection_attributes.json](cloud_inspection_attributes.json)
and run it. The upper PCD scan supplies intensity and confidence. **Track
source rows** records the original zero-based file row, so the first valid
point after an invalid XYZ row still carries index 2. The side PLY scan
supplies confidence and explicit source indices.

Eight rows remain after overlap suppression. Filtering confidence from
0.7 through 1.0 retains five rows. The 18 mm voxel sampler keeps three
measured representatives, with the following aligned attributes.

| Source scan | Source index | Intensity | Confidence |
| --- | ---: | ---: | ---: |
| 0 | 2 | 41 | 0.95 |
| 0 | 5 | 55 | 0.75 |
| 1 | 8 | Missing | 0.77 |

In **Measure point cloud**, the results show the field types and the
number of known values. To export the sampled cloud, connect **Save point
cloud** and choose a new PLY or PCD file. Both ASCII and uncompressed binary
carry the four scalar columns. PCD stores intensity and confidence as
64-bit floats and source identifiers as 32-bit integers. Missing values
use NaN for the measurements and −1 for IDs. The same conventions apply to
PLY. [PCD field reference](https://pointclouds.org/documentation/tutorials/pcd_file_format.html)

**Merge registered point-cloud scans** assigns a distinct source namespace
to each contributing scan while preserving its original row IDs. **Filter
point cloud** can keep or exclude missing values. Voxel centroid mode
instead averages known measurements and clears row IDs because it creates
new points.

The scans and [builder](build_example.py) are original OpenCVDataflowGUI
contributions under the repository's MIT license.
