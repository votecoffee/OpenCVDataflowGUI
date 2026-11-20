# Compare organized PCD encodings

Open [organized_pcd.json](organized_pcd.json) and run it. Both branches
render the same stepped coupon with two areas of missing depth. Each file
describes a 64 × 44 scan grid and supplies millimeter coordinates, RGB and
surface normals. The valid points keep their original pixel coordinates,
so the holes stay at the same positions.

To export a cloud, connect **Save point cloud** and choose a new `.pcd`
path. Set **PCD encoding** to **binary** for uncompressed little-endian
records or **ascii** for text rows. Set **PCD organization** to
**source_pixels** to restore the scan grid and fill empty locations with
NaN. This requires unique pixels and a multi-row source grid. The
**unorganized** setting writes one row per surviving point.

The loader accepts scalar XYZ, optional normals and separate or packed
RGB. It ignores alpha and other fields such as descriptor vectors, while
using the header's field widths to decode the records. Nonfinite XYZ and
invalid normals leave grid holes. An acquisition **VIEWPOINT** is
preserved as metadata and does not move the coordinates. External files
without a Dataflow comment use the input block's **Units** and **Frame**
settings. Convert compressed PCD to uncompressed binary or ASCII with
PCL before loading. [PCD format reference](https://pointclouds.org/documentation/tutorials/pcd_file_format.html)

The scans and [builder](build_example.py) are original OpenCVDataflowGUI
contributions under the repository's MIT license.
