# Compare OBJ and PLY point-cloud interchange

Open [cloud_obj_interchange.json](cloud_obj_interchange.json) and run it.
The two branches load equivalent OBJ and PLY scans of a folded coupon.
Both contain the same 12 point rows, RGB colors, and unit normals.
The file metadata supplies millimeters and the `folded_coupon` frame.
Each measurement block reports the matching centroid and extents.

The **Load point cloud** and **Save point cloud** tools select OBJ by
the `.obj` extension. An OBJ cloud has XYZ vertices in source order,
optional normalized inline RGB on every vertex, and optional `vn`
rows in the same order as the vertices. The loader normalizes normals.
For external OBJ files without a Dataflow metadata comment, use the
load block's Units and Frame settings. The save block writes ASCII OBJ
and requires a new destination.

This interchange handles point rows. Files with faces, lines, or free-form
surfaces require a mesh workflow. Image pixel mappings and arbitrary
metadata stay with the in-memory cloud. [OpenCV point-cloud API](https://docs.opencv.org/5.0/main_modules/ptcloud.html)

The app codec checks record shape and attribute alignment. Its writer
uses enough digits for float32 XYZ round trips. Tests also exchange
aligned OBJ and PLY attributes with OpenCV's native I/O.

The assets and [builder](build_example.py) are original OpenCVDataflowGUI
contributions under the repository's MIT license.
