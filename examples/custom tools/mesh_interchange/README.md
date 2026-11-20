# Exchange triangle meshes in OBJ and PLY

Open [mesh_interchange.json](mesh_interchange.json) and run it. Both branches
load the same folded panel. Each reports six vertices, four triangles, RGB
colors, normals, and two-component texture coordinates. Coordinates are in
millimeters in the `folded_panel` frame.

**Load triangle mesh** accepts triangulated ASCII OBJ and PLY with scalar
vertex attributes. PLY imports can be ASCII, binary little endian, or
binary big endian. OBJ face corners can index normals and texture
coordinates separately. At seams, the loader splits vertices so the
result retains each corner's attributes and triangle winding.

**Save triangle mesh** selects ASCII OBJ or binary PLY from the
destination extension and publishes to a new filename. Both formats
preserve units and frame through Dataflow comments. External files can
supply units and frame in the loader settings. Vertex RGB uses normalized
inline values in OBJ and byte channels in PLY. The supported texture
fields carry coordinates for each vertex.

OpenCV native interoperability was checked on aligned attributed meshes.
The pinned native writer required three-component texture coordinates in
the test, while the app preserves the input coordinate width. [OpenCV mesh API](https://docs.opencv.org/5.0/main_modules/ptcloud.html)

The assets and [builder](build_example.py) are original OpenCVDataflowGUI
contributions under the repository's MIT license.
