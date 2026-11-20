# Triangle mesh validity and measurement

Open `mesh_validity.json` and run the graph. Inspect the output report and
**Topology** image on each of the three inspection blocks. Use the Debug profile
to see the wireframes. Orange highlights free edges, purple highlights
inconsistent winding, and red highlights an edge shared by more than two
triangles. The report gives exact counts and sampled edge and face indices.

| Fixture | Expected outcome | Measurement |
| --- | --- | --- |
| Closed 80 × 40 × 20 mm box | Pass, outward winding | Area 11,200 mm², volume 64,000 mm³ |
| Box with its lid removed | Fail, four free edges form one loop | Area 8,000 mm² |
| Closed box plus a flap attached to an existing edge | Fail, one edge has three incident faces | Volume unavailable |

The two deliberately damaged controls make the headless run return a failed
inspection. Enable **Require volume** when the downstream decision depends on
a confirmed volume. Volume qualification currently covers one closed,
consistently wound convex shell. It checks face support planes and compares
surface area and signed volume against the
[SciPy convex hull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html).
The area output sums the input triangles, including any reported defects.

The PLY fixtures are original synthetic geometry by OpenCVDataflowGUI
contributors under the repository's MIT license. Run `build_example.py --output
<empty directory>` to regenerate the fixtures and project.
