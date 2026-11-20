# Separate independent parts in a point cloud

Open [cloud_clustering.json](cloud_clustering.json) and run it. The graph loads
a colored PLY with 37 interleaved scan rows in millimeters. Each of the three
pads has nine measured points 5 mm apart. Separate groups are at least 20 mm
apart. Two stray rows form a small group and a 35 mm edge forms a long group.

**Cluster point cloud by distance** connects pairs within 6 mm and measures
each connected component. The minimum of four points rejects the stray group.
The 30 mm maximum on the longest XYZ axis extent rejects the edge. The output
has three clusters and 27 points. The rejected cloud has ten points. The
optional labels align with the original scan rows and use 0 for rejected
rows. Each cluster record lists its original row indices and metric bounds.
The result preview colors the accepted clusters. Cloud outputs retain their
original RGB colors, normals and frame.

Increasing the radius can join parts by a chain of intervening points. If the
pair budget or the accepted-cluster budget is exceeded, the block fails
without emitting partial labels.

The built-in block uses [SciPy radius pairs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_pairs.html)
and [sparse connected components](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html).
The synthetic PLY and its [builder](build_example.py) are original
OpenCVDataflowGUI assets under the repository's MIT license.
