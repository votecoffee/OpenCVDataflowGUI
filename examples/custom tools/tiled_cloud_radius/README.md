# Radius filtering for a large binary PLY scan

Open [tiled_cloud_radius.json](tiled_cloud_radius.json) and run it. The scan
has eight original points, including two coincident rows and two isolated
points. The search radius is 125 mm. A point exactly that far away counts
as a neighbor, including across negative and positive tile boundaries.
The report should retain six rows and reject two.

Use **Output path** to save kept rows to a new binary PLY. The example
preserves source IDs 31, 9, 89, 12, 14 and 22 in that order, along with RGB
and other stored scalar columns. For scans with no `source_index`, the tool
adds a zero-based file row index. Nonfinite XYZ rows are excluded and
counted separately in the report.

The tool reads vertex-only, little-endian binary PLY files up to 256 MiB.
It builds a temporary SQLite index on disk, then streams occupied cubic
tiles. Each radius halo includes all possible neighbors of the core rows.
One bounded local tree counts neighbors and excludes the query row.
**Tile size** must be at least twice the radius. The block removes its
temporary index after the scan and publishes the optional export only
after successful completion. [SciPy radius count reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html)

The [builder](build_example.py) can generate a large analytic grid in a
new directory. From the repository root, with the app environment active:

```powershell
python "examples/custom tools/tiled_cloud_radius/build_example.py" --output "PATH_TO_NEW_EXAMPLE" --grid 144 144 100
```

The large project has 2,073,601 input rows, including one isolated point.
It retains 2,073,592 rows. The eight grid corners and isolated point fall
below four neighbors. The largest halo contains 1,331 points.

The builder and all data are original OpenCVDataflowGUI contributions
under the repository's MIT license.
