# Hausdorff contour distance

Open [hausdorff_distance.json](hausdorff_distance.json) and choose **Run Once**.
The comparison blocks measure a copy shifted by (3, 4) pixels and an otherwise
identical part with a 10-pixel tab.

| Comparison | Metric and rank | Distance |
| --- | --- | ---: |
| Shifted part | L2, full rank | 5 px |
| Shifted part | L1, full rank | 7 px |
| Part with tab | L2, full rank | 10 px |
| Part with tab | L2, 90% rank | 0 px |

**Compare contours (Hausdorff)** is under **OpenCV → Shape Distance and Matching →
Contour matching**. Connect contour collections or nonempty `[x,y]` point lists
in a common measurement grid. The block keeps their positions, so align and
rescale upstream when needed. Use **Find contours** with approximation **none**
for a dense pixel boundary. Distances use the supplied points, so vertex
sampling affects the measurement.

Each distinct point is matched to its closest point on the other outline.
The block selects zero-based index `floor(rank × (count − 1))` from the
nearest distances in ascending order in each direction and returns the larger
of the two. Use **1.0** to catch the largest deviation. Lower ranks can ignore
small defects, as the tab demonstrates. The directional outputs help show
which outline has the larger deviation.

The block uses exact nearest-point queries and an explicit rank selection.
The OpenCV 5.0.0 extractor sorts a column of distances row by row, which
allows point order to change its result.
[OpenCV 5.0.0 implementation](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/shape/src/haus_dis.cpp)

Enable **Check distance** and leave the maximum at **5** to pass the 5-pixel
measurement and fail the 7- and 10-pixel measurements. The project and its
three silhouettes are original MIT assets. Rebuild from the repository root.

    python "examples/opencv tools/hausdorff_distance/build_example.py" --output "examples/opencv tools/hausdorff_distance"
