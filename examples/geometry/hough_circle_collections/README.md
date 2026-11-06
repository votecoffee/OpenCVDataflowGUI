# Hough circle collections

Open [hough_circle_collections.json](hough_circle_collections.json) with
**File → Open…** and click **Run once**. The project finds two synthetic disks,
ranks them by **Strength**, and displays both with rank labels. A second branch
keeps only the disk with a fitted radius of at least 24 pixels. Its drawing
block also produces a filled mask of that disk.

Find **Hough circles** under **Inspection → Features**. An optional mask or
region selects detections by center. The existing **Circles** and **Centers**
ports report `[x, y, radius]` and `[x, y]` rows. **Votes** follows the same
order and gives the native support value for each fitted circle. Use
**Collection** to filter or sort by **Radius** and **Strength**.

Gradient and Gradient ALT measure support differently, so set strength
limits for the chosen method. To try Gradient ALT on this example, set
**Method** to **gradient_alt**, **Edge threshold** to 300, and
**Center threshold** to 0.9. To locate centers in Gradient mode, set
**Max radius** to -1. The legacy circle rows then have zero radius, and
the new vote and collection radius values are null.
[OpenCV Hough circles reference](https://docs.opencv.org/5.0/main_modules/imgproc_feature.html#houghcircles)

```powershell
opencv-dataflow-run examples/geometry/hough_circle_collections/hough_circle_collections.json
```

[All examples](../../README.md)
