# Watershed separation and seed selection

Open [`watershed.json`](watershed.json) and click **Run once**. The bundled
synthetic image contains two touching parts. Both branches should report two
foreground regions.

- **Watershed seeds from distance peaks** finds seed cores in the thresholded
  foreground mask. Its preview shows the cores and the known background.
- **Watershed seeds from points** uses one foreground point per region. Select
  the block after a Debug run and click **Edit watershed seeds…** to place
  foreground or background points on the captured image. Click **Use points**
  to stage the edits, then **Run block** to preview or **Apply configuration**
  to keep them. You can also wire point lists in image coordinates.
- **Watershed seeds from masks** is available for separate foreground-core and
  known-background masks. The masks must share the same frame and leave
  uncertain pixels clear in both. Each connected foreground core gets a label.

The seed blocks use label `1` for known background, `0` for unknown pixels,
and `2` onward for foreground. Point seeds can use an inset background ring or
explicit background points. Foreground points can be restricted by a mask or
region, and their seed areas must stay separate.

Connect the original image and the seed block's `markers` output to
**Marker-controlled watershed**. Choose **labels** as the marker mode and set
**Background Label** to `1`. Its color preview shows white boundaries and
distinct foreground regions. The raw int32 `labels` output retains region IDs,
and `regions` lists each foreground label and its area in pixels. Use **Output
Label** to isolate one label in the binary `mask`.
[OpenCV watershed tutorial](https://docs.opencv.org/5.0/tutorials/imgproc/imgtrans/distance_transformation/distance_transform.html)

```powershell
opencv-dataflow-run examples/image_processing/watershed/watershed.json
```

[All examples](../../README.md)
