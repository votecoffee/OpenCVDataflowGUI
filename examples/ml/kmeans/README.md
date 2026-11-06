# K-means clustering

Open [`kmeans.json`](kmeans.json) and select **Run once**. The project uses a
synthetic image and a built-in table of sample points.

- Select **K-means image pixels**. The thumbnail shows the false-color
  `cluster_map`. Inspect `quantized` to see pixels replaced by their cluster
  centers and `labels` for the integer class IDs. A connected mask or region
  restricts the fit. Outside the selection, labels are `-1`, the color map is
  black, and the quantized image retains the input pixels. Alpha is preserved.
- Select **K-means samples**. The plot colors the samples by cluster and marks
  the numbered centers with crosses. Change the plot axes to inspect other
  features. Enter one row per sample in **Samples** or wire a table to the
  `samples` input.

Both blocks expose the cluster count, initialization, number of attempts, and
termination criteria. An optional `initial_labels` input supplies zero-based
labels for the first attempt. The image block accepts an integer label image in
the same coordinate frame and uses the labels within the selection. Both return
centers, counts, and compactness, the sum of squared distances to the assigned
centers. Samples are fitted in float32 and feature scales affect the result.
[OpenCV 5 K-means implementation](https://github.com/opencv/opencv/blob/5.0.0/modules/core/src/kmeans.cpp)

Both blocks fit new centers on each run. IDs follow the order of the fitted
center values and can shift as the input changes.

Image clustering uses the input's encoded channels. The quantized image keeps
the input color format and pixel type. The block accepts uint8, uint16, float32,
and float64 images up to one million pixels. For a larger image, connect a
smaller pyramid level.

```powershell
opencv-dataflow-run examples/ml/kmeans/kmeans.json
```

[All examples](../../README.md)
