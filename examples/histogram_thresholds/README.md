# Histogram percentiles and adaptive thresholds

Open [`histogram_thresholds.json`](histogram_thresholds.json) and click **Run
once**. The bundled grayscale ramp has more dark pixels than bright pixels, so
its mean and median differ.

The **Histogram** marks the 20th and 80th percentiles, mean, and median, and
shades the interval between the cutoffs. Inspect the three output masks to see
the lower tail, middle interval, and upper tail. The lower tail includes its
cutoff, the middle interval includes both bounds, and the upper tail selects
values strictly above its cutoff. Tied pixel values can change the selected
fraction or appear in two masks at a shared bound.

For your own graph:

- Expose the Histogram's `threshold` output and **Threshold**'s `value` input
  to use the selected statistic. Choose a percentile, mean, or median in
  **Threshold Statistic**.
- Expose `lower_threshold` and `upper_threshold` on **Histogram**, and wire
  them to the `lower` and `upper` inputs of **In range mask** for the middle
  band. Wired integer ranges round the lower bound up and the upper bound down
  to preserve the inclusive interval.
- For color images, the histogram data includes statistics for every measured
  channel. Choose **Statistics Channel** for scalar outputs and plot markers.
  Measure and threshold the same channel or grayscale plane.

Statistics use the finite pixels inside the connected mask or region, including
values outside a manually chosen plotting range. Percentiles use linear
interpolation. Each channel reports its minimum, maximum, mean, median, and
population standard deviation alongside the cutoffs. When a selection has no
finite samples, scalar outputs emit no value and wired threshold inputs wait
for a measurement. [NumPy percentile reference](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)

```powershell
opencv-dataflow-run examples/histogram_thresholds/histogram_thresholds.json
```

[All examples](../README.md)
