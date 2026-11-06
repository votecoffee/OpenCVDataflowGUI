# Earth Mover's Distance

Open [earth_movers_distance.json](earth_movers_distance.json) with
**File → Open…**, then click **Run once**. The sample has the same relative
proportions as the reference, with each gray level increased by seven and
a larger pixel count. The normalized distance is **7 sample levels**.

In **Inspection → Measurements → Compare histograms**, choose
**Earth movers distance** and connect two **Histogram** outputs. The
block measures the average distance that intensity mass must move between
occupied bin centers. The histograms must have matching channel names and
can use different bin edges. Use comparable intensity units on both inputs.

**Normalize** is on by default and compares relative counts. With it off,
each channel must have equal total counts. **Reduction** returns the mean
or largest channel distance. **Check threshold** passes when the selected
distance is at or below the limit. The block accepts up to 512 occupied
bins per channel, and the native solver uses single-precision signatures.
[OpenCV EMD reference](https://docs.opencv.org/5.0/main_modules/imgproc_hist.html#emd)

```powershell
opencv-dataflow-run examples/inspection/earth_movers_distance/earth_movers_distance.json
```

[All examples](../../README.md)
